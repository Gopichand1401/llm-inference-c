# RAG: answering questions from Excel and CSV files

Retrieval-augmented generation (RAG) lets the engine answer questions about
a folder of spreadsheets, such as the institute's course timetables in
`data/`. The relevant rows are found first, then either printed directly
or handed to the language model, which writes the answer.

```
$ ./build/llm_rag --retrieve-only
> Who is the instructor for ML?
Swapnil Hingmire
Source: 04_UG_DS_Aug-Dec-2026.xlsx, tab Core, row 10 (B. Tech. Data Science Core Courses)
Do you want the slot?
> yes
D [Mon,Wed], E [Fri] + PA5
Do you want the room?
> and the mail id
swapnilh@iitpkd.ac.in
> How many courses are in slot D?
15 found for d:
  1. DS3010 - Introduction to Machine Learning  (04_UG_DS_Aug-Dec-2026.xlsx, Core, row 10)
  ...
```
*(Real output on the files in `data/`.)*

## 1. Three modes, one program

| Command | What answers |
|---|---|
| `llm_rag --retrieve-only` | the rows themselves; no model is loaded, so it is instant and exact |
| `llm_rag <model.gguf> --rag` | the best rows go into the prompt and the chat model (TinyLlama) writes the answer, with the sources printed under it |
| `llm_rag <model file>` | the model alone: chat for GGUF models, text completion for llama2.c models |

**Recommended for `--rag`:** `-k 1 --compact`. Then the model gets only
the best row, and of that row only the code, the name and the columns the
question asks about. See §5 for why this matters for speed.

`--docs PATH` chooses the folders or files (default `data/`; repeatable).
The folders are checked before every question, so a file that is added,
edited or deleted while the program runs is picked up at once. All options
are listed at the top of `examples/09_rag/llm_rag.c`. `rag_inspect` shows
each stage separately (`files`, `chunks`, `tokens`, `resolve`, `search`).

## 2. The pipeline

```
 .xlsx / .csv files
        │  xlsx_reader.c      ZIP + XML → clean cell text (RAG step 1)
        ▼
 rag_chunk.c   find each table's header row and title; one chunk per data
               row (rows of the same item merged), notes kept separately
        │                                                    (step 2)
        ▼
 rag_index.c   terms (lower-case, stop words, light stemming, joined codes
               "DS 3010" → ds3010, short forms ML → ~ml), BM25 index;
               typo correction within 1–2 edits                (step 3)
        │
        ├──► rag_answer.c   --retrieve-only: answer straight from the rows
        │                   (lookup, notes, lists/counts, follow-ups, offers)
        │
        └──► rag_prompt.c   --rag: render the best rows into a TinyLlama chat
                            prompt that fits the context window → model
                                                               (step 4)
```

| Module | Header | Role |
|---|---|---|
| `xlsx_reader.c` | `xlsx_reader.h` | reads `.xlsx` workbooks (only zlib's `inflate` is external) |
| `rag_chunk.c` | `rag_chunk.h` | folder scan, CSV parsing, header and title detection, row and note chunks, change detection |
| `rag_index.c` | `rag_index.h` | tokenisation, BM25 index and search, short-form and typo resolution |
| `rag_answer.c` | `rag_answer.h` | the no-model question answerer and its conversation state |
| `rag_prompt.c` | `rag_prompt.h` | prompt construction and context budget for `--rag` |
| `pocket.c` | `pocket.h` | the same features as plain library calls; see [POCKETLLM.md](POCKETLLM.md) |

None of these modules knows about courses. They work from each table's
header row, its first two "key" columns and the column labels, so any
spreadsheet works.

## 3. Algorithms

**Header detection** (`rag_chunk.h`). In each tab, the header is the first
of the top 30 non-empty rows that has at least 2 non-empty cells, is at
least half as wide as the widest row, and has at least 60% non-numeric
cells. The last non-empty row above it is the table title. Labels are
shortened ("Availability (Degree: …)" → "Availability"). Consecutive rows
with the same first-column value (one course split into batches) are
merged into one chunk, keeping (slot, room) pairs together.

**Terms** (`rag_index.h`). Text is lower-cased and split into runs of
letters and digits. Stop words ("the", "who", "teach" …) are dropped, and
a light stemmer handles plurals (courses → course, labs → lab). A letter
part followed by a digit part is also joined, so `DS3010`, `DS-3010` and
`ds 3010` all match. The initials of the words in key columns are indexed
as short forms: "Introduction to Machine Learning" → `~ml`, and "Theory of
Computation" → `~toc` and `~tc`.

**Resolving a question word.** Each word is tried in turn as: an exact
index term; a short form (2–6 letters); a typo correction (the closest term
within 1 edit, or 2 edits from 8 letters on; never for words containing
digits, so DS3010 cannot turn into DS3030); otherwise unknown.

**Ranking: Okapi BM25** with two re-ranking rules:
```
idf(t)     = ln(1 + (N − df + 0.5) / (df + 0.5))
bm25(q, d) = Σ_t  w(t) · idf(t) · tf·(k1 + 1) / (tf + k1·(1 − b + b·len/avg_len))
k1 = 1.2, b = 0.75; key columns count 3× in tf and len
w(t) = 0.3 if t is a word of some column label ("instructor", "slot"): it
       says which column is wanted, not which row; otherwise 1
name bonus: + Σ idf for every key field whose terms are all in the question
```

**Answering without a model** (`rag_answer.h`). Each question term is
classified against a candidate row: key column, column label, other cell,
title/file name, filler, or missing. The rules are then tried in order:
1. a **follow-up** that only names columns ("and its room?") uses the
   previous row;
2. a **lookup**: rows where every term is a key term or a label, with the
   best coverage of the key column winning (ties give a numbered choice);
3. a **note** containing every term;
4. a **list or count** ("which courses does X teach", "how many are in
   slot D");
5. a **loose match** by BM25;
6. otherwise "Sorry, I couldn't find …", plus an offer to list related
   rows.

Only the asked-for cells are printed, with a `Source:` line and an offer of
one more column ("Do you want the slot?").

**Prompt for the model** (`rag_prompt.h`). The system instruction ("answer
using only the rows given; copy names, codes, times and rooms exactly …")
is fed once and kept in the KV cache. Each question then adds only the
retrieved rows and the question. In each row, the key fields come first,
then the fields the question asks about. If the prompt does not fit the
context window, the lowest-ranked row is dropped, then long values are
shortened. If nothing scores above `--min-score`, the program replies "I
could not find this in the files." without running the model at all.

## 4. Tests

`make rag-test` (`tests/run_rag_checks.sh`) runs 35 checks; all pass:

| Check | What it proves |
|---|---|
| C vs Python reference | `tests/rag_reference.py` re-implements chunking, terms and BM25 independently; files, chunks (185 on `data/`), terms, typo/short-form resolution and every score match byte for byte |
| retrieval set | 35 of 35 questions in `tests/rag_questions.tsv` find the right row; off-topic questions find nothing |
| conversations | 131 checks in 14 scripted conversations (`tests/rag_dialogues.txt`): answers, sources, offers, lists, counts, typos, short forms, CSV |
| live folder | a file added, edited, broken, locked or deleted during a session is handled at the next question |
| the three modes | end to end with tiny generated models, under AddressSanitizer + UBSan |
| context budget | rows dropped to fit; a clear error when even one row cannot fit |
| `--compact` | the prompt keeps the key and asked-for columns (slot, room) and drops the rest |
| rows for the model | with `-k 1`, the full DS3010 row is chosen over the CSE sheet's "Check DS sheet" stub |
| mutation test | 35 bugs planted in `rag_chunk.c` / `rag_index.c`, every one caught |

`make pocket-test` checks that the Python package gives exactly the
same text as the C programs (16 checks, also under the sanitizers).

**Bugs found during the merge and fixed:**

1. A count or list question asked
right after an answer ("How many courses are in slot D?" after the slot of
DS3010 was shown) was treated as a follow-up, so it printed the previous
row's slot instead of the 15 courses. Count and list questions, and
questions with a plural noun such as "courses", now skip the follow-up
rule. A regression conversation was added to `rag_dialogues.txt`; the
old code fails it and the fixed code passes.
2. With `--rag -k 1`, the model was given the **wrong row** for DS3010: the
   CSE sheet's stub ("Credits: Check DS sheet for course details") instead
   of the full row in the DS sheet, so it rightly answered that it had no
   instructor information. `rag_order_duplicates()` (which puts the
   fullest row of a course first) ran only after the hits were cut to `k`.
   With `k = 1` there was nothing left to reorder. It now runs on all
   strong candidates before the cut (`select_rows()` in `pocket.c`). A
   check in `run_rag_checks.sh` covers it; the old code fails it.

**Tip for `--rag`:** `-k 1` (only the best row) usually gives TinyLlama
the best answers, because a 1.1B model easily mixes up several similar rows.

## 5. Speed of `--rag`

The engine processes the prompt one token at a time, at about the same speed
as it generates (about 8 tokens/s for TinyLlama on 8 threads; see
`PERFORMANCE.md`). The time per question is therefore roughly
(prompt tokens + answer tokens) ÷ 8. The system instruction (57 tokens)
is processed only once and stays in the KV cache.

Measured prompt sizes for three typical questions (TinyLlama tokenizer):

| Question | default prompt | `--compact` |
|---|---:|---:|
| Who teaches DS3010? | 257 tokens | 125 tokens |
| When and where is the Introduction to Machine Learning class? | 258 | 136 |
| Who is the instructor for C Programming? | 279 | 149 |
| **total** | **794** | **410 (−48%)** |

With `-k 3` (the default) the prompt is about three times longer again.
Long answers cost time too: TinyLlama often repeats every column, so use
`-n 80` to cap the answer. If the model file is read from a Windows or
OneDrive folder through WSL (`/mnt/c/...`), the OS may re-read weight pages
from disk during a session, which can make it several times slower. Copy
the model to the Linux file system (`~/models`).

The real fix is **batched prompt processing**: run all the prompt tokens
through each layer together as one matrix-matrix product, so each weight is
read once per prompt instead of once per token. That is listed as future
work in `PERFORMANCE.md`.

**Data quality matters as much as the model.** In the DS sheet the
instructor column is labelled "Primay instructor email ID" (a typo), and
TinyLlama failed to connect "who teaches" to it. For the C Programming row,
labelled "Primary Instructor Email ID", it answered correctly.

## 6. Limitations

- Retrieval is by keywords (BM25), not embeddings: the question must share
  words, codes, short forms or near-typos with the rows. Synonyms the
  files never use are not found.
- With `--rag`, a 1.1B model can copy facts from the rows but cannot
  reliably count or reason over many rows, so lists and counts are best
  answered in `--retrieve-only` mode.
- `.xlsx` and `.csv` only (no `.xls`, PDF or Word files).
- The `--rag` prompt uses the TinyLlama/Zephyr chat format.
