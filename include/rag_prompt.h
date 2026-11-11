#ifndef RAG_PROMPT_H
#define RAG_PROMPT_H

/*
 * rag_prompt.h - turn retrieved chunks into a TinyLlama chat prompt that
 *                fits the context window (RAG Step 4: context budget +
 *                prompt construction).
 *
 * The prompt is built in two parts so the first can stay in the KV cache:
 *
 *   rag_prompt_system()    "<|system|>\n{instruction}</s>"
 *                          fed once, with BOS, at position 0
 *   rag_prompt_question()  "\n<|user|>\nSpreadsheet rows:\n\n[1] ...\n\n"
 *                          "Question: {question}</s>\n<|assistant|>\n"
 *                          fed after it for every question
 *
 * "</s>" is a special token, so the split gives exactly the same tokens as
 * the one-piece chat format of examples/01_apps/llm_generate.c.
 *
 * A chunk is written as
 *   [1] 04_UG_DS_Aug-Dec-2026.xlsx, tab Core, row 10 (B. Tech. Data Science Core Courses)
 *   Code: DS3010
 *   Course Name: Introduction to Machine Learning
 *   Primay instructor email ID: Swapnil Hingmire swapnilh@iitpkd.ac.in
 *   ...
 * Key fields come first, then the fields the question asks about (a
 * question word appears in the label, or a question word such as "who",
 * "teach", "when", "where" maps to a label word such as "instructor",
 * "slot", "room"), then the rest in sheet order. Merged rows list their
 * differing fields one line per row. Values longer than max_value_bytes
 * are cut with "...".
 *
 * Context budget: if the prompt has more than max_tokens tokens, the
 * lowest-ranked chunk is dropped and the prompt rebuilt; with one chunk
 * left, long values are cut shorter (150, 80, 40 bytes) before giving up.
 */

#include "rag_chunk.h"
#include "rag_index.h"

#include <stddef.h>

/* Token count of text (no BOS), or -1 on error. */
typedef int (*RagCountTokens)(const char *text, void *user);

typedef struct {
    const char *system;    /* NULL: RAG_DEFAULT_SYSTEM */
    int max_tokens;        /* budget for rag_prompt_question(); required */
    int max_value_bytes;   /* 0: 300 */
    int compact;           /* 1: per row only the key fields and the fields
                              the question asks about (all fields when the
                              question names none); a much shorter prompt */
} RagPromptOptions;

#define RAG_DEFAULT_SYSTEM \
    "You answer questions using only the spreadsheet rows given by the " \
    "user. Copy names, codes, times and rooms exactly as written. If " \
    "several rows match, mention each of them. If the rows do not contain " \
    "the answer, reply: I could not find this in the files."

/* "<|system|>\n...</s>" (malloc'd), or NULL when out of memory. */
char *rag_prompt_system(const RagPromptOptions *options);

/*
 * The question part for hits[0 .. n_hits) (best first). On success returns
 * the malloc'd text and sets *n_used (chunks that fit, >= 1) and
 * *n_tokens. NULL with a message if even one chunk cannot fit.
 */
char *rag_prompt_question(const RagCorpus *corpus, const RagHit *hits,
                          int n_hits, const char *question,
                          const RagPromptOptions *options,
                          RagCountTokens count, void *count_user,
                          int *n_used, int *n_tokens,
                          char *error, size_t error_capacity);

/*
 * Rows of the same thing in several files (the same value in the first key
 * field, e.g. a course code listed by two departments): within each such
 * group, move the row with the most fields to the group's best position,
 * so a short "see the other sheet" row never pushes out the full one.
 * Other hits keep their places. Works in place on hits[0 .. n).
 */
void rag_order_duplicates(const RagCorpus *corpus, RagHit *hits, int n);

/*
 * One chunk as prompt text (number = its [n] label), with the field order
 * described above. malloc'd; NULL when out of memory.
 */
char *rag_render_chunk(const RagChunk *chunk, const char *question,
                       int number, int max_value_bytes);

#endif
