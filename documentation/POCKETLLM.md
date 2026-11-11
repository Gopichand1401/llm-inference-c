# PocketLLM: the library interface and Python package

*Part of the project; see the main [README](../README.md). The RAG pipeline
behind `Rag` is described in [RAG.md](RAG.md).*

A LLaMA-style inference engine written in C, with question answering over
Excel and CSV files (RAG), usable from the command line, from C and from
Python.

```python
import pocketllm

model = pocketllm.Model("models/gguf/tinyllama.gguf")
print(model.generate("My name is", sampler="top-k", temperature=0.7))

rag = pocketllm.Rag("data")
answer = rag.ask("Who is the instructor for ML?")
print(answer.text)      # Swapnil Hingmire
print(answer.source)    # 04_UG_DS_Aug-Dec-2026.xlsx, tab Core, row 10 (...)
```

## 1. Install

Works on Linux and on Windows through WSL (Ubuntu).

```bash
sudo apt install build-essential zlib1g-dev python3-pip
cd llm-inference-c            # the folder with the Makefile and setup.py
pip install .                 # newer Ubuntu: pip install --break-system-packages .
```

`pip install .` compiles the C code and installs the `pocketllm` package.
After that `import pocketllm` works in any Python file or notebook on that
computer. If it reports a missing `Python.h`, run
`sudo apt install python3-dev` and install again.

The model files are not part of the package. Download a model and pass its
path, for example `models/gguf/tinyllama.gguf`.

To use the package from the project folder without installing it:

```bash
make lib
export POCKETLLM_LIB=$PWD/build/libpocketllm.so PYTHONPATH=$PWD/python
```

## 2. Models

```python
model = pocketllm.Model("models/gguf/tinyllama.gguf")      # a chat model
story = pocketllm.Model("models/stories42M.bin")           # a story model
```

| Argument      | Meaning                                                           |
|---------------|-------------------------------------------------------------------|
| `path`        | a llama2.c `.bin` file or a GGUF file                             |
| `tokenizer`   | `tokenizer.bin` for `.bin` models (default: next to the model)    |
| `context`     | positions to allocate (default: the model's, at most 4096)        |
| `chat_format` | `"zephyr"` (default for GGUF), `"llama2"`, `"mistral"`, `"none"`  |
| `system`      | system message for `chat()`                                       |
| `seed`        | random seed; the same seed gives the same text                    |

### Text completion

```python
text = model.generate("Once upon a time",
                      sampler="top-k",      # "greedy", "temperature" or "top-k"
                      temperature=0.7,
                      top_k=40,
                      max_tokens=256,
                      seed=1)
```

`generate()` returns the continuation without the prompt.
`stream=True` also prints the text while it is produced, and
`on_token=function` calls your function with every new piece.

### Chat

```python
print(model.chat("What is machine learning?", sampler="greedy"))
print(model.chat("Give one example."))     # the first turn is remembered
model.reset()                               # start a new conversation
model.chat()                                # interactive: type, /reset, /exit
```

`model.stats` gives the token counts and timings of the last call.

### Which models load

| Model                                             | Status                         |
|---------------------------------------------------|--------------------------------|
| stories15M / 42M / 110M (`.bin`)                  | tested                         |
| TinyLlama 1.1B chat (GGUF)                        | tested                         |
| Llama 2, Mistral 7B, Zephyr 7B (GGUF)             | same format, not tested; slow on a laptop |
| Llama 3 and newer, Phi, Gemma, Qwen               | not supported (other tokenizer or architecture) |

GGUF weights may be F32, F16, Q4_0, Q8_0, Q4_K, Q5_K or Q6_K. A file that
cannot be loaded raises `pocketllm.PocketError` with the reason.

## 3. Answers from Excel and CSV files

```python
rag = pocketllm.Rag("data")          # a folder, a file, or several of them
```

Every `.xlsx` and `.csv` in the folder is read. Files added, edited or
deleted later are picked up before the next question.

### Straight from the rows (no model)

```python
a = rag.ask("Who is the instructor for ML?")
a.text      # 'Swapnil Hingmire'
a.source    # '04_UG_DS_Aug-Dec-2026.xlsx, tab Core, row 10 (...)'
a.offer     # 'Do you want the slot?'
a.found     # True

rag.ask("yes").text                         # 'D [Mon,Wed], E [Fri] + PA5'
rag.ask("and the mail id").text             # 'swapnilh@iitpkd.ac.in'
rag.ask("How many courses are in slot D?")  # a counted list
print(rag.ask("slot and room for Discrete Mathematics"))   # the whole reply
```

The conversation is remembered inside the `rag` object; `rag.reset()`
forgets it. `rag.chat()` starts an interactive session.

### With a model writing the answer (RAG)

```python
a = rag.ask("Who teaches DS3010?", model=model, sampler="greedy", max_tokens=160)
a.text      # the sentence written by the model
a.source    # the rows that were given to it
```

The best `k` rows (default 3) are put into the prompt and the chat model
writes the answer. `compact=True` gives the model only the key columns and
the columns the question asks about, which halves the prompt and makes the
answer faster; `k=1, compact=True` is the fastest and usually the most
accurate with TinyLlama. This needs a chat model; a story model is refused.
The prompt uses the TinyLlama / Zephyr chat format.

### Search only

```python
for row in rag.search("gpu programming", k=3):
    print(row.score, row.source)
    print(row.text)
```

`rag.files` lists what is loaded, `rag.warnings` what was skipped.

## 4. Command line

```bash
make && make rag-tools
./build/llm_rag --retrieve-only                       # answers from the files
./build/llm_rag models/gguf/tinyllama.gguf --rag      # files + model
./build/llm_rag models/gguf/tinyllama.gguf            # the model alone
./build/llm_rag models/stories42M.bin                 # text completion
```

## 5. From C or another language

`make lib` builds `build/libpocketllm.so`. Its functions are declared in
`include/pocket.h`; the Python package is a thin wrapper around them, and a
wrapper for another language can be written the same way.

```c
#include "pocket.h"

char error[1024];
PocketModel *m = pocket_model_open("tinyllama.gguf", NULL, 0, error, sizeof(error));
PocketGenOptions o = {"top-k", 0.7f, 40, 128};
puts(pocket_model_chat(m, "What is RAG?", &o, NULL, NULL));
pocket_model_close(m);
```

```bash
gcc -Iinclude my_program.c -Lbuild -lpocketllm -o my_program
LD_LIBRARY_PATH=build ./my_program
```

## 6. Checks

```bash
pip install --break-system-packages openpyxl gguf numpy
make pocket-test                # Python gives the same text as the programs
make rag-test                   # reading, search, answers
bash tests/run_xlsx_checks.sh   # the Excel reader
```

## 7. Limits

- Linux and WSL only; plain Windows is not supported yet.
- A C compiler is needed to install.
- PDF files are not read.
- Answers need the words of the question to be in the files; a 1.1B model
  cannot count or reason over many rows, so lists and counts come from the
  rows directly (`rag.ask()` without a model).
