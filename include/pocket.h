#ifndef POCKET_H
#define POCKET_H

/*
 * pocket.h - the library interface of the project ("pocketllm").
 *
 * Everything the command-line programs do, as plain C functions that return
 * values instead of printing, so other programs and other languages can use
 * the engine. python/pocketllm wraps exactly these functions with ctypes;
 * examples/09_rag/llm_rag.c is written on top of them too.
 *
 *   char error[512];
 *   PocketModel *m = pocket_model_open("tinyllama.gguf", NULL, 0, error, sizeof(error));
 *   PocketGenOptions o = {"top-k", 0.7f, 40, 128};
 *   const char *text = pocket_model_generate(m, "My name is", &o, NULL, NULL);
 *   const char *reply = pocket_model_chat(m, "What is RAG?", &o, NULL, NULL);
 *
 *   const char *docs[] = {"data"};
 *   PocketRag *r = pocket_rag_open(docs, 1, 0, 0, error, sizeof(error));
 *   const char *answer = pocket_rag_ask(r, "Who is the instructor for ML?");
 *   const char *source = pocket_rag_last_source(r);
 *   const char *written = pocket_rag_ask_model(r, m, "Who teaches DS3010?", NULL, NULL, NULL, NULL);
 *
 * Conventions (as in engine.h):
 *   - open functions return NULL on failure and write a message into the
 *     caller's buffer; other functions return NULL / -1 and keep the message
 *     for pocket_model_error() / pocket_rag_error();
 *   - returned strings belong to the handle and stay valid until the next
 *     call on that handle;
 *   - nothing is printed; a PocketTokenFn receives each piece of generated
 *     text as it is produced (pieces may split a UTF-8 character).
 * A handle must not be used from two threads at once.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PocketModel PocketModel;
typedef struct PocketRag PocketRag;

typedef void (*PocketTokenFn)(const char *piece, void *user);

typedef struct {
    const char *sampler;   /* "greedy" | "temperature" | "top-k"; NULL = "top-k" */
    float temperature;     /* <= 0: 0.7 */
    int top_k;             /* <= 0: 40 (capped at the vocabulary size) */
    int max_tokens;        /* <= 0: 256 */
} PocketGenOptions;

typedef struct {
    int prompt_tokens;
    int generated_tokens;
    int stopped_by_eos;    /* 0: the token limit or the context ended it */
    double prompt_seconds;
    double generate_seconds;
    int position;          /* KV-cache positions in use */
    int context;           /* positions allocated */
} PocketStats;

typedef struct {
    int k;                 /* rows given to the model; <= 0: 3 */
    double min_score;      /* best BM25 score needed to answer; <= 0: 1.0 */
    double keep;           /* keep rows scoring >= keep x best; <= 0: 0.35 */
    const char *system;    /* instruction; NULL: RAG_DEFAULT_SYSTEM */
    int compact;           /* 1: only the key and asked-for fields of each row
                              (shorter prompt, faster answer); 0: all fields */
} PocketRagOptions;

/* ------------------------------------------------------------------ */
/* Model                                                               */
/* ------------------------------------------------------------------ */

/*
 * Load a model: a llama2.c .bin (stories15M/42M/110M) or a GGUF file of the
 * "llama" architecture with a SentencePiece tokenizer and F32, F16, Q4_0,
 * Q8_0, Q4_K, Q5_K or Q6_K weights. tokenizer_path is only for .bin models
 * (NULL: tokenizer.bin next to the model, then the current directory).
 * max_context 0 = the model's own, capped at 4096.
 * The random seed is taken from the clock; see pocket_model_seed().
 */
PocketModel *pocket_model_open(const char *model_path, const char *tokenizer_path,
                               int max_context, char *error, size_t error_capacity);

void pocket_model_close(PocketModel *model);

/* e.g. "GGUF, weights Q4_K x135 Q6_K x21 F32 x45" or "llama2.c FP32". */
const char *pocket_model_description(const PocketModel *model);
int pocket_model_context(const PocketModel *model);
int pocket_model_vocab_size(const PocketModel *model);

/* 1 for GGUF chat models, 0 for llama2.c models (text completion only). */
int pocket_model_is_chat(const PocketModel *model);

void pocket_model_seed(PocketModel *model, unsigned long seed);

/*
 * Prompt format of pocket_model_chat():
 *   "zephyr"   <|system|>\n{s}</s>\n<|user|>\n{m}</s>\n<|assistant|>\n
 *              (TinyLlama-1.1B-Chat, Zephyr; the default for GGUF models)
 *   "llama2"   [INST] <<SYS>>\n{s}\n<</SYS>>\n\n{m} [/INST]   (BOS every turn)
 *   "mistral"  [INST] {m} [/INST]   (the system text is put before the first message)
 *   "none"     the message as it is (the default for llama2.c models)
 * Returns 0, or -1 for an unknown name. Changing it resets the conversation.
 */
int pocket_model_set_chat_format(PocketModel *model, const char *name);
const char *pocket_model_chat_format(const PocketModel *model);

/* System message for chat (copied; NULL or "" for none). Resets the conversation. */
int pocket_model_set_system(PocketModel *model, const char *text);

/* Forget the conversation (empties the KV cache). */
void pocket_model_reset(PocketModel *model);

/*
 * Text completion: starts from an empty context, feeds the prompt and
 * returns the continuation (without the prompt). NULL on error.
 */
const char *pocket_model_generate(PocketModel *model, const char *prompt,
                                  const PocketGenOptions *options,
                                  PocketTokenFn on_token, void *user);

/*
 * One chat turn in the current chat format. The KV cache keeps the earlier
 * turns, so only the new message is run through the model. Returns the
 * reply, or NULL on error (a full context: call pocket_model_reset()).
 * llama2.c models are not chat models: with the "none" format every call
 * is a fresh completion of the message.
 */
const char *pocket_model_chat(PocketModel *model, const char *message,
                              const PocketGenOptions *options,
                              PocketTokenFn on_token, void *user);

/* Numbers of the last generate / chat / ask_model call. */
void pocket_model_stats(const PocketModel *model, PocketStats *stats);

/* Message of the last failed call on this model ("" if none). */
const char *pocket_model_error(const PocketModel *model);

/* ------------------------------------------------------------------ */
/* RAG over .xlsx / .csv files                                         */
/* ------------------------------------------------------------------ */

/*
 * Read every .xlsx and .csv in the given folders / files and build the
 * search index. header_row 0 = detect. See rag_chunk.h and rag_index.h.
 */
PocketRag *pocket_rag_open(const char *const *paths, int n_paths, int header_row,
                           int include_hidden, char *error, size_t error_capacity);

void pocket_rag_close(PocketRag *rag);

/*
 * Check the folders: if a file was added, edited or removed, read
 * everything again. Returns 1 if it reloaded (pocket_rag_change() says
 * what changed), 0 if nothing changed, -1 on error. The ask functions call
 * this themselves unless auto refresh is turned off.
 */
int pocket_rag_refresh(PocketRag *rag);
int pocket_rag_reload(PocketRag *rag);                 /* 0 or -1 */
void pocket_rag_set_auto_refresh(PocketRag *rag, int on);
const char *pocket_rag_change(const PocketRag *rag);   /* of the last reload, or "" */

/*
 * Answer straight from the rows, no model (rag_answer.h). The conversation
 * is remembered: "yes", "no", "and its room?" or a number work as replies.
 * Returns the whole reply as it is printed by llm_rag --retrieve-only.
 */
const char *pocket_rag_ask(PocketRag *rag, const char *question);

/* The parts of the last pocket_rag_ask() reply. */
const char *pocket_rag_last_answer(const PocketRag *rag);  /* without source and offer */
const char *pocket_rag_last_source(const PocketRag *rag);  /* "" if none */
const char *pocket_rag_last_offer(const PocketRag *rag);   /* "" if none */
int pocket_rag_last_found(const PocketRag *rag);           /* 1: rows were shown */

/* Forget the conversation of pocket_rag_ask(). */
void pocket_rag_reset(PocketRag *rag);

/*
 * Retrieval-augmented generation: the best rows are put into the prompt
 * and the chat model writes the answer (rag_prompt.h). The instruction is
 * kept in the model's KV cache between questions. Returns the answer, or
 * "I could not find this in the files." without running the model when no
 * row is good enough; NULL on error. pocket_rag_last_source() then lists
 * the rows used, one per line. Needs a chat (GGUF) model.
 * options / rag_options may be NULL (greedy, 160 tokens, 3 rows).
 */
const char *pocket_rag_ask_model(PocketRag *rag, PocketModel *model,
                                 const char *question,
                                 const PocketGenOptions *options,
                                 const PocketRagOptions *rag_options,
                                 PocketTokenFn on_token, void *user);

/* The prompt given to the model by the last pocket_rag_ask_model() call. */
const char *pocket_rag_last_prompt(const PocketRag *rag);

/*
 * BM25 search only: the best max_hits rows (at most 64). Returns how many;
 * then for hit 0 .. n-1: its text ("Label: value" lines), where it is from
 * and its score. With for_model = 1 the rows are filtered and ordered as
 * pocket_rag_ask_model() would use them (rag_options may be NULL).
 */
int pocket_rag_search(PocketRag *rag, const char *question, int max_hits,
                      int for_model, const PocketRagOptions *rag_options);
const char *pocket_rag_hit_text(PocketRag *rag, int hit);
const char *pocket_rag_hit_source(PocketRag *rag, int hit);
double pocket_rag_hit_score(const PocketRag *rag, int hit);

/* What is loaded. */
int pocket_rag_row_count(const PocketRag *rag);        /* rows + notes */
int pocket_rag_term_count(const PocketRag *rag);
int pocket_rag_file_count(const PocketRag *rag);
const char *pocket_rag_file_name(const PocketRag *rag, int file);
int pocket_rag_file_readable(const PocketRag *rag, int file);
int pocket_rag_file_rows(const PocketRag *rag, int file);
int pocket_rag_file_tables(const PocketRag *rag, int file);
int pocket_rag_warning_count(const PocketRag *rag);
const char *pocket_rag_warning(const PocketRag *rag, int index);

const char *pocket_rag_error(const PocketRag *rag);

/* "0.3.0" */
const char *pocket_version(void);

#ifdef __cplusplus
}
#endif

#endif
