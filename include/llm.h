
#ifndef LLM_H
#define LLM_H

/*
 * llm.h - LOW-LEVEL API: llama2.c ("Stories") FP32 checkpoints only.
 *
 * Most programs should use engine.h instead: engine_open() detects the
 * format (llama2.c .bin or GGUF), loads the matching tokenizer and gives
 * one forward() for both. This header stays for code that wants direct
 * access to the FP32 Transformer struct (weights, RunState), e.g.
 * examples/01_apps/demo.c and showcase.c.
 *
 * Ownership: llm_load() allocates the Transformer and maps the file;
 * the caller owns it and must release it with llm_free_model().
 * On failure llm_load() frees everything it allocated, returns NULL and
 * writes a message into error (if error != NULL and error_capacity > 0).
 */

#include <stddef.h>
#include "transformer.h"

Transformer *llm_load(const char *model_path,
                      char *error,
                      size_t error_capacity);

const Config *llm_model_config(const Transformer *model);

size_t llm_model_file_size(const Transformer *model);

void llm_free_model(Transformer *model);

#endif

