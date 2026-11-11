CC = gcc
.DEFAULT_GOAL := all

CFLAGS = -std=c11 -Wall -Wextra -Wpedantic -Werror -O2 -fopenmp

# Optional: build for this CPU's full instruction set (e.g. AVX2), which
# lets GCC vectorise 8 floats at a time instead of 4:   make NATIVE=1
# The binaries then only run on CPUs with the same features.
ifdef NATIVE
CFLAGS += -march=native
endif
CPPFLAGS = -Iinclude

BUILD = build
LIB = $(BUILD)/libllm.a
OBJS = $(BUILD)/model.o $(BUILD)/transformer.o $(BUILD)/tensor.o $(BUILD)/config.o $(BUILD)/checkpoint.o $(BUILD)/checkpoint_stories.o $(BUILD)/sampler.o $(BUILD)/tokenizer.o $(BUILD)/checkpoint_gguf.o $(BUILD)/gguf_quant.o $(BUILD)/gguf_model.o $(BUILD)/gguf_forward.o $(BUILD)/gguf_tokenizer.o $(BUILD)/engine.o $(BUILD)/xlsx_reader.o $(BUILD)/rag_chunk.o $(BUILD)/rag_index.o $(BUILD)/rag_prompt.o $(BUILD)/rag_answer.o $(BUILD)/pocket.o
$(BUILD)/checkpoint.o: src/checkpoint.c include/checkpoint.h include/transformer.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/checkpoint.c -o $@
$(BUILD)/tensor.o: src/tensor.c include/transformer.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/tensor.c -o $@
$(BUILD)/checkpoint_stories.o: src/checkpoint_stories.c include/checkpoint.h include/transformer.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/checkpoint_stories.c -o $@
$(BUILD)/config.o: src/config.c include/transformer.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/config.c -o $@
$(BUILD)/checkpoint_gguf.o: src/checkpoint_gguf.c include/checkpoint.h include/transformer.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/checkpoint_gguf.c -o $@
$(BUILD)/gguf_quant.o: src/gguf_quant.c include/gguf_quant.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/gguf_quant.c -o $@
$(BUILD)/engine.o: src/engine.c include/engine.h include/gguf_forward.h include/gguf_model.h include/tokenizer.h include/transformer.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/engine.c -o $@
$(BUILD)/gguf_tokenizer.o: src/gguf_tokenizer.c include/tokenizer.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/gguf_tokenizer.c -o $@
$(BUILD)/gguf_forward.o: src/gguf_forward.c include/gguf_forward.h include/gguf_model.h include/transformer.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/gguf_forward.c -o $@
$(BUILD)/gguf_model.o: src/gguf_model.c include/gguf_model.h include/gguf_quant.h include/checkpoint.h include/transformer.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/gguf_model.c -o $@
DEMO = $(BUILD)/llm_demo

.PHONY: all clean test test-model asan valgrind examples showcase bench gguf-tools rag-tools rag-test pocket-test lib

all: $(DEMO) $(BUILD)/llm_generate $(BUILD)/llm_showcase $(BUILD)/llm_bench

# One command for every supported model file (llama2.c .bin and GGUF).
$(BUILD)/llm_generate: examples/01_apps/llm_generate.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/01_apps/llm_generate.c $(LIB) -lm -o $@

$(BUILD)/gguf_validate_demo: examples/02_gguf_format/gguf_validate_demo.c $(BUILD)/checkpoint_gguf.o $(BUILD)/config.o
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/02_gguf_format/gguf_validate_demo.c $(BUILD)/checkpoint_gguf.o $(BUILD)/config.o -o $@

# GGUF quantization demos (build with: make gguf-tools)
GGUF_TOOLS = $(BUILD)/gguf_q4k_demo $(BUILD)/gguf_q4k_real_demo \
             $(BUILD)/gguf_q6k_demo $(BUILD)/gguf_q6k_real_demo \
             $(BUILD)/gguf_tensor_report $(BUILD)/gguf_model_demo \
             $(BUILD)/gguf_matvec_demo $(BUILD)/gguf_forward_demo \
             $(BUILD)/gguf_tokenizer_demo $(BUILD)/gguf_generate \
             $(BUILD)/gguf_bench $(BUILD)/gguf_quant_check \
             $(BUILD)/llm_logits_dump

gguf-tools: $(GGUF_TOOLS)

$(BUILD)/gguf_q4k_demo: examples/03_quantization/gguf_q4k_demo.c $(BUILD)/gguf_quant.o
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/03_quantization/gguf_q4k_demo.c $(BUILD)/gguf_quant.o -lm -o $@

$(BUILD)/gguf_q4k_real_demo: examples/03_quantization/gguf_q4k_real_demo.c $(BUILD)/checkpoint_gguf.o $(BUILD)/config.o $(BUILD)/gguf_quant.o
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/03_quantization/gguf_q4k_real_demo.c $(BUILD)/checkpoint_gguf.o $(BUILD)/config.o $(BUILD)/gguf_quant.o -lm -o $@

$(BUILD)/gguf_q6k_demo: examples/03_quantization/gguf_q6k_demo.c $(BUILD)/gguf_quant.o
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/03_quantization/gguf_q6k_demo.c $(BUILD)/gguf_quant.o -lm -o $@

$(BUILD)/gguf_q6k_real_demo: examples/03_quantization/gguf_q6k_real_demo.c $(BUILD)/checkpoint_gguf.o $(BUILD)/config.o $(BUILD)/gguf_quant.o
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/03_quantization/gguf_q6k_real_demo.c $(BUILD)/checkpoint_gguf.o $(BUILD)/config.o $(BUILD)/gguf_quant.o -lm -o $@

$(BUILD)/gguf_tensor_report: examples/04_model_loading/gguf_tensor_report.c $(BUILD)/checkpoint_gguf.o $(BUILD)/config.o
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/04_model_loading/gguf_tensor_report.c $(BUILD)/checkpoint_gguf.o $(BUILD)/config.o -o $@

$(BUILD)/gguf_model_demo: examples/04_model_loading/gguf_model_demo.c $(BUILD)/gguf_model.o $(BUILD)/checkpoint_gguf.o $(BUILD)/config.o $(BUILD)/gguf_quant.o
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/04_model_loading/gguf_model_demo.c $(BUILD)/gguf_model.o $(BUILD)/checkpoint_gguf.o $(BUILD)/config.o $(BUILD)/gguf_quant.o -lm -o $@

$(BUILD)/gguf_matvec_demo: examples/05_matrix_math/gguf_matvec_demo.c $(BUILD)/gguf_model.o $(BUILD)/checkpoint_gguf.o $(BUILD)/config.o $(BUILD)/gguf_quant.o
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/05_matrix_math/gguf_matvec_demo.c $(BUILD)/gguf_model.o $(BUILD)/checkpoint_gguf.o $(BUILD)/config.o $(BUILD)/gguf_quant.o -lm -o $@

$(BUILD)/gguf_forward_demo: examples/06_forward_pass/gguf_forward_demo.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/06_forward_pass/gguf_forward_demo.c $(LIB) -lm -o $@

$(BUILD)/gguf_tokenizer_demo: examples/07_tokenizer/gguf_tokenizer_demo.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/07_tokenizer/gguf_tokenizer_demo.c $(LIB) -lm -o $@

$(BUILD)/gguf_generate: examples/01_apps/gguf_generate.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/01_apps/gguf_generate.c $(LIB) -lm -o $@

$(BUILD)/gguf_bench: examples/08_benchmark/gguf_bench.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/08_benchmark/gguf_bench.c $(LIB) -lm -o $@

$(BUILD)/gguf_quant_check: examples/03_quantization/gguf_quant_check.c $(BUILD)/gguf_quant.o
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/03_quantization/gguf_quant_check.c $(BUILD)/gguf_quant.o -o $@

$(BUILD)/llm_logits_dump: examples/06_forward_pass/llm_logits_dump.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/06_forward_pass/llm_logits_dump.c $(LIB) -lm -o $@


$(BUILD):
	mkdir -p $(BUILD)


# ---- RAG: retrieval-augmented generation over .xlsx / .csv files --------
$(BUILD)/xlsx_reader.o: src/xlsx_reader.c include/xlsx_reader.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/xlsx_reader.c -o $@

$(BUILD)/rag_chunk.o: src/rag_chunk.c include/rag_chunk.h include/xlsx_reader.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/rag_chunk.c -o $@

$(BUILD)/rag_index.o: src/rag_index.c include/rag_index.h include/rag_chunk.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/rag_index.c -o $@

$(BUILD)/rag_prompt.o: src/rag_prompt.c include/rag_prompt.h include/rag_index.h include/rag_chunk.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/rag_prompt.c -o $@

$(BUILD)/rag_answer.o: src/rag_answer.c include/rag_answer.h include/rag_index.h include/rag_chunk.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/rag_answer.c -o $@

# RAG programs (build with: make rag-tools). zlib is needed: -lz
RAG_TOOLS = $(BUILD)/xlsx_dump $(BUILD)/rag_inspect $(BUILD)/llm_rag

rag-tools: $(RAG_TOOLS)

$(BUILD)/xlsx_dump: examples/09_rag/xlsx_dump.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/09_rag/xlsx_dump.c $(LIB) -lm -lz -o $@

$(BUILD)/rag_inspect: examples/09_rag/rag_inspect.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/09_rag/rag_inspect.c $(LIB) -lm -lz -o $@

$(BUILD)/llm_rag: examples/09_rag/llm_rag.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/09_rag/llm_rag.c $(LIB) -lm -lz -o $@

# RAG and library-interface verification scripts (need python3 + openpyxl,
# gguf, numpy; they build tiny test models, so no real model is needed)
rag-test:
	bash tests/run_rag_checks.sh

pocket-test:
	bash tests/run_pocket_checks.sh

# ---- PocketLLM: simple library interface + shared library ---------------
# include/pocket.h wraps engine + RAG in a few calls; `make lib` builds
# build/libpocketllm.so, which the Python package in python/pocketllm loads.
$(BUILD)/pocket.o: src/pocket.c include/pocket.h include/engine.h include/rag_answer.h include/rag_prompt.h include/rag_index.h include/rag_chunk.h include/sampler.h include/tokenizer.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/pocket.c -o $@

SHARED = $(BUILD)/libpocketllm.so
PIC_OBJS = $(patsubst $(BUILD)/%.o,$(BUILD)/pic/%.o,$(OBJS))

lib: $(SHARED)

$(BUILD)/pic:
	mkdir -p $(BUILD)/pic

$(BUILD)/pic/%.o: src/%.c | $(BUILD)/pic
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -c $< -o $@

$(SHARED): $(PIC_OBJS)
	$(CC) -shared -fopenmp $(PIC_OBJS) -lm -lz -o $@

POCKET_TOOLS = $(BUILD)/pocket_chat $(BUILD)/pocket_demo

$(BUILD)/pocket_%: examples/10_library/pocket_%.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -lm -lz -o $@

# ---- core library objects -----------------------------------------------
$(BUILD)/model.o: src/model.c include/llm.h include/transformer.h include/checkpoint.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/model.c -o $@

$(BUILD)/transformer.o: src/transformer.c include/transformer.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/transformer.c -o $@

$(BUILD)/sampler.o: src/sampler.c include/sampler.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/sampler.c -o $@

$(BUILD)/tokenizer.o: src/tokenizer.c include/tokenizer.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/tokenizer.c -o $@

# ---- tests ----------------------------------------------------------------
#
#   make test                 unit + engine + edge-case tests (no model needed)
#   make test MODEL=f.bin     also the forward-pass tests on a real checkpoint
#   make asan                 the same tests built with AddressSanitizer + UBSan
#   make valgrind             the same tests under Valgrind (leak check)
#
# The tests need models/tokenizer.bin (TOKENIZER=... to use another file).

TOKENIZER ?= models/tokenizer.bin
MODEL ?= models/stories42M.bin

UNIT_TESTS = test_sampler test_tokenizer test_math test_engine test_edge_cases
MODEL_TESTS = test_validation test_forward
TEST_BINS = $(addprefix $(BUILD)/,$(UNIT_TESTS) $(MODEL_TESTS))

# tests that take the tokenizer path as their only argument
TOKENIZER_ARG_test_tokenizer = $(TOKENIZER)
TOKENIZER_ARG_test_engine = $(TOKENIZER)
TOKENIZER_ARG_test_edge_cases = $(TOKENIZER)

$(BUILD)/test_%: tests/test_%.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -lm -lz -o $@

# Runs one test binary set: $(1) = command prefix, $(2) = binary directory.
define run_tests
	@set -e; for t in $(UNIT_TESTS); do \
	    echo "--- $$t"; \
	    case $$t in \
	        test_tokenizer|test_engine|test_edge_cases) $(1) $(2)/$$t $(TOKENIZER);; \
	        *) $(1) $(2)/$$t;; \
	    esac; \
	done
	@if [ -f "$(MODEL)" ]; then \
	    set -e; for t in $(MODEL_TESTS); do \
	        echo "--- $$t $(MODEL)"; $(1) $(2)/$$t $(MODEL); \
	    done; \
	else \
	    echo "--- SKIP $(MODEL_TESTS): $(MODEL) not found (make test MODEL=path/to/model.bin)"; \
	fi
endef

test: $(TEST_BINS)
	$(call run_tests,,$(BUILD))
	@echo "=== all tests passed"

test-model: test

# Sanitizer build: every library source compiled with the test, so the
# library code itself is instrumented too.
SAN_FLAGS = -std=c11 -Wall -Wextra -Wpedantic -Werror -g -O1 -fopenmp \
            -fsanitize=address,undefined -fno-sanitize-recover=all \
            -fno-omit-frame-pointer
LIB_SRCS = $(wildcard src/*.c)
ASAN_BINS = $(addprefix $(BUILD)/asan/,$(UNIT_TESTS) $(MODEL_TESTS))

$(BUILD)/asan:
	mkdir -p $@

$(BUILD)/asan/test_%: tests/test_%.c $(LIB_SRCS) | $(BUILD)/asan
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $< $(LIB_SRCS) -lm -lz -o $@

asan: $(ASAN_BINS)
	$(call run_tests,ASAN_OPTIONS=detect_leaks=1:abort_on_error=1,$(BUILD)/asan)
	@echo "=== all tests passed under AddressSanitizer + UBSan (no leaks)"

VALGRIND = valgrind --quiet --leak-check=full --errors-for-leak-kinds=definite \
           --error-exitcode=1

valgrind: $(TEST_BINS)
	@command -v valgrind >/dev/null || { echo "valgrind is not installed"; exit 1; }
	$(call run_tests,$(VALGRIND),$(BUILD))
	@echo "=== all tests passed under Valgrind (no leaks, no invalid accesses)"

# ---- example programs -------------------------------------------------------
# Speed and memory for any model: ./build/llm_bench <model> [prompt] [gen]
# Full table for several models: bash tests/run_benchmark.sh <models...>
$(BUILD)/llm_bench: examples/08_benchmark/llm_bench.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/08_benchmark/llm_bench.c $(LIB) -lm -o $@

showcase: $(BUILD)/llm_showcase

$(BUILD)/llm_showcase: examples/01_apps/showcase.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/01_apps/showcase.c $(LIB) -lm -o $@

$(BUILD)/minimal: examples/01_apps/minimal.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/01_apps/minimal.c $(LIB) -lm -o $@

$(BUILD)/generate: examples/01_apps/generate.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/01_apps/generate.c $(LIB) -lm -o $@

# make bench MODELS="models/stories42M.bin models/gguf/x.gguf"
MODELS ?= $(wildcard models/*.bin models/gguf/*.gguf)
bench: $(BUILD)/llm_bench
	bash tests/run_benchmark.sh $(filter-out %tokenizer.bin,$(MODELS))

# GGUF file-format walkthroughs (examples/02_gguf_format)
FORMAT_DEMOS = $(patsubst examples/02_gguf_format/%.c,$(BUILD)/%,$(wildcard examples/02_gguf_format/*.c))

$(BUILD)/gguf_%_demo: examples/02_gguf_format/gguf_%_demo.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -lm -o $@

# every example program, so none of them silently stops compiling
examples: all gguf-tools rag-tools lib $(POCKET_TOOLS) $(BUILD)/generate $(BUILD)/minimal $(FORMAT_DEMOS)

$(LIB): $(OBJS)
	ar rcs $@ $(OBJS)

$(DEMO): examples/01_apps/demo.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) examples/01_apps/demo.c $(LIB) -lm -fopenmp -o $@

clean:
	rm -rf $(BUILD)
