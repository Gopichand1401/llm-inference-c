/*
 * test_engine.c - tests for the public engine API (include/engine.h).
 *
 * Needs no downloaded model: the test writes small llama2.c-format
 * checkpoints with random weights into a temporary folder, plus broken
 * variants of them, and runs the engine against each one.
 *
 * Covered (problem statement, section 3):
 *   normal      load, config, forward pass, determinism, generation loop
 *   boundary    first/last position, first/last token, context cap of 1
 *   invalid     NULL arguments, bad header values, bad token / position
 *   empty       0-byte model file, header-only file
 *   large       a full context of positions, prompt longer than context
 *   failure     missing files, truncated weights, wrong tokenizer
 *   memory      open/close cycles, close(NULL), every failure path frees
 *               what it allocated (run under `make asan` / `make valgrind`)
 *
 * Usage: test_engine [tokenizer.bin]   (default: models/tokenizer.bin)
 */
#define _POSIX_C_SOURCE 200809L
#include "engine.h"
#include "sampler.h"
#include "tokenizer.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures = 0;
static int checks = 0;

#define CHECK(cond, msg) do {                                        \
    ++checks;                                                        \
    if (!(cond)) {                                                   \
        ++failures;                                                  \
        fprintf(stderr, "  FAIL %s (line %d)\n", msg, __LINE__);     \
    }                                                                \
} while (0)

/* ------------------------------------------------------------------ */
/* Synthetic llama2.c checkpoint                                      */
/* ------------------------------------------------------------------ */

typedef struct {
    int32_t dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size,
            seq_len;
} Header;

/* A tiny model with grouped-query attention (4 query heads, 2 KV heads). */
static const Header TINY = {32, 64, 2, 4, 2, 32000, 16};

static char tmpdir[64];
static const char *tokenizer_path = "models/tokenizer.bin";

static const char *tmp_path(const char *name)
{
    static char buffers[8][128];
    static int next = 0;
    char *p = buffers[next++ % 8];
    snprintf(p, 128, "%s/%s", tmpdir, name);
    return p;
}

/* Number of floats after the header, as the llama2.c format lays them out. */
static size_t weight_count(const Header *h, int shared)
{
    size_t dim = (size_t)h->dim, hid = (size_t)h->hidden_dim;
    size_t L = (size_t)h->n_layers;
    size_t vocab = (size_t)abs(h->vocab_size);
    size_t head = dim / (size_t)h->n_heads;
    size_t kv = head * (size_t)h->n_kv_heads;
    size_t n = vocab * dim                       /* token embeddings      */
             + L * dim                           /* rms_att               */
             + L * dim * dim + 2 * L * dim * kv  /* wq, wk, wv            */
             + L * dim * dim                     /* wo                    */
             + L * dim                           /* rms_ffn               */
             + 3 * L * dim * hid                 /* w1, w2, w3            */
             + dim                               /* rms_final             */
             + (size_t)h->seq_len * head;        /* old RoPE tables       */
    if (!shared) {
        n += vocab * dim;                        /* separate classifier   */
    }
    return n;
}

/*
 * Write a checkpoint. floats = how many weight floats to write
 * ((size_t)-1 = all of them). vocab_size < 0 in the header marks an
 * unshared classifier, as in llama2.c.
 */
static int write_model(const char *path, const Header *h, size_t floats,
                       uint32_t seed)
{
    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        return 0;
    }
    fwrite(h, sizeof(*h), 1, f);
    size_t total = weight_count(h, h->vocab_size > 0);
    if (floats > total) {
        floats = total;
    }
    uint32_t s = seed ? seed : 1;
    for (size_t i = 0; i < floats; ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        /* small values; RMSNorm weights near 1 would also be fine */
        float v = ((float)(s % 2001) - 1000.0f) / 10000.0f;
        fwrite(&v, sizeof(v), 1, f);
    }
    fclose(f);
    return 1;
}

static void write_bytes(const char *path, const void *data, size_t n)
{
    FILE *f = fopen(path, "wb");
    if (f != NULL) {
        if (n > 0) {
            fwrite(data, 1, n, f);
        }
        fclose(f);
    }
}

static LLMModel *open_model(const char *path, int max_context,
                            char *error, size_t cap)
{
    LLMOptions o = {0};
    o.model_path = path;
    o.tokenizer_path = tokenizer_path;
    o.max_context = max_context;
    return engine_open(&o, error, cap);
}

static int all_finite(const float *x, int n)
{
    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            return 0;
        }
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* Tests                                                              */
/* ------------------------------------------------------------------ */

static void test_normal(void)
{
    printf("== normal cases\n");
    char err[256] = "";
    const char *path = tmp_path("tiny.bin");
    CHECK(write_model(path, &TINY, (size_t)-1, 7), "write tiny model");

    LLMModel *m = open_model(path, 0, err, sizeof(err));
    CHECK(m != NULL, "engine_open on a valid model");
    if (m == NULL) {
        fprintf(stderr, "    error: %s\n", err);
        return;
    }
    const Config *c = engine_config(m);
    CHECK(engine_format(m) == LLM_FORMAT_STORIES, "format detected as llama2.c");
    CHECK(c->dim == 32 && c->n_layers == 2 && c->n_heads == 4 &&
          c->n_kv_heads == 2 && c->vocab_size == 32000 && c->seq_len == 16,
          "config read from header");
    CHECK(engine_file_context(m) == 16, "file context");
    size_t kv_dim = (size_t)c->dim * c->n_kv_heads / c->n_heads;
    CHECK(engine_kv_cache_bytes(m) ==
          2 * (size_t)c->n_layers * c->seq_len * kv_dim * sizeof(float),
          "KV cache size = 2 * layers * context * kv_dim * 4");
    CHECK(engine_description(m) != NULL &&
          engine_description(m)[0] != '\0', "description");
    CHECK(engine_tokenizer(m) != NULL &&
          engine_tokenizer(m)->vocab_size == 32000, "tokenizer loaded");
    CHECK(engine_is_stop_token(m, engine_tokenizer(m)->eos_id), "EOS stops");
    CHECK(!engine_is_stop_token(m, 1000), "ordinary token does not stop");

    /* Forward pass gives finite logits; same input -> same output. */
    float *first = malloc(sizeof(float) * (size_t)c->vocab_size);
    float *logits = engine_forward(m, 1, 0);
    CHECK(logits != NULL && all_finite(logits, c->vocab_size),
          "forward at pos 0 gives finite logits");
    if (logits != NULL && first != NULL) {
        memcpy(first, logits, sizeof(float) * (size_t)c->vocab_size);
        LLMModel *m2 = open_model(path, 0, err, sizeof(err));
        float *again = m2 ? engine_forward(m2, 1, 0) : NULL;
        CHECK(again != NULL &&
              memcmp(again, first, sizeof(float) * (size_t)c->vocab_size) == 0,
              "two models, same token -> identical logits");
        engine_close(m2);
    }
    free(first);

    /* Prompt -> greedy generation over the whole context. */
    Tokenizer *tok = engine_tokenizer(m);
    int ids[32];
    size_t n = 0;
    CHECK(tokenizer_encode(tok, "Once upon a time", 1, 0, ids, 32, &n,
                           err, sizeof(err)) == 0 && n > 1,
          "encode prompt");
    int token = ids[0], ok = 1;
    for (int pos = 0; pos < c->seq_len; ++pos) {
        float *lg = engine_forward(m, token, pos);
        if (lg == NULL || !all_finite(lg, c->vocab_size)) {
            ok = 0;
            break;
        }
        int next = (size_t)(pos + 1) < n ? ids[pos + 1]
                                          : sampler_greedy(lg, (size_t)c->vocab_size);
        if (next < 0 || next >= c->vocab_size) {
            ok = 0;
            break;
        }
        token = next;
    }
    CHECK(ok, "prompt + greedy generation fills the whole context");
    engine_close(m);

    /* Unshared classifier (negative vocab_size in the header). */
    Header un = TINY;
    un.vocab_size = -32000;
    path = tmp_path("unshared.bin");
    write_model(path, &un, (size_t)-1, 9);
    m = open_model(path, 0, err, sizeof(err));
    CHECK(m != NULL && engine_forward(m, 5, 0) != NULL,
          "unshared classifier weights load and run");
    engine_close(m);

    /* Multi-query attention (1 KV head) and plain MHA (kv == heads). */
    Header mqa = TINY, mha = TINY;
    mqa.n_kv_heads = 1;
    mha.n_kv_heads = 4;
    write_model(tmp_path("mqa.bin"), &mqa, (size_t)-1, 3);
    write_model(tmp_path("mha.bin"), &mha, (size_t)-1, 3);
    m = open_model(tmp_path("mqa.bin"), 0, err, sizeof(err));
    CHECK(m != NULL && engine_forward(m, 5, 0) != NULL, "1 KV head (MQA)");
    engine_close(m);
    m = open_model(tmp_path("mha.bin"), 0, err, sizeof(err));
    CHECK(m != NULL && engine_forward(m, 5, 0) != NULL, "KV heads == heads (MHA)");
    engine_close(m);
}

static void test_boundaries(void)
{
    printf("== boundary cases\n");
    char err[256];
    const char *path = tmp_path("tiny.bin");
    LLMModel *m = open_model(path, 0, err, sizeof(err));
    CHECK(m != NULL, "open");
    if (m == NULL) {
        return;
    }
    const Config *c = engine_config(m);
    int last = c->seq_len - 1;
    for (int pos = 0; pos < last; ++pos) {
        engine_forward(m, 1, pos);
    }
    CHECK(engine_forward(m, 1, last) != NULL, "last position (seq_len - 1)");
    CHECK(engine_forward(m, 1, c->seq_len) == NULL, "position == seq_len rejected");
    CHECK(engine_forward(m, 0, 0) != NULL, "token 0");
    CHECK(engine_forward(m, c->vocab_size - 1, 0) != NULL, "token vocab_size - 1");
    CHECK(engine_forward(m, c->vocab_size, 0) == NULL, "token == vocab_size rejected");
    engine_close(m);

    /* Context cap. */
    m = open_model(path, 1, err, sizeof(err));
    CHECK(m != NULL && engine_config(m)->seq_len == 1, "max_context = 1");
    if (m != NULL) {
        CHECK(engine_forward(m, 1, 0) != NULL, "pos 0 with context 1");
        CHECK(engine_forward(m, 1, 1) == NULL, "pos 1 with context 1 rejected");
        CHECK(engine_file_context(m) == 16, "file context unchanged by cap");
    }
    engine_close(m);

    m = open_model(path, 100000, err, sizeof(err));
    CHECK(m != NULL && engine_config(m)->seq_len <= 16,
          "max_context above the file's is capped to the file's");
    engine_close(m);

    /* Smallest possible model: 1 layer, 1 head, dim 2. */
    Header tiny1 = {2, 2, 1, 1, 1, 32000, 1};
    write_model(tmp_path("min.bin"), &tiny1, (size_t)-1, 5);
    m = open_model(tmp_path("min.bin"), 0, err, sizeof(err));
    CHECK(m != NULL && engine_forward(m, 1, 0) != NULL,
          "minimal model (dim 2, 1 layer, 1 head, context 1)");
    engine_close(m);
}

static void test_invalid(void)
{
    printf("== invalid input\n");
    char err[256];
    LLMOptions o = {0};

    err[0] = '\0';
    CHECK(engine_open(NULL, err, sizeof(err)) == NULL && err[0] != '\0',
          "NULL options -> NULL + message");
    err[0] = '\0';
    CHECK(engine_open(&o, err, sizeof(err)) == NULL && err[0] != '\0',
          "NULL model path -> NULL + message");
    o.model_path = "";
    CHECK(engine_open(&o, err, sizeof(err)) == NULL, "empty model path");
    o.model_path = tmp_path("tiny.bin");
    o.tokenizer_path = tokenizer_path;
    o.max_context = -5;
    CHECK(engine_open(&o, err, sizeof(err)) == NULL, "negative max_context");
    o.model_path = NULL;
    CHECK(engine_open(&o, NULL, 0) == NULL, "NULL error buffer does not crash");
    CHECK(engine_open(&o, err, 1) == NULL && err[0] == '\0',
          "1-byte error buffer stays NUL-terminated");

    /* Functions that take a model are safe on NULL. */
    CHECK(engine_forward(NULL, 1, 0) == NULL, "forward(NULL)");
    CHECK(engine_config(NULL) == NULL, "config(NULL)");
    CHECK(engine_tokenizer(NULL) == NULL, "tokenizer(NULL)");
    CHECK(engine_kv_cache_bytes(NULL) == 0, "kv_cache_bytes(NULL)");
    CHECK(engine_file_context(NULL) == 0, "file_context(NULL)");
    engine_close(NULL);
    CHECK(1, "close(NULL)");

    /* Bad token / position on a real model. */
    LLMModel *m = open_model(tmp_path("tiny.bin"), 0, err, sizeof(err));
    if (m != NULL) {
        CHECK(engine_forward(m, -1, 0) == NULL, "token -1 rejected");
        CHECK(engine_forward(m, 1, -1) == NULL, "position -1 rejected");
        CHECK(engine_forward(m, 1, 0) != NULL, "model still usable after bad calls");
    }
    engine_close(m);

    /* Corrupt headers: every one must be refused with a message. */
    struct { const char *name; Header h; } bad[] = {
        {"zero dim",            {0, 64, 2, 4, 2, 32000, 16}},
        {"negative dim",        {-32, 64, 2, 4, 2, 32000, 16}},
        {"zero layers",         {32, 64, 0, 4, 2, 32000, 16}},
        {"zero heads",          {32, 64, 2, 0, 2, 32000, 16}},
        {"dim not / heads",     {30, 64, 2, 4, 2, 32000, 16}},
        {"kv heads > heads",    {32, 64, 2, 4, 8, 32000, 16}},
        {"heads not / kv",      {32, 64, 2, 4, 3, 32000, 16}},
        {"zero kv heads",       {32, 64, 2, 4, 0, 32000, 16}},
        {"zero vocab",          {32, 64, 2, 4, 2, 0, 16}},
        {"zero seq_len",        {32, 64, 2, 4, 2, 32000, 0}},
        {"zero hidden",         {32, 0, 2, 4, 2, 32000, 16}},
        {"huge sizes (overflow)", {1 << 30, 1 << 30, 1 << 30, 1, 1, 32000, 1 << 30}},
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
        char name[64];
        snprintf(name, sizeof(name), "bad%zu.bin", i);
        /* header plus plenty of bytes, so only the header can be at fault */
        FILE *f = fopen(tmp_path(name), "wb");
        if (f != NULL) {
            fwrite(&bad[i].h, sizeof(Header), 1, f);
            static float zeros[4096];
            for (int k = 0; k < 1100; ++k) {   /* ~18 MB */
                fwrite(zeros, sizeof(zeros), 1, f);
            }
            fclose(f);
        }
        err[0] = '\0';
        m = open_model(tmp_path(name), 0, err, sizeof(err));
        char msg[96];
        snprintf(msg, sizeof(msg), "header '%s' rejected with a message",
                 bad[i].name);
        CHECK(m == NULL && err[0] != '\0', msg);
        engine_close(m);
        unlink(tmp_path(name));
    }

    /* Random bytes. */
    unsigned char junk[4096];
    for (size_t i = 0; i < sizeof(junk); ++i) {
        junk[i] = (unsigned char)(i * 131u + 7u);
    }
    write_bytes(tmp_path("junk.bin"), junk, sizeof(junk));
    CHECK(open_model(tmp_path("junk.bin"), 0, err, sizeof(err)) == NULL,
          "random bytes rejected");

    /* GGUF magic with nothing behind it. */
    write_bytes(tmp_path("fake.gguf"), "GGUF", 4);
    err[0] = '\0';
    m = open_model(tmp_path("fake.gguf"), 0, err, sizeof(err));
    CHECK(m == NULL && err[0] != '\0', "GGUF magic only -> rejected");
    engine_close(m);
}

static void test_empty(void)
{
    printf("== empty input\n");
    char err[256];
    write_bytes(tmp_path("empty.bin"), "", 0);
    err[0] = '\0';
    CHECK(open_model(tmp_path("empty.bin"), 0, err, sizeof(err)) == NULL &&
          err[0] != '\0', "0-byte model file");

    write_bytes(tmp_path("header_only.bin"), &TINY, sizeof(TINY));
    err[0] = '\0';
    CHECK(open_model(tmp_path("header_only.bin"), 0, err, sizeof(err)) == NULL &&
          err[0] != '\0', "header with no weights");

    write_bytes(tmp_path("empty_tok.bin"), "", 0);
    LLMOptions o = {0};
    o.model_path = tmp_path("tiny.bin");
    o.tokenizer_path = tmp_path("empty_tok.bin");
    err[0] = '\0';
    CHECK(engine_open(&o, err, sizeof(err)) == NULL && err[0] != '\0',
          "0-byte tokenizer file");

    /* Empty prompt: BOS alone is still a valid generation start. */
    LLMModel *m = open_model(tmp_path("tiny.bin"), 0, err, sizeof(err));
    if (m != NULL) {
        int ids[4];
        size_t n = 99;
        CHECK(tokenizer_encode(engine_tokenizer(m), "", 1, 0, ids, 4, &n,
                               err, sizeof(err)) == 0 && n == 1 &&
              engine_forward(m, ids[0], 0) != NULL,
              "empty prompt -> BOS only -> forward works");
    }
    engine_close(m);
}

static void test_large(void)
{
    printf("== large input\n");
    char err[256];
    Header big = TINY;
    big.seq_len = 512;
    write_model(tmp_path("long.bin"), &big, (size_t)-1, 11);
    LLMModel *m = open_model(tmp_path("long.bin"), 0, err, sizeof(err));
    CHECK(m != NULL, "open model with 512-token context");
    if (m == NULL) {
        return;
    }
    int ok = 1;
    for (int pos = 0; pos < 512 && ok; ++pos) {
        float *lg = engine_forward(m, (pos * 37) % 32000, pos);
        ok = lg != NULL && all_finite(lg, 32000);
    }
    CHECK(ok, "512 positions: all logits finite (full KV cache)");

    /* A prompt longer than the context: the caller must see it does not fit. */
    size_t len = 20000;
    char *text = malloc(len + 1);
    int *ids = malloc(sizeof(int) * 32768);
    if (text != NULL && ids != NULL) {
        for (size_t i = 0; i < len; ++i) {
            text[i] = "the quick brown fox "[i % 20];
        }
        text[len] = '\0';
        size_t n = 0;
        CHECK(tokenizer_encode(engine_tokenizer(m), text, 1, 0, ids, 32768, &n,
                               err, sizeof(err)) == 0 &&
              n > (size_t)engine_config(m)->seq_len,
              "20 KB prompt encodes to more tokens than the context");
        CHECK(engine_forward(m, ids[0], engine_config(m)->seq_len) == NULL,
              "position past the context is refused, not written");
    }
    free(text);
    free(ids);
    engine_close(m);
}

static void test_failures(void)
{
    printf("== failure conditions\n");
    char err[256];

    err[0] = '\0';
    CHECK(open_model(tmp_path("does-not-exist.bin"), 0, err, sizeof(err)) == NULL &&
          err[0] != '\0', "missing model file");
    CHECK(open_model(tmpdir, 0, err, sizeof(err)) == NULL,
          "a directory instead of a file");

    /* Truncated at several points. */
    size_t total = weight_count(&TINY, 1);
    size_t cut[] = {1, total / 4, total / 2, total - 1};
    for (size_t i = 0; i < sizeof(cut) / sizeof(cut[0]); ++i) {
        write_model(tmp_path("trunc.bin"), &TINY, cut[i], 7);
        err[0] = '\0';
        LLMModel *m = open_model(tmp_path("trunc.bin"), 0, err, sizeof(err));
        char msg[96];
        snprintf(msg, sizeof(msg), "truncated to %zu of %zu floats rejected",
                 cut[i], total);
        CHECK(m == NULL && err[0] != '\0', msg);
        engine_close(m);
    }

    /* Tokenizer problems. */
    LLMOptions o = {0};
    o.model_path = tmp_path("tiny.bin");
    o.tokenizer_path = tmp_path("no-tokenizer.bin");
    err[0] = '\0';
    CHECK(engine_open(&o, err, sizeof(err)) == NULL && err[0] != '\0',
          "missing tokenizer file");

    Header small_vocab = TINY;
    small_vocab.vocab_size = 100;   /* tokenizer.bin has 32000 entries */
    write_model(tmp_path("v100.bin"), &small_vocab, (size_t)-1, 7);
    o.model_path = tmp_path("v100.bin");
    o.tokenizer_path = tokenizer_path;
    err[0] = '\0';
    LLMModel *m = engine_open(&o, err, sizeof(err));
    CHECK(m == NULL || engine_tokenizer(m)->vocab_size == 100,
          "model / tokenizer vocabulary mismatch is handled");
    engine_close(m);

    /* Truncated tokenizer. */
    FILE *in = fopen(tokenizer_path, "rb");
    if (in != NULL) {
        char buf[2000];
        size_t got = fread(buf, 1, sizeof(buf), in);
        fclose(in);
        write_bytes(tmp_path("trunc_tok.bin"), buf, got);
        o.model_path = tmp_path("tiny.bin");
        o.tokenizer_path = tmp_path("trunc_tok.bin");
        err[0] = '\0';
        CHECK(engine_open(&o, err, sizeof(err)) == NULL && err[0] != '\0',
              "truncated tokenizer file");
    }
}

static void test_memory(void)
{
    printf("== memory management\n");
    char err[256];
    /* Repeated open/close: leaks show up under ASan/Valgrind. */
    int ok = 1;
    for (int i = 0; i < 25 && ok; ++i) {
        LLMModel *m = open_model(tmp_path("tiny.bin"), 0, err, sizeof(err));
        ok = m != NULL && engine_forward(m, 1, 0) != NULL;
        engine_close(m);
    }
    CHECK(ok, "25 open / forward / close cycles");

    /* Two models alive at once do not share state. */
    LLMModel *a = open_model(tmp_path("tiny.bin"), 0, err, sizeof(err));
    LLMModel *b = open_model(tmp_path("tiny.bin"), 0, err, sizeof(err));
    if (a != NULL && b != NULL) {
        float *la = engine_forward(a, 10, 0);
        float va = la[0];
        engine_forward(b, 20, 0);
        CHECK(la != engine_forward(b, 20, 0) && la[0] == va,
              "two open models keep separate buffers");
    }
    engine_close(a);
    engine_close(b);

    /* Failure paths, repeated: each must free its partial allocations. */
    for (int i = 0; i < 10; ++i) {
        open_model(tmp_path("header_only.bin"), 0, err, sizeof(err));
        open_model(tmp_path("trunc.bin"), 0, err, sizeof(err));
        open_model(tmp_path("does-not-exist.bin"), 0, err, sizeof(err));
    }
    CHECK(1, "repeated failing opens (checked for leaks by the sanitizers)");
}

int main(int argc, char **argv)
{
    if (argc > 1) {
        tokenizer_path = argv[1];
    }
    FILE *t = fopen(tokenizer_path, "rb");
    if (t == NULL) {
        fprintf(stderr, "test_engine: cannot open %s (run from the project "
                        "root or pass the tokenizer path)\n", tokenizer_path);
        return 1;
    }
    fclose(t);

    snprintf(tmpdir, sizeof(tmpdir), "/tmp/llm_test_XXXXXX");
    if (mkdtemp(tmpdir) == NULL) {
        perror("mkdtemp");
        return 1;
    }

    test_normal();
    test_boundaries();
    test_invalid();
    test_empty();
    test_large();
    test_failures();
    test_memory();

    /* clean up the temporary folder */
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "rm -rf '%s'", tmpdir);
    if (system(cmd) != 0) {
        fprintf(stderr, "warning: could not remove %s\n", tmpdir);
    }

    if (failures == 0) {
        printf("PASS: engine tests (%d checks)\n", checks);
        return 0;
    }
    printf("FAIL: engine tests (%d of %d checks failed)\n", failures, checks);
    return 1;
}
