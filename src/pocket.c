/*
 * pocket.c - the library interface (see include/pocket.h).
 *
 * The generation steps are the ones of examples/01_apps/llm_generate.c:
 * encode, feed token by token at increasing positions, sample, stop at the
 * end token. tests/run_pocket_checks.sh checks that the same model, prompt
 * and seed give the same text through this file, through llm_generate and
 * through the Python package.
 */
#define _POSIX_C_SOURCE 200809L

#include "pocket.h"

#include "engine.h"
#include "rag_answer.h"
#include "rag_chunk.h"
#include "rag_index.h"
#include "rag_prompt.h"
#include "sampler.h"
#include "tokenizer.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define POCKET_VERSION "0.3.0"
#define MAX_SEARCH_HITS 64

const char *pocket_version(void)
{
    return POCKET_VERSION;
}

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ */
/* Growable text                                                       */
/* ------------------------------------------------------------------ */

typedef struct {
    char *s;
    size_t len, cap;
} Text;

static int text_reserve(Text *t, size_t extra)
{
    if (t->len + extra + 1 > t->cap) {
        size_t cap = t->cap ? t->cap : 256;
        while (cap < t->len + extra + 1) {
            cap *= 2;
        }
        char *s = realloc(t->s, cap);
        if (s == NULL) {
            return -1;
        }
        t->s = s;
        t->cap = cap;
    }
    return 0;
}

static int text_add(Text *t, const char *s, size_t n)
{
    if (text_reserve(t, n) != 0) {
        return -1;
    }
    memcpy(t->s + t->len, s, n);
    t->len += n;
    t->s[t->len] = '\0';
    return 0;
}

static int text_set(Text *t, const char *s)
{
    t->len = 0;
    return text_add(t, s, strlen(s));
}

static void text_clear(Text *t)
{
    t->len = 0;
    if (text_reserve(t, 0) == 0) {
        t->s[0] = '\0';
    }
}

static const char *text_get(const Text *t)
{
    return t->s ? t->s : "";
}

/* ------------------------------------------------------------------ */
/* Model                                                               */
/* ------------------------------------------------------------------ */

enum { FORMAT_NONE, FORMAT_ZEPHYR, FORMAT_LLAMA2, FORMAT_MISTRAL };
enum { CACHE_EMPTY, CACHE_CHAT, CACHE_RAG };

static const char *const FORMAT_NAMES[] = {"none", "zephyr", "llama2", "mistral"};

struct PocketModel {
    LLMModel *model;
    Tokenizer *tok;
    int is_chat;
    int format;
    char *system;
    int pos;                /* next free KV-cache position */
    int cache;              /* what the KV cache holds: CACHE_* */
    int rag_prefix_pos;     /* CACHE_RAG: positions of the instruction */
    char *rag_system;       /* CACHE_RAG: the instruction that was fed */
    uint32_t rng;
    Text out;
    PocketStats stats;
    char error[512];
};

static void model_fail(PocketModel *m, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(m->error, sizeof(m->error), fmt, ap);
    va_end(ap);
}

PocketModel *pocket_model_open(const char *model_path, const char *tokenizer_path,
                               int max_context, char *error, size_t error_capacity)
{
    PocketModel *m = calloc(1, sizeof(*m));
    LLMOptions options;
    char reason[400];

    if (m == NULL) {
        snprintf(error, error_capacity, "out of memory");
        return NULL;
    }
    if (model_path == NULL || model_path[0] == '\0') {
        snprintf(error, error_capacity, "no model file given");
        free(m);
        return NULL;
    }
    memset(&options, 0, sizeof(options));
    options.model_path = model_path;
    options.tokenizer_path = (tokenizer_path && tokenizer_path[0]) ? tokenizer_path : NULL;
    options.max_context = max_context > 0 ? max_context : 0;

    reason[0] = '\0';
    m->model = engine_open(&options, reason, sizeof(reason));
    if (m->model == NULL) {
        snprintf(error, error_capacity,
                 "cannot load %s: %s\n(pocketllm reads llama2.c .bin models and "
                 "GGUF models of the \"llama\" architecture with a SentencePiece "
                 "tokenizer and F32, F16, Q4_0, Q8_0, Q4_K, Q5_K or Q6_K weights)",
                 model_path, reason[0] ? reason : "unknown reason");
        free(m);
        return NULL;
    }
    m->tok = engine_tokenizer(m->model);
    if (m->tok->vocab_size != (size_t)engine_config(m->model)->vocab_size) {
        snprintf(error, error_capacity,
                 "the tokenizer has %zu tokens but the model has %d",
                 m->tok->vocab_size, engine_config(m->model)->vocab_size);
        engine_close(m->model);
        free(m);
        return NULL;
    }
    m->is_chat = engine_format(m->model) == LLM_FORMAT_GGUF;
    m->format = m->is_chat ? FORMAT_ZEPHYR : FORMAT_NONE;
    pocket_model_seed(m, (unsigned long)time(NULL));
    text_clear(&m->out);
    return m;
}

void pocket_model_close(PocketModel *m)
{
    if (m == NULL) {
        return;
    }
    engine_close(m->model);
    free(m->system);
    free(m->rag_system);
    free(m->out.s);
    free(m);
}

const char *pocket_model_description(const PocketModel *m)
{
    return engine_description(m->model);
}

int pocket_model_context(const PocketModel *m)
{
    return engine_config(m->model)->seq_len;
}

int pocket_model_vocab_size(const PocketModel *m)
{
    return engine_config(m->model)->vocab_size;
}

int pocket_model_is_chat(const PocketModel *m)
{
    return m->is_chat;
}

void pocket_model_seed(PocketModel *m, unsigned long seed)
{
    m->rng = (uint32_t)seed ? (uint32_t)seed : 1u;
}

void pocket_model_reset(PocketModel *m)
{
    m->pos = 0;
    m->cache = CACHE_EMPTY;
}

int pocket_model_set_chat_format(PocketModel *m, const char *name)
{
    for (int i = 0; i < 4; i++) {
        if (name != NULL && strcmp(name, FORMAT_NAMES[i]) == 0) {
            m->format = i;
            pocket_model_reset(m);
            return 0;
        }
    }
    model_fail(m, "unknown chat format '%s' (use zephyr, llama2, mistral or none)",
               name ? name : "");
    return -1;
}

const char *pocket_model_chat_format(const PocketModel *m)
{
    return FORMAT_NAMES[m->format];
}

static char *copy_string(const char *s)
{
    size_t n = strlen(s);
    char *c = malloc(n + 1);
    if (c != NULL) {
        memcpy(c, s, n + 1);
    }
    return c;
}

int pocket_model_set_system(PocketModel *m, const char *text)
{
    char *copy = (text != NULL && text[0] != '\0') ? copy_string(text) : NULL;
    if (text != NULL && text[0] != '\0' && copy == NULL) {
        model_fail(m, "out of memory");
        return -1;
    }
    free(m->system);
    m->system = copy;
    pocket_model_reset(m);
    return 0;
}

void pocket_model_stats(const PocketModel *m, PocketStats *stats)
{
    *stats = m->stats;
    stats->position = m->pos;
    stats->context = engine_config(m->model)->seq_len;
}

const char *pocket_model_error(const PocketModel *m)
{
    return m->error;
}

/* ---- the steps of llm_generate.c ---- */

static float *feed(PocketModel *m, const int *ids, size_t n)
{
    const int seq_len = engine_config(m->model)->seq_len;
    float *logits = NULL;

    if (n == 0) {
        model_fail(m, "nothing to feed to the model");
        return NULL;
    }
    if ((size_t)m->pos + n > (size_t)seq_len) {
        model_fail(m, "context full: %d + %zu of %d positions; reset the "
                      "conversation or use a larger context", m->pos, n, seq_len);
        return NULL;
    }
    for (size_t i = 0; i < n; i++) {
        logits = engine_forward(m->model, ids[i], m->pos);
        if (logits == NULL) {
            model_fail(m, "forward pass failed at position %d", m->pos);
            return NULL;
        }
        m->pos++;
    }
    return logits;
}

/* Encode text into a malloc'd id array. Returns the count, -1 on error. */
static int encode(PocketModel *m, const char *text, int add_bos, int **out)
{
    char error[256];
    size_t capacity = strlen(text) + 8;
    size_t count = 0;
    int *ids = malloc(capacity * sizeof(*ids));

    if (ids == NULL) {
        model_fail(m, "out of memory");
        return -1;
    }
    if (tokenizer_encode(m->tok, text, add_bos, 0, ids, capacity, &count, error,
                         sizeof(error)) != 0) {
        model_fail(m, "cannot encode the text: %s", error);
        free(ids);
        return -1;
    }
    *out = ids;
    return (int)count;
}

static float *feed_text(PocketModel *m, const char *text, int add_bos)
{
    int *ids = NULL;
    int n = encode(m, text, add_bos, &ids);
    float *logits = NULL;
    double t0 = now_seconds();

    if (n > 0) {
        logits = feed(m, ids, (size_t)n);
    } else if (n == 0) {
        model_fail(m, "the prompt is empty");
    }
    free(ids);
    m->stats.prompt_tokens += n > 0 ? n : 0;
    m->stats.prompt_seconds += now_seconds() - t0;
    return logits;
}

typedef struct {
    const char *sampler;
    float temperature;
    int top_k;
    int max_tokens;
} Settings;

static int settings_from(PocketModel *m, const PocketGenOptions *o,
                         const char *default_sampler, int default_tokens,
                         Settings *s)
{
    s->sampler = (o && o->sampler && o->sampler[0]) ? o->sampler : default_sampler;
    s->temperature = (o && o->temperature > 0.0f) ? o->temperature : 0.7f;
    s->top_k = (o && o->top_k > 0) ? o->top_k : 40;
    s->max_tokens = (o && o->max_tokens > 0) ? o->max_tokens : default_tokens;
    if (s->top_k > engine_config(m->model)->vocab_size) {
        s->top_k = engine_config(m->model)->vocab_size;
    }
    if (strcmp(s->sampler, "greedy") != 0 && strcmp(s->sampler, "temperature") != 0 &&
        strcmp(s->sampler, "top-k") != 0) {
        model_fail(m, "unknown sampler '%s' (use greedy, temperature or top-k)",
                   s->sampler);
        return -1;
    }
    return 0;
}

static int sample(PocketModel *m, const Settings *s, const float *logits)
{
    const size_t n = (size_t)engine_config(m->model)->vocab_size;

    if (strcmp(s->sampler, "greedy") == 0) {
        return sampler_greedy(logits, n);
    }
    if (strcmp(s->sampler, "temperature") == 0) {
        return sampler_temperature(logits, n, s->temperature, &m->rng);
    }
    return sampler_top_k(logits, n, (size_t)s->top_k, s->temperature, &m->rng);
}

/*
 * Sample until the end token, the token limit or a full context. The text
 * goes to m->out and, piece by piece, to on_token.
 * Returns 0, or -1 on error.
 */
static int generate(PocketModel *m, float *logits, const Settings *s,
                    int strip_first_space, PocketTokenFn on_token, void *user)
{
    const int seq_len = engine_config(m->model)->seq_len;
    double t0 = now_seconds();
    int n = 0, status = 0;

    while (n < s->max_tokens) {
        int next = sample(m, s, logits);
        if (next < 0) {
            model_fail(m, "the sampler failed: invalid logits");
            status = -1;
            break;
        }
        if (engine_is_stop_token(m->model, next)) {
            m->stats.stopped_by_eos = 1;
            break;
        }
        char piece[256];
        if (tokenizer_decode(m->tok, &next, 1, piece, sizeof(piece), NULL, 0) == 0) {
            const char *out = piece;
            if (strip_first_space && n == 0 && out[0] == ' ') {
                out++;      /* SentencePiece leading space of a reply */
            }
            if (text_add(&m->out, out, strlen(out)) != 0) {
                model_fail(m, "out of memory");
                status = -1;
                break;
            }
            if (on_token != NULL && out[0] != '\0') {
                on_token(out, user);
            }
        }
        n++;
        if (m->pos >= seq_len) {
            break;          /* context full: the reply ends here */
        }
        logits = feed(m, &next, 1);
        if (logits == NULL) {
            status = -1;
            break;
        }
    }
    m->stats.generated_tokens = n;
    m->stats.generate_seconds = now_seconds() - t0;
    return status;
}

static void begin_call(PocketModel *m)
{
    memset(&m->stats, 0, sizeof(m->stats));
    m->error[0] = '\0';
    text_clear(&m->out);
}

const char *pocket_model_generate(PocketModel *m, const char *prompt,
                                  const PocketGenOptions *options,
                                  PocketTokenFn on_token, void *user)
{
    Settings s;

    begin_call(m);
    if (prompt == NULL || settings_from(m, options, "top-k", 256, &s) != 0) {
        return NULL;
    }
    pocket_model_reset(m);
    float *logits = feed_text(m, prompt, 1);
    if (logits == NULL || generate(m, logits, &s, 0, on_token, user) != 0) {
        return NULL;
    }
    return text_get(&m->out);
}

/* The text of one chat turn (malloc'd) and whether it starts with BOS. */
static char *chat_turn_text(const PocketModel *m, const char *message, int first,
                            int *add_bos)
{
    const char *system = m->system;
    size_t size = strlen(message) + (system ? strlen(system) : 0) + 128;
    char *text = malloc(size);

    if (text == NULL) {
        return NULL;
    }
    *add_bos = first;
    switch (m->format) {
    case FORMAT_ZEPHYR:
        if (first && system != NULL) {
            snprintf(text, size,
                     "<|system|>\n%s</s>\n<|user|>\n%s</s>\n<|assistant|>\n",
                     system, message);
        } else if (first) {
            snprintf(text, size, "<|user|>\n%s</s>\n<|assistant|>\n", message);
        } else {
            /* the previous reply's EOS is already in the cache */
            snprintf(text, size, "\n<|user|>\n%s</s>\n<|assistant|>\n", message);
        }
        break;
    case FORMAT_LLAMA2:
        *add_bos = 1;       /* every [INST] block opens with BOS */
        if (first && system != NULL) {
            snprintf(text, size, "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]",
                     system, message);
        } else {
            snprintf(text, size, "[INST] %s [/INST]", message);
        }
        break;
    case FORMAT_MISTRAL:
        if (first && system != NULL) {
            snprintf(text, size, "[INST] %s\n\n%s [/INST]", system, message);
        } else {
            snprintf(text, size, "[INST] %s [/INST]", message);
        }
        break;
    default:
        snprintf(text, size, "%s", message);
        break;
    }
    return text;
}

const char *pocket_model_chat(PocketModel *m, const char *message,
                              const PocketGenOptions *options,
                              PocketTokenFn on_token, void *user)
{
    Settings s;
    int add_bos = 1;

    begin_call(m);
    if (message == NULL || settings_from(m, options, "top-k", 256, &s) != 0) {
        return NULL;
    }
    if (m->format == FORMAT_NONE) {
        /* not a chat format: a fresh completion of the message */
        pocket_model_reset(m);
        float *logits = feed_text(m, message, 1);
        if (logits == NULL || generate(m, logits, &s, 0, on_token, user) != 0) {
            return NULL;
        }
        return text_get(&m->out);
    }

    if (m->cache != CACHE_CHAT) {
        pocket_model_reset(m);
    }
    const int first = (m->pos == 0);
    char *text = chat_turn_text(m, message, first, &add_bos);
    if (text == NULL) {
        model_fail(m, "out of memory");
        return NULL;
    }
    m->cache = CACHE_CHAT;
    float *logits = feed_text(m, text, add_bos);
    free(text);
    if (logits == NULL || generate(m, logits, &s, 1, on_token, user) != 0) {
        return NULL;
    }
    /* close the assistant turn with EOS so the next turn follows it */
    if (m->pos < engine_config(m->model)->seq_len) {
        int eos_id = m->tok->eos_id;
        if (feed(m, &eos_id, 1) == NULL) {
            return NULL;
        }
    }
    return text_get(&m->out);
}

/* ------------------------------------------------------------------ */
/* RAG                                                                 */
/* ------------------------------------------------------------------ */

struct PocketRag {
    char **paths;
    int n_paths;
    RagLoadOptions load;
    RagCorpus *corpus;
    RagIndex *index;
    RagSession *session;
    int auto_refresh;
    int last_found;
    Text reply, answer, source, offer, prompt, change, scratch;
    RagHit hits[MAX_SEARCH_HITS];
    int n_hits;
    char *hit_question;
    char error[512];
};

static void rag_fail(PocketRag *r, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(r->error, sizeof(r->error), fmt, ap);
    va_end(ap);
}

/* Read and index everything; on failure the loaded documents stay. */
static int rag_load(PocketRag *r, char *error, size_t cap)
{
    RagCorpus *c = rag_corpus_load((const char *const *)r->paths, r->n_paths,
                                   &r->load, error, cap);
    if (c == NULL) {
        return -1;
    }
    RagIndex *ix = rag_index_build(c, NULL, error, cap);
    if (ix == NULL) {
        rag_corpus_free(c);
        return -1;
    }
    rag_index_free(r->index);
    rag_corpus_free(r->corpus);
    r->corpus = c;
    r->index = ix;
    r->n_hits = 0;
    rag_session_reset(r->session);   /* row numbers of the old load are gone */
    return 0;
}

PocketRag *pocket_rag_open(const char *const *paths, int n_paths, int header_row,
                           int include_hidden, char *error, size_t error_capacity)
{
    PocketRag *r = calloc(1, sizeof(*r));

    if (r == NULL || n_paths <= 0) {
        snprintf(error, error_capacity, r ? "no folder or file given" : "out of memory");
        free(r);
        return NULL;
    }
    r->paths = calloc((size_t)n_paths, sizeof(char *));
    r->session = rag_session_new();
    if (r->paths == NULL || r->session == NULL) {
        snprintf(error, error_capacity, "out of memory");
        pocket_rag_close(r);
        return NULL;
    }
    for (int i = 0; i < n_paths; i++) {
        r->paths[i] = copy_string(paths[i]);
        if (r->paths[i] == NULL) {
            snprintf(error, error_capacity, "out of memory");
            pocket_rag_close(r);
            return NULL;
        }
        r->n_paths = i + 1;
    }
    r->load.header_row = header_row > 0 ? header_row : 0;
    r->load.include_hidden = include_hidden != 0;
    r->auto_refresh = 1;
    if (rag_load(r, error, error_capacity) != 0) {
        pocket_rag_close(r);
        return NULL;
    }
    return r;
}

void pocket_rag_close(PocketRag *r)
{
    if (r == NULL) {
        return;
    }
    rag_session_free(r->session);
    rag_index_free(r->index);
    rag_corpus_free(r->corpus);
    for (int i = 0; i < r->n_paths; i++) {
        free(r->paths[i]);
    }
    free(r->paths);
    free(r->reply.s);
    free(r->answer.s);
    free(r->source.s);
    free(r->offer.s);
    free(r->prompt.s);
    free(r->change.s);
    free(r->scratch.s);
    free(r->hit_question);
    free(r);
}

int pocket_rag_reload(PocketRag *r)
{
    return rag_load(r, r->error, sizeof(r->error));
}

int pocket_rag_refresh(PocketRag *r)
{
    char summary[1024];
    int changed = rag_corpus_changed(r->corpus, summary, sizeof(summary));

    if (changed == 0) {
        return 0;
    }
    if (changed < 0) {
        rag_fail(r, "cannot check the documents: %s", summary);
        return -1;
    }
    if (text_set(&r->change, summary) != 0 || pocket_rag_reload(r) != 0) {
        return -1;
    }
    return 1;
}

void pocket_rag_set_auto_refresh(PocketRag *r, int on)
{
    r->auto_refresh = on != 0;
}

const char *pocket_rag_change(const PocketRag *r)
{
    return text_get(&r->change);
}

void pocket_rag_reset(PocketRag *r)
{
    rag_session_reset(r->session);
}

const char *pocket_rag_error(const PocketRag *r)
{
    return r->error;
}

/* Copy text without the line that equals drop (if any). */
static int add_without_line(Text *to, const char *text, const char *drop)
{
    const size_t dl = strlen(drop);
    const char *p = text;

    while (*p != '\0') {
        const char *end = strchr(p, '\n');
        size_t len = end ? (size_t)(end - p) : strlen(p);
        if (!(dl > 0 && len == dl && memcmp(p, drop, dl) == 0)) {
            if (text_add(to, p, len) != 0 || text_add(to, "\n", 1) != 0) {
                return -1;
            }
        }
        p += len + (end ? 1 : 0);
    }
    return 0;
}

const char *pocket_rag_ask(PocketRag *r, const char *question)
{
    char *captured = NULL;
    size_t size = 0;

    r->error[0] = '\0';
    if (r->auto_refresh) {
        pocket_rag_refresh(r);
    }
    FILE *out = open_memstream(&captured, &size);
    if (out == NULL) {
        rag_fail(r, "out of memory");
        return NULL;
    }
    r->last_found = rag_answer(r->session, r->corpus, r->index,
                               question ? question : "", out);
    fclose(out);

    /* the reply, and the same text without its source and offer lines */
    char source_line[700];
    const char *source = rag_session_last_source(r->session);
    const char *closing = rag_session_last_offer(r->session);
    int bad = captured == NULL || text_set(&r->reply, captured) != 0 ||
              text_set(&r->source, source) != 0 || text_set(&r->offer, closing) != 0;
    if (!bad) {
        snprintf(source_line, sizeof(source_line), "Source: %s", source);
        r->scratch.len = 0;
        r->answer.len = 0;
        bad = add_without_line(&r->scratch, captured, source[0] ? source_line : "") != 0 ||
              text_reserve(&r->scratch, 0) != 0;
        if (!bad) {
            r->scratch.s[r->scratch.len] = '\0';
            bad = add_without_line(&r->answer, r->scratch.s, closing) != 0 ||
                  text_reserve(&r->answer, 0) != 0;
        }
        if (!bad) {
            while (r->answer.len > 0 && r->answer.s[r->answer.len - 1] == '\n') {
                r->answer.len--;
            }
            r->answer.s[r->answer.len] = '\0';
        }
    }
    free(captured);
    if (bad) {
        rag_fail(r, "out of memory");
        return NULL;
    }
    return text_get(&r->reply);
}

const char *pocket_rag_last_answer(const PocketRag *r)
{
    return text_get(&r->answer);
}

const char *pocket_rag_last_source(const PocketRag *r)
{
    return text_get(&r->source);
}

const char *pocket_rag_last_offer(const PocketRag *r)
{
    return text_get(&r->offer);
}

int pocket_rag_last_found(const PocketRag *r)
{
    return r->last_found;
}

const char *pocket_rag_last_prompt(const PocketRag *r)
{
    return text_get(&r->prompt);
}

/* The rows the model gets for a question: strong hits only, best first. */
static int select_rows(PocketRag *r, const char *question,
                       const PocketRagOptions *o, int max)
{
    const int k = (o && o->k > 0) ? o->k : 3;
    const double min_score = (o && o->min_score > 0) ? o->min_score : 1.0;
    const double keep = (o && o->keep > 0) ? o->keep : 0.35;
    int n = rag_index_search(r->index, question, r->hits,
                             max < MAX_SEARCH_HITS ? max : MAX_SEARCH_HITS);

    if (n > 0 && r->hits[0].score < min_score) {
        n = 0;
    }
    /*
     * Order the duplicates among ALL strong candidates before cutting to k:
     * otherwise, with k = 1, a short "see the other sheet" row ranked first
     * would be the only row left, and the full row of the same course
     * (ranked second) could never take its place.
     */
    int strong = 0;
    while (strong < n && r->hits[strong].score >= keep * r->hits[0].score) {
        strong++;
    }
    rag_order_duplicates(r->corpus, r->hits, strong);
    return strong < k ? strong : k;
}

static int remember_question(PocketRag *r, const char *question)
{
    char *copy = copy_string(question);
    if (copy == NULL) {
        return -1;
    }
    free(r->hit_question);
    r->hit_question = copy;
    return 0;
}

int pocket_rag_search(PocketRag *r, const char *question, int max_hits,
                      int for_model, const PocketRagOptions *rag_options)
{
    r->error[0] = '\0';
    r->n_hits = 0;
    if (question == NULL || max_hits <= 0) {
        return 0;
    }
    if (r->auto_refresh) {
        pocket_rag_refresh(r);
    }
    if (remember_question(r, question) != 0) {
        rag_fail(r, "out of memory");
        return -1;
    }
    if (for_model) {
        r->n_hits = select_rows(r, question, rag_options, MAX_SEARCH_HITS);
    } else {
        r->n_hits = rag_index_search(r->index, question, r->hits,
                                     max_hits < MAX_SEARCH_HITS ? max_hits
                                                                : MAX_SEARCH_HITS);
    }
    return r->n_hits;
}

static const RagChunk *hit_chunk(const PocketRag *r, int hit)
{
    return (hit >= 0 && hit < r->n_hits) ? rag_corpus_chunk(r->corpus, r->hits[hit].chunk)
                                         : NULL;
}

const char *pocket_rag_hit_text(PocketRag *r, int hit)
{
    const RagChunk *ch = hit_chunk(r, hit);
    if (ch == NULL) {
        return "";
    }
    char *text = rag_render_chunk(ch, r->hit_question ? r->hit_question : "",
                                  hit + 1, 0);
    int bad = text == NULL || text_set(&r->scratch, text) != 0;
    free(text);
    return bad ? "" : text_get(&r->scratch);
}

/* 04_UG_DS.xlsx, tab "Core", row 10 */
static int add_hit_source(Text *t, const RagChunk *ch)
{
    char buf[512];
    int n = snprintf(buf, sizeof(buf), "%.250s", ch->file);

    if (ch->sheet[0] != '\0') {
        n += snprintf(buf + n, sizeof(buf) - (size_t)n, ", tab \"%.150s\"", ch->sheet);
    }
    if (ch->row_first == ch->row_last) {
        n += snprintf(buf + n, sizeof(buf) - (size_t)n, ", row %d", ch->row_first);
    } else {
        n += snprintf(buf + n, sizeof(buf) - (size_t)n, ", rows %d-%d",
                      ch->row_first, ch->row_last);
    }
    if (ch->kind == RAG_CHUNK_NOTE) {
        n += snprintf(buf + n, sizeof(buf) - (size_t)n, " (note)");
    }
    return text_add(t, buf, (size_t)n);
}

const char *pocket_rag_hit_source(PocketRag *r, int hit)
{
    const RagChunk *ch = hit_chunk(r, hit);
    r->scratch.len = 0;
    if (ch == NULL || add_hit_source(&r->scratch, ch) != 0) {
        return "";
    }
    return text_get(&r->scratch);
}

double pocket_rag_hit_score(const PocketRag *r, int hit)
{
    return (hit >= 0 && hit < r->n_hits) ? r->hits[hit].score : 0.0;
}

static int count_tokens(const char *text, void *user)
{
    int *ids = NULL;
    int n = encode(user, text, 0, &ids);
    free(ids);
    return n;
}

const char *pocket_rag_ask_model(PocketRag *r, PocketModel *m, const char *question,
                                 const PocketGenOptions *options,
                                 const PocketRagOptions *rag_options,
                                 PocketTokenFn on_token, void *user)
{
    Settings s;
    RagPromptOptions po;
    char error[512];

    r->error[0] = '\0';
    r->n_hits = 0;
    text_clear(&r->source);
    text_clear(&r->prompt);
    begin_call(m);
    if (question == NULL) {
        rag_fail(r, "no question given");
        return NULL;
    }
    if (!m->is_chat) {
        rag_fail(r, "answering from the files with a model needs a chat model "
                    "that can follow instructions (a GGUF model such as "
                    "TinyLlama); a llama2.c story model can only continue text");
        return NULL;
    }
    if (settings_from(m, options, "greedy", 160, &s) != 0) {
        rag_fail(r, "%s", m->error);
        return NULL;
    }
    if (r->auto_refresh) {
        pocket_rag_refresh(r);
    }

    memset(&po, 0, sizeof(po));
    po.system = (rag_options && rag_options->system && rag_options->system[0])
                    ? rag_options->system : NULL;
    po.compact = rag_options ? rag_options->compact : 0;
    const char *system = po.system ? po.system : RAG_DEFAULT_SYSTEM;

    /* The instruction is fed once and stays in the KV cache. */
    if (m->cache != CACHE_RAG || m->rag_system == NULL ||
        strcmp(m->rag_system, system) != 0) {
        char *prefix = rag_prompt_system(&po);
        char *copy = copy_string(system);
        if (prefix == NULL || copy == NULL) {
            free(prefix);
            free(copy);
            rag_fail(r, "out of memory");
            return NULL;
        }
        pocket_model_reset(m);
        float *ok = feed_text(m, prefix, 1);
        free(prefix);
        if (ok == NULL) {
            free(copy);
            rag_fail(r, "%s", m->error);
            return NULL;
        }
        free(m->rag_system);
        m->rag_system = copy;
        m->rag_prefix_pos = m->pos;
        m->cache = CACHE_RAG;
        memset(&m->stats, 0, sizeof(m->stats));   /* count the question only */
    }

    if (remember_question(r, question) != 0) {
        rag_fail(r, "out of memory");
        return NULL;
    }
    int n = select_rows(r, question, rag_options, MAX_SEARCH_HITS);
    if (n == 0) {
        if (text_set(&m->out, "I could not find this in the files.") != 0) {
            rag_fail(r, "out of memory");
            return NULL;
        }
        if (on_token != NULL) {
            on_token(m->out.s, user);
        }
        return text_get(&m->out);
    }

    int used = 0, prompt_tokens = 0;
    po.max_tokens = engine_config(m->model)->seq_len - m->rag_prefix_pos - s.max_tokens;
    char *prompt = rag_prompt_question(r->corpus, r->hits, n, question, &po,
                                       count_tokens, m, &used, &prompt_tokens,
                                       error, sizeof(error));
    if (prompt == NULL) {
        rag_fail(r, "%s", error);
        return NULL;
    }
    r->n_hits = used;
    int bad = text_set(&r->prompt, prompt) != 0;
    for (int i = 0; i < used && !bad; i++) {
        char number[16];
        int len = snprintf(number, sizeof(number), "%s[%d] ", i ? "\n" : "", i + 1);
        bad = text_add(&r->source, number, (size_t)len) != 0 ||
              add_hit_source(&r->source, rag_corpus_chunk(r->corpus, r->hits[i].chunk)) != 0;
    }
    if (bad) {
        free(prompt);
        rag_fail(r, "out of memory");
        return NULL;
    }

    m->pos = m->rag_prefix_pos;       /* forget the previous question */
    float *logits = feed_text(m, prompt, 0);
    free(prompt);
    if (logits == NULL || generate(m, logits, &s, 1, on_token, user) != 0) {
        rag_fail(r, "%s", m->error);
        return NULL;
    }
    return text_get(&m->out);
}

int pocket_rag_row_count(const PocketRag *r)
{
    return rag_corpus_chunk_count(r->corpus);
}

int pocket_rag_term_count(const PocketRag *r)
{
    return rag_index_term_count(r->index);
}

int pocket_rag_file_count(const PocketRag *r)
{
    return rag_corpus_file_count(r->corpus);
}

const char *pocket_rag_file_name(const PocketRag *r, int file)
{
    const RagFileInfo *f = rag_corpus_file(r->corpus, file);
    return f ? f->name : "";
}

int pocket_rag_file_readable(const PocketRag *r, int file)
{
    const RagFileInfo *f = rag_corpus_file(r->corpus, file);
    return f ? f->ok : 0;
}

int pocket_rag_file_rows(const PocketRag *r, int file)
{
    const RagFileInfo *f = rag_corpus_file(r->corpus, file);
    return f ? f->chunks : 0;
}

int pocket_rag_file_tables(const PocketRag *r, int file)
{
    const RagFileInfo *f = rag_corpus_file(r->corpus, file);
    return f ? f->tabs_used : 0;
}

int pocket_rag_warning_count(const PocketRag *r)
{
    return rag_corpus_warning_count(r->corpus);
}

const char *pocket_rag_warning(const PocketRag *r, int index)
{
    const char *w = rag_corpus_warning(r->corpus, index);
    return w ? w : "";
}
