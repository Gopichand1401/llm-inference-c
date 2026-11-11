/*
 * rag_prompt.c - chunk rendering, field ordering and the context budget.
 * See include/rag_prompt.h.
 */
#include "rag_prompt.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Growable string                                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    char *s;
    size_t len, cap;
    int failed;
} Buf;

static void buf_add(Buf *b, const char *text, size_t n)
{
    if (b->failed) {
        return;
    }
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 256;
        while (cap < b->len + n + 1) {
            cap *= 2;
        }
        char *s = realloc(b->s, cap);
        if (s == NULL) {
            b->failed = 1;
            return;
        }
        b->s = s;
        b->cap = cap;
    }
    memcpy(b->s + b->len, text, n);
    b->len += n;
    b->s[b->len] = '\0';
}

static void buf_puts(Buf *b, const char *text)
{
    buf_add(b, text, strlen(text));
}

static void buf_printf(Buf *b, const char *fmt, ...)
{
    char tmp[512];
    va_list ap;

    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) {
        b->failed = 1;
        return;
    }
    buf_add(b, tmp, (size_t)n < sizeof(tmp) ? (size_t)n : sizeof(tmp) - 1);
}

/* Append value, cut at max bytes (on a UTF-8 boundary) with "...". */
static void buf_value(Buf *b, const char *value, int max)
{
    size_t n = strlen(value);
    if (max <= 0 || n <= (size_t)max) {
        buf_add(b, value, n);
        return;
    }
    size_t cut = (size_t)max;
    while (cut > 0 && ((unsigned char)value[cut] & 0xC0) == 0x80) {
        cut--;
    }
    while (cut > 0 && value[cut - 1] == ' ') {
        cut--;
    }
    buf_add(b, value, cut);
    buf_puts(b, "...");
}

/* ------------------------------------------------------------------ */
/* Which fields does the question ask for?                             */
/* ------------------------------------------------------------------ */

/* Question word -> label term (label terms are stemmed, see rag_index.h). */
static const char *const SYNONYMS[][2] = {
    {"who", "instructor"}, {"who", "faculty"}, {"who", "teacher"},
    {"teach", "instructor"}, {"teaches", "instructor"},
    {"teaching", "instructor"}, {"taught", "instructor"},
    {"teacher", "instructor"}, {"professor", "instructor"},
    {"faculty", "instructor"}, {"teach", "faculty"}, {"teaches", "faculty"},
    {"taught", "faculty"}, {"instructor", "faculty"},
    {"when", "slot"}, {"when", "time"}, {"when", "timing"}, {"when", "day"},
    {"when", "date"}, {"time", "slot"}, {"timing", "slot"},
    {"timings", "slot"}, {"schedule", "slot"}, {"day", "slot"},
    {"where", "room"}, {"where", "venue"}, {"where", "location"},
    {"venue", "room"}, {"location", "room"}, {"classroom", "room"},
    {"place", "room"}, {"hall", "room"}
};

#define MAX_WORDS 64

typedef struct {
    char words[MAX_WORDS][32];   /* raw lower-case words of the question */
    int n_words;
    char terms[MAX_WORDS][32];   /* rag_tokenize terms of the question */
    int n_terms;
} QuestionWords;

static void collect_term(const char *term, void *user)
{
    QuestionWords *q = user;
    if (q->n_terms < MAX_WORDS && strlen(term) < 32) {
        strcpy(q->terms[q->n_terms++], term);
    }
}

static void question_words(const char *question, QuestionWords *q)
{
    memset(q, 0, sizeof(*q));
    rag_tokenize(question, collect_term, q);
    const char *p = question;
    while (*p && q->n_words < MAX_WORDS) {
        while (*p && !isalnum((unsigned char)*p)) {
            p++;
        }
        size_t n = 0;
        char w[32];
        while (*p && isalnum((unsigned char)*p)) {
            if (n < sizeof(w) - 1) {
                w[n++] = (char)tolower((unsigned char)*p);
            }
            p++;
        }
        if (n > 0) {
            w[n] = '\0';
            strcpy(q->words[q->n_words++], w);
        }
    }
}

typedef struct {
    const QuestionWords *q;
    int hit;
} LabelMatch;

static void match_label_term(const char *term, void *user)
{
    LabelMatch *m = user;
    if (m->hit) {
        return;
    }
    for (int i = 0; i < m->q->n_terms; i++) {
        if (strcmp(term, m->q->terms[i]) == 0) {
            m->hit = 1;
            return;
        }
    }
    for (size_t s = 0; s < sizeof(SYNONYMS) / sizeof(SYNONYMS[0]); s++) {
        if (strcmp(term, SYNONYMS[s][1]) != 0) {
            continue;
        }
        for (int i = 0; i < m->q->n_words; i++) {
            if (strcmp(m->q->words[i], SYNONYMS[s][0]) == 0) {
                m->hit = 1;
                return;
            }
        }
    }
}

static int label_asked(const char *label, const QuestionWords *q)
{
    LabelMatch m = {q, 0};
    rag_tokenize(label, match_label_term, &m);
    return m.hit;
}

/* ------------------------------------------------------------------ */
/* Rendering                                                           */
/* ------------------------------------------------------------------ */

static void render(Buf *b, const RagChunk *ch, const QuestionWords *q,
                   int number, int max_value, int compact)
{
    char where[400];
    if (ch->sheet[0] != '\0') {
        snprintf(where, sizeof(where), "%.200s, tab %.150s", ch->file, ch->sheet);
    } else {
        snprintf(where, sizeof(where), "%.200s", ch->file);
    }
    if (ch->kind == RAG_CHUNK_NOTE) {
        buf_printf(b, "[%d] Note in %s: ", number, where);
        buf_value(b, ch->note, max_value);
        buf_puts(b, "\n");
        return;
    }

    if (ch->row_first == ch->row_last) {
        buf_printf(b, "[%d] %s, row %d", number, where, ch->row_first);
    } else {
        buf_printf(b, "[%d] %s, rows %d-%d", number, where, ch->row_first,
                   ch->row_last);
    }
    if (ch->title[0] != '\0') {
        buf_puts(b, " (");
        buf_value(b, ch->title, 120);
        buf_puts(b, ")");
    }
    buf_puts(b, "\n");

    /* key fields, then asked fields, then the rest (common fields only).
     * compact: leave out the rest when the question asked for any field. */
    int n_asked = 0;
    for (int f = 0; f < ch->n_fields; f++) {
        n_asked += !ch->fields[f].key && label_asked(ch->fields[f].label, q);
    }
    const int passes = (compact && n_asked > 0) ? 2 : 3;
    for (int pass = 0; pass < passes; pass++) {
        for (int f = 0; f < ch->n_fields; f++) {
            const RagField *fd = &ch->fields[f];
            if (fd->variant != 0) {
                continue;
            }
            int asked = !fd->key && label_asked(fd->label, q);
            int want = pass == 0 ? fd->key : pass == 1 ? asked : (!fd->key && !asked);
            if (!want) {
                continue;
            }
            buf_puts(b, fd->label);
            buf_puts(b, ": ");
            buf_value(b, fd->value, max_value);
            buf_puts(b, "\n");
        }
    }

    if (ch->n_variants > 1) {
        buf_puts(b, "Separate rows:\n");
        for (int v = 1; v <= ch->n_variants; v++) {
            int first = 1;
            for (int f = 0; f < ch->n_fields; f++) {
                const RagField *fd = &ch->fields[f];
                if (fd->variant != v) {
                    continue;
                }
                if (first) {
                    buf_printf(b, "- row %d: ", ch->variant_rows[v - 1]);
                    first = 0;
                } else {
                    buf_puts(b, "; ");
                }
                buf_puts(b, fd->label);
                buf_puts(b, ": ");
                buf_value(b, fd->value, max_value);
            }
            if (!first) {
                buf_puts(b, "\n");
            }
        }
    }
}

char *rag_render_chunk(const RagChunk *chunk, const char *question,
                       int number, int max_value_bytes)
{
    QuestionWords q;
    Buf b = {0};

    question_words(question ? question : "", &q);
    render(&b, chunk, &q, number, max_value_bytes > 0 ? max_value_bytes : 300, 0);
    if (b.failed) {
        free(b.s);
        return NULL;
    }
    return b.s;
}

static const char *first_key(const RagChunk *ch)
{
    for (int f = 0; f < ch->n_fields; f++) {
        if (ch->fields[f].key && ch->fields[f].variant == 0) {
            return ch->fields[f].value;
        }
    }
    return NULL;
}

void rag_order_duplicates(const RagCorpus *corpus, RagHit *hits, int n)
{
    for (int i = 0; i < n; i++) {
        const RagChunk *a = rag_corpus_chunk(corpus, hits[i].chunk);
        const char *key = first_key(a);
        if (key == NULL) {
            continue;
        }
        /* best (most fields) member of this key's group at or after i */
        int best = i;
        for (int j = i + 1; j < n; j++) {
            const RagChunk *b = rag_corpus_chunk(corpus, hits[j].chunk);
            const char *kb = first_key(b);
            if (kb != NULL && strcmp(kb, key) == 0 &&
                b->n_fields > rag_corpus_chunk(corpus, hits[best].chunk)->n_fields) {
                best = j;
            }
        }
        if (best != i) {
            /* move hits[best] to position i, shifting i .. best-1 down */
            RagHit keep = hits[best];
            memmove(&hits[i + 1], &hits[i], (size_t)(best - i) * sizeof(*hits));
            hits[i] = keep;
        }
    }
}

char *rag_prompt_system(const RagPromptOptions *options)
{
    const char *system = (options && options->system) ? options->system
                                                      : RAG_DEFAULT_SYSTEM;
    Buf b = {0};
    buf_puts(&b, "<|system|>\n");
    buf_puts(&b, system);
    buf_puts(&b, "</s>");
    if (b.failed) {
        free(b.s);
        return NULL;
    }
    return b.s;
}

static char *build_question(const RagCorpus *corpus, const RagHit *hits, int n,
                            const char *question, const QuestionWords *q,
                            int max_value, int compact)
{
    Buf b = {0};

    buf_puts(&b, "\n<|user|>\nSpreadsheet rows:\n\n");
    for (int i = 0; i < n; i++) {
        render(&b, rag_corpus_chunk(corpus, hits[i].chunk), q, i + 1, max_value,
               compact);
        buf_puts(&b, "\n");
    }
    buf_puts(&b, "Question: ");
    buf_puts(&b, question);
    buf_puts(&b, "</s>\n<|assistant|>\n");
    if (b.failed) {
        free(b.s);
        return NULL;
    }
    return b.s;
}

char *rag_prompt_question(const RagCorpus *corpus, const RagHit *hits,
                          int n_hits, const char *question,
                          const RagPromptOptions *options,
                          RagCountTokens count, void *count_user,
                          int *n_used, int *n_tokens,
                          char *error, size_t error_capacity)
{
    QuestionWords q;
    const int budget = options ? options->max_tokens : 0;
    int max_value = (options && options->max_value_bytes > 0)
                        ? options->max_value_bytes : 300;
    static const int shorter[] = {150, 80, 40};

    if (n_hits <= 0) {
        snprintf(error, error_capacity, "no rows to put in the prompt");
        return NULL;
    }
    question_words(question, &q);

    int n = n_hits;
    int step = 0;
    for (;;) {
        char *text = build_question(corpus, hits, n, question, &q, max_value,
                                    options ? options->compact : 0);
        if (text == NULL) {
            snprintf(error, error_capacity, "out of memory");
            return NULL;
        }
        int tokens = count(text, count_user);
        if (tokens < 0) {
            free(text);
            snprintf(error, error_capacity, "cannot count prompt tokens");
            return NULL;
        }
        if (tokens <= budget) {
            *n_used = n;
            *n_tokens = tokens;
            return text;
        }
        free(text);
        if (n > 1) {
            n--;                               /* drop the lowest-ranked row */
        } else if (step < 3 && shorter[step] < max_value) {
            max_value = shorter[step++];       /* then cut long values */
        } else if (step < 3) {
            step++;
        } else {
            snprintf(error, error_capacity,
                     "the question and one row need %d tokens, more than the "
                     "%d available; use a shorter question, a smaller -n or "
                     "a larger --ctx", tokens, budget);
            return NULL;
        }
    }
}
