/*
 * rag_index.c - term splitting and BM25 search for RAG.
 * The rules are documented in include/rag_index.h.
 */
#include "rag_index.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Terms                                                               */
/* ------------------------------------------------------------------ */

static const char *const STOP_WORDS[] = {
    "a", "about", "an", "and", "any", "are", "as", "at", "be", "by", "can",
    "could", "do", "does", "find", "for", "from", "get", "give", "handle",
    "handles", "handling", "has", "have", "he", "her", "his", "how", "i",
    "in", "is", "it", "its", "know", "list", "me", "my", "name", "need",
    "of", "on", "or", "please", "she", "show", "take", "taken", "takes",
    "taking", "taught", "teach", "teaches", "teaching", "tell", "that",
    "the", "their", "them", "there", "these", "they", "this", "those", "to",
    "want", "was", "were", "what", "when", "where", "which", "who", "whom",
    "whose", "why", "will", "with", "would", "you", "your"
};

static int is_stop_word(const char *w)
{
    size_t lo = 0, hi = sizeof(STOP_WORDS) / sizeof(STOP_WORDS[0]);
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        int c = strcmp(w, STOP_WORDS[mid]);
        if (c == 0) {
            return 1;
        }
        if (c < 0) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return 0;
}

static int is_piece_char(unsigned char ch)
{
    return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
           (ch >= '0' && ch <= '9') || ch >= 0x80;
}

static int ends_with(const char *w, size_t len, const char *suffix)
{
    size_t n = strlen(suffix);
    return len >= n && memcmp(w + len - n, suffix, n) == 0;
}

/* Light stemming of a lower-case alphabetic word, in place. */
static void stem(char *w)
{
    size_t len = strlen(w);

    if (len > 4 && ends_with(w, len, "ies")) {
        w[len - 3] = 'y';
        w[len - 2] = '\0';
    } else if (len > 4 && (ends_with(w, len, "sses") || ends_with(w, len, "xes") ||
                           ends_with(w, len, "ches") || ends_with(w, len, "shes"))) {
        w[len - 2] = '\0';
    } else if (len > 3 && w[len - 1] == 's' && !ends_with(w, len, "ss") &&
               !ends_with(w, len, "us") && !ends_with(w, len, "is")) {
        w[len - 1] = '\0';
    }
}

void rag_tokenize(const char *text,
                  void (*emit)(const char *term, void *user), void *user)
{
    size_t n = strlen(text);
    char *piece = malloc(n + 1);
    char *prev = malloc(n + 1);
    char *joined = malloc(2 * n + 2);
    int have_prev = 0, prev_has_letter = 0;
    size_t prev_end = 0;

    if (piece == NULL || prev == NULL || joined == NULL) {
        free(piece);
        free(prev);
        free(joined);
        return;
    }

    size_t i = 0;
    while (i < n) {
        if (!is_piece_char((unsigned char)text[i])) {
            i++;
            continue;
        }
        size_t start = i, len = 0;
        int alpha = 1, digits = 1, has_letter = 0;
        while (i < n && is_piece_char((unsigned char)text[i])) {
            unsigned char ch = (unsigned char)text[i];
            if (ch >= 'A' && ch <= 'Z') {
                ch = (unsigned char)(ch - 'A' + 'a');
            }
            piece[len++] = (char)ch;
            if (ch >= 'a' && ch <= 'z') {
                has_letter = 1;
                digits = 0;
            } else if (ch >= '0' && ch <= '9') {
                alpha = 0;
            } else {
                alpha = 0;
                digits = 0;
            }
            i++;
        }
        piece[len] = '\0';

        /* Join "DS" + "3010" style identifiers across a short separator. */
        int join = 0;
        if (have_prev && prev_has_letter && digits) {
            size_t gap = start - prev_end;
            join = gap >= 1 && gap <= 2;
            for (size_t g = prev_end; g < start && join; g++) {
                join = text[g] == ' ' || text[g] == '-' || text[g] == '_';
            }
        }
        if (join) {
            snprintf(joined, 2 * n + 2, "%s%s", prev, piece);
        }
        memcpy(prev, piece, len + 1);
        prev_has_letter = has_letter;
        prev_end = i;
        have_prev = 1;

        if (alpha) {
            if (!is_stop_word(piece)) {
                char *w = joined + (join ? strlen(joined) + 1 : 0);
                /* stem a copy so prev keeps the raw piece */
                memcpy(w, piece, len + 1);
                stem(w);
                if (!is_stop_word(w)) {     /* "names" -> "name": also a stop word */
                    emit(w, user);
                }
            }
        } else {
            emit(piece, user);
        }
        if (join) {
            emit(joined, user);
        }
    }
    free(piece);
    free(prev);
    free(joined);
}

/* ------------------------------------------------------------------ */
/* Index                                                               */
/* ------------------------------------------------------------------ */

typedef struct {
    int chunk;
    double tf;
} Posting;

typedef struct {
    char *text;
    Posting *postings;
    int n_postings, cap;
    int is_label;         /* also occurs in a column label */
} Term;

struct RagIndex {
    const RagCorpus *corpus;
    double k1, b;
    Term *terms;
    int n_terms, terms_cap;
    int *slots;           /* hash: term index + 1, 0 = free */
    int slot_cap;         /* power of two */
    double *doc_len;
    int n_docs;
    double avg_len;
    double label_weight, name_bonus;
    /* Distinct term ids of every key field: field f of the whole index is
     * key_terms[key_start[f] .. key_start[f + 1]); chunk d owns fields
     * chunk_key[d] .. chunk_key[d + 1]. */
    int *key_terms, n_key_terms, key_terms_cap;
    int *key_start, n_key_fields, key_fields_cap;
    int *chunk_key;
    int failed;
};

static unsigned long hash_term(const char *s)
{
    unsigned long h = 2166136261u;
    for (; *s; s++) {
        h = (h ^ (unsigned char)*s) * 16777619u;
    }
    return h;
}

static int *find_slot(const RagIndex *ix, const char *term)
{
    unsigned long mask = (unsigned long)ix->slot_cap - 1;
    unsigned long i = hash_term(term) & mask;
    for (;;) {
        int *slot = &ix->slots[i];
        if (*slot == 0 || strcmp(ix->terms[*slot - 1].text, term) == 0) {
            return slot;
        }
        i = (i + 1) & mask;
    }
}

static int grow_slots(RagIndex *ix)
{
    int cap = ix->slot_cap ? ix->slot_cap * 2 : 1024;
    int *old = ix->slots;
    ix->slots = calloc((size_t)cap, sizeof(int));
    if (ix->slots == NULL) {
        ix->slots = old;
        return -1;
    }
    ix->slot_cap = cap;
    for (int t = 0; t < ix->n_terms; t++) {
        *find_slot(ix, ix->terms[t].text) = t + 1;
    }
    free(old);
    return 0;
}

typedef struct {
    RagIndex *ix;
    int chunk;
    double weight;
    int virtual_term;     /* 1: not counted in the chunk length (acronyms) */
} AddCtx;

static void add_term(const char *term, void *user)
{
    AddCtx *a = user;
    RagIndex *ix = a->ix;

    if (ix->failed) {
        return;
    }
    if (2 * (ix->n_terms + 1) > ix->slot_cap && grow_slots(ix) != 0) {
        ix->failed = 1;
        return;
    }
    int *slot = find_slot(ix, term);
    if (*slot == 0) {
        if (ix->n_terms == ix->terms_cap) {
            int cap = ix->terms_cap ? ix->terms_cap * 2 : 1024;
            Term *t = realloc(ix->terms, (size_t)cap * sizeof(*t));
            if (t == NULL) {
                ix->failed = 1;
                return;
            }
            ix->terms = t;
            ix->terms_cap = cap;
        }
        Term *t = &ix->terms[ix->n_terms];
        size_t len = strlen(term);
        t->text = malloc(len + 1);
        if (t->text == NULL) {
            ix->failed = 1;
            return;
        }
        memcpy(t->text, term, len + 1);
        t->postings = NULL;
        t->n_postings = t->cap = 0;
        t->is_label = 0;
        *slot = ++ix->n_terms;
    }
    Term *t = &ix->terms[*slot - 1];
    if (t->n_postings > 0 && t->postings[t->n_postings - 1].chunk == a->chunk) {
        t->postings[t->n_postings - 1].tf += a->weight;
    } else {
        if (t->n_postings == t->cap) {
            int cap = t->cap ? t->cap * 2 : 4;
            Posting *p = realloc(t->postings, (size_t)cap * sizeof(*p));
            if (p == NULL) {
                ix->failed = 1;
                return;
            }
            t->postings = p;
            t->cap = cap;
        }
        t->postings[t->n_postings].chunk = a->chunk;
        t->postings[t->n_postings].tf = a->weight;
        t->n_postings++;
    }
    if (!a->virtual_term) {
        ix->doc_len[a->chunk] += a->weight;
    }
}

static void index_text(AddCtx *a, const char *text, double weight)
{
    a->weight = weight;
    a->virtual_term = 0;
    rag_tokenize(text, add_term, a);
}

/* Find a term, or create it without postings. Returns its id or -1. */
static int intern_term(RagIndex *ix, const char *term)
{
    if (2 * (ix->n_terms + 1) > ix->slot_cap && grow_slots(ix) != 0) {
        ix->failed = 1;
        return -1;
    }
    int *slot = find_slot(ix, term);
    if (*slot == 0) {
        if (ix->n_terms == ix->terms_cap) {
            int cap = ix->terms_cap ? ix->terms_cap * 2 : 1024;
            Term *t = realloc(ix->terms, (size_t)cap * sizeof(*t));
            if (t == NULL) {
                ix->failed = 1;
                return -1;
            }
            ix->terms = t;
            ix->terms_cap = cap;
        }
        Term *t = &ix->terms[ix->n_terms];
        size_t len = strlen(term);
        t->text = malloc(len + 1);
        if (t->text == NULL) {
            ix->failed = 1;
            return -1;
        }
        memcpy(t->text, term, len + 1);
        t->postings = NULL;
        t->n_postings = t->cap = 0;
        t->is_label = 0;
        *slot = ++ix->n_terms;
    }
    return *slot - 1;
}

/* ------------------------------------------------------------------ */
/* Short forms: initials of the words of a name                        */
/* ------------------------------------------------------------------ */

#define ACR_MAX_WORDS 16
#define ACR_MIN 2
#define ACR_MAX 6

static int is_alpha_term(const char *t)
{
    if (t[0] == '\0') {
        return 0;
    }
    for (; *t; t++) {
        if (*t < 'a' || *t > 'z') {
            return 0;
        }
    }
    return 1;
}

/*
 * Call emit("~xy...") for the initials of every run of 2 to 6 consecutive
 * words of text. Words are the purely alphabetic pieces; the runs are
 * taken twice: over all words (a run may not start or end on a stop word)
 * and with the stop words left out:
 * "Theory of Computation"            -> ~toc, ~tc
 * "Introduction to Machine Learning" -> ~itm ~itml ~ml, ~im ~iml (~ml again)
 * A short form can be emitted more than once.
 */
static void acronyms_of(const char *text,
                        void (*emit)(const char *term, void *user), void *user)
{
    char all[ACR_MAX_WORDS + 1], sig[ACR_MAX_WORDS + 1];
    char all_stop[ACR_MAX_WORDS + 1];
    int n_all = 0, n_sig = 0;
    char word[RAG_TERM_MAX];
    const size_t n = strlen(text);
    size_t i = 0;

    while (i < n) {
        if (!is_piece_char((unsigned char)text[i])) {
            i++;
            continue;
        }
        size_t len = 0;
        int alpha = 1;
        while (i < n && is_piece_char((unsigned char)text[i])) {
            unsigned char ch = (unsigned char)text[i++];
            if (ch >= 'A' && ch <= 'Z') {
                ch = (unsigned char)(ch - 'A' + 'a');
            }
            if (ch < 'a' || ch > 'z') {
                alpha = 0;
            }
            if (len < sizeof(word) - 1) {
                word[len++] = (char)ch;
            }
        }
        word[len] = '\0';
        if (!alpha) {
            continue;
        }
        const int stop = is_stop_word(word);
        if (n_all < ACR_MAX_WORDS) {
            all_stop[n_all] = (char)stop;
            all[n_all++] = word[0];
        }
        if (n_sig < ACR_MAX_WORDS && !stop) {
            sig[n_sig++] = word[0];
        }
    }

    for (int pass = 0; pass < 2; pass++) {
        const char *initials = pass == 0 ? all : sig;
        const int count = pass == 0 ? n_all : n_sig;
        char acr[ACR_MAX + 2];
        for (int a = 0; a < count; a++) {
            for (int l = ACR_MIN; l <= ACR_MAX && a + l <= count; l++) {
                if (pass == 0 && (all_stop[a] || all_stop[a + l - 1])) {
                    continue;      /* a run never starts or ends on a stop word */
                }
                acr[0] = '~';
                memcpy(acr + 1, initials + a, (size_t)l);
                acr[l + 1] = '\0';
                emit(acr, user);
            }
        }
    }
}

typedef struct {
    AddCtx add;
    char seen[2 * ACR_MAX_WORDS * ACR_MAX][ACR_MAX + 2];
    int n_seen;
} AcrCtx;

/* Each short form counts once per key field. */
static void add_acronym(const char *term, void *user)
{
    AcrCtx *a = user;
    for (int i = 0; i < a->n_seen; i++) {
        if (strcmp(a->seen[i], term) == 0) {
            return;
        }
    }
    strcpy(a->seen[a->n_seen++], term);
    add_term(term, &a->add);
}

typedef struct {
    const char *want;
    int found;
} HasCtx;

static void has_term_cb(const char *term, void *user)
{
    HasCtx *h = user;
    if (!h->found && strcmp(term, h->want) == 0) {
        h->found = 1;
    }
}

int rag_text_has_term(const char *text, const char *term)
{
    HasCtx h = {term, 0};
    if (term[0] == '~') {
        acronyms_of(text, has_term_cb, &h);
    } else {
        rag_tokenize(text, has_term_cb, &h);
    }
    return h.found;
}

/* Label words are searchable vocabulary even when no cell contains them. */
static void mark_label(const char *term, void *user)
{
    RagIndex *ix = user;
    int id = ix->failed ? -1 : intern_term(ix, term);
    if (id >= 0) {
        ix->terms[id].is_label = 1;
    }
}

static void add_key_term(const char *term, void *user)
{
    RagIndex *ix = user;
    int slot = *find_slot(ix, term);
    int start = ix->key_start[ix->n_key_fields];

    if (ix->failed || slot == 0) {
        return;
    }
    for (int i = start; i < ix->n_key_terms; i++) {
        if (ix->key_terms[i] == slot - 1) {
            return;
        }
    }
    if (ix->n_key_terms == ix->key_terms_cap) {
        int cap = ix->key_terms_cap ? ix->key_terms_cap * 2 : 1024;
        int *k = realloc(ix->key_terms, (size_t)cap * sizeof(*k));
        if (k == NULL) {
            ix->failed = 1;
            return;
        }
        ix->key_terms = k;
        ix->key_terms_cap = cap;
    }
    ix->key_terms[ix->n_key_terms++] = slot - 1;
}

/* Record the distinct terms of one key field (after all terms exist). */
static void add_key_field(RagIndex *ix, const char *value)
{
    if (ix->n_key_fields + 2 > ix->key_fields_cap) {
        int cap = ix->key_fields_cap ? ix->key_fields_cap * 2 : 256;
        int *k = realloc(ix->key_start, (size_t)cap * sizeof(*k));
        if (k == NULL) {
            ix->failed = 1;
            return;
        }
        ix->key_start = k;
        ix->key_fields_cap = cap;
    }
    ix->key_start[ix->n_key_fields] = ix->n_key_terms;
    rag_tokenize(value, add_key_term, ix);
    ix->n_key_fields++;
    ix->key_start[ix->n_key_fields] = ix->n_key_terms;
}

void rag_index_default_options(RagIndexOptions *o)
{
    o->k1 = 1.2;
    o->b = 0.75;
    o->key_weight = 3.0;
    o->label_weight = 0.3;
    o->name_bonus = 1.0;
}

RagIndex *rag_index_build(const RagCorpus *corpus,
                          const RagIndexOptions *options,
                          char *error, size_t error_capacity)
{
    RagIndex *ix = calloc(1, sizeof(*ix));
    double key_weight = 3.0;

    if (ix == NULL) {
        snprintf(error, error_capacity, "out of memory");
        return NULL;
    }
    ix->corpus = corpus;
    ix->k1 = 1.2;
    ix->b = 0.75;
    ix->label_weight = 0.3;
    ix->name_bonus = 1.0;
    if (options != NULL) {
        if (options->label_weight >= 0) {
            ix->label_weight = options->label_weight;
        }
        if (options->name_bonus >= 0) {
            ix->name_bonus = options->name_bonus;
        }
        if (options->k1 > 0) {
            ix->k1 = options->k1;
        }
        if (options->b >= 0) {
            ix->b = options->b;
        }
        if (options->key_weight > 0) {
            key_weight = options->key_weight;
        }
    }

    ix->n_docs = rag_corpus_chunk_count(corpus);
    ix->doc_len = calloc((size_t)ix->n_docs + 1, sizeof(double));
    if (ix->doc_len == NULL || grow_slots(ix) != 0) {
        rag_index_free(ix);
        snprintf(error, error_capacity, "out of memory");
        return NULL;
    }

    AddCtx a;
    a.ix = ix;
    for (int d = 0; d < ix->n_docs && !ix->failed; d++) {
        const RagChunk *ch = rag_corpus_chunk(corpus, d);
        a.chunk = d;
        if (ch->kind == RAG_CHUNK_NOTE) {
            index_text(&a, ch->note, 1.0);
            continue;
        }
        for (int f = 0; f < ch->n_fields; f++) {
            index_text(&a, ch->fields[f].value,
                       ch->fields[f].key ? key_weight : 1.0);
        }
        index_text(&a, ch->title, 1.0);
        index_text(&a, ch->sheet, 1.0);
        index_text(&a, ch->file, 1.0);
    }
    if (ix->failed) {
        rag_index_free(ix);
        snprintf(error, error_capacity, "out of memory while indexing");
        return NULL;
    }

    /* Column labels and key fields, once every term exists. */
    ix->chunk_key = calloc((size_t)ix->n_docs + 1, sizeof(int));
    if (ix->chunk_key == NULL || (ix->key_start = malloc(256 * sizeof(int))) == NULL) {
        ix->failed = 1;
    } else {
        ix->key_fields_cap = 256;
        ix->key_start[0] = 0;
    }
    for (int d = 0; d < ix->n_docs && !ix->failed; d++) {
        const RagChunk *ch = rag_corpus_chunk(corpus, d);
        ix->chunk_key[d] = ix->n_key_fields;
        for (int f = 0; f < ch->n_fields; f++) {
            rag_tokenize(ch->fields[f].label, mark_label, ix);
            if (ch->fields[f].key) {
                add_key_field(ix, ch->fields[f].value);
            }
        }
    }
    if (ix->failed) {
        rag_index_free(ix);
        snprintf(error, error_capacity, "out of memory while indexing");
        return NULL;
    }
    ix->chunk_key[ix->n_docs] = ix->n_key_fields;

    /* Short forms of the key fields ("~ml" for "... Machine Learning"). */
    for (int d = 0; d < ix->n_docs && !ix->failed; d++) {
        const RagChunk *ch = rag_corpus_chunk(corpus, d);
        for (int f = 0; f < ch->n_fields; f++) {
            if (ch->fields[f].key) {
                AcrCtx ac;
                ac.add.ix = ix;
                ac.add.chunk = d;
                ac.add.weight = 1.0;
                ac.add.virtual_term = 1;
                ac.n_seen = 0;
                acronyms_of(ch->fields[f].value, add_acronym, &ac);
            }
        }
    }
    if (ix->failed) {
        rag_index_free(ix);
        snprintf(error, error_capacity, "out of memory while indexing");
        return NULL;
    }

    double total = 0.0;
    for (int d = 0; d < ix->n_docs; d++) {
        total += ix->doc_len[d];
    }
    ix->avg_len = ix->n_docs > 0 ? total / ix->n_docs : 0.0;
    return ix;
}

void rag_index_free(RagIndex *ix)
{
    if (ix == NULL) {
        return;
    }
    for (int t = 0; t < ix->n_terms; t++) {
        free(ix->terms[t].text);
        free(ix->terms[t].postings);
    }
    free(ix->terms);
    free(ix->slots);
    free(ix->doc_len);
    free(ix->key_terms);
    free(ix->key_start);
    free(ix->chunk_key);
    free(ix);
}

int rag_index_term_count(const RagIndex *ix)
{
    return ix ? ix->n_terms : 0;
}

/* ------------------------------------------------------------------ */
/* Query terms: exact, short form, corrected typo, or unknown          */
/* ------------------------------------------------------------------ */

/* Optimal string alignment distance (edits + adjacent swaps), capped. */
static int edit_distance(const char *a, const char *b, int limit)
{
    int la = (int)strlen(a), lb = (int)strlen(b);
    int prev2[RAG_TERM_MAX + 1], prev[RAG_TERM_MAX + 1], cur[RAG_TERM_MAX + 1];

    if (la > RAG_TERM_MAX - 1 || lb > RAG_TERM_MAX - 1 || abs(la - lb) > limit) {
        return limit + 1;
    }
    for (int j = 0; j <= lb; j++) {
        prev[j] = j;
        prev2[j] = 0;
    }
    for (int i = 1; i <= la; i++) {
        cur[0] = i;
        for (int j = 1; j <= lb; j++) {
            int cost = a[i - 1] != b[j - 1];
            int v = prev[j - 1] + cost;
            if (prev[j] + 1 < v) {
                v = prev[j] + 1;
            }
            if (cur[j - 1] + 1 < v) {
                v = cur[j - 1] + 1;
            }
            if (i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] &&
                prev2[j - 2] + 1 < v) {
                v = prev2[j - 2] + 1;
            }
            cur[j] = v;
        }
        memcpy(prev2, prev, sizeof(int) * (size_t)(lb + 1));
        memcpy(prev, cur, sizeof(int) * (size_t)(lb + 1));
    }
    return prev[lb];
}

typedef struct {
    const RagIndex *ix;
    RagQueryTerm *out;
    int n, max;
} ResolveCtx;

static void resolve_term(const char *typed, void *user)
{
    ResolveCtx *r = user;
    const RagIndex *ix = r->ix;
    const size_t len = strlen(typed);
    RagQueryTerm q;

    if (len >= RAG_TERM_MAX - 1 || typed[0] == '~') {
        return;
    }
    memset(&q, 0, sizeof(q));
    strcpy(q.typed, typed);
    q.id = -1;
    q.kind = RAG_TERM_UNKNOWN;

    int slot = *find_slot(ix, typed);
    if (slot != 0) {
        q.kind = RAG_TERM_EXACT;
        q.id = slot - 1;
    } else if (is_alpha_term(typed)) {
        char acr[RAG_TERM_MAX + 1];
        acr[0] = '~';
        strcpy(acr + 1, typed);
        if (len >= ACR_MIN && len <= ACR_MAX && (slot = *find_slot(ix, acr)) != 0) {
            q.kind = RAG_TERM_SHORT_FORM;
            q.id = slot - 1;
        } else if (len >= 4) {
            /* closest alphabetic term: 1 edit, or 2 for 8+ letters */
            const int limit = len >= 8 ? 2 : 1;
            int best = -1, best_d = limit + 1;
            for (int t = 0; t < ix->n_terms && best_d > 1; t++) {
                const char *cand = ix->terms[t].text;
                if (cand[0] == '~' || !is_alpha_term(cand)) {
                    continue;
                }
                int d = edit_distance(typed, cand, limit);
                if (d < best_d) {
                    best_d = d;
                    best = t;
                }
            }
            if (best >= 0) {
                q.kind = RAG_TERM_CORRECTED;
                q.id = best;
            }
        }
    }
    if (q.id >= 0) {
        snprintf(q.term, sizeof(q.term), "%s", ix->terms[q.id].text);
        q.is_label = ix->terms[q.id].is_label;
    } else {
        strcpy(q.term, typed);
    }
    for (int i = 0; i < r->n; i++) {
        if (strcmp(r->out[i].term, q.term) == 0) {
            return;                   /* distinct terms only */
        }
    }
    if (r->n < r->max) {
        r->out[r->n++] = q;
    }
}

int rag_index_resolve(const RagIndex *ix, const char *text,
                      RagQueryTerm *out, int max)
{
    ResolveCtx r = {ix, out, 0, max};
    if (ix == NULL || max <= 0) {
        return 0;
    }
    rag_tokenize(text, resolve_term, &r);
    return r.n;
}

typedef struct {
    int ids[RAG_QUERY_MAX];
    int n;
} QueryCtx;

static double term_idf(double N, const Term *t)
{
    double df = t->n_postings;
    return log(1.0 + (N - df + 0.5) / (df + 0.5));
}

static int compare_hits(const void *a, const void *b)
{
    const RagHit *x = a, *y = b;
    if (x->score > y->score) {
        return -1;
    }
    if (x->score < y->score) {
        return 1;
    }
    return (x->chunk > y->chunk) - (x->chunk < y->chunk);
}

int rag_index_search(const RagIndex *ix, const char *query,
                     RagHit *hits, int max_hits)
{
    QueryCtx q;

    if (ix == NULL || ix->n_docs == 0 || max_hits <= 0) {
        return 0;
    }
    {
        RagQueryTerm terms[RAG_QUERY_MAX];
        int n = rag_index_resolve(ix, query, terms, RAG_QUERY_MAX);
        q.n = 0;
        for (int i = 0; i < n; i++) {
            if (terms[i].id >= 0 && ix->terms[terms[i].id].n_postings > 0) {
                q.ids[q.n++] = terms[i].id;
            }
        }
    }
    if (q.n == 0) {
        return 0;
    }

    double *score = calloc((size_t)ix->n_docs, sizeof(double));
    if (score == NULL) {
        return 0;
    }
    const double N = ix->n_docs;
    for (int i = 0; i < q.n; i++) {
        const Term *t = &ix->terms[q.ids[i]];
        double w = t->is_label ? ix->label_weight : 1.0;
        double widf = w * term_idf(N, t);
        for (int p = 0; p < t->n_postings; p++) {
            int d = t->postings[p].chunk;
            double tf = t->postings[p].tf;
            double norm = ix->k1 * (1.0 - ix->b + ix->b * ix->doc_len[d] / ix->avg_len);
            score[d] += widf * tf * (ix->k1 + 1.0) / (tf + norm);
        }
    }

    /* Name bonus: a key field all of whose terms were asked for. */
    if (ix->name_bonus > 0.0) {
        for (int d = 0; d < ix->n_docs; d++) {
            if (score[d] <= 0.0) {
                continue;
            }
            for (int f = ix->chunk_key[d]; f < ix->chunk_key[d + 1]; f++) {
                int a = ix->key_start[f], z = ix->key_start[f + 1];
                if (a == z) {
                    continue;
                }
                int covered = 1;
                double sum = 0.0;
                for (int j = a; j < z && covered; j++) {
                    covered = 0;
                    for (int i = 0; i < q.n; i++) {
                        if (q.ids[i] == ix->key_terms[j]) {
                            covered = 1;
                            break;
                        }
                    }
                    sum += term_idf(N, &ix->terms[ix->key_terms[j]]);
                }
                if (covered) {
                    score[d] += ix->name_bonus * sum;
                }
            }
        }
    }

    int n_pos = 0;
    for (int d = 0; d < ix->n_docs; d++) {
        n_pos += score[d] > 0.0;
    }
    RagHit *all = malloc((size_t)(n_pos + 1) * sizeof(*all));
    if (all == NULL) {
        free(score);
        return 0;
    }
    int k = 0;
    for (int d = 0; d < ix->n_docs; d++) {
        if (score[d] > 0.0) {
            all[k].chunk = d;
            all[k].score = score[d];
            k++;
        }
    }
    qsort(all, (size_t)k, sizeof(*all), compare_hits);
    int out = k < max_hits ? k : max_hits;
    memcpy(hits, all, (size_t)out * sizeof(*hits));
    free(all);
    free(score);
    return out;
}
