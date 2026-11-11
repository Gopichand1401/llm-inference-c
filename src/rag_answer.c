/*
 * rag_answer.c - direct answers from table rows (no language model).
 * The method is described in include/rag_answer.h.
 */
#include "rag_answer.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TERMS RAG_QUERY_MAX
#define MAX_WORDS 80
#define MAX_HITS 256
#define MAX_LIST 15
#define MAX_RELATED 5
#define MAX_FIELDS 256
#define MAX_PEOPLE 16

/* ------------------------------------------------------------------ */
/* Word lists                                                          */
/* ------------------------------------------------------------------ */

/* Terms (as rag_tokenize writes them) that say nothing about which row. */
static const char *const FILLER[] = {
    "actually", "address", "all", "also", "available", "check", "class",
    "classroom", "complete", "conducted", "contact", "count", "course",
    "currently", "date", "day", "detail", "display", "each", "email",
    "entire", "every", "exactly", "faculty", "fetch", "full", "gmail",
    "going", "gonna", "hall", "happen", "happening", "held", "hello", "hey",
    "hi", "ids", "info", "information", "just", "kind", "kindly", "lecturer",
    "like", "location", "look", "looking", "madam", "mail", "mam", "many",
    "mark", "mention", "more", "much", "no", "now", "number", "offered",
    "ok", "okay", "one", "only", "paper", "place", "pls", "plz", "print", "prof",
    "professor", "provide", "really", "regarding", "say", "schedule",
    "search", "see", "send", "share", "sir", "subject", "teacher", "thank",
    "thanks", "thing", "time", "timing", "total", "u", "ur", "venue",
    "wanna", "ye", "yes"
};

/* Question word -> label terms it stands for. */
static const char *const SYNONYMS[][2] = {
    {"who", "instructor"}, {"who", "faculty"}, {"who", "teacher"},
    {"whom", "instructor"}, {"whom", "faculty"},
    {"teach", "instructor"}, {"teaches", "instructor"},
    {"teaching", "instructor"}, {"taught", "instructor"},
    {"teach", "faculty"}, {"teaches", "faculty"}, {"taught", "faculty"},
    {"teacher", "instructor"}, {"prof", "instructor"},
    {"professor", "instructor"}, {"sir", "instructor"},
    {"madam", "instructor"}, {"mam", "instructor"},
    {"lecturer", "instructor"}, {"faculty", "instructor"},
    {"instructor", "faculty"}, {"instructor", "teacher"},
    {"takes", "instructor"}, {"taking", "instructor"},
    {"handles", "instructor"}, {"handling", "instructor"},
    {"when", "slot"}, {"when", "time"}, {"when", "timing"}, {"when", "day"},
    {"when", "date"}, {"time", "slot"}, {"timing", "slot"},
    {"timings", "slot"}, {"schedule", "slot"}, {"day", "slot"},
    {"where", "room"}, {"where", "venue"}, {"where", "location"},
    {"venue", "room"}, {"location", "room"}, {"classroom", "room"},
    {"place", "room"}, {"hall", "room"},
    {"mail", "email"}, {"gmail", "email"}, {"contact", "email"},
    {"email", "mail"},
    {"prereq", "prerequisite"}, {"prereqs", "prerequisite"},
    {"required", "prerequisite"}, {"requirement", "prerequisite"},
    {"requirements", "prerequisite"},
    {"grade", "grading"}, {"grading", "grade"},
    {"capacity", "cap"}, {"seats", "cap"}, {"strength", "cap"},
    {"capacity", "availability"}, {"seats", "availability"}
};

/* Label terms offered first as a follow-up, most useful first. */
static const char *const IMPORTANT[] = {
    "slot", "time", "timing", "date", "day", "room", "venue", "instructor",
    "faculty", "teacher", "credit", "prerequisite", "grade", "grading"
};

static const char *const YES_WORDS[] = {
    "y", "yes", "yeah", "yep", "yup", "ya", "ok", "okay", "sure", "haan"
};
static const char *const NO_WORDS[] = {"n", "no", "nope", "nah"};
static const char *const ALL_WORDS[] = {
    "details", "detail", "everything", "full", "complete", "entire"
};

#define COUNT(a) ((int)(sizeof(a) / sizeof((a)[0])))

static int in_list(const char *w, const char *const *list, int n)
{
    for (int i = 0; i < n; i++) {
        if (strcmp(w, list[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Session                                                             */
/* ------------------------------------------------------------------ */

enum { PENDING_NONE, PENDING_SUGGEST, PENDING_CHOICE, PENDING_RELATED };
enum { SHOWN_VALUE = 1, SHOWN_EMAIL = 2 };
#define SUGGEST_VARIANTS (-2)

struct RagSession {
    int ctx;                         /* chunk answered last, -1 if none */
    unsigned char shown[MAX_FIELDS]; /* per field of ctx: SHOWN_* bits */
    int variants_shown;
    int pending;
    int suggest_field;               /* field index or SUGGEST_VARIANTS */
    int suggest_email;               /* 1: the e-mail part of that field */
    int list[MAX_LIST];              /* numbered choice / related rows */
    int n_list;
    char list_question[1024];        /* question to answer for the choice */
    char last_source[640];           /* "file, tab T, row N (title)" of the reply */
    char last_offer[256];            /* closing question of the reply, if any */
};

const char *rag_session_last_source(const RagSession *s)
{
    return s ? s->last_source : "";
}

const char *rag_session_last_offer(const RagSession *s)
{
    return s ? s->last_offer : "";
}

/* Print the closing question of a reply and remember it. */
static void offer(RagSession *s, FILE *out, const char *text)
{
    snprintf(s->last_offer, sizeof(s->last_offer), "%s", text);
    fprintf(out, "%s\n", text);
}

RagSession *rag_session_new(void)
{
    RagSession *s = calloc(1, sizeof(*s));
    if (s != NULL) {
        s->ctx = -1;
    }
    return s;
}

void rag_session_free(RagSession *s)
{
    free(s);
}

void rag_session_reset(RagSession *s)
{
    if (s != NULL) {
        memset(s, 0, sizeof(*s));
        s->ctx = -1;
    }
}

/* ------------------------------------------------------------------ */
/* Question analysis                                                   */
/* ------------------------------------------------------------------ */

typedef struct {
    char text[1024];                 /* cleaned question */
    char words[MAX_WORDS][32];       /* lower-case words as typed */
    int n_words;
    RagQueryTerm terms[MAX_TERMS];
    int n_terms;
    int joined_parts[MAX_TERMS][2];  /* a joined term's two parts, or -1 */
    int skip[MAX_TERMS];             /* never counts (unknown joined term) */
    int filler[MAX_TERMS];
    const char *syn[64];
    int n_syn;
    int want_email;   /* 0: names, 1: mail ids, 2: both ("instructor and his mail id") */
    int want_all, count_intent, list_intent;
} Question;

static int has_word(const Question *q, const char *w)
{
    for (int i = 0; i < q->n_words; i++) {
        if (strcmp(q->words[i], w) == 0) {
            return 1;
        }
    }
    return 0;
}

static void analyse(const RagIndex *ix, const char *input, Question *q)
{
    memset(q, 0, sizeof(*q));

    /* copy without possessive 's (ASCII ' or U+2019) */
    size_t w = 0;
    for (size_t i = 0; input[i] != '\0' && w < sizeof(q->text) - 1; i++) {
        size_t skip = 0;
        if (input[i] == '\'') {
            skip = 1;
        } else if (strncmp(input + i, "\xE2\x80\x99", 3) == 0) {
            skip = 3;
        }
        if (skip && (input[i + skip] == 's' || input[i + skip] == 'S') &&
            !isalnum((unsigned char)input[i + skip + 1])) {
            i += skip;
            continue;
        }
        q->text[w++] = input[i];
    }
    q->text[w] = '\0';

    const char *p = q->text;
    while (*p && q->n_words < MAX_WORDS) {
        while (*p && !isalnum((unsigned char)*p)) {
            p++;
        }
        size_t n = 0;
        while (*p && isalnum((unsigned char)*p)) {
            if (n < sizeof(q->words[0]) - 1) {
                q->words[q->n_words][n++] = (char)tolower((unsigned char)*p);
            }
            p++;
        }
        if (n > 0) {
            q->words[q->n_words++][n] = '\0';
        }
    }

    q->n_terms = rag_index_resolve(ix, q->text, q->terms, MAX_TERMS);
    for (int i = 0; i < q->n_terms; i++) {
        q->joined_parts[i][0] = q->joined_parts[i][1] = -1;
        q->filler[i] = in_list(q->terms[i].typed, FILLER, COUNT(FILLER));
    }
    /* "ds 3010" gives ds, 3010 and the joined ds3010 */
    for (int i = 0; i < q->n_terms; i++) {
        const size_t li = strlen(q->terms[i].typed);
        for (int a = 0; a < q->n_terms; a++) {
            const size_t la = strlen(q->terms[a].typed);
            if (a == i || la >= li ||
                strncmp(q->terms[i].typed, q->terms[a].typed, la) != 0) {
                continue;
            }
            for (int b = 0; b < q->n_terms; b++) {
                if (b != i && b != a &&
                    strcmp(q->terms[i].typed + la, q->terms[b].typed) == 0) {
                    q->joined_parts[i][0] = a;
                    q->joined_parts[i][1] = b;
                }
            }
        }
        if (q->joined_parts[i][0] >= 0 && q->terms[i].kind == RAG_TERM_UNKNOWN) {
            q->skip[i] = 1;      /* the parts speak for themselves */
        }
    }

    for (int i = 0; i < q->n_words; i++) {
        for (int s = 0; s < COUNT(SYNONYMS) && q->n_syn < 64; s++) {
            if (strcmp(q->words[i], SYNONYMS[s][0]) == 0) {
                q->syn[q->n_syn++] = SYNONYMS[s][1];
            }
        }
        if (in_list(q->words[i], ALL_WORDS, COUNT(ALL_WORDS))) {
            q->want_all = 1;
        }
    }
    q->want_email = has_word(q, "mail") || has_word(q, "email") ||
                    has_word(q, "gmail") || has_word(q, "mailid") ||
                    has_word(q, "emailid");
    if (q->want_email &&
        (has_word(q, "name") || has_word(q, "names") ||
         (has_word(q, "and") &&
          (has_word(q, "who") || has_word(q, "instructor") ||
           has_word(q, "instructors") || has_word(q, "faculty") ||
           has_word(q, "teacher") || has_word(q, "professor") ||
           has_word(q, "prof"))))) {
        q->want_email = 2;
    }
    q->count_intent = has_word(q, "count") ||
                      (has_word(q, "how") && has_word(q, "many")) ||
                      (has_word(q, "number") && has_word(q, "of"));
    q->list_intent = q->count_intent || has_word(q, "list") ||
                     has_word(q, "all") || has_word(q, "every");
}

/* ------------------------------------------------------------------ */
/* Matching a question against one row                                 */
/* ------------------------------------------------------------------ */

enum { C_NONE, C_KEY, C_LABEL, C_VALUE, C_TEXT, C_IGNORED, C_MISSING };

typedef struct {
    int cls[MAX_TERMS];
    int field[MAX_TERMS];     /* C_VALUE: the field, -1 for a note */
    int n_key, n_label, n_value, n_text, n_missing;
    double coverage;          /* best share of a key field's terms asked for */
} Match;

typedef struct {
    char seen[48][RAG_TERM_MAX];
    int n;
} TermSet;

static void collect_distinct(const char *term, void *user)
{
    TermSet *t = user;
    for (int i = 0; i < t->n; i++) {
        if (strcmp(t->seen[i], term) == 0) {
            return;
        }
    }
    if (t->n < 48 && strlen(term) < RAG_TERM_MAX) {
        strcpy(t->seen[t->n++], term);
    }
}

static int label_has(const RagChunk *ch, const char *term)
{
    if (term[0] == '~') {
        return 0;
    }
    for (int f = 0; f < ch->n_fields; f++) {
        if (rag_text_has_term(ch->fields[f].label, term)) {
            return 1;
        }
    }
    return 0;
}

static void match_row(const Question *q, const RagChunk *ch, Match *m)
{
    memset(m, 0, sizeof(*m));

    for (int i = 0; i < q->n_terms; i++) {
        const char *t = q->terms[i].term;
        int cls = C_MISSING, field = -1;

        if (q->skip[i]) {
            cls = C_IGNORED;
        } else if (q->filler[i] && q->terms[i].kind != RAG_TERM_EXACT) {
            /* a filler word must not turn into a short form or a "typo" */
            cls = C_IGNORED;
        } else if (q->terms[i].kind == RAG_TERM_UNKNOWN) {
            cls = C_MISSING;
        } else {
            for (int f = 0; f < ch->n_fields && cls == C_MISSING; f++) {
                if (ch->fields[f].key && rag_text_has_term(ch->fields[f].value, t)) {
                    cls = C_KEY;
                }
            }
            if (cls == C_MISSING && label_has(ch, t)) {
                cls = C_LABEL;
            }
            if (cls == C_MISSING && q->filler[i]) {
                cls = C_IGNORED;
            }
            if (cls == C_MISSING && t[0] != '~') {
                if (ch->kind == RAG_CHUNK_NOTE) {
                    if (rag_text_has_term(ch->note, t)) {
                        cls = C_VALUE;
                    }
                } else {
                    for (int f = 0; f < ch->n_fields; f++) {
                        if (!ch->fields[f].key &&
                            rag_text_has_term(ch->fields[f].value, t)) {
                            cls = C_VALUE;
                            field = f;
                            break;
                        }
                    }
                }
                if (cls == C_MISSING &&
                    (rag_text_has_term(ch->title, t) ||
                     rag_text_has_term(ch->sheet, t) ||
                     rag_text_has_term(ch->file, t))) {
                    cls = C_TEXT;
                }
            }
        }
        m->cls[i] = cls;
        m->field[i] = field;
    }

    /* coverage of each key field, before the parts of joined terms go */
    for (int f = 0; f < ch->n_fields; f++) {
        if (!ch->fields[f].key) {
            continue;
        }
        TermSet set;
        double got = 0.0;
        set.n = 0;
        rag_tokenize(ch->fields[f].value, collect_distinct, &set);
        for (int i = 0; i < q->n_terms; i++) {
            if (m->cls[i] == C_KEY &&
                rag_text_has_term(ch->fields[f].value, q->terms[i].term)) {
                got += q->terms[i].term[0] == '~'
                           ? (double)(strlen(q->terms[i].term) - 1) : 1.0;
            }
        }
        if (set.n > 0) {
            double c = got / set.n;
            if (c > 1.0) {
                c = 1.0;
            }
            if (c > m->coverage) {
                m->coverage = c;
            }
        }
    }

    /* the parts of a joined term that matched say nothing new */
    for (int i = 0; i < q->n_terms; i++) {
        if (q->joined_parts[i][0] >= 0 && m->cls[i] == C_KEY) {
            m->cls[q->joined_parts[i][0]] = C_IGNORED;
            m->cls[q->joined_parts[i][1]] = C_IGNORED;
        }
    }
    for (int i = 0; i < q->n_terms; i++) {
        switch (m->cls[i]) {
        case C_KEY:     m->n_key++; break;
        case C_LABEL:   m->n_label++; break;
        case C_VALUE:   m->n_value++; break;
        case C_TEXT:    m->n_text++; break;
        case C_MISSING: m->n_missing++; break;
        default: break;
        }
    }
}

/* Terms found in no table at all (not filler). */
static int unknown_terms(const Question *q)
{
    int n = 0;
    for (int i = 0; i < q->n_terms; i++) {
        n += q->terms[i].kind == RAG_TERM_UNKNOWN && !q->filler[i] && !q->skip[i];
    }
    return n;
}

static const char *entity_key(const RagChunk *ch)
{
    for (int f = 0; f < ch->n_fields; f++) {
        if (ch->fields[f].key && ch->fields[f].variant == 0) {
            return ch->fields[f].value;
        }
    }
    return NULL;
}

static int same_text(const char *a, const char *b)
{
    for (; *a && *b; a++, b++) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) {
            return 0;
        }
    }
    return *a == *b;
}

static int same_entity(const RagCorpus *c, int a, int b)
{
    const char *ka = entity_key(rag_corpus_chunk(c, a));
    const char *kb = entity_key(rag_corpus_chunk(c, b));
    return a == b || (ka != NULL && kb != NULL && same_text(ka, kb));
}

/* ------------------------------------------------------------------ */
/* Cells with people: "Name (mail) Name2, mail2 ..."                   */
/* ------------------------------------------------------------------ */

typedef struct {
    char name[MAX_PEOPLE][96];
    char email[MAX_PEOPLE][96];
    int n;
} People;

static int is_email_char(char ch)
{
    return isalnum((unsigned char)ch) || ch == '.' || ch == '_' || ch == '%' ||
           ch == '+' || ch == '-';
}

static void clean_name(const char *from, size_t n, char *out, size_t cap)
{
    while (n > 0 && strchr(" ,;()[]/:&", from[0]) != NULL) {
        from++;
        n--;
    }
    if (n >= 4 && strncmp(from, "and ", 4) == 0) {
        from += 4;
        n -= 4;
    }
    while (n > 0 && strchr(" ,;()[]/:&", from[n - 1]) != NULL) {
        n--;
    }
    if (n >= 4 && strncmp(from + n - 4, " and", 4) == 0) {
        n -= 4;
    }
    if (n >= cap) {
        n = cap - 1;
    }
    memcpy(out, from, n);
    out[n] = '\0';

    int letters = 0;
    for (size_t i = 0; i < n; i++) {
        letters += isalpha((unsigned char)out[i]) != 0;
    }
    if (letters < 3 || strcmp(out, "Dr.") == 0 || strcmp(out, "Prof.") == 0 ||
        strcmp(out, "and Dr.") == 0 || strcmp(out, "and") == 0) {
        out[0] = '\0';
    }
}

/* Split a cell into people. Returns 0 if it has no e-mail address. */
static int parse_people(const char *value, People *p)
{
    const size_t n = strlen(value);
    size_t seg = 0;

    p->n = 0;
    for (size_t i = 0; i < n; i++) {
        if (value[i] != '@' || i == 0 || !is_email_char(value[i - 1])) {
            continue;
        }
        size_t a = i, b = i + 1;
        while (a > seg && is_email_char(value[a - 1])) {
            a--;
        }
        while (b < n && is_email_char(value[b])) {
            b++;
        }
        while (b > i + 1 && (value[b - 1] == '.' || value[b - 1] == '-')) {
            b--;
        }
        if (b == i + 1 || p->n == MAX_PEOPLE) {
            continue;
        }
        clean_name(value + seg, a - seg, p->name[p->n], sizeof(p->name[0]));
        size_t len = b - a < sizeof(p->email[0]) ? b - a : sizeof(p->email[0]) - 1;
        memcpy(p->email[p->n], value + a, len);
        p->email[p->n][len] = '\0';
        p->n++;
        seg = b;
        i = b - 1;
    }
    if (p->n == 0) {
        return 0;
    }
    if (seg < n && p->n < MAX_PEOPLE) {       /* a name without an address */
        clean_name(value + seg, n - seg, p->name[p->n], sizeof(p->name[0]));
        if (p->name[p->n][0] != '\0') {
            p->email[p->n][0] = '\0';
            p->n++;
        }
    }
    return 1;
}

/*
 * "A, B and C". emails 0: names (the address when a name is missing),
 * 1: addresses, 2: both, as "Name (address)".
 */
static void print_people(FILE *out, const People *p, int emails)
{
    int n = 0, shown = 0;
    for (int i = 0; i < p->n; i++) {
        n += emails == 1 ? p->email[i][0] != '\0' : 1;
    }
    for (int i = 0; i < p->n; i++) {
        const char *text = emails == 1 ? p->email[i]
                                       : (p->name[i][0] ? p->name[i] : p->email[i]);
        if (text[0] == '\0') {
            continue;
        }
        fprintf(out, "%s%s", shown == 0 ? "" : (shown == n - 1 ? " and " : ", "), text);
        if (emails == 2 && p->name[i][0] != '\0' && p->email[i][0] != '\0') {
            fprintf(out, " (%s)", p->email[i]);
        }
        shown++;
    }
}

static int people_have_email(const People *p)
{
    for (int i = 0; i < p->n; i++) {
        if (p->email[i][0] != '\0') {
            return 1;
        }
    }
    return 0;
}

static int people_have_name(const People *p)
{
    for (int i = 0; i < p->n; i++) {
        if (p->name[i][0] != '\0') {
            return 1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Labels for display                                                  */
/* ------------------------------------------------------------------ */

/* Lower-case a label for a sentence; words like "OE", "ID" keep their case. */
static void display_label(const char *label, int strip_mail, char *out, size_t cap)
{
    char tmp[160];
    size_t n = strlen(label);

    if (n >= sizeof(tmp)) {
        n = sizeof(tmp) - 1;
    }
    memcpy(tmp, label, n);
    tmp[n] = '\0';

    if (strip_mail) {          /* "Primary Instructor Email ID" -> "... Instructor" */
        for (;;) {
            while (n > 0 && tmp[n - 1] == ' ') {
                tmp[--n] = '\0';
            }
            char *sp = strrchr(tmp, ' ');
            const char *last = sp ? sp + 1 : tmp;
            char low[32];
            size_t l = 0;
            for (; last[l] && l < sizeof(low) - 1; l++) {
                low[l] = (char)tolower((unsigned char)last[l]);
            }
            low[l] = '\0';
            if (sp != NULL && (strcmp(low, "email") == 0 || strcmp(low, "e-mail") == 0 ||
                               strcmp(low, "mail") == 0 || strcmp(low, "id") == 0 ||
                               strcmp(low, "ids") == 0)) {
                *sp = '\0';
                n = strlen(tmp);
            } else {
                break;
            }
        }
    }

    size_t w = 0;
    for (size_t i = 0; tmp[i] != '\0' && w + 1 < cap;) {
        size_t j = i;
        int upper = 0;
        while (tmp[j] != '\0' && tmp[j] != ' ') {
            upper += isupper((unsigned char)tmp[j]) != 0;
            j++;
        }
        for (; i < j && w + 1 < cap; i++) {
            out[w++] = upper >= 2 ? tmp[i] : (char)tolower((unsigned char)tmp[i]);
        }
        if (tmp[i] == ' ' && w + 1 < cap) {
            out[w++] = ' ';
            i++;
        }
    }
    out[w] = '\0';
}

static void print_source(RagSession *s, FILE *out, const RagChunk *ch)
{
    char *t = s->last_source;
    const size_t cap = sizeof(s->last_source);
    size_t n = 0;

    n += (size_t)snprintf(t + n, cap - n, "%.200s", ch->file);
    if (ch->sheet[0] != '\0') {
        n += (size_t)snprintf(t + n, cap - n, ", tab %.150s", ch->sheet);
    }
    if (ch->row_first == ch->row_last) {
        n += (size_t)snprintf(t + n, cap - n, ", row %d", ch->row_first);
    } else {
        n += (size_t)snprintf(t + n, cap - n, ", rows %d-%d", ch->row_first,
                              ch->row_last);
    }
    if (ch->title[0] != '\0') {
        snprintf(t + n, cap - n, " (%.120s)", ch->title);
    }
    fprintf(out, "Source: %s\n", t);
}

/* "DS3010 - Introduction to Machine Learning" */
static void print_identity(FILE *out, const RagChunk *ch)
{
    int n = 0;
    if (ch->kind == RAG_CHUNK_NOTE) {
        fprintf(out, "%s", ch->note);
        return;
    }
    for (int f = 0; f < ch->n_fields; f++) {
        if (ch->fields[f].key && ch->fields[f].variant == 0) {
            fprintf(out, "%s%s", n++ ? " - " : "", ch->fields[f].value);
        }
    }
    if (n == 0 && ch->n_fields > 0) {
        fprintf(out, "%s", ch->fields[0].value);
    }
}

static void print_list_line(FILE *out, int number, const RagChunk *ch)
{
    fprintf(out, "  %d. ", number);
    print_identity(out, ch);
    fprintf(out, "  (%s", ch->file);
    if (ch->sheet[0] != '\0') {
        fprintf(out, ", %s", ch->sheet);
    }
    fprintf(out, ", row %d)\n", ch->row_first);
}

/* The rows of a merged chunk as a small aligned table. */
static void print_variants(FILE *out, const RagChunk *ch)
{
    const char *labels[32];
    int n_labels = 0;
    size_t width[32];

    for (int f = 0; f < ch->n_fields; f++) {
        if (ch->fields[f].variant == 0) {
            continue;
        }
        int known = 0;
        for (int l = 0; l < n_labels; l++) {
            known |= strcmp(labels[l], ch->fields[f].label) == 0;
        }
        if (!known && n_labels < 32) {
            labels[n_labels++] = ch->fields[f].label;
        }
    }
    for (int l = 0; l < n_labels; l++) {
        width[l] = strlen(labels[l]);
        for (int f = 0; f < ch->n_fields; f++) {
            if (ch->fields[f].variant != 0 &&
                strcmp(ch->fields[f].label, labels[l]) == 0 &&
                strlen(ch->fields[f].value) > width[l]) {
                width[l] = strlen(ch->fields[f].value);
            }
        }
        if (width[l] > 44) {
            width[l] = 44;
        }
    }

    fprintf(out, "  %-5s", "Row");
    for (int l = 0; l < n_labels; l++) {
        fprintf(out, "  %-*s", l + 1 < n_labels ? (int)width[l] : 0, labels[l]);
    }
    fprintf(out, "\n");
    for (int v = 1; v <= ch->n_variants; v++) {
        fprintf(out, "  %-5d", ch->variant_rows[v - 1]);
        for (int l = 0; l < n_labels; l++) {
            const char *value = "";
            for (int f = 0; f < ch->n_fields; f++) {
                if (ch->fields[f].variant == v &&
                    strcmp(ch->fields[f].label, labels[l]) == 0) {
                    value = ch->fields[f].value;
                }
            }
            fprintf(out, "  %-*s", l + 1 < n_labels ? (int)width[l] : 0, value);
        }
        fprintf(out, "\n");
    }
}

/* ------------------------------------------------------------------ */
/* Which cells does the question ask for?                              */
/* ------------------------------------------------------------------ */

typedef struct {
    int fields[32];
    int n;
    int variants;        /* the per-row cells of a merged chunk */
} Asked;

static void asked_add(Asked *a, int f)
{
    for (int i = 0; i < a->n; i++) {
        if (a->fields[i] == f) {
            return;
        }
    }
    if (a->n < 32) {
        a->fields[a->n++] = f;
    }
}

static void asked_fields(const Question *q, const RagChunk *ch, const Match *m,
                         Asked *a)
{
    memset(a, 0, sizeof(*a));
    if (ch->kind == RAG_CHUNK_NOTE) {
        return;
    }

    /* 1. a cell that contains words of the question ("exam slot") */
    for (int i = 0; i < q->n_terms; i++) {
        if (m->cls[i] == C_VALUE && m->field[i] >= 0) {
            if (ch->fields[m->field[i]].variant != 0) {
                a->variants = 1;
            } else {
                asked_add(a, m->field[i]);
            }
        }
    }
    if (a->n > 0 || a->variants) {
        return;
    }

    /* 2. the columns whose label matches the question best */
    int score[MAX_FIELDS];
    int best = 0;
    const int nf = ch->n_fields < MAX_FIELDS ? ch->n_fields : MAX_FIELDS;
    for (int f = 0; f < nf; f++) {
        const RagField *fd = &ch->fields[f];
        score[f] = 0;
        if (fd->key) {      /* a key column counts only if it was not the way in */
            int used = 0;
            for (int i = 0; i < q->n_terms; i++) {
                used |= m->cls[i] == C_KEY &&
                        rag_text_has_term(fd->value, q->terms[i].term);
            }
            if (used) {
                continue;
            }
        }
        for (int i = 0; i < q->n_terms; i++) {
            if (m->cls[i] == C_LABEL &&
                rag_text_has_term(fd->label, q->terms[i].term)) {
                score[f]++;
            }
        }
        for (int s = 0; s < q->n_syn; s++) {
            int repeat = 0;
            for (int r = 0; r < s; r++) {
                repeat |= strcmp(q->syn[r], q->syn[s]) == 0;
            }
            for (int i = 0; i < q->n_terms; i++) {
                repeat |= m->cls[i] == C_LABEL &&
                          strcmp(q->terms[i].term, q->syn[s]) == 0;
            }
            if (!repeat && rag_text_has_term(fd->label, q->syn[s])) {
                score[f]++;
            }
        }
        if (score[f] > best) {
            best = score[f];
        }
    }
    if (best > 0) {
        for (int f = 0; f < nf; f++) {
            if (score[f] != best) {
                continue;
            }
            if (ch->fields[f].variant != 0) {
                a->variants = 1;
            } else {
                asked_add(a, f);
            }
        }
    }

    /* 3. "mail id of ..." with no column named: every cell with an address */
    if (a->n == 0 && !a->variants && q->want_email) {
        for (int f = 0; f < nf; f++) {
            if (ch->fields[f].variant == 0 && strchr(ch->fields[f].value, '@')) {
                asked_add(a, f);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Printing an answer and the follow-up offer                          */
/* ------------------------------------------------------------------ */

static void set_context(RagSession *s, int chunk)
{
    if (s->ctx != chunk) {
        s->ctx = chunk;
        memset(s->shown, 0, sizeof(s->shown));
        s->variants_shown = 0;
    }
}

static int has_variants(const RagChunk *ch)
{
    for (int f = 0; f < ch->n_fields; f++) {
        if (ch->fields[f].variant != 0) {
            return 1;
        }
    }
    return 0;
}

static int label_rank(const char *label)
{
    for (int i = 0; i < COUNT(IMPORTANT); i++) {
        if (rag_text_has_term(label, IMPORTANT[i])) {
            return i;
        }
    }
    return -1;
}

/* Offer one more cell of the current row. */
static void suggest(RagSession *s, const RagCorpus *c, FILE *out)
{
    const RagChunk *ch = rag_corpus_chunk(c, s->ctx);
    const int nf = ch->n_fields < MAX_FIELDS ? ch->n_fields : MAX_FIELDS;
    int pick = -1, pick_rank = COUNT(IMPORTANT), email = 0;
    char label[160];

    s->pending = PENDING_NONE;
    if (ch->kind == RAG_CHUNK_NOTE) {
        return;
    }
    /* a: important columns not shown yet */
    for (int f = 0; f < nf; f++) {
        const RagField *fd = &ch->fields[f];
        int rank = label_rank(fd->label);
        if (fd->variant == 0 && !fd->key && !(s->shown[f] & SHOWN_VALUE) &&
            rank >= 0 && rank < pick_rank) {
            pick = f;
            pick_rank = rank;
        }
    }
    /* b: the rows of a merged chunk */
    if (pick < 0 && has_variants(ch) && !s->variants_shown) {
        s->pending = PENDING_SUGGEST;
        s->suggest_field = SUGGEST_VARIANTS;
        s->suggest_email = 0;
        char text[96];
        snprintf(text, sizeof(text), "Do you want the details of each of its %d rows?",
                 ch->n_variants);
        offer(s, out, text);
        return;
    }
    /* c: the mail id behind a name that was shown */
    for (int f = 0; f < nf && pick < 0; f++) {
        People p;
        if (ch->fields[f].variant == 0 && (s->shown[f] & SHOWN_VALUE) &&
            !(s->shown[f] & SHOWN_EMAIL) && parse_people(ch->fields[f].value, &p) &&
            people_have_email(&p)) {
            pick = f;
            email = 1;
        }
    }
    /* d: any other column */
    for (int f = 0; f < nf && pick < 0; f++) {
        if (ch->fields[f].variant == 0 && !ch->fields[f].key &&
            !(s->shown[f] & SHOWN_VALUE)) {
            pick = f;
        }
    }
    if (pick < 0) {
        return;
    }
    s->pending = PENDING_SUGGEST;
    s->suggest_field = pick;
    s->suggest_email = email;
    People p;
    int is_people = parse_people(ch->fields[pick].value, &p);
    display_label(ch->fields[pick].label, is_people, label, sizeof(label));
    char text[224];
    snprintf(text, sizeof(text), "Do you want the %.160s%s?", label,
             email ? " mail id" : "");
    offer(s, out, text);
}

/*
 * Print the asked cells of a row, its source and the next offer.
 * emails: print addresses rather than names.
 */
static void print_answer(RagSession *s, const RagCorpus *c, int chunk,
                         const Asked *a, int emails, int with_source, FILE *out)
{
    const RagChunk *ch = rag_corpus_chunk(c, chunk);
    People people[32];
    int is_people[32];
    int all_people = a->n > 0;

    set_context(s, chunk);

    if (ch->kind == RAG_CHUNK_NOTE) {
        fprintf(out, "%s\n", ch->note);
        print_source(s, out, ch);
        s->pending = PENDING_NONE;
        return;
    }

    for (int i = 0; i < a->n; i++) {
        is_people[i] = parse_people(ch->fields[a->fields[i]].value, &people[i]);
        all_people &= is_people[i];
    }

    if (a->n == 0 && !a->variants) {
        print_identity(out, ch);
        fprintf(out, "\n");
    } else if (a->n == 1) {
        if (!is_people[0]) {
            fprintf(out, "%s", ch->fields[a->fields[0]].value);
        } else if (emails == 1 && !people_have_email(&people[0])) {
            fprintf(out, "No mail id is listed for ");
            print_people(out, &people[0], 0);
        } else {
            print_people(out, &people[0], emails);
        }
        fprintf(out, "\n");
    } else if (a->n > 1 && all_people && emails != 1) {
        /* "A is the primary instructor and B is the additional instructor." */
        for (int i = 0; i < a->n; i++) {
            char role[160];
            display_label(ch->fields[a->fields[i]].label, 1, role, sizeof(role));
            fprintf(out, "%s", i == 0 ? "" : (i == a->n - 1 ? " and " : ", "));
            print_people(out, &people[i], emails);
            fprintf(out, " %s the %s%s", people[i].n > 1 ? "are" : "is", role,
                    people[i].n > 1 ? "s" : "");
        }
        fprintf(out, ".\n");
    } else {
        for (int i = 0; i < a->n; i++) {
            fprintf(out, "%s: ", ch->fields[a->fields[i]].label);
            if (is_people[i] && (emails != 1 || people_have_email(&people[i]))) {
                print_people(out, &people[i], emails);
            } else {
                fprintf(out, "%s", ch->fields[a->fields[i]].value);
            }
            fprintf(out, "\n");
        }
    }
    if (a->variants) {
        if (a->n > 0) {
            fprintf(out, "\n");
        }
        print_variants(out, ch);
        s->variants_shown = 1;
    }
    for (int i = 0; i < a->n; i++) {
        const int f = a->fields[i];
        if (f < MAX_FIELDS) {
            if (!is_people[i]) {
                s->shown[f] |= SHOWN_VALUE | SHOWN_EMAIL;
            } else if (emails == 2) {
                s->shown[f] |= SHOWN_VALUE | SHOWN_EMAIL;
            } else if (emails == 1) {
                s->shown[f] |= SHOWN_EMAIL;
                if (!people_have_name(&people[i])) {
                    s->shown[f] |= SHOWN_VALUE;
                }
            } else {
                s->shown[f] |= SHOWN_VALUE;
                if (!people_have_name(&people[i])) {
                    s->shown[f] |= SHOWN_EMAIL;   /* addresses were printed */
                }
            }
        }
    }
    if (with_source) {
        print_source(s, out, ch);
    }
    suggest(s, c, out);
}

/* Every cell of the row that was not shown yet. */
static void print_all(RagSession *s, const RagCorpus *c, int chunk, FILE *out)
{
    const RagChunk *ch = rag_corpus_chunk(c, chunk);
    const int nf = ch->n_fields < MAX_FIELDS ? ch->n_fields : MAX_FIELDS;

    set_context(s, chunk);
    for (int f = 0; f < nf; f++) {
        if (ch->fields[f].variant == 0) {
            fprintf(out, "%s: %s\n", ch->fields[f].label, ch->fields[f].value);
            s->shown[f] = SHOWN_VALUE | SHOWN_EMAIL;
        }
    }
    if (has_variants(ch)) {
        print_variants(out, ch);
        s->variants_shown = 1;
    }
    print_source(s, out, ch);
    s->pending = PENDING_NONE;
}

/* Answer the question for one chosen row. */
static void answer_row(RagSession *s, const RagCorpus *c, int chunk,
                       const Question *q, FILE *out)
{
    const RagChunk *ch = rag_corpus_chunk(c, chunk);
    Match m;
    Asked a;

    match_row(q, ch, &m);
    asked_fields(q, ch, &m, &a);
    if (q->want_all && a.n == 0 && !a.variants) {
        print_all(s, c, chunk, out);
        return;
    }
    print_answer(s, c, chunk, &a, q->want_email, 1, out);
}

/* ------------------------------------------------------------------ */
/* Lists                                                               */
/* ------------------------------------------------------------------ */

/* Add a row to a list unless the same thing is in it; keep the fuller row. */
static void list_add(const RagCorpus *c, int *list, int *n, int *total, int chunk)
{
    for (int i = 0; i < *n; i++) {
        if (same_entity(c, list[i], chunk)) {
            if (rag_corpus_chunk(c, chunk)->n_fields >
                rag_corpus_chunk(c, list[i])->n_fields) {
                list[i] = chunk;
            }
            return;
        }
    }
    (*total)++;
    if (*n < MAX_LIST) {
        list[(*n)++] = chunk;
    }
}

static void print_numbered(RagSession *s, const RagCorpus *c, const int *list,
                           int n, int total, const char *question, FILE *out)
{
    for (int i = 0; i < n; i++) {
        print_list_line(out, i + 1, rag_corpus_chunk(c, list[i]));
        s->list[i] = list[i];
    }
    if (total > n) {
        fprintf(out, "  ... and %d more\n", total - n);
    }
    s->n_list = n;
    s->pending = PENDING_CHOICE;
    snprintf(s->list_question, sizeof(s->list_question), "%s", question);
    offer(s, out, "Type a number for one of them, or ask another question.");
}

/* ------------------------------------------------------------------ */
/* The tiers                                                           */
/* ------------------------------------------------------------------ */

/* 1. Follow-up on the current row: only label words (or "its mail id"). */
static int try_follow_up(RagSession *s, const RagCorpus *c, const Question *q,
                         int allow_values, FILE *out)
{
    if (s->ctx < 0) {
        return 0;
    }
    const RagChunk *ch = rag_corpus_chunk(c, s->ctx);
    Match m;
    Asked a;

    match_row(q, ch, &m);
    if (m.n_key > 0 || m.n_missing > 0 || m.n_text > 0 ||
        (!allow_values && m.n_value > 0) || (allow_values && m.n_value == 0)) {
        return 0;
    }
    asked_fields(q, ch, &m, &a);
    if (a.n == 0 && !a.variants) {
        if (q->want_all) {
            print_all(s, c, s->ctx, out);
            return 1;
        }
        return 0;
    }
    print_answer(s, c, s->ctx, &a, q->want_email, 0, out);
    return 1;
}

static int answer_question(RagSession *s, const RagCorpus *c, const RagIndex *ix,
                           const char *input, FILE *out);

/*
 * 1 if the question asks about several rows: a count / list intent, or a
 * filler word typed in the plural ("which courses are in slot D"). Such a
 * question is never a follow-up on the current row, even when it quotes
 * one of that row's cells.
 */
static int asks_for_many(const Question *q)
{
    if (q->count_intent || q->list_intent) {
        return 1;
    }
    /* q->words keeps the words as typed (the terms are already stemmed) */
    for (int i = 0; i < q->n_words; i++) {
        char stem[32];
        size_t len = strlen(q->words[i]);
        if (len < 4 || q->words[i][len - 1] != 's') {
            continue;
        }
        memcpy(stem, q->words[i], len - 1);
        stem[len - 1] = '\0';
        if (in_list(stem, FILLER, COUNT(FILLER))) {
            return 1;
        }
    }
    return 0;
}

/* The whole method for a fresh question. Returns 1 if rows were shown. */
static int answer_fresh(RagSession *s, const RagCorpus *c, const RagIndex *ix,
                        const Question *q, FILE *out)
{
    static RagHit hits[MAX_HITS];
    static Match match[MAX_HITS];
    const int n_hits = rag_index_search(ix, q->text, hits, MAX_HITS);
    const int unknown = unknown_terms(q);
    int list[MAX_LIST], n = 0, total = 0;

    for (int h = 0; h < n_hits; h++) {
        match_row(q, rag_corpus_chunk(c, hits[h].chunk), &match[h]);
    }

    /* ---- 2. lookup: every term is a key term or a label word ---- */
    double best = 0.0;
    int best_h = -1;
    for (int h = 0; h < n_hits; h++) {
        const Match *m = &match[h];
        if (m->n_key > 0 && m->n_missing == 0 && m->n_value == 0 &&
            m->n_text == 0 && m->coverage > best &&
            (unknown == 0 || m->coverage >= 1.0)) {
            best = m->coverage;
            best_h = h;
        }
    }
    if (best > 0.0) {
        /* "list all physics courses" lists; "how many credits is X" does not */
        const int listing = q->list_intent && match[best_h].n_label == 0 &&
                            q->n_syn == 0;
        for (int h = 0; h < n_hits; h++) {
            const Match *m = &match[h];
            if (m->n_key > 0 && m->n_missing == 0 && m->n_value == 0 &&
                m->n_text == 0 &&
                (listing ? (unknown == 0 || m->coverage >= 1.0)
                         : m->coverage >= best - 1e-9)) {
                list_add(c, list, &n, &total, hits[h].chunk);
            }
        }
        if (total == 1) {
            answer_row(s, c, list[0], q, out);
            return 1;
        }
        if (listing) {
            fprintf(out, "%d found:\n", total);
        } else {
            fprintf(out, "%d rows match equally well. Which one do you mean?\n",
                    total);
        }
        print_numbered(s, c, list, n, total, q->text, out);
        return 1;
    }

    /* ---- 3. a follow-up that quotes a cell of the current row ----
     * Not when several rows are asked for: "how many courses are in slot
     * D" asked after a row whose slot is D is about every row in slot D,
     * not about that row's slot. */
    if (!asks_for_many(q) && try_follow_up(s, c, q, 1, out)) {
        return 1;
    }

    if (unknown == 0) {
        /* ---- 4. a footnote that has every term ---- */
        for (int h = 0; h < n_hits; h++) {
            const RagChunk *ch = rag_corpus_chunk(c, hits[h].chunk);
            if (ch->kind == RAG_CHUNK_NOTE && match[h].n_value > 0 &&
                match[h].n_missing == 0 && match[h].n_text == 0) {
                Asked none;
                memset(&none, 0, sizeof(none));
                print_answer(s, c, hits[h].chunk, &none, 0, 1, out);
                return 1;
            }
        }

        /* ---- 5. list / count: one other column has all the terms ---- */
        for (int h = 0; h < n_hits; h++) {
            const RagChunk *ch = rag_corpus_chunk(c, hits[h].chunk);
            const Match *m = &match[h];
            if (ch->kind != RAG_CHUNK_ROW || m->n_key > 0 || m->n_missing > 0 ||
                m->n_text > 0 || m->n_value == 0) {
                continue;
            }
            /* is the search limited to named columns ("in slot D")? */
            int limited = 0;
            for (int f = 0; f < ch->n_fields && !limited; f++) {
                if (ch->fields[f].key) {
                    continue;
                }
                for (int i = 0; i < q->n_terms; i++) {
                    limited |= m->cls[i] == C_LABEL &&
                               rag_text_has_term(ch->fields[f].label, q->terms[i].term);
                }
                for (int y = 0; y < q->n_syn; y++) {
                    limited |= rag_text_has_term(ch->fields[f].label, q->syn[y]);
                }
            }
            for (int f = 0; f < ch->n_fields; f++) {
                const RagField *fd = &ch->fields[f];
                if (fd->key) {
                    continue;
                }
                if (limited) {
                    int named = 0;
                    for (int i = 0; i < q->n_terms; i++) {
                        named |= m->cls[i] == C_LABEL &&
                                 rag_text_has_term(fd->label, q->terms[i].term);
                    }
                    for (int y = 0; y < q->n_syn; y++) {
                        named |= rag_text_has_term(fd->label, q->syn[y]);
                    }
                    if (!named) {
                        continue;
                    }
                }
                int all = 1;
                for (int i = 0; i < q->n_terms && all; i++) {
                    if (m->cls[i] == C_VALUE) {
                        all = rag_text_has_term(fd->value, q->terms[i].term);
                    }
                }
                if (all) {
                    list_add(c, list, &n, &total, hits[h].chunk);
                    break;
                }
            }
        }
        if (total > 0) {
            /* "3 found for santhakumar:" */
            fprintf(out, "%d found for", total);
            for (int i = 0; i < q->n_terms; i++) {
                int value_term = 0;
                for (int h = 0; h < n_hits && !value_term; h++) {
                    value_term = hits[h].chunk == list[0] && match[h].cls[i] == C_VALUE;
                }
                if (value_term && q->joined_parts[i][0] < 0) {
                    fprintf(out, " %s", q->terms[i].typed);
                }
            }
            fprintf(out, ":\n");
            print_numbered(s, c, list, n, total, "", out);
            return 1;
        }

        /* ---- 6. loose: a key term, the rest anywhere in the row ---- */
        double top = 0.0;
        for (int h = 0; h < n_hits; h++) {
            if (match[h].n_key > 0 && match[h].n_missing == 0 &&
                hits[h].score > top) {
                top = hits[h].score;
            }
        }
        if (top > 0.0) {
            for (int h = 0; h < n_hits; h++) {
                if (match[h].n_key > 0 && match[h].n_missing == 0 &&
                    hits[h].score >= 0.97 * top) {
                    list_add(c, list, &n, &total, hits[h].chunk);
                }
            }
            if (total == 1) {
                answer_row(s, c, list[0], q, out);
                return 1;
            }
            fprintf(out, "%d rows match equally well. Which one do you mean?\n",
                    total);
            print_numbered(s, c, list, n, total, q->text, out);
            return 1;
        }
    }

    /* ---- 7. nothing exact ---- */
    int usable = 0;
    for (int i = 0; i < q->n_terms; i++) {
        usable += !q->skip[i] && !q->filler[i] && !q->terms[i].is_label;
    }
    if (usable == 0 && q->n_terms > 0 && unknown == 0) {
        fprintf(out, "Please say which one you mean, for example a code or a name.\n");
        s->pending = PENDING_NONE;
        return 0;
    }
    fprintf(out, "Sorry, I couldn't find any data about your question.\n");
    s->pending = PENDING_NONE;
    s->n_list = 0;
    if (n_hits > 0) {
        const double floor = hits[0].score * 0.35 > 1.0 ? hits[0].score * 0.35 : 1.0;
        int rel_total = 0;
        for (int h = 0; h < n_hits && s->n_list < MAX_RELATED; h++) {
            if (hits[h].score >= floor && match[h].n_key > 0) {
                int count = s->n_list;
                list_add(c, s->list, &count, &rel_total, hits[h].chunk);
                s->n_list = count < MAX_RELATED ? count : MAX_RELATED;
            }
        }
    }
    if (s->n_list > 0) {
        s->pending = PENDING_RELATED;
        offer(s, out, "There is some related information. Do you want to see it? (Y/N)");
    }
    return 0;
}

static void note_corrections(const Question *q, FILE *out)
{
    int n = 0;
    for (int i = 0; i < q->n_terms; i++) {
        if (q->terms[i].kind == RAG_TERM_CORRECTED && !q->filler[i]) {
            fprintf(out, "%s\"%s\" as \"%s\"", n++ ? ", " : "(Reading ",
                    q->terms[i].typed, q->terms[i].term);
        }
    }
    if (n > 0) {
        fprintf(out, ")\n");
    }
}

static int answer_question(RagSession *s, const RagCorpus *c, const RagIndex *ix,
                           const char *input, FILE *out)
{
    static Question q;

    analyse(ix, input, &q);
    if (try_follow_up(s, c, &q, 0, out)) {
        return 1;
    }
    note_corrections(&q, out);
    return answer_fresh(s, c, ix, &q, out);
}

/* ------------------------------------------------------------------ */
/* Entry point: replies to offers, then questions                      */
/* ------------------------------------------------------------------ */

int rag_answer(RagSession *s, const RagCorpus *c, const RagIndex *ix,
               const char *input, FILE *out)
{
    char line[1024], first[32];
    size_t n = 0, f = 0;
    const char *rest;

    /* lower-case copy without surrounding blanks and closing punctuation */
    while (*input == ' ' || *input == '\t') {
        input++;
    }
    for (; input[n] != '\0' && n < sizeof(line) - 1; n++) {
        line[n] = (char)tolower((unsigned char)input[n]);
    }
    while (n > 0 && strchr(" \t.!?,", line[n - 1]) != NULL) {
        n--;
    }
    line[n] = '\0';
    while (line[f] != '\0' && isalnum((unsigned char)line[f]) && f < sizeof(first) - 1) {
        first[f] = line[f];
        f++;
    }
    first[f] = '\0';
    rest = line + f;
    while (*rest != '\0' && !isalnum((unsigned char)*rest)) {
        rest++;
    }

    const int yes = in_list(first, YES_WORDS, COUNT(YES_WORDS)) ||
                    strcmp(first, "please") == 0;
    const int no = in_list(first, NO_WORDS, COUNT(NO_WORDS)) &&
                   (*rest == '\0' || strncmp(rest, "thank", 5) == 0);
    const int pending = s->pending;

    s->last_source[0] = '\0';
    s->last_offer[0] = '\0';
    if (n == 0) {
        return 0;
    }
    if (rag_corpus_chunk_count(c) == 0) {
        fprintf(out, "No rows are loaded yet. Put .xlsx or .csv files in the "
                     "documents folder.\n");
        return 0;
    }

    if (pending == PENDING_CHOICE && line[strspn(line, "0123456789")] == '\0') {
        int pick = atoi(line);
        if (pick < 1 || pick > s->n_list) {
            fprintf(out, "Please type a number from 1 to %d, or ask another "
                         "question.\n", s->n_list);
            return 0;
        }
        static Question q;
        analyse(ix, s->list_question, &q);
        s->pending = PENDING_NONE;
        answer_row(s, c, s->list[pick - 1], &q, out);
        return 1;
    }

    if (pending == PENDING_RELATED && (yes || no)) {
        s->pending = PENDING_NONE;
        if (yes) {
            int list[MAX_LIST];
            const int count = s->n_list;
            memcpy(list, s->list, sizeof(list));
            print_numbered(s, c, list, count, count, "", out);
            return 1;
        }
        return 0;
    }

    if (pending == PENDING_SUGGEST && no) {
        s->pending = PENDING_NONE;
        return 0;
    }
    if (pending == PENDING_SUGGEST && yes) {
        /* "yes give the room" names another column: try that first */
        if (*rest != '\0') {
            static Question q;
            analyse(ix, rest, &q);
            if (try_follow_up(s, c, &q, 0, out) || try_follow_up(s, c, &q, 1, out)) {
                return 1;
            }
        }
        Asked a;
        memset(&a, 0, sizeof(a));
        if (s->suggest_field == SUGGEST_VARIANTS) {
            a.variants = 1;
        } else {
            a.fields[a.n++] = s->suggest_field;
        }
        print_answer(s, c, s->ctx, &a, s->suggest_email, 0, out);
        return 1;
    }

    if ((yes || no) && *rest == '\0') {
        fprintf(out, "Please ask a question.\n");
        return 0;
    }
    return answer_question(s, c, ix, input, out);
}
