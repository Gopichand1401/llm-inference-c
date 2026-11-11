/*
 * rag_chunk.c - folder scan, header detection and row-to-chunk conversion.
 * The rules are documented in include/rag_chunk.h; tests/rag_reference.py
 * implements the same rules on top of openpyxl, and the two are compared
 * chunk by chunk in tests/run_rag_checks.sh.
 */
#define _POSIX_C_SOURCE 200809L

#include "rag_chunk.h"
#include "xlsx_reader.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define HEADER_SEARCH_ROWS 30

/* ------------------------------------------------------------------ */
/* Small arena: every string and field array of a corpus lives here,   */
/* so freeing the corpus is a walk over a few blocks.                  */
/* ------------------------------------------------------------------ */

typedef struct ArenaBlock {
    struct ArenaBlock *next;
    size_t used;
    size_t capacity;
    unsigned char data[];
} ArenaBlock;

typedef struct {
    ArenaBlock *head;
} Arena;

static void *arena_alloc(Arena *a, size_t size)
{
    size = (size + 15u) & ~(size_t)15u;
    if (a->head == NULL || a->head->capacity - a->head->used < size) {
        size_t cap = size > 65536u ? size : 65536u;
        ArenaBlock *b = malloc(sizeof(*b) + cap);
        if (b == NULL) {
            return NULL;
        }
        b->next = a->head;
        b->used = 0;
        b->capacity = cap;
        a->head = b;
    }
    void *p = a->head->data + a->head->used;
    a->head->used += size;
    return p;
}

static char *arena_strndup(Arena *a, const char *s, size_t n)
{
    char *p = arena_alloc(a, n + 1);
    if (p != NULL) {
        memcpy(p, s, n);
        p[n] = '\0';
    }
    return p;
}

static char *arena_strdup(Arena *a, const char *s)
{
    return arena_strndup(a, s, strlen(s));
}

static void arena_free(Arena *a)
{
    while (a->head != NULL) {
        ArenaBlock *next = a->head->next;
        free(a->head);
        a->head = next;
    }
}

/* ------------------------------------------------------------------ */
/* Corpus                                                              */
/* ------------------------------------------------------------------ */

typedef struct {
    char *path;
    long long size;
    long long mtime_sec;
    long mtime_nsec;
} ScanEntry;

struct RagCorpus {
    Arena arena;
    RagChunk *chunks;
    int n_chunks, chunk_cap;
    RagFileInfo *files;
    int n_files;
    char **warnings;
    int n_warnings, warning_cap;
    ScanEntry *scan;          /* what was loaded, for change detection */
    int n_scan;
    char **inputs;
    int n_inputs;
    RagLoadOptions options;
    int *note_slots;          /* hash set of note chunk index + 1, 0 = free */
    int note_cap, n_notes;
    int out_of_memory;
};

static unsigned long hash_text(const char *s)
{
    unsigned long h = 2166136261u;
    for (; *s; s++) {
        h = (h ^ (unsigned char)*s) * 16777619u;
    }
    return h;
}

/* Slot holding this note text, or the free slot where it belongs. */
static int *note_slot(RagCorpus *c, const char *text)
{
    unsigned long mask = (unsigned long)c->note_cap - 1;
    unsigned long i = hash_text(text) & mask;
    for (;;) {
        int *slot = &c->note_slots[i];
        if (*slot == 0 || strcmp(c->chunks[*slot - 1].note, text) == 0) {
            return slot;
        }
        i = (i + 1) & mask;
    }
}

static int note_set_grow(RagCorpus *c)
{
    int cap = c->note_cap ? c->note_cap * 2 : 64;
    int *old = c->note_slots;
    int old_cap = c->note_cap;
    c->note_slots = calloc((size_t)cap, sizeof(int));
    if (c->note_slots == NULL) {
        c->note_slots = old;
        return -1;
    }
    c->note_cap = cap;
    for (int i = 0; i < old_cap; i++) {
        if (old[i] != 0) {
            *note_slot(c, c->chunks[old[i] - 1].note) = old[i];
        }
    }
    free(old);
    return 0;
}

static void add_warning(RagCorpus *c, const char *fmt, ...)
{
    char buf[512];
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    if (c->n_warnings == c->warning_cap) {
        int cap = c->warning_cap ? c->warning_cap * 2 : 16;
        char **w = realloc(c->warnings, (size_t)cap * sizeof(*w));
        if (w == NULL) {
            c->out_of_memory = 1;
            return;
        }
        c->warnings = w;
        c->warning_cap = cap;
    }
    char *copy = arena_strdup(&c->arena, buf);
    if (copy == NULL) {
        c->out_of_memory = 1;
        return;
    }
    c->warnings[c->n_warnings++] = copy;
}

static RagChunk *new_chunk(RagCorpus *c)
{
    if (c->n_chunks == c->chunk_cap) {
        int cap = c->chunk_cap ? c->chunk_cap * 2 : 256;
        RagChunk *n = realloc(c->chunks, (size_t)cap * sizeof(*n));
        if (n == NULL) {
            c->out_of_memory = 1;
            return NULL;
        }
        c->chunks = n;
        c->chunk_cap = cap;
    }
    RagChunk *ch = &c->chunks[c->n_chunks++];
    memset(ch, 0, sizeof(*ch));
    return ch;
}

/* ------------------------------------------------------------------ */
/* Folder scanning                                                     */
/* ------------------------------------------------------------------ */

/* Case-insensitive file extension test (ext includes the dot). */
static int ends_with_ext(const char *name, const char *ext)
{
    size_t n = strlen(name), e = strlen(ext);
    if (n <= e) {
        return 0;
    }
    for (size_t i = 0; i < e; i++) {
        if (tolower((unsigned char)name[n - e + i]) != ext[i]) {
            return 0;
        }
    }
    return 1;
}

static int is_data_file(const char *name)
{
    return ends_with_ext(name, ".xlsx") || ends_with_ext(name, ".csv");
}

static const char *base_name(const char *path)
{
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

static int compare_entries(const void *a, const void *b)
{
    const ScanEntry *x = a, *y = b;
    return strcmp(x->path, y->path);
}

static void free_scan(ScanEntry *e, int n)
{
    for (int i = 0; i < n; i++) {
        free(e[i].path);
    }
    free(e);
}

static int push_entry(ScanEntry **list, int *n, int *cap, const char *path,
                      const struct stat *st)
{
    if (*n == *cap) {
        int nc = *cap ? *cap * 2 : 32;
        ScanEntry *l = realloc(*list, (size_t)nc * sizeof(*l));
        if (l == NULL) {
            return -1;
        }
        *list = l;
        *cap = nc;
    }
    ScanEntry *e = &(*list)[*n];
    size_t len = strlen(path);
    e->path = malloc(len + 1);
    if (e->path == NULL) {
        return -1;
    }
    memcpy(e->path, path, len + 1);
    e->size = (long long)st->st_size;
    e->mtime_sec = (long long)st->st_mtim.tv_sec;
    e->mtime_nsec = (long)st->st_mtim.tv_nsec;
    (*n)++;
    return 0;
}

/*
 * List the .xlsx files of every input, each folder sorted by name, in input
 * order. Returns 0 or -1 with a message.
 */
static int scan_inputs(char *const *inputs, int n_inputs, ScanEntry **out,
                       int *out_n, char *error, size_t cap)
{
    ScanEntry *list = NULL;
    int n = 0, lcap = 0;

    for (int i = 0; i < n_inputs; i++) {
        struct stat st;
        const char *in = inputs[i];

        if (stat(in, &st) != 0) {
            snprintf(error, cap, "%s: %s", in, strerror(errno));
            free_scan(list, n);
            return -1;
        }
        if (S_ISREG(st.st_mode)) {
            if (push_entry(&list, &n, &lcap, in, &st) != 0) {
                goto oom;
            }
            continue;
        }
        if (!S_ISDIR(st.st_mode)) {
            snprintf(error, cap, "%s: not a file or folder", in);
            free_scan(list, n);
            return -1;
        }

        DIR *dir = opendir(in);
        if (dir == NULL) {
            snprintf(error, cap, "%s: %s", in, strerror(errno));
            free_scan(list, n);
            return -1;
        }
        int first = n;
        struct dirent *de;
        while ((de = readdir(dir)) != NULL) {
            const char *name = de->d_name;
            if (name[0] == '.' || (name[0] == '~' && name[1] == '$') ||
                !is_data_file(name)) {
                continue;
            }
            size_t len = strlen(in) + strlen(name) + 2;
            char *full = malloc(len);
            if (full == NULL) {
                closedir(dir);
                goto oom;
            }
            size_t in_len = strlen(in);
            if (in_len > 0 && in[in_len - 1] == '/') {
                snprintf(full, len, "%s%s", in, name);
            } else {
                snprintf(full, len, "%s/%s", in, name);
            }
            struct stat fst;
            int okay = stat(full, &fst) == 0 && S_ISREG(fst.st_mode);
            int pushed = okay ? push_entry(&list, &n, &lcap, full, &fst) : 0;
            free(full);
            if (pushed != 0) {
                closedir(dir);
                goto oom;
            }
        }
        closedir(dir);
        qsort(list + first, (size_t)(n - first), sizeof(*list), compare_entries);
    }
    *out = list;
    *out_n = n;
    return 0;

oom:
    snprintf(error, cap, "out of memory while scanning");
    free_scan(list, n);
    return -1;
}

/* ------------------------------------------------------------------ */
/* Cell helpers                                                        */
/* ------------------------------------------------------------------ */

/* "", "-", en dash and em dash count as empty. */
static int is_blank(const char *s)
{
    return s[0] == '\0' || strcmp(s, "-") == 0 ||
           strcmp(s, "\xE2\x80\x93") == 0 || strcmp(s, "\xE2\x80\x94") == 0;
}

static const char *cell_value(const XlsxRow *row, int col)
{
    const char *s = xlsx_cell(row, col);
    return is_blank(s) ? "" : s;
}

static int count_filled(const XlsxRow *row)
{
    int n = 0;
    for (int c = 0; c < row->ncols; c++) {
        if (cell_value(row, c)[0] != '\0') {
            n++;
        }
    }
    return n;
}

static int is_number(const char *s)
{
    char *end;
    errno = 0;
    (void)strtod(s, &end);
    return end != s && *end == '\0';
}

static int count_text(const XlsxRow *row)
{
    int n = 0;
    for (int c = 0; c < row->ncols; c++) {
        const char *v = cell_value(row, c);
        if (v[0] != '\0' && !is_number(v)) {
            n++;
        }
    }
    return n;
}

static void column_letters(int col, char *out)
{
    char tmp[8];
    int n = 0;

    for (int c = col + 1; c > 0 && n < 7; c = (c - 1) / 26) {
        tmp[n++] = (char)('A' + (c - 1) % 26);
    }
    for (int i = 0; i < n; i++) {
        out[i] = tmp[n - 1 - i];
    }
    out[n] = '\0';
}

/*
 * Header text -> short label: cut at the first '(' '?' or ',', trim, drop a
 * trailing "Yes/No", trim ' ' ':' '-' '.' at the end. If that leaves less
 * than 2 characters, the full text is used.
 */
static char *short_label(Arena *a, const char *text)
{
    size_t n = strcspn(text, "(?,");
    while (n > 0 && text[n - 1] == ' ') {
        n--;
    }
    if (n >= 6 && strncmp(text + n - 6, "Yes/No", 6) == 0) {
        n -= 6;
    }
    while (n > 0 && strchr(" :-.", text[n - 1]) != NULL) {
        n--;
    }
    if (n < 2) {
        return arena_strdup(a, text);
    }
    return arena_strndup(a, text, n);
}

/* Join the non-empty cells of a row with " | ". */
static char *join_row(Arena *a, const XlsxRow *row)
{
    size_t len = 0;
    for (int c = 0; c < row->ncols; c++) {
        const char *v = cell_value(row, c);
        if (v[0] != '\0') {
            len += strlen(v) + 3;
        }
    }
    char *out = arena_alloc(a, len + 1);
    if (out == NULL) {
        return NULL;
    }
    out[0] = '\0';
    size_t pos = 0;
    for (int c = 0; c < row->ncols; c++) {
        const char *v = cell_value(row, c);
        if (v[0] == '\0') {
            continue;
        }
        if (pos > 0) {
            memcpy(out + pos, " | ", 3);
            pos += 3;
        }
        size_t vl = strlen(v);
        memcpy(out + pos, v, vl);
        pos += vl;
    }
    out[pos] = '\0';
    return out;
}

/* ------------------------------------------------------------------ */
/* Chunk building                                                      */
/* ------------------------------------------------------------------ */

typedef struct {
    RagCorpus *c;
    RagFileInfo *file;
    const char *sheet;     /* arena copy */
    const char *title;     /* arena copy, "" if none */
    char **labels;         /* arena labels per header column ("" = none) */
    int n_labels;
    int key_cols[2];
    const XlsxRow **group;
    int n_group, group_cap;
    int chunks_before;
} SheetCtx;

static void add_note(SheetCtx *s, const char *text, int row)
{
    RagCorpus *c = s->c;

    if (2 * (c->n_notes + 1) > c->note_cap && note_set_grow(c) != 0) {
        c->out_of_memory = 1;
        return;
    }
    int *slot = note_slot(c, text);
    if (*slot != 0) {
        c->chunks[*slot - 1].note_count++;
        return;
    }
    RagChunk *ch = new_chunk(c);
    char *copy = arena_strdup(&c->arena, text);
    if (ch == NULL || copy == NULL) {
        c->out_of_memory = 1;
        return;
    }
    ch->kind = RAG_CHUNK_NOTE;
    ch->file = s->file->name;
    ch->sheet = s->sheet;
    ch->title = s->title;
    ch->row_first = ch->row_last = row;
    ch->note = copy;
    ch->note_count = 1;
    *slot = c->n_chunks;
    c->n_notes++;
    s->file->chunks++;
}

static const char *label_for(SheetCtx *s, int col, char *buf, size_t cap)
{
    if (col < s->n_labels && s->labels[col][0] != '\0') {
        return s->labels[col];
    }
    char letters[8];
    column_letters(col, letters);
    snprintf(buf, cap, "Column %s", letters);
    return buf;
}

static int push_field(RagCorpus *c, RagField *fields, int *n, const char *label,
                      const char *value, int key, int variant)
{
    char *l = arena_strdup(&c->arena, label);
    char *v = arena_strdup(&c->arena, value);
    if (l == NULL || v == NULL) {
        c->out_of_memory = 1;
        return -1;
    }
    fields[*n].label = l;
    fields[*n].value = v;
    fields[*n].key = key;
    fields[*n].variant = variant;
    (*n)++;
    return 0;
}

/* Turn the collected run of rows into one chunk. */
static void flush_group(SheetCtx *s)
{
    RagCorpus *c = s->c;
    const int g = s->n_group;

    if (g == 0) {
        return;
    }
    s->n_group = 0;

    int ncols = s->n_labels;
    int max_fields = 0;
    for (int r = 0; r < g; r++) {
        if (s->group[r]->ncols > ncols) {
            ncols = s->group[r]->ncols;
        }
    }
    max_fields = ncols * (g + 1);

    RagChunk *ch = new_chunk(c);
    RagField *fields = arena_alloc(&c->arena, (size_t)max_fields * sizeof(*fields) + 1);
    int *vrows = arena_alloc(&c->arena, (size_t)g * sizeof(int));
    if (ch == NULL || fields == NULL || vrows == NULL) {
        c->out_of_memory = 1;
        return;
    }

    int nf = 0;
    char lbuf[32];
    /* Pass 1: values that are the same in every row of the run. */
    unsigned char *common = arena_alloc(&c->arena, (size_t)ncols + 1);
    if (common == NULL) {
        c->out_of_memory = 1;
        return;
    }
    for (int col = 0; col < ncols; col++) {
        const char *v0 = cell_value(s->group[0], col);
        int same = 1;
        for (int r = 1; r < g && same; r++) {
            same = strcmp(v0, cell_value(s->group[r], col)) == 0;
        }
        common[col] = (unsigned char)same;
        if (same && v0[0] != '\0') {
            int key = col == s->key_cols[0] || col == s->key_cols[1];
            if (push_field(c, fields, &nf, label_for(s, col, lbuf, sizeof(lbuf)),
                           v0, key, 0) != 0) {
                return;
            }
        }
    }
    /* Pass 2: the rest, row by row. */
    for (int r = 0; r < g; r++) {
        vrows[r] = s->group[r]->row;
        if (g == 1) {
            continue;
        }
        for (int col = 0; col < ncols; col++) {
            const char *v = cell_value(s->group[r], col);
            if (common[col] || v[0] == '\0') {
                continue;
            }
            int key = col == s->key_cols[0] || col == s->key_cols[1];
            if (push_field(c, fields, &nf, label_for(s, col, lbuf, sizeof(lbuf)),
                           v, key, r + 1) != 0) {
                return;
            }
        }
    }

    ch->kind = RAG_CHUNK_ROW;
    ch->file = s->file->name;
    ch->sheet = s->sheet;
    ch->title = s->title;
    ch->row_first = s->group[0]->row;
    ch->row_last = s->group[g - 1]->row;
    ch->n_variants = g;
    ch->variant_rows = vrows;
    ch->n_fields = nf;
    ch->fields = fields;
    s->file->chunks++;
}

static void group_push(SheetCtx *s, const XlsxRow *row)
{
    if (s->n_group == s->group_cap) {
        int cap = s->group_cap ? s->group_cap * 2 : 16;
        const XlsxRow **g = realloc(s->group, (size_t)cap * sizeof(*g));
        if (g == NULL) {
            s->c->out_of_memory = 1;
            return;
        }
        s->group = g;
        s->group_cap = cap;
    }
    s->group[s->n_group++] = row;
}

typedef struct {
    const char *name;        /* tab name, "" for a CSV file */
    const XlsxRow **rows;    /* non-empty rows in ascending order */
    int n;
} Table;

/* Index of the header row among the tab's rows, or -1. */
static int find_header(const Table *t, int forced_row)
{
    const int n = t->n;

    if (forced_row > 0) {
        for (int i = 0; i < n; i++) {
            if (t->rows[i]->row == forced_row) {
                return i;
            }
        }
    }

    int widest = 0;
    for (int i = 0; i < n; i++) {
        int f = count_filled(t->rows[i]);
        if (f > widest) {
            widest = f;
        }
    }
    for (int i = 0; i < n && i < HEADER_SEARCH_ROWS; i++) {
        const XlsxRow *row = t->rows[i];
        int f = count_filled(row);
        if (f >= 2 && 2 * f >= widest && 5 * count_text(row) >= 3 * f) {
            return i;
        }
    }
    return -1;
}

static void load_table(RagCorpus *c, RagFileInfo *file, const Table *t)
{
    const int n = t->n;
    SheetCtx s;

    memset(&s, 0, sizeof(s));
    s.c = c;
    s.file = file;
    s.key_cols[0] = s.key_cols[1] = -1;
    s.chunks_before = file->chunks;
    s.sheet = arena_strdup(&c->arena, t->name);
    s.title = "";
    if (s.sheet == NULL) {
        c->out_of_memory = 1;
        return;
    }

    int h = find_header(t, c->options.header_row);
    if (h < 0) {
        /* No table shape: every row is a note. */
        for (int i = 0; i < n && !c->out_of_memory; i++) {
            const XlsxRow *row = t->rows[i];
            if (count_filled(row) == 0) {
                continue;
            }
            char *text = join_row(&c->arena, row);
            if (text == NULL) {
                c->out_of_memory = 1;
                break;
            }
            add_note(&s, text, row->row);
        }
        goto done;
    }

    for (int i = h - 1; i >= 0; i--) {
        const XlsxRow *row = t->rows[i];
        if (count_filled(row) > 0) {
            char *tt = join_row(&c->arena, row);
            if (tt == NULL) {
                c->out_of_memory = 1;
                return;
            }
            s.title = tt;
            break;
        }
    }

    const XlsxRow *header = t->rows[h];
    s.n_labels = header->ncols;
    s.labels = arena_alloc(&c->arena, (size_t)(s.n_labels + 1) * sizeof(char *));
    if (s.labels == NULL) {
        c->out_of_memory = 1;
        return;
    }
    int keys = 0;
    for (int col = 0; col < s.n_labels; col++) {
        const char *ht = cell_value(header, col);
        s.labels[col] = ht[0] ? short_label(&c->arena, ht) : (char *)"";
        if (s.labels[col] == NULL) {
            c->out_of_memory = 1;
            return;
        }
        if (ht[0] && keys < 2) {
            s.key_cols[keys++] = col;
        }
    }

    const char *group_key = NULL;
    for (int i = h + 1; i < n && !c->out_of_memory; i++) {
        const XlsxRow *row = t->rows[i];
        int f = count_filled(row);

        if (f == 0) {
            continue;
        }
        if (f == 1) {
            flush_group(&s);
            group_key = NULL;
            for (int col = 0; col < row->ncols; col++) {
                const char *v = cell_value(row, col);
                if (v[0] != '\0') {
                    add_note(&s, v, row->row);
                    break;
                }
            }
            continue;
        }
        const char *key = cell_value(row, s.key_cols[0]);
        if (s.n_group > 0 && key[0] != '\0' && group_key != NULL &&
            strcmp(key, group_key) == 0) {
            group_push(&s, row);
            continue;
        }
        flush_group(&s);
        group_push(&s, row);
        group_key = key[0] != '\0' ? key : NULL;
    }
    flush_group(&s);

done:
    free(s.group);
    if (file->chunks > s.chunks_before) {
        file->tabs_used++;
    }
}

/* ------------------------------------------------------------------ */
/* CSV                                                                 */
/* ------------------------------------------------------------------ */

typedef struct {
    XlsxRow *rows;
    int n, cap;
    const char **cells;      /* all cell pointers, row after row */
    size_t n_cells, cells_cap;
    char *text;              /* cleaned cell texts, NUL separated */
    size_t text_len;
} CsvData;

static int csv_is_space(char ch)
{
    return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' ||
           ch == '\f' || ch == '\v';
}

/*
 * Parse CSV text (RFC 4180 quoting; delimiter ',' ';' or tab, whichever is
 * most frequent outside quotes in the first line; \n, \r\n or \r line
 * ends). Cells get runs of white space collapsed to one space and their
 * ends trimmed. Records are numbered from 1; records without any non-empty
 * cell are dropped but still counted. Returns 0, or -1 when out of memory.
 */
static int csv_parse(const char *in, size_t len, CsvData *d)
{
    size_t i = 0;
    char delim = ',';

    if (len >= 3 && memcmp(in, "\xEF\xBB\xBF", 3) == 0) {
        i = 3;
    }
    {
        size_t count[3] = {0, 0, 0};
        int quoted = 0;
        for (size_t j = i; j < len; j++) {
            char ch = in[j];
            if (ch == '"') {
                quoted = !quoted;
            } else if (!quoted && (ch == '\n' || ch == '\r')) {
                break;
            } else if (!quoted) {
                count[0] += ch == ',';
                count[1] += ch == ';';
                count[2] += ch == '\t';
            }
        }
        if (count[1] > count[0] && count[1] >= count[2]) {
            delim = ';';
        } else if (count[2] > count[0] && count[2] > count[1]) {
            delim = '\t';
        }
    }

    d->text = malloc(len + 2);
    /* at most one cell per input byte, plus one */
    size_t *starts = malloc((len + 2) * sizeof(*starts));
    int *row_of = malloc((len + 2) * sizeof(*row_of));
    if (d->text == NULL || starts == NULL || row_of == NULL) {
        free(starts);
        free(row_of);
        return -1;
    }

    size_t w = 0, n_cells = 0;
    int record = 1;
    while (i < len) {
        /* one record */
        int done_record = 0;
        while (!done_record) {
            /* one field */
            size_t start = w;
            int pending_space = 0, started = 0;
            int quoted = i < len && in[i] == '"';
            if (quoted) {
                i++;
            }
            for (;;) {
                if (i >= len) {
                    done_record = 1;
                    break;
                }
                char ch = in[i];
                if (quoted) {
                    if (ch == '"') {
                        if (i + 1 < len && in[i + 1] == '"') {
                            i += 2;          /* "" -> " */
                        } else {
                            quoted = 0;
                            i++;
                            continue;
                        }
                    } else {
                        i++;
                    }
                } else if (ch == delim) {
                    i++;
                    break;
                } else if (ch == '\n' || ch == '\r') {
                    i++;
                    if (ch == '\r' && i < len && in[i] == '\n') {
                        i++;
                    }
                    done_record = 1;
                    break;
                } else {
                    i++;
                }
                if (csv_is_space(ch)) {
                    pending_space = started;
                } else {
                    if (pending_space) {
                        d->text[w++] = ' ';
                        pending_space = 0;
                    }
                    d->text[w++] = ch;
                    started = 1;
                }
            }
            d->text[w++] = '\0';
            starts[n_cells] = start;
            row_of[n_cells] = record;
            n_cells++;
        }
        record++;
    }

    d->cells = malloc((n_cells + 1) * sizeof(*d->cells));
    d->rows = malloc(((size_t)record + 1) * sizeof(*d->rows));
    if (d->cells == NULL || d->rows == NULL) {
        free(starts);
        free(row_of);
        return -1;
    }
    for (size_t k = 0; k < n_cells; k++) {
        d->cells[k] = d->text + starts[k];
    }
    size_t k = 0;
    while (k < n_cells) {
        size_t first = k;
        int last_filled = -1;
        while (k < n_cells && row_of[k] == row_of[first]) {
            if (d->cells[k][0] != '\0') {
                last_filled = (int)(k - first);
            }
            k++;
        }
        if (last_filled >= 0) {
            XlsxRow *r = &d->rows[d->n++];
            r->row = row_of[first];
            r->ncols = last_filled + 1;
            r->cells = &d->cells[first];
        }
    }
    free(starts);
    free(row_of);
    return 0;
}

static void csv_free(CsvData *d)
{
    free(d->rows);
    free(d->cells);
    free(d->text);
}

static void load_csv(RagCorpus *c, RagFileInfo *file)
{
    FILE *f = fopen(file->path, "rb");
    CsvData d;
    char *buf = NULL;
    size_t len = 0, cap = 0;

    memset(&d, 0, sizeof(d));
    if (f == NULL) {
        add_warning(c, "%s: skipped (%s)", file->name, strerror(errno));
        return;
    }
    for (;;) {
        if (len + 65536 > cap) {
            cap = cap ? cap * 2 : 131072;
            char *nb = realloc(buf, cap);
            if (nb == NULL) {
                free(buf);
                fclose(f);
                c->out_of_memory = 1;
                return;
            }
            buf = nb;
        }
        size_t got = fread(buf + len, 1, 65536, f);
        len += got;
        if (got < 65536) {
            break;
        }
    }
    fclose(f);

    if (memchr(buf, '\0', len) != NULL) {
        add_warning(c, "%s: skipped (not a text file)", file->name);
        free(buf);
        return;
    }
    if (csv_parse(buf, len, &d) != 0) {
        c->out_of_memory = 1;
    } else {
        Table t;
        const XlsxRow **rows = malloc(((size_t)d.n + 1) * sizeof(*rows));
        if (rows == NULL) {
            c->out_of_memory = 1;
        } else {
            for (int i = 0; i < d.n; i++) {
                rows[i] = &d.rows[i];
            }
            t.name = "";
            t.rows = rows;
            t.n = d.n;
            file->ok = 1;
            load_table(c, file, &t);
            free(rows);
            if (file->chunks == 0) {
                add_warning(c, "%s: no rows found", file->name);
            }
        }
    }
    csv_free(&d);
    free(buf);
}

static void load_file(RagCorpus *c, RagFileInfo *file)
{
    char error[256];

    if (ends_with_ext(file->name, ".csv")) {
        load_csv(c, file);
        return;
    }
    XlsxBook *b = xlsx_open(file->path, error, sizeof(error));

    if (b == NULL) {
        add_warning(c, "%s: skipped (%s)", file->name, error);
        return;
    }
    file->ok = 1;
    for (int s = 0; s < xlsx_sheet_count(b) && !c->out_of_memory; s++) {
        if (xlsx_sheet_hidden(b, s) && !c->options.include_hidden) {
            add_warning(c, "%s: hidden tab \"%s\" skipped", file->name,
                        xlsx_sheet_name(b, s));
            continue;
        }
        Table t;
        const int n = xlsx_row_count(b, s);
        const XlsxRow **rows = malloc(((size_t)n + 1) * sizeof(*rows));
        if (rows == NULL) {
            c->out_of_memory = 1;
            break;
        }
        for (int i = 0; i < n; i++) {
            rows[i] = xlsx_row(b, s, i);
        }
        t.name = xlsx_sheet_name(b, s);
        t.rows = rows;
        t.n = n;
        load_table(c, file, &t);
        free(rows);
    }
    if (file->chunks == 0) {
        add_warning(c, "%s: no rows found", file->name);
    }
    xlsx_close(b);
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

RagCorpus *rag_corpus_load(const char *const *inputs, int n_inputs,
                           const RagLoadOptions *options,
                           char *error, size_t error_capacity)
{
    RagCorpus *c = calloc(1, sizeof(*c));

    if (c == NULL) {
        snprintf(error, error_capacity, "out of memory");
        return NULL;
    }
    if (options != NULL) {
        c->options = *options;
    }
    if (n_inputs <= 0) {
        snprintf(error, error_capacity, "no folder or file given");
        free(c);
        return NULL;
    }

    c->inputs = calloc((size_t)n_inputs, sizeof(char *));
    if (c->inputs == NULL) {
        goto oom;
    }
    c->n_inputs = n_inputs;
    for (int i = 0; i < n_inputs; i++) {
        c->inputs[i] = arena_strdup(&c->arena, inputs[i]);
        if (c->inputs[i] == NULL) {
            goto oom;
        }
    }

    if (scan_inputs(c->inputs, c->n_inputs, &c->scan, &c->n_scan, error,
                    error_capacity) != 0) {
        rag_corpus_free(c);
        return NULL;
    }

    c->files = calloc((size_t)c->n_scan + 1, sizeof(*c->files));
    if (c->files == NULL) {
        goto oom;
    }
    for (int i = 0; i < c->n_scan && !c->out_of_memory; i++) {
        RagFileInfo *f = &c->files[c->n_files++];
        f->path = arena_strdup(&c->arena, c->scan[i].path);
        f->name = f->path ? base_name(f->path) : NULL;
        if (f->path == NULL) {
            goto oom;
        }
        load_file(c, f);
    }
    if (c->out_of_memory) {
        goto oom;
    }
    return c;

oom:
    snprintf(error, error_capacity, "out of memory while loading");
    rag_corpus_free(c);
    return NULL;
}

void rag_corpus_free(RagCorpus *c)
{
    if (c == NULL) {
        return;
    }
    free(c->chunks);
    free(c->files);
    free(c->warnings);
    free(c->inputs);
    free(c->note_slots);
    free_scan(c->scan, c->n_scan);
    arena_free(&c->arena);
    free(c);
}

int rag_corpus_chunk_count(const RagCorpus *c)
{
    return c ? c->n_chunks : 0;
}

const RagChunk *rag_corpus_chunk(const RagCorpus *c, int i)
{
    return (c && i >= 0 && i < c->n_chunks) ? &c->chunks[i] : NULL;
}

int rag_corpus_file_count(const RagCorpus *c)
{
    return c ? c->n_files : 0;
}

const RagFileInfo *rag_corpus_file(const RagCorpus *c, int i)
{
    return (c && i >= 0 && i < c->n_files) ? &c->files[i] : NULL;
}

int rag_corpus_warning_count(const RagCorpus *c)
{
    return c ? c->n_warnings : 0;
}

const char *rag_corpus_warning(const RagCorpus *c, int i)
{
    return (c && i >= 0 && i < c->n_warnings) ? c->warnings[i] : NULL;
}

int rag_corpus_input_count(const RagCorpus *c)
{
    return c ? c->n_inputs : 0;
}

const char *rag_corpus_input(const RagCorpus *c, int i)
{
    return (c && i >= 0 && i < c->n_inputs) ? c->inputs[i] : NULL;
}

const RagLoadOptions *rag_corpus_options(const RagCorpus *c)
{
    return c ? &c->options : NULL;
}

/* Append "word name" to a summary, keeping it NUL-terminated. */
static void summary_add(char *out, size_t cap, int *first_of_kind,
                        const char *word, const char *name)
{
    size_t len = strlen(out);
    if (len + 4 >= cap) {
        return;
    }
    if (*first_of_kind) {
        snprintf(out + len, cap - len, "%s%s %s", len ? "; " : "", word,
                 base_name(name));
        *first_of_kind = 0;
    } else {
        snprintf(out + len, cap - len, ", %s", base_name(name));
    }
}

int rag_corpus_changed(const RagCorpus *c, char *summary, size_t cap)
{
    ScanEntry *now = NULL;
    int n_now = 0;
    char error[256];

    if (cap > 0) {
        summary[0] = '\0';
    }
    if (scan_inputs(c->inputs, c->n_inputs, &now, &n_now, error,
                    sizeof(error)) != 0) {
        snprintf(summary, cap, "%s", error);
        return -1;
    }

    /* Both lists are in the same order (input order, then by name), but
     * compare by path so the summary can say what was added or removed. */
    int changed = 0;
    int first_add = 1, first_rm = 1, first_mod = 1;
    for (int i = 0; i < n_now; i++) {
        int found = -1;
        for (int j = 0; j < c->n_scan; j++) {
            if (strcmp(now[i].path, c->scan[j].path) == 0) {
                found = j;
                break;
            }
        }
        if (found < 0) {
            summary_add(summary, cap, &first_add, "added", now[i].path);
            changed = 1;
        } else if (now[i].size != c->scan[found].size ||
                   now[i].mtime_sec != c->scan[found].mtime_sec ||
                   now[i].mtime_nsec != c->scan[found].mtime_nsec) {
            summary_add(summary, cap, &first_mod, "changed", now[i].path);
            changed = 1;
        }
    }
    for (int j = 0; j < c->n_scan; j++) {
        int found = 0;
        for (int i = 0; i < n_now && !found; i++) {
            found = strcmp(now[i].path, c->scan[j].path) == 0;
        }
        if (!found) {
            summary_add(summary, cap, &first_rm, "removed", c->scan[j].path);
            changed = 1;
        }
    }
    free_scan(now, n_now);
    return changed;
}
