#ifndef RAG_CHUNK_H
#define RAG_CHUNK_H

/*
 * rag_chunk.h - turn a folder of .xlsx / .csv files into retrieval chunks
 *               (RAG Step 2: ingestion + chunking).
 *
 *   const char *inputs[] = { "data" };          // folders and/or files
 *   RagCorpus *c = rag_corpus_load(inputs, 1, NULL, error, sizeof(error));
 *   for (int i = 0; i < rag_corpus_chunk_count(c); i++) {
 *       const RagChunk *ch = rag_corpus_chunk(c, i);
 *       ...
 *   }
 *   if (rag_corpus_changed(c, summary, sizeof(summary)) == 1) {
 *       // a file was added, edited or removed: load again
 *   }
 *   rag_corpus_free(c);
 *
 * Nothing in here knows about courses. The rules work on any table:
 *
 *   Header row   per tab, the first of the top 30 non-empty rows with at
 *                least 2 non-empty cells, at least half as many as the
 *                widest row of the tab, and at least 60% non-numeric cells.
 *                RagLoadOptions.header_row forces an Excel row number.
 *                A tab with no such row becomes notes, one per row.
 *   Title        the last non-empty row above the header (e.g. "B. Tech.
 *                Data Science Core Courses"), attached to every chunk.
 *   Labels       header text shortened at the first '(' '?' or ',', and a
 *                trailing "Yes/No" dropped: "Availability (Degree:...)"
 *                -> "Availability". Columns without a header text are
 *                labelled "Column X".
 *   Values       cells that are only "-", en dash or em dash count as empty.
 *   Row chunk    every data row with 2 or more non-empty cells. Consecutive
 *                rows with the same non-empty value in the first header
 *                column (one course split into batches, say) are merged into
 *                one chunk: values equal in every row are kept once, the
 *                rest are kept per row so pairs (slot, room) stay together.
 *   Note chunk   a row below the header with exactly one non-empty cell
 *                (footnotes such as "Primary instructor is the faculty
 *                who..."). Identical notes are stored once, with a count.
 *   Hidden tabs  skipped unless RagLoadOptions.include_hidden is set.
 *
 * CSV files   one table per file, tab name "". RFC 4180 quoting; the
 *             delimiter is ',' ';' or tab, whichever is most frequent in
 *             the first line; cells get white space collapsed and trimmed;
 *             row numbers are record numbers starting at 1.
 *
 * Folders are scanned for *.xlsx and *.csv (any case); Excel lock files
 * ("~$...") and hidden files are ignored. A file that cannot be read is skipped with
 * a warning; the others still load.
 *
 * Errors follow the project style: NULL / non-zero return plus a message in
 * the caller's buffer. Nothing is printed by the library.
 */

#include <stddef.h>

typedef enum {
    RAG_CHUNK_ROW = 1,    /* one table row, or a merged run of rows */
    RAG_CHUNK_NOTE = 2    /* a single-cell row (footnote, remark) */
} RagChunkKind;

typedef struct {
    const char *label;    /* shortened header text */
    const char *value;    /* cell text, never empty */
    int key;              /* 1 for the first two header columns */
    int variant;          /* 0: same in every merged row; k: only row k (1-based) */
} RagField;

typedef struct {
    RagChunkKind kind;
    const char *file;     /* file name without the folder */
    const char *sheet;    /* tab name; "" for a CSV file */
    const char *title;    /* title line above the header, "" if none */
    int row_first;        /* Excel row numbers */
    int row_last;
    int n_variants;       /* rows merged into this chunk (1 = plain row) */
    const int *variant_rows;  /* Excel row of each variant, n_variants entries */
    int n_fields;
    const RagField *fields;   /* common fields first, then variants in order */
    const char *note;     /* RAG_CHUNK_NOTE: the text; NULL for rows */
    int note_count;       /* RAG_CHUNK_NOTE: how many tabs had this note */
} RagChunk;

typedef struct {
    int header_row;       /* Excel row of the header in every tab; 0 = detect */
    int include_hidden;   /* 1: also read hidden tabs */
} RagLoadOptions;

typedef struct {
    const char *path;     /* as found while scanning */
    const char *name;     /* file name without the folder */
    int ok;               /* 1 if the workbook was read */
    int chunks;           /* chunks taken from this file */
    int tabs_used;        /* tabs that produced at least one chunk */
} RagFileInfo;

typedef struct RagCorpus RagCorpus;

/*
 * Load every .xlsx in the given folders / files. options may be NULL.
 * Fails (NULL) only when an input does not exist or memory runs out;
 * an empty folder gives an empty corpus.
 */
RagCorpus *rag_corpus_load(const char *const *inputs, int n_inputs,
                           const RagLoadOptions *options,
                           char *error, size_t error_capacity);

/* Release everything. Safe on NULL. Invalidates every returned pointer. */
void rag_corpus_free(RagCorpus *corpus);

int rag_corpus_chunk_count(const RagCorpus *corpus);
const RagChunk *rag_corpus_chunk(const RagCorpus *corpus, int index);

/* Files seen by the scan, in load order (also the unreadable ones). */
int rag_corpus_file_count(const RagCorpus *corpus);
const RagFileInfo *rag_corpus_file(const RagCorpus *corpus, int index);

/* Messages about skipped files and tabs, e.g. "x.xlsx: not a ZIP file". */
int rag_corpus_warning_count(const RagCorpus *corpus);
const char *rag_corpus_warning(const RagCorpus *corpus, int index);

/*
 * Scan the inputs again and compare names, sizes and modification times
 * with what was loaded. Returns 1 if anything changed (summary describes
 * it, e.g. "added 09_UG_X.xlsx; changed 04_UG_DS.xlsx"), 0 if not, and -1
 * if an input can no longer be read (summary holds the reason).
 */
int rag_corpus_changed(const RagCorpus *corpus,
                       char *summary, size_t summary_capacity);

/* The inputs and options this corpus was loaded with (for reloading). */
int rag_corpus_input_count(const RagCorpus *corpus);
const char *rag_corpus_input(const RagCorpus *corpus, int index);
const RagLoadOptions *rag_corpus_options(const RagCorpus *corpus);

#endif
