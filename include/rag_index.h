#ifndef RAG_INDEX_H
#define RAG_INDEX_H

/*
 * rag_index.h - BM25 keyword search over the chunks of a RagCorpus
 *               (RAG Step 3: indexing + retrieval).
 *
 *   RagIndex *ix = rag_index_build(corpus, NULL, error, sizeof(error));
 *   RagHit hits[4];
 *   int n = rag_index_search(ix, "Who teaches DS3010?", hits, 4);
 *   // hits[0].chunk is an index into the corpus, hits[0].score its BM25
 *   rag_index_free(ix);
 *
 * Text -> terms (the same for chunks and questions):
 *   1. ASCII letters are lower-cased. A "piece" is a maximal run of
 *      [a-z0-9] and bytes >= 0x80 (so UTF-8 words stay whole).
 *   2. Alphabetic pieces that are stop words ("the", "who", "is", "teach",
 *      "take" ...; the list is in rag_index.c) are
 *      dropped; other alphabetic pieces get a light stem:
 *        -ies -> -y; -sses/-xes/-ches/-shes lose "es"; a final -s is
 *        dropped unless the word ends in -ss, -us or -is.
 *      (length limits: -ies and -es rules need > 4 letters, -s needs > 3)
 *      A word whose stem is a stop word ("names", "gives") is dropped too.
 *   3. Identifiers split by a space, '-' or '_' (1 or 2 characters) are
 *      also joined: when a piece containing a letter is followed by an
 *      all-digit piece, the two together are one extra term. So "DS3010",
 *      "DS-3010" and "ds 3010" all contain the term "ds3010", and
 *      "A01-212" contains "a01212".
 *
 * What is indexed for a chunk, with a weight per occurrence:
 *   values of the first two header columns (the row's identifier and
 *   name, e.g. code + course name): key_weight (default 3);
 *   all other values, the table title, the tab name and the file name: 1.
 *   Column labels are not indexed (they are the same in every row).
 *
 *   Notes: their text only.
 *   Short forms: for every key field, the initials of each run of 2 to 6
 *   consecutive alphabetic words are indexed as "~" terms, taken over all
 *   words (a run may not start or end on a stop word) and again without
 *   the stop words ("Theory of Computation" -> ~toc ~tc), once per field,
 *   weight 1, not counted in the chunk length.
 *   Words of column labels belong to the vocabulary even if no cell
 *   contains them (they then match nothing, but are not "unknown").
 *
 * A question term is resolved in this order (rag_index_resolve):
 *   exact       it is a term of the index;
 *   short form  alphabetic, 2 to 6 letters, and "~term" is in the index;
 *   corrected   alphabetic, 4+ letters: the closest alphabetic index term
 *               within 1 edit (2 edits from 8 letters; an edit is an
 *               insert, delete, replace or swap of neighbours); the
 *               smallest distance wins, then the term indexed first;
 *   unknown     otherwise (ignored by the search).
 *   Terms with a digit are never corrected, so DS3010 cannot become DS3030.
 *
 * Score (Okapi BM25 with the always-positive Lucene idf, plus two re-ranking
 * rules):
 *   idf(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))
 *   bm25(q, d)  = sum over distinct query terms t (in query order) of
 *                 w(t) * idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg))
 *   tf = weighted count of t in d, len = weighted term count of d.
 *   w(t) = label_weight (default 0.3) when t also occurs in a column label
 *          anywhere in the corpus ("instructor", "slot", "room" ...): such
 *          words say which column is wanted, not which row; otherwise 1.
 *   Name bonus: for every key field of d whose terms are all in the query
 *   (the whole course code or the whole course name was asked for), add
 *   name_bonus (default 1) * the sum of idf over that field's distinct terms.
 *   Defaults k1 = 1.2, b = 0.75. Ties keep corpus order.
 *
 * tests/rag_reference.py implements the same rules in Python; the scores of
 * a set of questions are compared with it in tests/run_rag_checks.sh.
 */

#include "rag_chunk.h"

#include <stddef.h>

typedef struct {
    double k1;            /* 0 = default 1.2 */
    double b;             /* < 0 = default 0.75 (0 is a valid value) */
    double key_weight;    /* 0 = default 3 */
    double label_weight;  /* < 0 = default 0.3; 1 turns the rule off */
    double name_bonus;    /* < 0 = default 1; 0 turns the rule off */
} RagIndexOptions;

/* Fill o with the defaults (k1 1.2, b 0.75, key 3, label 0.3, bonus 1). */
void rag_index_default_options(RagIndexOptions *o);

typedef struct {
    int chunk;            /* index into the corpus */
    double score;
} RagHit;

typedef struct RagIndex RagIndex;

/* Build the index (the corpus must outlive it). options may be NULL. */
RagIndex *rag_index_build(const RagCorpus *corpus,
                          const RagIndexOptions *options,
                          char *error, size_t error_capacity);

void rag_index_free(RagIndex *index);

/*
 * Best max_hits chunks for the query, best first. Returns how many were
 * written (only chunks with a score > 0, i.e. at least one shared term).
 */
int rag_index_search(const RagIndex *index, const char *query,
                     RagHit *hits, int max_hits);

#define RAG_TERM_MAX 64      /* bytes of a term, with the NUL */
#define RAG_QUERY_MAX 64     /* distinct terms of one question */

typedef enum {
    RAG_TERM_EXACT = 0,
    RAG_TERM_SHORT_FORM = 1,
    RAG_TERM_CORRECTED = 2,
    RAG_TERM_UNKNOWN = 3
} RagTermKind;

typedef struct {
    char term[RAG_TERM_MAX];   /* term used for the search ("~ml" for a short form) */
    char typed[RAG_TERM_MAX];  /* the term as it came out of rag_tokenize */
    RagTermKind kind;
    int is_label;              /* also a word of some column label */
    int id;                    /* internal; -1 when unknown */
} RagQueryTerm;

/*
 * The distinct search terms of a question, in order, resolved as described
 * above. Returns how many were written (at most max).
 */
int rag_index_resolve(const RagIndex *index, const char *text,
                      RagQueryTerm *out, int max);

/*
 * 1 if text contains the term (as produced by rag_tokenize). A "~xy" term
 * matches when xy are the initials of consecutive words of text.
 */
int rag_text_has_term(const char *text, const char *term);

/* Number of distinct terms in the index. */
int rag_index_term_count(const RagIndex *index);

/*
 * Split text into terms with the rules above. Calls emit(term, user) for
 * every term in order (a term may repeat). Exposed for tests and for
 * matching question words against column labels.
 */
void rag_tokenize(const char *text,
                  void (*emit)(const char *term, void *user), void *user);

#endif
