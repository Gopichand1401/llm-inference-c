/*
 * rag_inspect.c - look inside the RAG pipeline without a model.
 *
 * Usage:
 *   rag_inspect chunks <folder|file.xlsx>... [--header-row N] [--include-hidden]
 *       every chunk, one tab-separated record per line:
 *         chunk <i> ROW|NOTE <file> <tab> <first row> <last row> <rows> <notes>
 *         title <text>
 *         field <key 0/1> <variant> <label> <value>      (ROW)
 *         note <text>                                    (NOTE)
 *   rag_inspect files <folder|file.xlsx>... [--header-row N] [--include-hidden]
 *       every file the scan found: file <name> <readable 0/1> <chunks>
 *   rag_inspect tokens "some text"
 *       the search terms of the text, one per line
 *   rag_inspect resolve <folder|file.xlsx>... "text"
 *       how each term of the text is resolved against the index:
 *       <typed> <term used> exact|short|corrected|unknown
 *   rag_inspect search <folder|file.xlsx>... -q "question" [-k N]
 *       <rank> <chunk> <BM25 score>, best first
 *
 * tests/rag_reference.py prints the same formats from Python (openpyxl),
 * so the outputs can be compared with cmp / diff.
 */
#include "rag_chunk.h"
#include "rag_index.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_INPUTS 64

static void usage(const char *p)
{
    fprintf(stderr,
            "Usage: %s chunks <folder|file>... [--header-row N] [--include-hidden]\n"
            "       %s files <folder|file>... [--header-row N] [--include-hidden]\n"
            "       %s tokens \"text\"\n"
            "       %s resolve <folder|file>... \"text\"\n"
            "       %s search <folder|file>... -q \"question\" [-k N]\n",
            p, p, p, p, p);
}

static void print_term(const char *term, void *user)
{
    (void)user;
    printf("%s\n", term);
}

int main(int argc, char **argv)
{
    const char *inputs[MAX_INPUTS];
    int n_inputs = 0, k = 10;
    const char *query = NULL;
    RagLoadOptions opt;
    char error[512];

    memset(&opt, 0, sizeof(opt));
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }
    const char *cmd = argv[1];

    if (strcmp(cmd, "tokens") == 0) {
        rag_tokenize(argv[2], print_term, NULL);
        return 0;
    }
    const int resolve = strcmp(cmd, "resolve") == 0;
    if (strcmp(cmd, "chunks") != 0 && strcmp(cmd, "search") != 0 &&
        strcmp(cmd, "files") != 0 && !resolve) {
        usage(argv[0]);
        return 1;
    }

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--header-row") == 0 && i + 1 < argc) {
            opt.header_row = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--include-hidden") == 0) {
            opt.include_hidden = 1;
        } else if (strcmp(argv[i], "-q") == 0 && i + 1 < argc) {
            query = argv[++i];
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            k = atoi(argv[++i]);
        } else if (argv[i][0] == '-' && argv[i][1] != '\0') {
            usage(argv[0]);
            return 1;
        } else if (n_inputs < MAX_INPUTS) {
            inputs[n_inputs++] = argv[i];
        }
    }
    if (resolve && n_inputs >= 2) {
        query = inputs[--n_inputs];       /* last argument is the text */
    }
    if (n_inputs == 0 || (resolve && query == NULL) ||
        (strcmp(cmd, "search") == 0 && (query == NULL || k <= 0))) {
        usage(argv[0]);
        return 1;
    }

    RagCorpus *c = rag_corpus_load(inputs, n_inputs, &opt, error, sizeof(error));
    if (c == NULL) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    for (int i = 0; i < rag_corpus_warning_count(c); i++) {
        fprintf(stderr, "warning: %s\n", rag_corpus_warning(c, i));
    }

    if (resolve) {
        static const char *const kinds[] = {"exact", "short", "corrected", "unknown"};
        RagQueryTerm terms[RAG_QUERY_MAX];
        RagIndex *ix = rag_index_build(c, NULL, error, sizeof(error));
        if (ix == NULL) {
            fprintf(stderr, "%s\n", error);
            rag_corpus_free(c);
            return 1;
        }
        int n = rag_index_resolve(ix, query, terms, RAG_QUERY_MAX);
        for (int i = 0; i < n; i++) {
            printf("%s\t%s\t%s\n", terms[i].typed, terms[i].term,
                   kinds[terms[i].kind]);
        }
        rag_index_free(ix);
    } else if (strcmp(cmd, "files") == 0) {
        for (int i = 0; i < rag_corpus_file_count(c); i++) {
            const RagFileInfo *f = rag_corpus_file(c, i);
            printf("file\t%s\t%d\t%d\n", f->name, f->ok, f->chunks);
        }
    } else if (strcmp(cmd, "chunks") == 0) {
        for (int i = 0; i < rag_corpus_chunk_count(c); i++) {
            const RagChunk *ch = rag_corpus_chunk(c, i);
            printf("chunk\t%d\t%s\t%s\t%s\t%d\t%d\t", i,
                   ch->kind == RAG_CHUNK_ROW ? "ROW" : "NOTE",
                   ch->file, ch->sheet, ch->row_first, ch->row_last);
            for (int v = 0; v < ch->n_variants; v++) {
                printf("%s%d", v ? "," : "", ch->variant_rows[v]);
            }
            printf("\t%d\n", ch->note_count);
            printf("title\t%s\n", ch->title);
            if (ch->kind == RAG_CHUNK_NOTE) {
                printf("note\t%s\n", ch->note);
            }
            for (int f = 0; f < ch->n_fields; f++) {
                printf("field\t%d\t%d\t%s\t%s\n", ch->fields[f].key,
                       ch->fields[f].variant, ch->fields[f].label,
                       ch->fields[f].value);
            }
        }
    } else {
        RagIndex *ix = rag_index_build(c, NULL, error, sizeof(error));
        if (ix == NULL) {
            fprintf(stderr, "%s\n", error);
            rag_corpus_free(c);
            return 1;
        }
        RagHit *hits = malloc((size_t)k * sizeof(*hits));
        int n = hits ? rag_index_search(ix, query, hits, k) : 0;
        for (int i = 0; i < n; i++) {
            printf("%d\t%d\t%.9f\n", i + 1, hits[i].chunk, hits[i].score);
        }
        free(hits);
        rag_index_free(ix);
    }
    rag_corpus_free(c);
    return 0;
}
