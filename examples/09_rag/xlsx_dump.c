/*
 * xlsx_dump.c - print every cell of every tab of one or more .xlsx files.
 *
 * Usage:
 *   xlsx_dump <file.xlsx> [more files] [--tsv] [--sheet NAME] [--max-rows N]
 *
 * Default output is for reading:
 *   [tab 1] "Core"  rows 36
 *     row 5: A=Code | B=Course Name | C=Credits | ...
 *
 * --tsv prints "#tab<TAB>name<TAB>visible|hidden" for each tab, then one line
 * per non-empty cell, "tab<TAB>row<TAB>column<TAB>text". tests/compare_xlsx.py
 * prints the same format from Python's openpyxl so the two can be diffed.
 */
#include "xlsx_reader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void usage(const char *program)
{
    fprintf(stderr,
            "Usage: %s <file.xlsx> [more files] [--tsv] [--sheet NAME] "
            "[--max-rows N]\n", program);
}

int main(int argc, char **argv)
{
    const char *files[64];
    int nfiles = 0;
    int tsv = 0;
    int max_rows = 0;
    const char *only_sheet = NULL;
    int status = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--tsv") == 0) {
            tsv = 1;
        } else if (strcmp(argv[i], "--sheet") == 0 && i + 1 < argc) {
            only_sheet = argv[++i];
        } else if (strcmp(argv[i], "--max-rows") == 0 && i + 1 < argc) {
            max_rows = atoi(argv[++i]);
        } else if (argv[i][0] == '-' && argv[i][1] != '\0') {
            usage(argv[0]);
            return 1;
        } else if (nfiles < 64) {
            files[nfiles++] = argv[i];
        }
    }
    if (nfiles == 0) {
        usage(argv[0]);
        return 1;
    }

    for (int f = 0; f < nfiles; f++) {
        char error[256];
        XlsxBook *book = xlsx_open(files[f], error, sizeof(error));

        if (book == NULL) {
            fprintf(stderr, "%s: %s\n", files[f], error);
            status = 1;
            continue;
        }
        if (!tsv) {
            printf("== %s: %d tab(s) ==\n", files[f], xlsx_sheet_count(book));
        }
        for (int s = 0; s < xlsx_sheet_count(book); s++) {
            const char *name = xlsx_sheet_name(book, s);
            int rows = xlsx_row_count(book, s);

            if (only_sheet != NULL && strcmp(only_sheet, name) != 0) {
                continue;
            }
            if (tsv) {
                printf("#tab\t%s\t%s\n", name,
                       xlsx_sheet_hidden(book, s) ? "hidden" : "visible");
            } else {
                printf("[tab %d] \"%s\"  rows %d%s\n", s + 1, name, rows,
                       xlsx_sheet_hidden(book, s) ? "  (hidden)" : "");
            }
            for (int r = 0; r < rows; r++) {
                const XlsxRow *row = xlsx_row(book, s, r);
                char letters[8];

                if (max_rows > 0 && r >= max_rows) {
                    break;
                }
                if (!tsv) {
                    printf("  row %d:", row->row);
                }
                int first = 1;
                for (int c = 0; c < row->ncols; c++) {
                    const char *text = xlsx_cell(row, c);
                    if (text[0] == '\0') {
                        continue;
                    }
                    column_letters(c, letters);
                    if (tsv) {
                        printf("%s\t%d\t%s\t%s\n", name, row->row, letters, text);
                    } else {
                        printf("%s%s=%s", first ? " " : " | ", letters, text);
                        first = 0;
                    }
                }
                if (!tsv) {
                    printf("\n");
                }
            }
        }
        xlsx_close(book);
    }
    return status;
}
