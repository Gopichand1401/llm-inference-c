#ifndef XLSX_READER_H
#define XLSX_READER_H

/*
 * xlsx_reader.h - read any .xlsx workbook into plain text cells.
 *
 *   XlsxBook *b = xlsx_open("timetable.xlsx", error, sizeof(error));
 *   for (int s = 0; s < xlsx_sheet_count(b); s++) {
 *       for (int r = 0; r < xlsx_row_count(b, s); r++) {
 *           const XlsxRow *row = xlsx_row(b, s, r);
 *           const char *cell = xlsx_cell(row, 0);     // column A
 *       }
 *   }
 *   xlsx_close(b);
 *
 * What it does (all in this project's own code; only zlib's inflate() is
 * used, for the DEFLATE decompression inside the ZIP container):
 *   - reads the ZIP directory, then workbook.xml, its relationships,
 *     sharedStrings.xml, styles.xml and every worksheet XML
 *   - returns every cell as clean one-line text:
 *       * line breaks, tabs and runs of spaces become one space, ends trimmed
 *       * XML entities (&amp; &#10; ...) and _x000D_-style escapes decoded
 *       * numbers print like Excel's General format (up to 15 digits)
 *       * cells with a date/time number format print as YYYY-MM-DD[ HH:MM]
 *       * booleans print TRUE / FALSE, error cells print as "#N/A" etc.
 *   - cells inside a vertically merged range repeat the top-left value
 *     (horizontal-only merges, usually titles, are left alone)
 *
 * Not supported (refused with a message): encrypted workbooks, ZIP64,
 * old binary .xls files. Formulas are read as their last saved value.
 *
 * Errors: functions that can fail return NULL / non-zero and write a
 * message into the caller's buffer. Nothing is printed by the library.
 * Accessors are safe on bad arguments: a NULL book or an out-of-range
 * sheet / row gives 0, NULL (names, rows) or "" (cells), never a crash.
 */

#include <stddef.h>

typedef struct XlsxBook XlsxBook;

typedef struct {
    int row;             /* Excel row number, 1-based */
    int ncols;           /* entries in cells[]; trailing empty cells dropped */
    const char **cells;  /* never NULL; "" means empty; owned by the book */
} XlsxRow;

/* Open and fully parse a workbook. NULL on failure, with a message. */
XlsxBook *xlsx_open(const char *path, char *error, size_t error_capacity);

/* Release everything. Safe on NULL. Invalidates every pointer returned. */
void xlsx_close(XlsxBook *book);

int xlsx_sheet_count(const XlsxBook *book);

/* Tab name, in workbook order. Valid until xlsx_close(). */
const char *xlsx_sheet_name(const XlsxBook *book, int sheet);

/* 1 if the tab is hidden or "very hidden". */
int xlsx_sheet_hidden(const XlsxBook *book, int sheet);

/*
 * Rows that contain at least one non-empty cell, in ascending Excel row
 * order. Rows are indexed 0 .. xlsx_row_count()-1; XlsxRow.row keeps the
 * original Excel row number.
 */
int xlsx_row_count(const XlsxBook *book, int sheet);
const XlsxRow *xlsx_row(const XlsxBook *book, int sheet, int index);

/* Cell text of a 0-based column; "" when the column is past the row's end. */
const char *xlsx_cell(const XlsxRow *row, int col);

#endif
