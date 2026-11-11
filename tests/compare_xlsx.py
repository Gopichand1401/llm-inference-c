#!/usr/bin/env python3
"""Reference dump of an .xlsx using openpyxl, in the same TSV format as
`xlsx_dump --tsv`:  "#tab <TAB> name <TAB> visible|hidden" per tab, then
tab <TAB> row <TAB> column letters <TAB> text per non-empty cell

The formatting rules are the ones documented in include/xlsx_reader.h, applied
to what openpyxl returns, so any difference points at a bug on one side:
  - text: Unicode whitespace runs -> one space, zero-width chars dropped, trimmed
  - numbers: up to 15 significant digits ("%.15g")
  - booleans: TRUE / FALSE
  - date -> YYYY-MM-DD, datetime -> YYYY-MM-DD HH:MM[:SS], time -> HH:MM[:SS]
  - a vertically merged range repeats its top-left value down its first column,
    into rows that already have some other content
Usage: compare_xlsx.py file.xlsx
"""
import datetime
import re
import sys

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter


def clean(text):
    # Excel escapes control characters as _xHHHH_; openpyxl leaves them as is.
    text = re.sub(r"_x([0-9A-Fa-f]{4})_", lambda m: chr(int(m.group(1), 16)), text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return " ".join(text.split())


def norm(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, int):
        return "%.15g" % v
    if isinstance(v, float):
        return "%.15g" % v
    if isinstance(v, datetime.datetime):
        if v.hour == 0 and v.minute == 0 and v.second == 0:
            return v.strftime("%Y-%m-%d")
        if v.second == 0:
            return v.strftime("%Y-%m-%d %H:%M")
        return v.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(v, datetime.date):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, datetime.time):
        if v.second == 0:
            return v.strftime("%H:%M")
        return v.strftime("%H:%M:%S")
    return clean(str(v))


def main():
    wb = load_workbook(sys.argv[1], data_only=True)
    out = []
    for ws in wb.worksheets:
        grid = {}
        for row in ws.iter_rows():
            for cell in row:
                text = norm(cell.value)
                if text:
                    grid.setdefault(cell.row, {})[cell.column] = text
        for rng in ws.merged_cells.ranges:
            if rng.max_row <= rng.min_row:
                continue
            top = grid.get(rng.min_row, {}).get(rng.min_col, "")
            if not top:
                continue
            for r in range(rng.min_row + 1, rng.max_row + 1):
                if r in grid and not grid[r].get(rng.min_col):
                    grid[r][rng.min_col] = top
        hidden = "visible" if ws.sheet_state == "visible" else "hidden"
        out.append("#tab\t%s\t%s" % (ws.title, hidden))
        for r in sorted(grid):
            for c in sorted(grid[r]):
                out.append("%s\t%d\t%s\t%s" % (ws.title, r, get_column_letter(c), grid[r][c]))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
