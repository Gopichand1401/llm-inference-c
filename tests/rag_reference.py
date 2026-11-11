#!/usr/bin/env python3
"""Independent Python reference for RAG Steps 2 and 3.

Implements the rules documented in include/rag_chunk.h and
include/rag_index.h on top of openpyxl (cells read with the same rules as
tests/compare_xlsx.py), and prints the same formats as build/rag_inspect:

  rag_reference.py chunks <folder|file>... [--header-row N] [--include-hidden]
  rag_reference.py files <folder|file>... [--header-row N] [--include-hidden]
  rag_reference.py tokens "text"
  rag_reference.py resolve <folder|file>... "text"
  rag_reference.py search <folder|file>... -q "question" [-k N]

tests/run_rag_checks.sh compares the two outputs byte for byte.
"""
import csv
import io
import math
import os
import re
import sys

from openpyxl import load_workbook

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_xlsx import norm  # noqa: E402  (same cell text rules)

HEADER_SEARCH_ROWS = 30
BLANKS = {"", "-", "–", "—"}


# ---------------------------------------------------------------- reading

def read_sheets(path):
    """[(name, hidden, [(excel_row, {col0: text})])] like xlsx_reader."""
    wb = load_workbook(path, data_only=True)
    out = []
    for ws in wb.worksheets:
        grid = {}
        for row in ws.iter_rows():
            for cell in row:
                text = norm(cell.value)
                if text:
                    grid.setdefault(cell.row, {})[cell.column - 1] = text
        for rng in ws.merged_cells.ranges:
            if rng.max_row <= rng.min_row:
                continue
            top = grid.get(rng.min_row, {}).get(rng.min_col - 1, "")
            if not top:
                continue
            for r in range(rng.min_row + 1, rng.max_row + 1):
                if r in grid and not grid[r].get(rng.min_col - 1):
                    grid[r][rng.min_col - 1] = top
        rows = [(r, grid[r]) for r in sorted(grid)]
        out.append((ws.title, ws.sheet_state != "visible", rows))
    return out


def clean_csv_cell(text):
    return " ".join(p for p in re.split(r"[ \t\n\r\f\v]+", text) if p)


def read_csv(path):
    """One table per CSV file, tab name "", with Python's csv module."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if b"\0" in raw:
        raise ValueError("not a text file")
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    text = raw.decode("utf-8", "surrogateescape")
    count = {",": 0, ";": 0, "\t": 0}
    quoted = False
    for ch in text:
        if ch == '"':
            quoted = not quoted
        elif not quoted and ch in "\r\n":
            break
        elif not quoted and ch in count:
            count[ch] += 1
    delim = ","
    if count[";"] > count[","] and count[";"] >= count["\t"]:
        delim = ";"
    elif count["\t"] > count[","] and count["\t"] > count[";"]:
        delim = "\t"
    rows = []
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delim)
    for number, record in enumerate(reader, 1):
        cells = {}
        for col, cell in enumerate(record):
            cell = clean_csv_cell(cell)
            if cell:
                cells[col] = cell
        if cells:
            rows.append((number, cells))
    return [("", False, rows)]


def scan(inputs):
    files = []
    for p in inputs:
        if os.path.isfile(p):
            files.append(p)
            continue
        names = []
        for name in os.listdir(p):
            if name.startswith(".") or name.startswith("~$"):
                continue
            if not name.lower().endswith((".xlsx", ".csv")):
                continue
            full = p + name if p.endswith("/") else p + "/" + name
            if os.path.isfile(full):
                names.append(full)
        files.extend(sorted(names, key=lambda s: s.encode()))
    return files


# ---------------------------------------------------------------- chunking

def val(cells, col):
    v = cells.get(col, "")
    return "" if v in BLANKS else v


def filled(cells):
    return [c for c in sorted(cells) if val(cells, c)]


def is_number(s):
    try:
        float(s)
    except ValueError:
        return False
    # C strtod accepts these too; Python float() also accepts "nan", "inf",
    # "infinity" and underscores, which strtod handles the same way except
    # for underscores.
    return "_" not in s


def short_label(text):
    cut = len(text)
    for ch in "(?,":
        i = text.find(ch)
        if i >= 0:
            cut = min(cut, i)
    s = text[:cut].rstrip(" ")
    if s.endswith("Yes/No"):
        s = s[:-6]
    s = s.rstrip(" :-.")
    if len(s.encode()) < 2:
        return text
    return s


def letters(col):
    out = ""
    c = col + 1
    while c > 0:
        out = chr(ord("A") + (c - 1) % 26) + out
        c = (c - 1) // 26
    return out


def find_header(rows, forced):
    if forced > 0:
        for i, (r, _) in enumerate(rows):
            if r == forced:
                return i
    widest = max((len(filled(c)) for _, c in rows), default=0)
    for i, (_, cells) in enumerate(rows[:HEADER_SEARCH_ROWS]):
        f = filled(cells)
        texts = [c for c in f if not is_number(val(cells, c))]
        if len(f) >= 2 and 2 * len(f) >= widest and 5 * len(texts) >= 3 * len(f):
            return i
    return -1


def join_row(cells):
    return " | ".join(val(cells, c) for c in filled(cells))


class Corpus:
    def __init__(self):
        self.chunks = []
        self.notes = {}

    def note(self, file, sheet, title, row, text):
        if text in self.notes:
            self.notes[text]["note_count"] += 1
            return
        ch = dict(kind="NOTE", file=file, sheet=sheet, title=title,
                  first=row, last=row, variants=[], note=text, note_count=1,
                  fields=[])
        self.notes[text] = ch
        self.chunks.append(ch)


def load_sheet(corpus, file, sheet, rows, forced):
    h = find_header(rows, forced)
    if h < 0:
        for r, cells in rows:
            if filled(cells):
                corpus.note(file, sheet, "", r, join_row(cells))
        return
    title = ""
    for r, cells in reversed(rows[:h]):
        if filled(cells):
            title = join_row(cells)
            break
    header = rows[h][1]
    labels = {c: short_label(val(header, c)) for c in filled(header)}
    keys = filled(header)[:2]

    def label(col):
        return labels.get(col) or "Column " + letters(col)

    group = []

    def flush():
        if not group:
            return
        cols = sorted(set().union(*[set(filled(c)) for _, c in group]))
        fields = []
        common = set()
        for col in cols:
            vals = [val(c, col) for _, c in group]
            if all(v == vals[0] for v in vals):
                common.add(col)
                fields.append((int(col in keys), 0, label(col), vals[0]))
        if len(group) > 1:
            for k, (_, cells) in enumerate(group, 1):
                for col in cols:
                    if col not in common and val(cells, col):
                        fields.append((int(col in keys), k, label(col), val(cells, col)))
        corpus.chunks.append(dict(kind="ROW", file=file, sheet=sheet, title=title,
                                  first=group[0][0], last=group[-1][0],
                                  variants=[r for r, _ in group], note=None,
                                  note_count=0, fields=fields))
        group.clear()

    group_key = None
    for r, cells in rows[h + 1:]:
        f = filled(cells)
        if not f:
            continue
        if len(f) == 1:
            flush()
            group_key = None
            corpus.note(file, sheet, title, r, val(cells, f[0]))
            continue
        key = val(cells, keys[0])
        if group and key and group_key is not None and key == group_key:
            group.append((r, cells))
            continue
        flush()
        group.append((r, cells))
        group_key = key or None
    flush()


def load(inputs, forced=0, include_hidden=False, files=None):
    corpus = Corpus()
    for path in scan(inputs):
        name = os.path.basename(path)
        try:
            if name.lower().endswith(".csv"):
                sheets = read_csv(path)
            else:
                sheets = read_sheets(path)
        except Exception:  # unreadable files are skipped, as in C
            if files is not None:
                files.append((name, 0))
            continue
        if files is not None:
            files.append((name, 1))
        for sheet, hidden, rows in sheets:
            if hidden and not include_hidden:
                continue
            load_sheet(corpus, name, sheet, rows, forced)
    return corpus.chunks


# ---------------------------------------------------------------- terms

STOP = set("""a about an and any are as at be by can could do does find for from get give
handle handles handling has have he her his how i in is it its know list me my
name need of on or please she show take taken takes taking taught teach
teaches teaching tell that the their them there these they this those to want
was were what when where which who whom whose why will with would you your""".split())


def stem(w):
    n = len(w)
    if n > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if n > 4 and (w.endswith("sses") or w.endswith("xes") or w.endswith("ches")
                  or w.endswith("shes")):
        return w[:-2]
    if n > 3 and w.endswith("s") and not (w.endswith("ss") or w.endswith("us")
                                          or w.endswith("is")):
        return w[:-1]
    return w


def piece_byte(b):
    return (97 <= b <= 122) or (65 <= b <= 90) or (48 <= b <= 57) or b >= 0x80


def tokenize(text):
    data = text.encode()
    terms = []
    i, n = 0, len(data)
    prev, prev_end = None, 0
    while i < n:
        if not piece_byte(data[i]):
            i += 1
            continue
        start = i
        while i < n and piece_byte(data[i]):
            i += 1
        raw = data[start:i]
        piece = bytes(b + 32 if 65 <= b <= 90 else b for b in raw)
        alpha = all(97 <= b <= 122 for b in piece)
        digits = all(48 <= b <= 57 for b in piece)
        has_letter = any(97 <= b <= 122 for b in piece)
        joined = None
        if prev is not None and prev[1] and digits:
            gap = data[prev_end:start]
            if 1 <= len(gap) <= 2 and all(g in b" -_" for g in gap):
                joined = prev[0] + piece
        prev, prev_end = (piece, has_letter), i
        word = piece.decode("utf-8", "surrogateescape")
        if alpha:
            if word not in STOP and stem(word) not in STOP:
                terms.append(stem(word))
        else:
            terms.append(word)
        if joined is not None:
            terms.append(joined.decode("utf-8", "surrogateescape"))
    return terms


# ---------------------------------------------------------------- BM25

def is_alpha(t):
    return bool(t) and all("a" <= c <= "z" for c in t)


def acronyms(text):
    """~initials of every run of 2..6 consecutive alphabetic words, taken
    over all words (not starting or ending on a stop word) and again with
    the stop words left out."""
    words = []
    data = text.encode()
    i, n = 0, len(data)
    while i < n:
        if not piece_byte(data[i]):
            i += 1
            continue
        start = i
        while i < n and piece_byte(data[i]):
            i += 1
        piece = bytes(b + 32 if 65 <= b <= 90 else b for b in data[start:i])
        if all(97 <= b <= 122 for b in piece):
            words.append(piece.decode())
    out = []
    for with_stops, seq in ((True, words[:16]),
                            (False, [w for w in words if w not in STOP][:16])):
        for a in range(len(seq)):
            for size in range(2, 7):
                if a + size > len(seq):
                    continue
                if with_stops and (seq[a] in STOP or seq[a + size - 1] in STOP):
                    continue    # a run never starts or ends on a stop word
                out.append("~" + "".join(w[0] for w in seq[a:a + size]))
    return out


def osa(a, b, limit):
    """Optimal string alignment distance, limit + 1 when clearly larger."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            v = min(prev[j - 1] + (a[i - 1] != b[j - 1]), prev[j] + 1, cur[j - 1] + 1)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                v = min(v, prev2[j - 2] + 1)
            cur[j] = v
        prev2, prev = prev, cur
    return prev[len(b)]


class Index:
    def __init__(self, chunks, key_weight=3.0):
        self.chunks = chunks
        self.postings = {}     # term -> {doc: tf}; dict order = index order
        self.length = [0.0] * len(chunks)
        self.labels = set()
        self.key_fields = [[] for _ in chunks]

        def add(doc, term, w, counted=True):
            p = self.postings.setdefault(term, {})
            p[doc] = p.get(doc, 0.0) + w
            if counted:
                self.length[doc] += w

        for d, ch in enumerate(chunks):
            if ch["kind"] == "NOTE":
                for t in tokenize(ch["note"]):
                    add(d, t, 1.0)
                continue
            for key, _, _, value in ch["fields"]:
                for t in tokenize(value):
                    add(d, t, key_weight if key else 1.0)
            for text in (ch["title"], ch["sheet"], ch["file"]):
                for t in tokenize(text):
                    add(d, t, 1.0)
        # label words join the vocabulary; key fields are recorded
        for d, ch in enumerate(chunks):
            for key, _, label, value in ch["fields"]:
                for t in tokenize(label):
                    self.postings.setdefault(t, {})
                    self.labels.add(t)
                if key:
                    terms_f = []
                    for t in tokenize(value):
                        if t not in terms_f:
                            terms_f.append(t)
                    self.key_fields[d].append(terms_f)
        # short forms of the key fields
        for d, ch in enumerate(chunks):
            for key, _, _, value in ch["fields"]:
                if key:
                    seen = []
                    for t in acronyms(value):
                        if t not in seen:
                            seen.append(t)
                            add(d, t, 1.0, counted=False)

    def resolve(self, text):
        """[(term used, typed, kind)], distinct by the term used."""
        out = []
        for typed in tokenize(text):
            if len(typed.encode()) >= 63 or typed.startswith("~"):
                continue
            term, kind = typed, "unknown"
            if typed in self.postings:
                kind = "exact"
            elif is_alpha(typed):
                if 2 <= len(typed) <= 6 and "~" + typed in self.postings:
                    term, kind = "~" + typed, "short"
                elif len(typed) >= 4:
                    limit = 2 if len(typed) >= 8 else 1
                    best, best_d = None, limit + 1
                    for cand in self.postings:
                        if best_d <= 1:
                            break
                        if cand.startswith("~") or not is_alpha(cand):
                            continue
                        d = osa(typed, cand, limit)
                        if d < best_d:
                            best, best_d = cand, d
                    if best is not None:
                        term, kind = best, "corrected"
            if all(term != o[0] for o in out) and len(out) < 64:
                out.append((term, typed, kind))
        return out


def search(chunks, query, k, k1=1.2, b=0.75, key_weight=3.0,
           label_weight=0.3, name_bonus=1.0):
    N = len(chunks)
    if N == 0:
        return []
    ix = Index(chunks, key_weight)
    postings, length = ix.postings, ix.length
    avg = sum(length) / N

    def idf(t):
        df = len(postings[t])
        return math.log(1.0 + (N - df + 0.5) / (df + 0.5))

    terms = [t for t, _, kind in ix.resolve(query)
             if kind != "unknown" and postings[t]]
    score = [0.0] * N
    for t in terms:
        widf = (label_weight if t in ix.labels else 1.0) * idf(t)
        for d in sorted(postings[t]):
            tf = postings[t][d]
            normf = k1 * (1.0 - b + b * length[d] / avg)
            score[d] += widf * tf * (k1 + 1.0) / (tf + normf)
    if name_bonus > 0:
        for d in range(N):
            if score[d] <= 0:
                continue
            for terms_f in ix.key_fields[d]:
                if terms_f and all(t in terms for t in terms_f):
                    total = 0.0
                    for t in terms_f:
                        total += idf(t)
                    score[d] += name_bonus * total
    hits = sorted((d for d in range(N) if score[d] > 0), key=lambda d: (-score[d], d))
    return [(d, score[d]) for d in hits[:k]]


# ---------------------------------------------------------------- main

def main(argv):
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 1
    cmd = argv[1]
    if cmd == "tokens":
        for t in tokenize(argv[2]):
            print(t)
        return 0
    if cmd == "resolve":
        ix = Index(load(argv[2:-1]))
        for term, typed, kind in ix.resolve(argv[-1]):
            print("%s\t%s\t%s" % (typed, term, kind))
        return 0
    inputs, forced, hidden, query, k = [], 0, False, None, 10
    i = 2
    while i < len(argv):
        a = argv[i]
        if a == "--header-row":
            forced = int(argv[i + 1]); i += 1
        elif a == "--include-hidden":
            hidden = True
        elif a == "-q":
            query = argv[i + 1]; i += 1
        elif a == "-k":
            k = int(argv[i + 1]); i += 1
        else:
            inputs.append(a)
        i += 1
    files = []
    chunks = load(inputs, forced, hidden, files)
    out = sys.stdout
    if cmd == "files":
        for name, ok in files:
            n = sum(1 for ch in chunks if ch["file"] == name) if ok else 0
            out.write("file\t%s\t%d\t%d\n" % (name, ok, n))
    elif cmd == "chunks":
        for i, ch in enumerate(chunks):
            out.write("chunk\t%d\t%s\t%s\t%s\t%d\t%d\t%s\t%d\n" % (
                i, ch["kind"], ch["file"], ch["sheet"], ch["first"], ch["last"],
                ",".join(str(r) for r in ch["variants"]), ch["note_count"]))
            out.write("title\t%s\n" % ch["title"])
            if ch["kind"] == "NOTE":
                out.write("note\t%s\n" % ch["note"])
            for key, variant, label, value in ch["fields"]:
                out.write("field\t%d\t%d\t%s\t%s\n" % (key, variant, label, value))
    else:
        for rank, (d, s) in enumerate(search(chunks, query, k), 1):
            out.write("%d\t%d\t%.9f\n" % (rank, d, s))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
