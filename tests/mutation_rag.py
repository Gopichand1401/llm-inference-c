#!/usr/bin/env python3
"""Mutation test for RAG Steps 2-3: plant one deliberate bug at a time in a
copy of src/rag_chunk.c or src/rag_index.c, rebuild rag_inspect, and check
that the comparison with tests/rag_reference.py notices. A check that cannot
fail proves nothing.

Usage: mutation_rag.py <project root> <test workbook folder> [real data folder]
"""
import os
import subprocess
import sys
import tempfile

ROOT = sys.argv[1]
FOLDERS = [sys.argv[2]] + ([sys.argv[3]] if len(sys.argv) > 3 else [])
REF = os.path.join(ROOT, "tests/rag_reference.py")
QUERIES = [q for q in open(os.path.join(ROOT, "tests/rag_check_queries.txt"),
                           encoding="utf-8").read().splitlines() if q]
TERMS = [t for t in open(os.path.join(ROOT, "tests/rag_check_terms.txt"),
                         encoding="utf-8").read().splitlines() if t]
CFLAGS = ["-std=c11", "-O1", "-I" + os.path.join(ROOT, "include")]

MUTATIONS = [
    ("rag_chunk.c", "header needs the full width",
     "2 * f >= widest", "f >= widest"),
    ("rag_chunk.c", "numeric rows accepted as header",
     "5 * count_text(row) >= 3 * f", "5 * count_text(row) >= 0 * f"),
    ("rag_chunk.c", "\"-\" not treated as empty",
     "return s[0] == '\\0' || strcmp(s, \"-\") == 0 ||",
     "return s[0] == '\\0' ||"),
    ("rag_chunk.c", "labels not cut at ','",
     "strcspn(text, \"(?,\")", "strcspn(text, \"(?\")"),
    ("rag_chunk.c", "trailing Yes/No kept",
     "strncmp(text + n - 6, \"Yes/No\", 6) == 0", "0"),
    ("rag_chunk.c", "batch rows never merged",
     "group_key != NULL &&\n            strcmp(key, group_key) == 0",
     "group_key != NULL &&\n            0"),
    ("rag_chunk.c", "differing batch values shown once",
     "same = strcmp(v0, cell_value(s->group[r], col)) == 0;", "same = 1;"),
    ("rag_chunk.c", "notes not de-duplicated",
     "if (*slot != 0) {\n        c->chunks[*slot - 1].note_count++;",
     "if (0) {\n        c->chunks[*slot - 1].note_count++;"),
    ("rag_chunk.c", "hidden tabs read",
     "if (xlsx_sheet_hidden(b, s) && !c->options.include_hidden) {",
     "if (0 && xlsx_sheet_hidden(b, s)) {"),
    ("rag_chunk.c", "title from the first row",
     "for (int i = h - 1; i >= 0; i--) {", "for (int i = 0; i < h; i++) {"),
    ("rag_chunk.c", "only one key column",
     "if (ht[0] && keys < 2) {", "if (ht[0] && keys < 1) {"),
    ("rag_chunk.c", "lock files not ignored",
     "(name[0] == '~' && name[1] == '$') ||", "0 ||"),
    ("rag_index.c", "stop words kept",
     "if (!is_stop_word(piece)) {", "if (1) {"),
    ("rag_index.c", "-ies stemmed to -i",
     "w[len - 3] = 'y';", "w[len - 3] = 'i';"),
    ("rag_index.c", "-us words lose their s",
     "!ends_with(w, len, \"us\") &&", ""),
    ("rag_index.c", "identifiers never joined",
     "join = gap >= 1 && gap <= 2;", "join = 0;"),
    ("rag_index.c", "identifiers joined across 3 characters",
     "join = gap >= 1 && gap <= 2;", "join = gap >= 1 && gap <= 3;"),
    ("rag_index.c", "idf with df + 1",
     "return log(1.0 + (N - df + 0.5) / (df + 0.5));",
     "return log(1.0 + (N - df + 0.5) / (df + 1.0));"),
    ("rag_index.c", "key fields weighted 2",
     "double key_weight = 3.0;", "double key_weight = 2.0;"),
    ("rag_index.c", "label words not down-weighted",
     "ix->label_weight = 0.3;\n    ix->name_bonus", "ix->label_weight = 1.0;\n    ix->name_bonus"),
    ("rag_index.c", "no name bonus",
     "score[d] += ix->name_bonus * sum;", "score[d] += 0.0 * sum;"),
    ("rag_index.c", "b = 0.5",
     "ix->b = 0.75;\n    ix->label_weight", "ix->b = 0.5;\n    ix->label_weight"),
    ("rag_chunk.c", "CSV: doubled quote kept twice",
     "i += 2;          /* \"\" -> \" */", "i += 1;"),
    ("rag_chunk.c", "CSV: semicolon never detected",
     "if (count[1] > count[0] && count[1] >= count[2]) {", "if (0) {"),
    ("rag_chunk.c", "CSV: blank lines not counted",
     "        record++;\n    }\n\n    d->cells",
     "        record += n_cells > 0 && d->text[starts[n_cells - 1]] != 0;\n    }\n\n    d->cells"),
    ("rag_chunk.c", "CSV: white space not collapsed",
     "pending_space = started;", "pending_space = 0;"),
    ("rag_index.c", "typos: 2 edits allowed for short words",
     "const int limit = len >= 8 ? 2 : 1;", "const int limit = 2;"),
    ("rag_index.c", "typos: 3-letter words corrected",
     "} else if (len >= 4) {", "} else if (len >= 3) {"),
    ("rag_index.c", "typos: neighbour swap not one edit",
     "if (i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] &&",
     "if (0 && i > 1 && j > 1 && a[i - 1] == b[j - 2] &&"),
    ("rag_index.c", "typos: last close term wins",
     "if (d < best_d) {\n                    best_d = d;", "if (d <= best_d && d <= limit) {\n                    best_d = d;"),
    ("rag_index.c", "short forms: stop-word run kept",
     "if (pass == 0 && (all_stop[a] || all_stop[a + l - 1])) {", "if (0) {"),
    ("rag_index.c", "short forms: only without stop words",
     "for (int pass = 0; pass < 2; pass++) {\n        const char *initials", "for (int pass = 1; pass < 2; pass++) {\n        const char *initials"),
    ("rag_index.c", "short forms counted in the row length",
     "ac.add.virtual_term = 1;", "ac.add.virtual_term = 0;"),
    ("rag_index.c", "label words not in the vocabulary",
     "int id = ix->failed ? -1 : intern_term(ix, term);\n    if (id >= 0) {\n        ix->terms[id].is_label = 1;",
     "int id = *find_slot(ix, term) - 1;\n    if (id >= 0) {\n        ix->terms[id].is_label = 1;"),
    ("rag_index.c", "ties in reverse order",
     "return (x->chunk > y->chunk) - (x->chunk < y->chunk);",
     "return (x->chunk < y->chunk) - (x->chunk > y->chunk);"),
]


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def commands():
    """Every comparable output: (name, arguments)."""
    for folder in FOLDERS:
        short = os.path.basename(folder)
        yield "files " + short, ["files", folder]
        yield "chunks " + short, ["chunks", folder]
        yield "chunks hidden " + short, ["chunks", folder, "--include-hidden"]
    for t in TERMS:
        yield "terms %r" % t, ["tokens", t]
    for folder in FOLDERS:
        for q in QUERIES:
            yield "search %r" % q, ["search", folder, "-q", q, "-k", "8"]
            yield "resolve %r" % q, ["resolve", folder, q]


def outputs(exe_or_ref):
    base = [exe_or_ref] if exe_or_ref != REF else [sys.executable, REF]
    return [(name, run(base + args)) for name, args in commands()]


def build(sources, tmp):
    paths = []
    for name, text in sources.items():
        p = os.path.join(tmp, name)
        open(p, "w").write(text)
        paths.append(p)
    exe = os.path.join(tmp, "m_inspect")
    r = subprocess.run(["gcc"] + CFLAGS + paths +
                       [os.path.join(ROOT, "src/xlsx_reader.c"),
                        os.path.join(ROOT, "examples/09_rag/rag_inspect.c"),
                        "-lz", "-lm", "-o", exe], capture_output=True, text=True)
    return exe if r.returncode == 0 else None


def first_difference(exe, reference):
    """Name of the first output that differs (stops there), or None."""
    for (name, args), (_, want) in zip(commands(), reference):
        if run([exe] + args) != want:
            return name
    return None


SRC = {n: open(os.path.join(ROOT, "src", n)).read() for n in ("rag_chunk.c", "rag_index.c")}
reference = outputs(REF)
with tempfile.TemporaryDirectory() as tmp:
    exe = build(SRC, tmp)
    assert exe and first_difference(exe, reference) is None, \
        "unmutated build must match the reference"
    missed = 0
    for file, name, old, new in MUTATIONS:
        assert SRC[file].count(old) == 1, "mutation target not found exactly once: " + name
        src = dict(SRC)
        src[file] = SRC[file].replace(old, new)
        exe = build(src, tmp)
        if exe is None:
            print("  [compile error] %s" % name)
            missed += 1
            continue
        hit = first_difference(exe, reference)
        print("  %-42s %s" % (name, "DETECTED by " + hit[:40] if hit else "*** MISSED ***"))
        missed += hit is None
    print("MUTATION: %d of %d planted bugs detected" % (len(MUTATIONS) - missed, len(MUTATIONS)))
    sys.exit(1 if missed else 0)
