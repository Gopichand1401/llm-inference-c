#!/usr/bin/env python3
"""Mutation test: plant one deliberate bug at a time in a copy of
src/xlsx_reader.c, rebuild, and check that the openpyxl comparison notices.
A check that cannot fail proves nothing.

Usage: mutation_xlsx.py <project root> <folder with the test workbooks>
"""
import os
import subprocess
import sys
import tempfile

ROOT = sys.argv[1]
DATA = sys.argv[2]
SRC = open(os.path.join(ROOT, "src/xlsx_reader.c")).read()
FILES = ["ds_like", "cs_like", "types", "raw_features"]
CFLAGS = ["-std=c11", "-O1", "-I" + os.path.join(ROOT, "include")]

MUTATIONS = [
    ("shared-string index off by one",
     "val = b->shared[idx];", "val = b->shared[idx > 0 ? idx - 1 : idx];"),
    ("leading spaces not trimmed",
     "if (pending && started) {\n                *w++ = ' ';",
     "if (pending) {\n                *w++ = ' ';"),
    ("date epoch off by one day (1900 system)",
     "days - (date1904 ? 24107L : 25569L)", "days - (date1904 ? 24107L : 25568L)"),
    ("date epoch off by one day (1904 system)",
     "days - (date1904 ? 24107L : 25569L)", "days - (date1904 ? 24106L : 25569L)"),
    ("merged cells not filled",
     "if (apply_merges(sh, merges, nmerge) != 0) {", "if (0) {"),
    ("phonetic text not skipped",
     "} else if (in_si && !in_rph && name_is(&t, \"t\")) {",
     "} else if (in_si && name_is(&t, \"t\")) {"),
    ("column letters base 27",
     "c = c * 26 + ((*p & 0x1F));", "c = c * 27 + ((*p & 0x1F));"),
    ("numbers keep only 6 digits",
     'snprintf(out, cap, "%.15g", d);', 'snprintf(out, cap, "%.6g", d);'),
    ("booleans inverted",
     "(vbuf.len > 0 && vbuf.data[0] == '1') ? \"TRUE\" : \"FALSE\"",
     "(vbuf.len > 0 && vbuf.data[0] == '1') ? \"FALSE\" : \"TRUE\""),
    ("_xHHHH_ escapes not decoded",
     "} else if (s[i] == '_' && i + 6 < n &&", "} else if (0 && s[i] == '_' && i + 6 < n &&"),
    ("&amp; not decoded",
     "out[w++] = '&';\n                } else if (len == 2 && memcmp(e, \"lt\", 2) == 0) {",
     "out[w++] = 'X';\n                } else if (len == 2 && memcmp(e, \"lt\", 2) == 0) {"),
    ("hidden flag ignored",
     "r->hidden = xml_attr(&t, \"state\", attr, sizeof(attr)) &&",
     "r->hidden = 0 && xml_attr(&t, \"state\", attr, sizeof(attr)) &&"),
    ("rows without r= not numbered",
     "atoi(attr)\n                                                                : cur_row + 1;",
     "atoi(attr)\n                                                                : cur_row;"),
    ("date detection ignores custom formats",
     "int is_date = builtin_is_date(id) || custom_is_date(fmts, nfmt, id);",
     "int is_date = builtin_is_date(id);"),
]

def build(src_text, tmp):
    c = os.path.join(tmp, "m.c")
    open(c, "w").write(src_text)
    exe = os.path.join(tmp, "m_dump")
    r = subprocess.run(["gcc"] + CFLAGS + [c, os.path.join(ROOT, "examples/09_rag/xlsx_dump.c"),
                        "-lz", "-o", exe], capture_output=True, text=True)
    return exe if r.returncode == 0 else None

def differs(exe):
    for f in FILES:
        path = os.path.join(DATA, f + ".xlsx")
        got = subprocess.run([exe, "--tsv", path], capture_output=True, text=True).stdout
        ref = subprocess.run([sys.executable, os.path.join(ROOT, "tests/compare_xlsx.py"), path],
                             capture_output=True, text=True).stdout
        if got != ref:
            return f
    return None

with tempfile.TemporaryDirectory() as tmp:
    exe = build(SRC, tmp)
    assert exe and differs(exe) is None, "unmutated build must match the reference"
    missed = 0
    for name, old, new in MUTATIONS:
        assert SRC.count(old) == 1, "mutation target not found exactly once: " + name
        exe = build(SRC.replace(old, new), tmp)
        if exe is None:
            print("  [compile error] %s" % name)
            missed += 1
            continue
        hit = differs(exe)
        print("  %-42s %s" % (name, "DETECTED by " + hit if hit else "*** MISSED ***"))
        missed += hit is None
    print("MUTATION: %d of %d planted bugs detected" % (len(MUTATIONS) - missed, len(MUTATIONS)))
    sys.exit(1 if missed else 0)
