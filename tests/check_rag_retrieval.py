#!/usr/bin/env python3
"""Run every question of a .tsv test set through `llm_rag --retrieve-only
--show-rows` (the rows that --rag gives the model) and check that the
expected row is among them.
Usage: check_rag_retrieval.py <llm_rag> <docs folder> <questions.tsv>"""
import re
import subprocess
import sys

exe, docs, path = sys.argv[1:4]
row_re = re.compile(r"^\s+\[\d+\] (\S+), tab \".*\", rows? (\d+)(?:-(\d+))?")
passed = failed = 0
for line in open(path, encoding="utf-8"):
    if not line.strip() or line.startswith("#"):
        continue
    question, prefix, row = line.rstrip("\n").split("\t")
    out = subprocess.run([exe, "--retrieve-only", "--show-rows", "--docs", docs, question],
                         capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout
    got = []
    for l in out.splitlines():
        m = row_re.match(l)
        if m:
            first = int(m.group(2))
            last = int(m.group(3) or first)
            got.append((m.group(1), first, last))
    if prefix == "-":
        ok = not got and "could not find" in out
    else:
        r = int(row)
        ok = any(f.startswith(prefix) and a <= r <= b for f, a, b in got)
    passed += ok
    failed += not ok
    if not ok:
        print("  MISS  %s -> %s" % (question, got or "nothing"))
print("retrieval: %d of %d questions found the right row" % (passed, passed + failed))
sys.exit(1 if failed else 0)
