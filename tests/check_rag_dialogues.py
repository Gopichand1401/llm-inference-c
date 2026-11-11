#!/usr/bin/env python3
"""Play the conversations of tests/rag_dialogues.txt against
`llm_rag --retrieve-only` and check every reply.
Usage: check_rag_dialogues.py <llm_rag> <dialogues.txt> [--skip-data]
(--skip-data leaves out the sessions that need the real files in data/)"""
import subprocess
import sys

exe, path = sys.argv[1:3]
skip_data = "--skip-data" in sys.argv

sessions, cur = [], None
for raw in open(path, encoding="utf-8"):
    line = raw.rstrip("\n")
    if line.startswith("#"):
        continue
    if not line.strip():
        cur = None
        continue
    if cur is None:
        cur = {"docs": "data", "turns": []}
        sessions.append(cur)
    tag, text = line[0], line[2:]
    if tag == "@":
        cur["docs"] = text.split(None, 1)[1]
    elif tag == ">":
        cur["turns"].append({"say": text, "want": [], "bad": []})
    elif tag == "=":
        cur["turns"][-1]["want"].append(text)
    elif tag == "!":
        cur["turns"][-1]["bad"].append(text)

checks = failed = skipped = 0
for s in sessions:
    if s["docs"] == "data" and skip_data:
        skipped += 1
        continue
    typed = "".join(t["say"] + "\n" for t in s["turns"])
    out = subprocess.run([exe, "--retrieve-only", "--docs", s["docs"]], input=typed,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True).stdout
    replies = out.split("\n> ")[1:]          # [0] is the banner
    for i, t in enumerate(s["turns"]):
        reply = replies[i] if i < len(replies) else "<no reply>"
        for w in t["want"]:
            checks += 1
            if w not in reply:
                failed += 1
                print("  FAIL  > %s\n        expected: %s\n        got: %s"
                      % (t["say"], w, reply.strip().replace("\n", "\n             ")))
        for b in t["bad"]:
            checks += 1
            if b in reply:
                failed += 1
                print("  FAIL  > %s\n        must not contain: %s\n        got: %s"
                      % (t["say"], b, reply.strip().replace("\n", "\n             ")))
print("dialogues: %d of %d checks passed in %d conversations%s"
      % (checks - failed, checks, len(sessions) - skipped,
         " (%d skipped: no data/)" % skipped if skipped else ""))
sys.exit(1 if failed else 0)
