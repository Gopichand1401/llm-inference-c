#!/usr/bin/env python3
"""Checks of the pocketllm Python package against the command-line programs:
the same model, prompt, sampler and seed must give the same text through
Python, through llm_generate and through llm_rag, and Rag.ask() must give
the replies of `llm_rag --retrieve-only` for every conversation of
tests/rag_dialogues.txt.

Usage: test_pocketllm.py <tiny.gguf> [tiny_stories.bin]
(run from the project folder by tests/run_pocket_checks.sh)"""
import os
import subprocess
import sys

import pocketllm

GGUF = sys.argv[1]
STORIES = sys.argv[2] if len(sys.argv) > 2 else None
HAVE_DATA = any(f.endswith(".xlsx") for f in os.listdir("data")) if os.path.isdir("data") else False
passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    passed += bool(ok)
    failed += not ok
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", name,
                          "" if ok or not detail else "\n        " + detail))


def cli(args, stdin=""):
    """stdout of a program, decoded like the package decodes text (the tiny
    random test models also produce bytes that are not valid UTF-8)."""
    out = subprocess.run(args, input=stdin.encode(), stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL).stdout
    return out.decode("utf-8", "replace")


# ---- text generation: Python == llm_generate ----
model = pocketllm.Model(GGUF)
same = True
for sampler, seed, prompt in (("greedy", 1, "hello there"), ("top-k", 7, "the course"),
                              ("temperature", 42, "who is")):
    want = cli(["./build/llm_generate", GGUF, "-n", "24", "--sampler", sampler,
                "--seed", str(seed), "--temperature", "0.9", "--top-k", "20", prompt])
    got = prompt + model.generate(prompt, sampler=sampler, seed=seed, temperature=0.9,
                                  top_k=20, max_tokens=24) + "\n"
    same &= got == want and len(want) > len(prompt) + 1
check("generate(): greedy, top-k and temperature give llm_generate's text", same)

want = cli(["./build/llm_generate", GGUF, "--chat", "-n", "20", "--sampler", "top-k",
            "--seed", "9", "hello"])
model.reset()
got = model.chat("hello", sampler="top-k", seed=9, max_tokens=20) + "\n"
check("chat(): one turn gives llm_generate --chat's reply", got == want and len(want) > 1)

want = cli(["./build/llm_generate", GGUF, "-i", "-n", "10", "--sampler", "top-k",
            "--seed", "5", "--system", "be brief"], "one\ntwo\nthree\n")
model.reset()
model.system = "be brief"
model.seed(5)
got = "".join("\n" + model.chat(m, sampler="top-k", max_tokens=10) for m in ("one", "two", "three"))
check("chat(): three turns with a system message give llm_generate -i's replies",
      got == want and model.stats["position"] > 60)
model.system = None

pieces = []
text = model.generate("abc", sampler="greedy", max_tokens=12, on_token=pieces.append)
check("streaming: the pieces add up to the returned text", "".join(pieces) == text and pieces)

# ---- chat formats: the turn is the documented template ----
ok = True
for fmt, template in (("zephyr", "<|system|>\nS</s>\n<|user|>\nhi</s>\n<|assistant|>\n"),
                      ("llama2", "[INST] <<SYS>>\nS\n<</SYS>>\n\nhi [/INST]"),
                      ("mistral", "[INST] S\n\nhi [/INST]"), ("none", "hi")):
    m2 = pocketllm.Model(GGUF, chat_format=fmt, system="S")
    a = m2.chat("hi", sampler="greedy", max_tokens=8)
    n_chat = m2.stats["prompt_tokens"]
    b = m2.generate(template, sampler="greedy", max_tokens=8)
    b = b if fmt == "none" else (b[1:] if b.startswith(" ") else b)
    ok &= m2.chat_format == fmt and a == b and n_chat == m2.stats["prompt_tokens"]
    m2.close()
check("chat formats zephyr / llama2 / mistral / none feed the documented prompt", ok)

if STORIES:
    story = pocketllm.Model(STORIES)
    want = cli(["./build/llm_generate", STORIES, "-n", "16", "--sampler", "top-k",
                "--seed", "3", "Once upon a time"])
    got = "Once upon a time" + story.generate("Once upon a time", seed=3, max_tokens=16) + "\n"
    check("llama2.c model: generate() gives llm_generate's text",
          got == want and not story.is_chat and story.chat_format == "none")
    try:
        pocketllm.Rag("build/rag_test_data").ask("x", model=story)
        check("llama2.c model refused for answers from files", False)
    except pocketllm.PocketError as exc:
        check("llama2.c model refused for answers from files", "chat model" in str(exc))
    story.close()

# ---- errors ----
errors = 0
for bad in (lambda: pocketllm.Model("build/no_such_model.gguf"),
            lambda: pocketllm.Model(__file__),
            lambda: pocketllm.Rag("build/no_such_folder"),
            lambda: model.generate("x", sampler="beam"),
            lambda: model.generate("x", max_tokens=0),
            lambda: setattr(model, "chat_format", "chatml")):
    try:
        bad()
    except (pocketllm.PocketError, ValueError):
        errors += 1
check("bad model file, folder, sampler, token count and chat format raise errors", errors == 6)

small = pocketllm.Model(GGUF, context=64)
try:
    small.generate("word " * 200, max_tokens=4)
    check("a prompt longer than the context raises an error", False)
except pocketllm.PocketError as exc:
    check("a prompt longer than the context raises an error", "context full" in str(exc))
small.close()

# ---- answers from files: Python == llm_rag --retrieve-only ----
sessions, cur = [], None
for raw in open("tests/rag_dialogues.txt", encoding="utf-8"):
    line = raw.rstrip("\n")
    if line.startswith("#"):
        continue
    if not line.strip():
        cur = None
        continue
    if cur is None:
        cur = {"docs": "data", "say": []}
        sessions.append(cur)
    if line[0] == "@":
        cur["docs"] = line[2:].split(None, 1)[1]
    elif line[0] == ">":
        cur["say"].append(line[2:])

turns = same = parts = 0
for s in sessions:
    if s["docs"] == "data" and not HAVE_DATA:
        continue
    out = subprocess.run(["./build/llm_rag", "--retrieve-only", "--docs", s["docs"]],
                         input="".join(q + "\n" for q in s["say"]).encode(),
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT).stdout.decode("utf-8", "replace")
    replies = out.split("\n> ")[1:]
    rag = pocketllm.Rag(s["docs"])
    for i, q in enumerate(s["say"]):
        want = replies[i] if i + 1 < len(replies) else replies[i].rstrip("\n") + "\n"
        a = rag.ask(q)
        turns += 1
        same += a.full == want or a.full == want.rstrip("\n") + "\n" or (not a.full and not want.strip())
        lines = [l for l in a.full.split("\n") if l]
        rebuilt = [l for l in a.text.split("\n") if l]
        rebuilt += ["Source: " + a.source] if a.source else []
        rebuilt += [a.offer] if a.offer else []
        parts += sorted(lines) == sorted(rebuilt)
    rag.close()
check("Rag.ask(): %d of %d replies equal llm_rag --retrieve-only" % (same, turns),
      same == turns and turns > 20)
check("Answer.text + source + offer make up the whole reply (%d of %d)" % (parts, turns),
      parts == turns)

# ---- files + model: Python == llm_rag --rag ----
docs = "build/rag_test_data"
rag = pocketllm.Rag(docs)
question = "Who is the instructor for LB205?"
want = cli(["./build/llm_rag", GGUF, "--rag", "--docs", docs, "-n", "12", question])
a = rag.ask(question, model=model, max_tokens=12)
got = a.text + "\n\nSources:\n" + "".join("  " + l + "\n" for l in a.source.split("\n"))
check("Rag.ask(model=...): the answer and sources of llm_rag --rag", got == want and a.source)
b = rag.ask("What is the capital of France?", model=model)
check("Rag.ask(model=...): nothing relevant, the model is not run",
      b.text == "I could not find this in the files." and not b.found
      and model.stats["generated_tokens"] == 0)
c = rag.ask(question, model=model, max_tokens=12)
check("the same question again gives the same answer (KV cache reset per question)",
      c.text == a.text)
rows = rag.search("physics lab", k=3)
check("Rag.search(): rows with text, source and score, best first",
      rows and "Physics Lab" in rows[0].text and all(rows[i].score >= rows[i + 1].score
                                                       for i in range(len(rows) - 1)))

# ---- a new file is picked up ----
live = "build/pocket_live"
os.makedirs(live, exist_ok=True)
for f in os.listdir(live):
    os.remove(os.path.join(live, f))
with open(os.path.join(live, "a.csv"), "w") as fh:
    fh.write("ID,Thing,Owner\nA-1,Lamp,Asha\n")
rag2 = pocketllm.Rag(live)
first = rag2.ask("owner of B-2").found
with open(os.path.join(live, "b.csv"), "w") as fh:
    fh.write("ID,Thing,Owner\nB-2,Globe,Balu\n")
second = rag2.ask("owner of B-2")
check("a file added after Rag() is answered from at the next question",
      not first and second.text == "Balu" and len(rag2.files) == 2)

print("\npocketllm checks: %d passed, %d failed" % (passed, failed))
sys.exit(1 if failed else 0)
