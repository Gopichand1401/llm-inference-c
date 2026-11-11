#!/usr/bin/env python3
"""Hostile-input tests for xlsx_dump (build it with AddressSanitizer + UBSan).

  1. fixed bad inputs must be refused with a clear message (exit code 1)
  2. ZIP-level fuzzing: random byte flips / truncations of a real workbook
  3. XML-level fuzzing: valid ZIP, but the XML inside is damaged

Pass = no crash, no sanitizer report, no hang: exit code is 0 (parsed) or 1
(refused with a message), never anything else.

Usage: fuzz_xlsx.py <xlsx_dump binary> <sample.xlsx> [iterations]
"""
import io
import os
import random
import subprocess
import sys
import tempfile
import zipfile

ENV = dict(os.environ,
           ASAN_OPTIONS="exitcode=99:detect_leaks=1",
           UBSAN_OPTIONS="halt_on_error=1:exitcode=98:print_stacktrace=1")


def run(binary, path):
    try:
        p = subprocess.run([binary, "--tsv", path], capture_output=True,
                           env=ENV, timeout=20)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", b""
    err = p.stderr.decode("utf-8", "replace")
    if "Sanitizer" in err or "runtime error" in err or p.returncode not in (0, 1):
        return "CRASH(exit %d)" % p.returncode, err.encode()
    return p.returncode, err.encode()


def fixed_cases(binary, sample, tmp):
    good = open(sample, "rb").read()
    cases = {
        "empty file": b"",
        "10 random bytes": os.urandom(10),
        "plain text renamed .xlsx": b"hello, this is not a workbook\n" * 50,
        "old .xls (OLE2 magic)": b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1" + b"\0" * 600,
        "truncated to half": good[: len(good) // 2],
        "truncated to 30 bytes": good[:30],
        "zip with no workbook.xml": None,
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("readme.txt", "no workbook here")
    cases["zip with no workbook.xml"] = buf.getvalue()

    ok = True
    for name, data in cases.items():
        path = os.path.join(tmp, "case.xlsx")
        with open(path, "wb") as f:
            f.write(data)
        code, err = run(binary, path)
        msg = err.decode().strip().splitlines()[-1:] or ["(no message)"]
        good_case = (code == 1 and msg != ["(no message)"])
        print("  %-28s -> %s  %s" % (name, "refused" if good_case else code, msg[0][:90]))
        ok &= good_case
    code, err = run(binary, os.path.join(tmp, "does_not_exist.xlsx"))
    print("  %-28s -> %s  %s" % ("missing file", "refused" if code == 1 else code,
                                 err.decode().strip()[:90]))
    return ok and code == 1


def fuzz_zip(binary, sample, tmp, n, rng):
    good = bytearray(open(sample, "rb").read())
    bad = 0
    for i in range(n):
        data = bytearray(good)
        kind = rng.choice(["flip", "flip", "trunc", "zero"])
        if kind == "flip":
            for _ in range(rng.randint(1, 8)):
                data[rng.randrange(len(data))] = rng.randrange(256)
        elif kind == "zero":
            a = rng.randrange(len(data))
            data[a:a + rng.randint(1, 64)] = b"\0" * rng.randint(1, 64)
        else:
            data = data[: rng.randrange(1, len(data))]
        path = os.path.join(tmp, "fz.xlsx")
        open(path, "wb").write(bytes(data))
        code, err = run(binary, path)
        if code not in (0, 1):
            bad += 1
            keep = os.path.join(tmp, "crash_zip_%d.xlsx" % i)
            os.replace(path, keep)
            print("  FAIL", code, keep, err.decode()[:300])
    return bad


def mutate_text(s, rng):
    kind = rng.choice(["cut", "delete", "insert", "dup", "trunc", "swap"])
    if not s:
        return s
    a = rng.randrange(len(s))
    b = min(len(s), a + rng.randint(1, 40))
    if kind == "cut":
        return s[:a] + s[b:]
    if kind == "delete":
        return s[:a] + s[a + 1:]
    if kind == "insert":
        junk = rng.choice(["<", ">", "</", "<c r=\"", "&", "&#", "&#x", "_x00",
                           "\"", "'", "<![CDATA[", "<!--", "]]>", "-->",
                           "<c><v>", "r=\"ZZZZZZ99999999999\"", "s=\"-5\"",
                           "t=\"s\"><v>99999999999</v>", "\0"])
        return s[:a] + junk + s[a:]
    if kind == "dup":
        return s[:b] + s[a:b] + s[b:]
    if kind == "trunc":
        return s[:a]
    return s[:a] + s[b:] + s[a:b]


def fuzz_xml(binary, sample, tmp, n, rng):
    with zipfile.ZipFile(sample) as z:
        parts = {name: z.read(name) for name in z.namelist()}
    names = [p for p in parts if p.endswith(".xml") or p.endswith(".rels")]
    weights = [3 if "sheet" in p or "shared" in p or "workbook" in p else 1 for p in names]
    bad = 0
    for i in range(n):
        mutated = dict(parts)
        for _ in range(rng.randint(1, 3)):
            name = rng.choices(names, weights)[0]
            text = mutated[name].decode("utf-8", "replace")
            mutated[name] = mutate_text(text, rng).encode("utf-8")
        path = os.path.join(tmp, "fx.xlsx")
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for name, data in mutated.items():
                z.writestr(name, data)
        code, err = run(binary, path)
        if code not in (0, 1):
            bad += 1
            keep = os.path.join(tmp, "crash_xml_%d.xlsx" % i)
            os.replace(path, keep)
            print("  FAIL", code, keep, err.decode()[:300])
    return bad


def main():
    binary, sample = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    rng = random.Random(12345)
    with tempfile.TemporaryDirectory() as tmp:
        print("1. fixed bad inputs")
        ok = fixed_cases(binary, sample, tmp)
        print("2. ZIP-level fuzzing, %d files" % n)
        bad_zip = fuzz_zip(binary, sample, tmp, n, rng)
        print("   crashes: %d" % bad_zip)
        print("3. XML-level fuzzing, %d files" % n)
        bad_xml = fuzz_xml(binary, sample, tmp, n, rng)
        print("   crashes: %d" % bad_xml)
        if not ok or bad_zip or bad_xml:
            print("FUZZ: FAIL")
            sys.exit(1)
    print("FUZZ: PASS")


if __name__ == "__main__":
    main()
