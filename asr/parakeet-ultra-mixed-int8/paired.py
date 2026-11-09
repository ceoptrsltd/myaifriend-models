# Per-utterance word errors of two runs: how many files each one wins. Usage: paired.py <refs.json> <a.json> <b.json>
import sys, json, re, unicodedata, jiwer

refs = json.load(open(sys.argv[1], encoding="utf-8"))
A = json.load(open(sys.argv[2], encoding="utf-8"))["res"]
B = json.load(open(sys.argv[3], encoding="utf-8"))["res"]


def norm(s):
    s = unicodedata.normalize("NFC", s.lower()); s = re.sub(r"[^\w\s']", " ", s); return " ".join(s.split())


def errs(r, h):
    o = jiwer.process_words(norm(r), norm(h)); return o.substitutions + o.deletions + o.insertions


for l in sorted({k.split("_")[0] for k in refs}):
    ks = [k for k in refs if k.startswith(l + "_")]
    d = [errs(refs[k], A[k]) - errs(refs[k], B[k]) for k in ks]
    print(f"{l}: b better in {sum(x > 0 for x in d)}, worse in {sum(x < 0 for x in d)}, same {sum(x == 0 for x in d)};"
          f" words saved {sum(d)}; identical text {sum(A[k] == B[k] for k in ks)}/{len(ks)}")
