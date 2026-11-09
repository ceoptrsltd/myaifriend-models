# WER per language for several decode results. Usage: wer.py <refs.json> <label=result.json> ...
import sys, json, re, unicodedata, jiwer

refs = json.load(open(sys.argv[1], encoding="utf-8"))


def norm(s):
    s = unicodedata.normalize("NFC", s.lower()); s = re.sub(r"[^\w\s']", " ", s); return " ".join(s.split())


runs = []
for arg in sys.argv[2:]:
    label, path = arg.split("=", 1)
    runs.append((label, json.load(open(path, encoding="utf-8"))))
langs = sorted({k.split("_")[0] for k in refs})
print(f"{'variant':10}" + "".join(f"{l:>9}" for l in langs) + "      rtf")
for label, r in runs:
    row = ""
    for l in langs:
        ks = sorted(k for k in refs if k.startswith(l + "_"))
        row += f"{100 * jiwer.wer([norm(refs[k]) for k in ks], [norm(r['res'][k]) for k in ks]):8.2f}%"
    print(f"{label:10}{row}    {r['rtf']:.3f}")
