# Mixes every wav of a set with babble (three other speakers from another set, summed) at a given SNR.
# Usage: make_noisy.py <clean_dir> <babble_dir> <out_dir> <snr_db> [max_per_lang]
import sys, os, glob, json, shutil
import numpy as np, soundfile as sf

clean, babble_dir, out, snr = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
cap = int(sys.argv[5]) if len(sys.argv) > 5 else 10 ** 9
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(7)
pool = [sf.read(p, dtype="float32")[0] for p in sorted(glob.glob(os.path.join(babble_dir, "*.wav")))[:60]]
refs = json.load(open(os.path.join(clean, "refs.json"), encoding="utf-8"))
kept, count = {}, {}
for p in sorted(glob.glob(os.path.join(clean, "*.wav"))):
    name = os.path.splitext(os.path.basename(p))[0]
    lang = name.split("_")[0]
    if count.get(lang, 0) >= cap:
        continue
    count[lang] = count.get(lang, 0) + 1
    x = sf.read(p, dtype="float32")[0]
    b = np.zeros_like(x)
    for i in rng.choice(len(pool), 3, replace=False):
        s = pool[i]
        s = np.tile(s, int(np.ceil(len(x) / len(s))))[: len(x)]
        b += s / (np.sqrt(np.mean(s ** 2)) + 1e-9)
    gain = np.sqrt(np.mean(x ** 2)) / (np.sqrt(np.mean(b ** 2)) + 1e-9) / (10 ** (snr / 20))
    y = x + gain * b
    y /= max(1.0, float(np.abs(y).max()) / 0.98)
    sf.write(os.path.join(out, name + ".wav"), y, 16000, subtype="PCM_16")
    kept[name] = refs[name]
json.dump(kept, open(os.path.join(out, "refs.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(out, count)
