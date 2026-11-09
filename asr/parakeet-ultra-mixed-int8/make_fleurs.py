# Builds an en/pl test set from the FLEURS test parquet files the OnlyMe session downloaded:
# one recording per sentence id, 3..15 s, N picked evenly. Usage: make_fleurs.py <parquet_dir> <out_dir> <N>
import io, os, json, sys
import soundfile as sf, pyarrow.parquet as pq

src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(out, exist_ok=True)
refs = {}
for lang in ("en_us", "pl_pl"):
    t = pq.read_table(os.path.join(src, f"{lang}.parquet"), columns=["id", "audio", "raw_transcription", "num_samples"]).to_pylist()
    seen, picked = set(), []
    for row in t:
        if row["id"] in seen or not (3 * 16000 <= row["num_samples"] <= 15 * 16000):
            continue
        seen.add(row["id"]); picked.append(row)
    print(lang, "eligible", len(picked))
    picked = picked[::max(1, len(picked) // n)][:n]
    for k, row in enumerate(picked):
        x, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        assert sr == 16000, sr
        name = f"{lang[:2]}_{k:03d}"
        sf.write(os.path.join(out, name + ".wav"), x, sr, subtype="PCM_16")
        refs[name] = row["raw_transcription"]
json.dump(refs, open(os.path.join(out, "refs.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(refs), "files")
