# Decodes every wav of a test set with one sherpa-onnx model (the app's 1.13.5: nemo_transducer, greedy, 16 kHz)
# and writes {name: text} plus timing. Usage: decode_set.py <model_dir> <suffix: int8|fp32|mixed> <set_dir> <out.json>
import sys, os, json, time, glob, wave
import numpy as np
import sherpa_onnx

d, suffix, set_dir, out = sys.argv[1:5]
enc = {"int8": "encoder.int8.onnx", "fp32": "encoder.onnx", "mixed": "encoder.mixed.onnx",
       "giga": "encoder.int8.onnx", "gigafp32": "encoder.onnx", "gigamixed": "encoder.mixed.onnx"}[suffix]
dj = "" if suffix == "fp32" or suffix.startswith("giga") else ".int8"  # GigaAM ships float decoder/joiner
t0 = time.time()
rec = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=os.path.join(d, enc), decoder=os.path.join(d, f"decoder{dj}.onnx"),
    joiner=os.path.join(d, f"joiner{dj}.onnx"), tokens=os.path.join(d, "tokens.txt"),
    model_type="nemo_transducer", num_threads=4, decoding_method="greedy_search")
load = time.time() - t0
res, audio_s, dec_s = {}, 0.0, 0.0
for w in sorted(glob.glob(os.path.join(set_dir, "*.wav"))):
    with wave.open(w) as f:
        a = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        sr = f.getframerate()
    s = rec.create_stream(); s.accept_waveform(sr, a)
    t = time.time(); rec.decode_stream(s); dec_s += time.time() - t
    audio_s += len(a) / sr
    res[os.path.splitext(os.path.basename(w))[0]] = s.result.text
json.dump({"model": d, "encoder": enc, "load_s": load, "rtf": dec_s / audio_s, "res": res},
          open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{os.path.basename(d)} {enc}: {len(res)} files, load {load:.1f}s, rtf {dec_s / audio_s:.3f}")
