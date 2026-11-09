# My AI friend — models

Model files the [My AI friend](https://ai-friend.club) Android app downloads, with the scripts that built them.
The files themselves are release assets; the app pins each one by its SHA-256 and also fetches it from
`https://ai-friend.club/models/` when GitHub is unavailable.

## Parakeet Ultra 0.6B, int8 with float32 depthwise and subsampling convolutions

Release [`asr-2026-11-08`](https://github.com/ceoptrsltd/myaifriend-models/releases/tag/asr-2026-11-08):
`parakeet-tdt-0.6b-ultra-mixed-int8.tar.bz2`, 487,310,667 bytes,
sha256 `452b74983764fb3cb41a1a2ef1371e0a79ec5a2cdb84af88fcdb373642b491bd`. Layout of sherpa-onnx's
`nemo_transducer` packages: `encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, `tokens.txt`.

**What and why.** sherpa-onnx's int8 package of [Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra)
quantises every encoder convolution with one uint8 scale per tensor (onnxruntime `quantize_dynamic`,
per-tensor). The 24 depthwise convolutions carry a folded BatchNorm, whose per-channel scales are far apart, and
lose most of their precision that way. This build starts from sherpa-onnx's float32 export and quantises it the
same way, except those 24 depthwise and the 5 subsampling convolutions, which stay float32. They are tiny, so
size and speed do not change.

**Measured** with sherpa-onnx 1.13.5 (greedy search, 4 threads, x86-64), FLEURS test split, one recording per
sentence of 3–15 s, 200 per language, WER after lower-casing and removing punctuation:

| | English | Polish | real-time factor |
|---|---|---|---|
| sherpa-onnx int8 | 7.51 % | 10.50 % | 0.135 |
| **this package** | **5.63 %** | **7.65 %** | 0.127 |
| float32 | 5.48 % | 7.38 % | — |

With babble of three other speakers mixed in at 10 dB SNR (100 clips per language): English 10.30 → 6.41 %,
Polish 17.35 → 11.92 %. The same change on GigaAM v3 (Russian), whose convolution module uses LayerNorm, gains
nothing measurable (5.72 → 5.61 %).

**Rebuild.** [`asr/parakeet-ultra-mixed-int8/`](asr/parakeet-ultra-mixed-int8/) — `quant_mixed.py` builds the
encoder; `make_fleurs.py`, `make_noisy.py`, `decode_set.py`, `wer.py`, `paired.py` reproduce the numbers.
Inputs: [csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-ultra](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-ultra)
(`encoder.onnx` sha256 `609ff3ce…3ae2`, `encoder.weights` `1209bc92…1c12`), sherpa-onnx's
`sherpa-onnx-nemo-parakeet-tdt-0.6b-ultra-int8.tar.bz2` for the decoder, joiner and tokens, onnxruntime 1.31.0.

    python quant_mixed.py encoder.onnx encoder.int8.onnx mixed

## Licences

- **Model weights:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Parakeet Ultra by Moondream, a
  post-training of [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
  (© NVIDIA Corporation). Converted to ONNX by the [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project
  (Apache-2.0). Modified by PTRS LTD: re-quantised as described above. Provided without warranties of any kind.
  Neither NVIDIA, Moondream nor the sherpa-onnx project endorses this package.
- **Scripts in this repository:** MIT, see [LICENSE](LICENSE).
