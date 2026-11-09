# Parakeet Ultra 0.6B, int8 with float32 depthwise and subsampling convolutions (sherpa-onnx layout)

Weights: [moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra), Moondream's post-training of
NVIDIA's parakeet-tdt-0.6b-v3, licensed CC BY 4.0. ONNX graphs: the sherpa-onnx project's float32 export
[csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-ultra](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-ultra)
(encoder.onnx sha256 609ff3ce…, encoder.weights 1209bc92…).

Changes made by My AI friend (PTRS LTD), 2026-11-08:

- encoder.int8.onnx is that float32 encoder quantised with onnxruntime 1.31.0 `quantize_dynamic`
  (uint8 weights, per-tensor) exactly as sherpa-onnx quantises it, except the 24 depthwise convolutions
  (BatchNorm folded in) and the 5 subsampling (pre_encode) convolutions, which stay float32. The encoder
  metadata is sherpa-onnx's own.
- decoder.int8.onnx, joiner.int8.onnx and tokens.txt are sherpa-onnx's int8 package files, unchanged.

FLEURS test, 200 clips per language, sherpa-onnx 1.13.5, greedy search, WER:

| | English | Polish |
|---|---|---|
| sherpa-onnx int8 (all convolutions int8) | 7.51 % | 10.50 % |
| this package | 5.63 % | 7.65 % |
| float32 | 5.48 % | 7.38 % |

Same size and speed as the all-int8 package. Neither NVIDIA, Moondream nor the sherpa-onnx project endorses it.
