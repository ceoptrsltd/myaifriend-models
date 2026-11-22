# Parakeet TDT 0.6B v2 (English), int8 with float32 depthwise and subsampling convolutions (sherpa-onnx layout)

Weights: [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), licensed CC BY 4.0.
ONNX graphs: the sherpa-onnx project's float32 export
[csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2)
(encoder.onnx sha256 266d8e81…, encoder.weights 90cd4bb6…).

Changes made by PTRS LTD, 2026-11-22:

- encoder.int8.onnx is that float32 encoder quantised with onnxruntime 1.28.0 `quantize_dynamic`
  (uint8 weights, per-tensor) exactly as sherpa-onnx quantises it, except the 24 depthwise convolutions
  (BatchNorm folded in) and the 5 subsampling (pre_encode) convolutions, which stay float32. The encoder
  metadata is sherpa-onnx's own.
- decoder.int8.onnx, joiner.int8.onnx and tokens.txt are the files of sherpa-onnx's
  `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8` package, unchanged.

Measured with sherpa-onnx 2.0.2, modified beam search with 2 active paths and length reward 0.5, 4 threads:

| | sherpa-onnx int8 | this package |
|---|---|---|
| FLEURS English, 60 clips, WER | 4.48 % | 4.72 % |
| the same clips at -55 dBFS | 4.48 % | 4.32 % |
| 32 long dictations with pauses (2,819 words), cut at 0.7 s pauses | 6.56 % | 6.63 % |
| real-time factor, x86-64 laptop | 0.150 | 0.079 |
| 20 s of speech, Snapdragon 7s Gen 2 phone | 0.81–0.88 s | 0.64–0.69 s |

Accuracy is the same within the noise of these sets; the package is faster. Same size.
Neither NVIDIA nor the sherpa-onnx project endorses it.
