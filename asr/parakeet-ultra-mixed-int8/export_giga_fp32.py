# Exports GigaAM v3 e2e RNNT's encoder to float32 ONNX exactly like sherpa-onnx's
# scripts/nemo/GigaAM/export-onnx-rnnt-v3-punct.py (same wrapper, same metadata), without quantising it.
# Usage: export_giga_fp32.py <out_dir>
import os, sys
import gigaam, onnx, torch
from gigaam.utils import onnx_converter

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
model = gigaam.load_model("v3_e2e_rnnt", fp16_encoder=False, device="cpu").eval()
onnx_converter(model_name="v3_e2e_rnnt_encoder", out_dir=out, module=model.encoder,
               dynamic_axes=model.encoder.dynamic_axes())
meta = {
    "vocab_size": model.cfg["head"]["decoder"]["num_classes"] - 1,
    "pred_rnn_layers": model.cfg["head"]["decoder"]["pred_rnn_layers"],
    "pred_hidden": model.cfg["head"]["decoder"]["pred_hidden"],
    "normalize_type": "", "subsampling_factor": 4, "model_type": "EncDecRNNTBPEModel", "version": "3",
    "model_author": "https://github.com/salute-developers/GigaAM",
    "license": "https://github.com/salute-developers/GigaAM/blob/main/LICENSE",
    "language": "Russian", "comment": "v3", "is_giga_am": 1,
}
p = os.path.join(out, "v3_e2e_rnnt_encoder.onnx")
m = onnx.load(p)
del m.metadata_props[:]
for k, v in meta.items():
    m.metadata_props.add(key=k, value=str(v))
onnx.save(m, os.path.join(out, "encoder.onnx"))
os.remove(p)
print("encoder.onnx written;", sum(n.op_type == "Conv" for n in m.graph.node), "Conv nodes")
for n in m.graph.node:
    if n.op_type == "Conv":
        print(" ", n.name)
