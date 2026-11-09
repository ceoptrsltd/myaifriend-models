# Quantises sherpa-onnx's float32 Parakeet Ultra encoder the way its own export does (onnxruntime quantize_dynamic,
# uint8 weights, per-tensor) — optionally keeping the depthwise and subsampling (pre_encode) convolutions in float32.
# Usage: quant_mixed.py <fp32_encoder.onnx> <out.onnx> <mixed|all> [expected kept convs, 29 = Parakeet, 18 = GigaAM]
import sys
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

src, dst, mode = sys.argv[1:4]
expected = int(sys.argv[4]) if len(sys.argv) > 4 else 29
m = onnx.load(src, load_external_data=False)
keep = [n.name for n in m.graph.node
        if n.op_type == "Conv" and ("/depthwise_conv/" in n.name or n.name.startswith("/pre_encode/"))]
assert len(keep) == expected, keep  # Parakeet: 24 depthwise + 5 subsampling; GigaAM: 16 + 2
quantize_dynamic(src, dst, weight_type=QuantType.QUInt8, nodes_to_exclude=keep if mode == "mixed" else [])

# sherpa-onnx reads vocab size, subsampling, feature normalisation etc. from the encoder's metadata
q = onnx.load(dst)
have = {p.key for p in q.metadata_props}
for p in m.metadata_props:
    if p.key not in have:
        q.metadata_props.add(key=p.key, value=p.value)
onnx.save(q, dst)
convs = sum(n.op_type == "Conv" for n in q.graph.node)
print(dst, "float Conv:", convs, "ConvInteger:", sum(n.op_type == "ConvInteger" for n in q.graph.node),
      "MatMulInteger:", sum(n.op_type == "MatMulInteger" for n in q.graph.node),
      "meta:", sorted(p.key for p in q.metadata_props))
