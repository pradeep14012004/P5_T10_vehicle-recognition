#!/usr/bin/env python3
"""Convert ConvInteger signed INT8 weights to equivalent UINT8 weights.

For each ConvInteger node, map q_int8 -> q_uint8 = q_int8 + 128 and
zero_point_int8 -> zero_point_uint8 = zero_point_int8 + 128.
This preserves (q - zero_point) exactly while using the UINT8 CPU kernel.
"""
from pathlib import Path
import os
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from huggingface_hub import snapshot_download, HfApi

REPO = os.environ["MODEL_REPO"]
TOKEN = os.environ["HF_TOKEN"]
ROOT = Path(snapshot_download(repo_id=REPO, token=TOKEN))
MODEL = ROOT / "classifier_int8_dynamic.onnx"
BACKUP = ROOT / "classifier_int8_dynamic_signed_backup.onnx"
OUT = ROOT / "classifier_int8_dynamic_uint8.onnx"

model = onnx.load(str(MODEL), load_external_data=True)
initializers = {x.name: x for x in model.graph.initializer}
converted = 0
for index, node in enumerate(model.graph.node):
    if node.op_type != "ConvInteger":
        continue
    if len(node.input) < 2 or not node.input[1]:
        raise RuntimeError(f"ConvInteger node {node.name or index} has no weight input")
    weight_name = node.input[1]
    if weight_name not in initializers:
        raise RuntimeError(f"ConvInteger weight is not a constant initializer: {weight_name}")
    weight_tensor = initializers[weight_name]
    weights = numpy_helper.to_array(weight_tensor)
    if weights.dtype == np.uint8:
        continue
    if weights.dtype != np.int8:
        raise RuntimeError(f"Unsupported ConvInteger weight dtype {weights.dtype} at {node.name or index}")
    new_weight_name = f"{weight_name}_uint8_convfix_{index}"
    new_weights = (weights.astype(np.int16) + 128).astype(np.uint8)
    new_tensor = numpy_helper.from_array(new_weights, name=new_weight_name)
    model.graph.initializer.append(new_tensor)
    node.input[1] = new_weight_name

    # ConvInteger input 3 is B_zero_point. If omitted, ONNX defaults it to
    # zero in B's type, so the equivalent UINT8 zero point is 128.
    if len(node.input) < 4 or not node.input[3]:
        zp_name = f"{new_weight_name}_zero_point"
        model.graph.initializer.append(numpy_helper.from_array(np.array(128, dtype=np.uint8), name=zp_name))
        while len(node.input) < 4:
            node.input.append("")
        node.input[3] = zp_name
    else:
        old_zp_name = node.input[3]
        if old_zp_name not in initializers:
            raise RuntimeError(f"ConvInteger B_zero_point is not a constant initializer: {old_zp_name}")
        old_zp = numpy_helper.to_array(initializers[old_zp_name])
        if old_zp.dtype == np.int8:
            new_zp = (old_zp.astype(np.int16) + 128).astype(np.uint8)
        elif old_zp.dtype == np.uint8:
            new_zp = old_zp
        else:
            raise RuntimeError(f"Unsupported B_zero_point dtype {old_zp.dtype} at {node.name or index}")
        zp_name = f"{old_zp_name}_uint8_convfix_{index}"
        model.graph.initializer.append(numpy_helper.from_array(new_zp, name=zp_name))
        node.input[3] = zp_name
    converted += 1

if converted == 0:
    raise RuntimeError("No signed-INT8 ConvInteger nodes were converted; refusing to upload.")

onnx.checker.check_model(model)
onnx.save(model, str(OUT))

# Validate that ONNX Runtime can load and execute the converted graph before upload.
session = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
input_meta = session.get_inputs()[0]
shape = [1 if not isinstance(d, int) or d <= 0 else d for d in input_meta.shape]
dtype = np.float32 if input_meta.type == "tensor(float)" else np.float32
dummy = np.random.default_rng(7).standard_normal(shape).astype(dtype)
outputs = session.run(None, {input_meta.name: dummy})
if not outputs or not np.isfinite(outputs[0]).all():
    raise RuntimeError("Converted model produced empty or non-finite output.")
print(f"Validated converted graph: {converted} ConvInteger nodes; output shape={outputs[0].shape}")

# Preserve original in the model repo, then replace the app's expected filename.
api = HfApi(token=TOKEN)
api.upload_file(path_or_fileobj=str(MODEL), path_in_repo="classifier_int8_dynamic_signed_backup.onnx",
                repo_id=REPO, repo_type="model", commit_message="Back up signed INT8 classifier before compatibility conversion")
api.upload_file(path_or_fileobj=str(OUT), path_in_repo="classifier_int8_dynamic.onnx",
                repo_id=REPO, repo_type="model", commit_message="Convert ConvInteger weights to equivalent UINT8 for CPU inference")
print("Uploaded validated UINT8-compatible classifier as classifier_int8_dynamic.onnx")
