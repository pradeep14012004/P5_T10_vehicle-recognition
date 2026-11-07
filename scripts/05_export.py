"""Export models to ONNX and apply the required optimisation (quantization)."""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
import timm
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic, quantize_static
from ultralytics import YOLO
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vehicle_id import classical
from vehicle_id.data import read_manifest
from vehicle_id.evaluation import gt_box, read_rgb
from vehicle_id.pipeline import classifier_tensor

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="checkpoints/classifier/best.pt")
parser.add_argument("--detector", default="checkpoints/detector/finetune/weights/best.pt")
parser.add_argument("--manifest", default="data/manifest.csv")
parser.add_argument("--out", default="models")
parser.add_argument("--calibration-images", type=int, default=200)
args = parser.parse_args()
out = Path(args.out)
out.mkdir(exist_ok=True)
state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
config = state["config"]
size = config["input_size"][1]
model = timm.create_model(state["name"], pretrained=False, num_classes=196, exportable=True)
model.load_state_dict(state["model"])
model.eval()
fp32 = out / "classifier_fp32.onnx"
dummy = torch.randn(2, 3, size, size)
export_args = dict(input_names=["image"], output_names=["logits"], dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
try:
    torch.onnx.export(model, dummy, fp32, dynamo=False, **export_args)
except TypeError:
    torch.onnx.export(model, dummy, fp32, **export_args)
session = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
difference = np.abs(session.run(None, {"image": dummy.numpy()})[0] - model(dummy).detach().numpy()).max()
print(f"ONNX vs PyTorch maximum logit difference: {difference:.6f}")
if difference > 1e-2:
    raise SystemExit("The ONNX export does not match the PyTorch model.")


class Calibration(CalibrationDataReader):
    def __init__(self, rows):
        self.rows = iter(rows)

    def get_next(self):
        row = next(self.rows, None)
        if row is None:
            return None
        image = read_rgb(row["path"])
        box = classical.expand_box(gt_box(row), image.shape[1], image.shape[0])
        return {"image": classifier_tensor(classical.crop(image, box), size, config["mean"], config["std"])[None]}


quantize_dynamic(str(fp32), str(out / "classifier_int8_dynamic.onnx"), weight_type=QuantType.QUInt8)
rows = read_manifest(args.manifest, "val")
rows = [rows[i] for i in np.random.default_rng(42).permutation(len(rows))[:args.calibration_images]]
try:
    quantize_static(str(fp32), str(out / "classifier_int8_static.onnx"), Calibration(rows), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
except Exception as error:
    print(f"Static quantization skipped: {error}")
shutil.copy(YOLO("yolo11n.pt").export(format="onnx", imgsz=640, opset=17, simplify=False), out / "detector_coco.onnx")
if Path(args.detector).exists():
    shutil.copy(YOLO(args.detector).export(format="onnx", imgsz=640, opset=17, simplify=False), out / "detector_finetuned.onnx")
names = json.load(open(Path(args.manifest).parent / "classes.json"))
(out / "meta.json").write_text(json.dumps({
    "classifier_name": state["name"], "classifier_file": fp32.name, "size": size, "mean": list(config["mean"]), "std": list(config["std"]),
    "detector_file": "detector_coco.onnx", "detector_classes": [2, 5, 7], "class_names": names}, indent=1))
for path in sorted(out.glob("*.onnx")):
    print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB")
