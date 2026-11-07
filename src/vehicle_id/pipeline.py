"""End-to-end inference: quality check, detection, ROI, classification, fusion, decision."""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from . import classical
from .decision import Hierarchy, decide, fuse, softmax

MAX_SIDE = 1600
MIN_SIDE = 32
VEHICLE_CLASSES_COCO = (2, 5, 7)  # car, bus, truck


class BadInput(ValueError):
    """An upload that cannot be processed. The message is shown to the user."""


def load_image(source) -> np.ndarray:
    """Accepts a path, PIL image or array. Applies phone EXIF rotation and limits size."""
    try:
        if isinstance(source, np.ndarray):
            image = Image.fromarray(source)
        elif isinstance(source, Image.Image):
            image = source
        else:
            image = Image.open(source)
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, TypeError) as error:
        raise BadInput("That file could not be read as an image. Please upload a JPG or PNG.") from error
    if min(image.size) < MIN_SIDE:
        raise BadInput(f"That image is too small. Please use one at least {MIN_SIDE}px on each side.")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    return np.asarray(image)


def letterbox(image: np.ndarray, size: int):
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    new_w, new_h = round(width * scale), round(height * scale)
    canvas = np.full((size, size, 3), 114, np.uint8)
    left, top = (size - new_w) // 2, (size - new_h) // 2
    canvas[top:top + new_h, left:left + new_w] = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return canvas, scale, left, top


def parse_yolo_output(raw: np.ndarray, size: int, scale: float, left: int, top: int, class_ids,
                      confidence: float = 0.25, iou: float = 0.5, shape=None) -> list[dict]:
    """Decodes an Ultralytics YOLO11 ONNX output (1, 4 + classes, N), with xywh centre boxes."""
    prediction = np.squeeze(raw, 0).T
    boxes, class_scores = prediction[:, :4].copy(), prediction[:, 4:]
    if boxes.size and boxes.max() <= 1.5:
        boxes *= size
    allowed = [i for i in class_ids if i < class_scores.shape[1]] or [0]
    scores = class_scores[:, allowed].max(axis=1)
    keep = scores >= confidence
    boxes, scores = boxes[keep], scores[keep]
    if not len(boxes):
        return []
    xywh = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2, boxes[:, 2], boxes[:, 3]], axis=1)
    picks = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), confidence, iou)
    results = []
    for index in np.array(picks).reshape(-1):
        x, y, w, h = xywh[index]
        box = [(x - left) / scale, (y - top) / scale, (x + w - left) / scale, (y + h - top) / scale]
        if shape is not None:
            box = classical.clip_box(box, shape[1], shape[0])
        results.append({"box": tuple(box), "score": float(scores[index])})
    return sorted(results, key=lambda r: -r["score"])


class OnnxDetector:
    def __init__(self, path, size: int = 640, class_ids=VEHICLE_CLASSES_COCO, confidence: float = 0.25, providers=None):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(path), providers=providers or ["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.size, self.class_ids, self.confidence = size, class_ids, confidence

    def __call__(self, image: np.ndarray) -> list[dict]:
        canvas, scale, left, top = letterbox(image, self.size)
        tensor = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        raw = self.session.run(None, {self.input: tensor})[0]
        return parse_yolo_output(raw, self.size, scale, left, top, self.class_ids, self.confidence, shape=image.shape)


def classifier_tensor(region: np.ndarray, size: int, mean, std) -> np.ndarray:
    resized = cv2.resize(region, (size, size), interpolation=cv2.INTER_CUBIC)
    array = (resized.astype(np.float32) / 255.0 - np.array(mean, np.float32)) / np.array(std, np.float32)
    return array.transpose(2, 0, 1)


class OnnxClassifier:
    def __init__(self, path, size: int, mean, std, providers=None):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(path), providers=providers or ["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.size, self.mean, self.std = size, mean, std

    def __call__(self, region: np.ndarray, flip: bool = False) -> np.ndarray:
        batch = [classifier_tensor(region, self.size, self.mean, self.std)]
        if flip:
            batch.append(batch[0][:, :, ::-1].copy())
        logits = np.concatenate([self.session.run(None, {self.input: b[None]})[0] for b in batch])
        return logits.mean(axis=0)


class ClassicalModel:
    """Scaler + PCA + softmax regression, stored as plain arrays so deployment needs no sklearn."""

    def __init__(self, path):
        weights = np.load(path)
        self.mean, self.scale = weights["mean"], weights["scale"]
        self.pca_mean, self.components = weights["pca_mean"], weights["components"]
        self.coef, self.intercept = weights["coef"], weights["intercept"]

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        scaled = (features - self.mean) / self.scale
        reduced = (scaled - self.pca_mean) @ self.components.T
        return softmax(reduced @ self.coef.T + self.intercept)


@dataclass
class Options:
    use_detector: bool = True
    use_refinement: bool = True
    use_enhancement: bool = True
    use_tta: bool = True
    use_fusion: bool = True
    max_vehicles: int = 3
    min_area_fraction: float = 0.02


@dataclass
class Result:
    status: str
    message: str = ""
    vehicles: list = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)
    intermediates: dict = field(default_factory=dict)
    geometry: dict = field(default_factory=dict)


class VehiclePipeline:
    def __init__(self, detector, classifier, classical_model, class_names, temperature=1.0, beta=0.0,
                 thresholds=None):
        self.detector, self.classifier, self.classical_model = detector, classifier, classical_model
        self.hierarchy = Hierarchy(class_names)
        self.temperature, self.beta = temperature, beta
        self.thresholds = thresholds or {"full": 0.6, "model": 0.6, "make": 0.6}

    def _classify(self, image, box, options) -> tuple[np.ndarray, np.ndarray]:
        region = classical.crop(image, box)
        logits = self.classifier(region, flip=options.use_tta)
        probs = softmax(logits, self.temperature)
        if options.use_fusion and self.classical_model is not None and self.beta > 0:
            features = classical.extract_features(region, box, image.shape)
            probs = fuse(probs, self.classical_model.probabilities(features), self.beta)
        return probs, region

    def run(self, source, options: Options | None = None) -> Result:
        options = options or Options()
        timings, started = {}, time.perf_counter()

        def lap(name):
            nonlocal started
            timings[name] = round((time.perf_counter() - started) * 1000, 1)
            started = time.perf_counter()

        try:
            original = load_image(source)
        except BadInput as error:
            return Result(status="bad_input", message=str(error))
        quality = classical.quality_metrics(original)
        image, steps = (classical.adaptive_enhance(original, quality) if options.use_enhancement else (original, []))
        intermediates = {"Enhanced input": image, "Edges (Canny)": classical.auto_canny(image)}
        lap("preprocess")
        height, width = image.shape[:2]
        fallback = False
        if options.use_detector and self.detector is not None:
            detections = self.detector(image)
            if not detections and image is not original:
                detections = self.detector(original)
            detections = [d for d in detections
                          if (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]) >= options.min_area_fraction * width * height]
            detections = sorted(detections, key=lambda d: -((d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])))[:options.max_vehicles]
        else:
            detections = [{"box": (0, 0, width, height), "score": None}]
        if not detections:
            detections, fallback = [{"box": classical.contour_roi(image), "score": None}], True
            intermediates["Contour ROI"] = _draw(image, detections[0]["box"])
        lap("detect")
        vehicles = []
        for detection in detections:
            box = detection["box"]
            if options.use_refinement and detection["score"] is not None:
                box = classical.refine_box(image, box)
            box = classical.expand_box(box, width, height)
            probs, region = self._classify(image, box, options)
            decision = decide(probs, self.hierarchy, self.thresholds)
            vehicles.append({"box": box, "detector_score": detection["score"], "region": region, **decision})
        lap("classify")
        if fallback and vehicles[0]["level"] != "full":
            return Result(status="no_vehicle", quality=quality, steps=steps, timings_ms=timings, intermediates=intermediates,
                          message="No vehicle found. Please try a clearer photo with the car in view.")
        intermediates["Regions"] = _draw(image, *[v["box"] for v in vehicles])
        intermediates["Classified crop"] = vehicles[0]["region"]
        intermediates.update(classical.feature_views(vehicles[0]["region"]))
        geometry = classical.geometry_summary(vehicles[0]["region"], vehicles[0]["box"], image.shape)
        status = "fallback_roi" if fallback else "ok"
        return Result(status=status, vehicles=vehicles, quality=quality, steps=steps, timings_ms=timings, intermediates=intermediates, geometry=geometry)


def _draw(image: np.ndarray, *boxes) -> np.ndarray:
    canvas = image.copy()
    thickness = max(2, min(image.shape[:2]) // 200)
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 120), thickness)
    return canvas


def annotate(image: np.ndarray, vehicles: list) -> np.ndarray:
    canvas = image.copy()
    scale = max(0.5, min(image.shape[:2]) / 900)
    thickness = max(2, int(scale * 2))
    colours = {"full": (0, 170, 90), "model": (0, 120, 220), "make": (230, 150, 0), "abstain": (210, 50, 50)}
    for vehicle in vehicles:
        x1, y1, x2, y2 = [int(v) for v in vehicle["box"]]
        colour = colours[vehicle["level"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)
        text = f"{vehicle['label']}  {vehicle['confidence']:.0%}"
        if len(text) > 48:
            text = text[:45] + "..."
        (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        w = min(w, image.shape[1] - x1)
        y = max(y1, h + base + 4)
        cv2.rectangle(canvas, (x1, y - h - base - 4), (x1 + w, y), colour, -1)
        cv2.putText(canvas, text, (x1, y - base - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return canvas


def load_pipeline(model_dir: str | Path, classifier_file: str | None = None, providers=None) -> VehiclePipeline:
    """Reads the files written by the training, export and calibration scripts."""
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "meta.json").read_text())
    detector = OnnxDetector(model_dir / meta["detector_file"], class_ids=meta["detector_classes"], providers=providers)
    classifier = OnnxClassifier(model_dir / (classifier_file or meta["classifier_file"]), meta["size"], meta["mean"], meta["std"], providers=providers)
    classical_path = model_dir / "classical.npz"
    classical_model = ClassicalModel(classical_path) if classical_path.exists() else None
    return VehiclePipeline(detector, classifier, classical_model, meta["class_names"], meta.get("temperature", 1.0),
                           meta.get("beta", 0.0), meta.get("thresholds"))
