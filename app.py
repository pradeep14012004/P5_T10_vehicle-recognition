"""Gradio app: upload a vehicle photo; get boxes, make / model / year, confidence and processing steps."""
from __future__ import annotations
import copy
import cv2
import html
import json
import logging
import os
from pathlib import Path
import gradio as gr
import numpy as np
from src.vehicle_id import classical, ui
from src.vehicle_id.pipeline import BadInput, OnnxDetector, Options, annotate, load_image, load_pipeline

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vehicle-id")
ROOT = Path(__file__).parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models"))
METRICS = ROOT / "results" / "metrics.json"
COCO_DETECTOR = Path(os.environ.get("COCO_DETECTOR", ROOT / "models_extra" / "detector_coco.onnx"))
BUSY_MIN_VEHICLES = 3      # this many vehicles (score >= 0.5, >= 2% of frame) makes a scene "busy"
WEAK_BOX_SCORE = 0.5       # a default-detector box below this score that covers 2+ vehicles is a merged box
NAMED_VEHICLES = 4         # in busy scenes / every-car mode, the largest this many get a name; all others get a plain box
LEVEL_TEXT = {"full": "Make, model and year identified", "model": "Year uncertain: showing make and model only",
              "make": "Low confidence: showing make only. This may be a vehicle outside the 196 known classes, or not a car at all", "abstain": "Not confident enough to name this vehicle"}


def fetch_models() -> None:
    repo = os.environ.get("MODEL_REPO")
    required_files = (
        "classifier_int8_dynamic.onnx",
        "detector_finetuned.onnx",
        "classical.npz",
        "meta.json",
    )
    missing = [name for name in required_files if not (MODEL_DIR / name).is_file()]
    if missing and repo:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo, local_dir=str(MODEL_DIR))
        missing = [name for name in required_files if not (MODEL_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(missing)
            + ". Set MODEL_REPO to a Hugging Face model repository containing these files."
        )


PIPELINE, STARTUP_ERROR = None, None
try:
    fetch_models()
    PIPELINE = load_pipeline(MODEL_DIR)
except Exception as error:  # The UI explains the problem rather than crashing at boot.
    log.exception("Could not load models")
    STARTUP_ERROR = f"The models are not available: {error}"

# Multi-vehicle route: same classifier, stock COCO detector (car, bus, truck) that finds every vehicle.
# Used when the user ticks the box, or automatically for busy scenes. Not part of the reported evaluation.
MULTI_PIPELINE = None
if PIPELINE is not None and COCO_DETECTOR.exists():
    try:
        MULTI_PIPELINE = copy.copy(PIPELINE)
        MULTI_PIPELINE.detector = OnnxDetector(COCO_DETECTOR, class_ids=(2, 5, 7), confidence=0.5)
    except Exception:
        log.exception("Could not load the COCO detector; multi-vehicle mode is off")
        MULTI_PIPELINE = None


def big_enough(detections, image) -> list:
    """Detections covering at least 2% of the frame (the same size filter the pipeline uses)."""
    height, width = image.shape[:2]
    return [d for d in detections if (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]) >= 0.02 * width * height]


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def draw_scene(image, all_vehicles, named) -> np.ndarray:
    """Thin neutral boxes for every detected vehicle, then the coloured, labelled boxes for the named ones."""
    canvas = image.copy()
    thickness = max(1, min(image.shape[:2]) // 400)
    for d in all_vehicles:
        if all(iou(d["box"], v["box"]) < 0.5 for v in named):
            x1, y1, x2, y2 = [int(c) for c in d["box"]]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (190, 198, 210), thickness, cv2.LINE_AA)
    return annotate(canvas, named)


def inside(inner, outer, slack=0.1) -> bool:
    """True if most of box `inner` lies within box `outer`."""
    x1, y1, x2, y2 = inner
    ox1, oy1, ox2, oy2 = outer
    w, h = x2 - x1, y2 - y1
    return x1 >= ox1 - slack * w and y1 >= oy1 - slack * h and x2 <= ox2 + slack * w and y2 <= oy2 + slack * h


# Hugging Face ZeroGPU hardware expects one decorated function at startup. Inference itself is ONNX on CPU.
try:
    import spaces

    @spaces.GPU(duration=1)
    def _zerogpu_placeholder():
        return None
except ImportError:
    pass


def analyse(upload, multi=False):
    if STARTUP_ERROR:
        raise gr.Error(STARTUP_ERROR)
    if upload is None:
        raise gr.Error("Please upload a photo first.")
    try:
        image = load_image(upload)
        route = "multi" if multi and MULTI_PIPELINE is not None else "default"
        all_vehicles = MULTI_PIPELINE.detector(image) if MULTI_PIPELINE is not None else []
        seen = big_enough(all_vehicles, image)
        if route == "default" and len(seen) >= BUSY_MIN_VEHICLES:
            route = "busy"
        many = Options(max_vehicles=NAMED_VEHICLES)
        result = MULTI_PIPELINE.run(upload, many) if route != "default" else PIPELINE.run(upload)
        if route == "default" and result.vehicles and seen:
            first = result.vehicles[0]
            score = first["detector_score"]
            if score is not None and score < WEAK_BOX_SCORE and sum(inside(d["box"], first["box"]) for d in seen) >= 2:
                route = "busy"
                result = MULTI_PIPELINE.run(upload, many)
        if result.status == "bad_input":
            raise gr.Error(result.message)
    except BadInput as error:
        raise gr.Error(str(error)) from error
    except gr.Error:
        raise
    except Exception as error:
        log.exception("Inference failed")
        raise gr.Error("Something went wrong with this image. Please try another one.") from error
    gallery = [(np.stack([v] * 3, axis=-1) if np.ndim(v) == 2 else np.asarray(v), k) for k, v in result.intermediates.items()]
    quality = result.quality
    details = (f"**Brightness** {quality['brightness']:.2f} · **Contrast** {quality['contrast']:.2f} · **Sharpness** {quality['sharpness']:.0f}  \n"
               f"**Adaptive steps:** {', '.join(result.steps) or 'none needed'}  \n"
               f"**Time:** {sum(result.timings_ms.values()) / 1000:.2f} s ({', '.join(f'{k} {v:.0f} ms' for k, v in result.timings_ms.items())})")
    if result.geometry:
        g = result.geometry
        details += (f"  \n**Geometry:** aspect ratio {g['aspect_ratio']:.2f} · share of frame {g['frame_fraction']:.0%} · "
                    f"edge density {g['edge_density']:.2f} · solidity {g['solidity']:.2f} · extent {g['extent']:.2f}")
    if not result.vehicles:
        return image, ui.empty_state(result.message), gallery, details
    note = '<div class="vi-note">The detector found nothing, so a contour-based region was used.</div>' if result.status == "fallback_roi" else ""
    count = (f'<div class="vi-note"><b>{len(all_vehicles)} vehicles detected · {len(result.vehicles)} largest identified.</b></div>'
             if route != "default" else "")
    if route == "busy":
        note = ('<div class="vi-note"><b>Busy scene detected.</b> Every vehicle is boxed; the largest were identified separately with a '
                'general-purpose detector. This route was not part of the reported evaluation.</div>') + note
    elif route == "multi":
        note = ('<div class="vi-note"><b>Multi-vehicle detection.</b> Every vehicle is boxed; the largest were identified separately with a '
                'general-purpose detector. This route was not part of the reported evaluation.</div>') + note
    cards = count + note + "".join(ui.vehicle_card(v, i, LEVEL_TEXT) for i, v in enumerate(result.vehicles))
    picture = draw_scene(image, all_vehicles, result.vehicles) if route != "default" else annotate(image, result.vehicles)
    return picture, cards, gallery, details


def compare(upload):
    if STARTUP_ERROR:
        raise gr.Error(STARTUP_ERROR)
    if upload is None:
        raise gr.Error("Please upload a photo first.")
    try:
        image = load_image(upload)
    except BadInput as error:
        raise gr.Error(str(error)) from error
    rows = []
    if PIPELINE.classical_model is not None:
        box = classical.contour_roi(image)
        features = classical.extract_features(classical.crop(image, box), box, image.shape)
        probs = PIPELINE.classical_model.probabilities(features)
        rows.append(["Classical only", PIPELINE.hierarchy.classes[int(probs.argmax())]["name"], f"{probs.max():.1%}", "Answer always shown; no abstaining"])
    deep = PIPELINE.run(upload, Options(use_refinement=False, use_enhancement=False, use_tta=False, use_fusion=False))
    hybrid = PIPELINE.run(upload)
    for name, result in (("Deep learning only", deep), ("Hybrid", hybrid)):
        vehicle = result.vehicles[0] if result.vehicles else None
        rows.append([name, vehicle["label"] if vehicle else result.message, f"{vehicle['confidence']:.1%}" if vehicle else "-",
                     LEVEL_TEXT[vehicle["level"]] if vehicle else result.status])
    return rows


def benchmark_markdown() -> str:
    if not METRICS.exists():
        return "Benchmark results have not been generated yet."
    data = json.loads(METRICS.read_text())
    if "comparison" not in data:
        return "Benchmark results are incomplete."
    lines = [f"Measured on {data['test_images']} unseen test images.", "", "| Approach | Top-1 | Top-5 |", "|---|---|---|"]
    for name, key in (("Classical only", "classical_only"), ("Deep learning only", "deep_only"), ("Hybrid", "hybrid")):
        row = data["comparison"][key]
        lines.append(f"| {name} | {row['top1']:.2%} | {row['top5']:.2%} |")
    if "detection" in data:
        lines += ["", f"Detection: mean IoU {data['detection']['mean_iou']:.3f}, mAP@0.5:0.95 {data['detection']['map50_95']:.3f}"]
    if "efficiency" in data:
        efficiency = data["efficiency"]
        size = efficiency["models"].get(efficiency["deployed_classifier"], {}).get("size_mb")
        size_text = f"{size:.0f} MB" if size else "size not recorded"
        lines += ["", f"Deployed classifier: {size_text}. End-to-end CPU time per image: {efficiency['end_to_end_cpu']['mean_ms'] / 1000:.2f} s."]
    return "\n".join(lines)


LIMITATIONS = "- Newer cars and models outside the catalogue cannot be named exactly. The app then usually gives a make-only answer marked as low confidence, but that make can still be wrong, and a non-car object it detects is also given a make.\n- The training data is mostly pre-2013 and US-market, so accuracy is lower elsewhere.\n- Confidence is not a guarantee, and adjacent model years can look almost identical.\n- In busy scenes every detected vehicle is boxed and the four largest are named, using a general-purpose detector. That route was not part of the reported evaluation, and cars outside the catalogue (many Indian-market models, for example) usually get only a make, which can be wrong.\n- This is a research demo. It is not for law enforcement, insurance decisions or identifying people."


def hero_stats():
    """Headline numbers for the hero tiles, read from results/metrics.json (never typed in)."""
    if not METRICS.exists():
        return [], None
    data = json.loads(METRICS.read_text())
    stats = []
    hybrid = data.get("comparison", {}).get("hybrid")
    if hybrid:
        stats += [(f"{hybrid['top1']:.2%}", "Top-1 accuracy"), (f"{hybrid['top5']:.2%}", "Top-5 accuracy"),
                  (f"{hybrid['make_accuracy']:.2%}", "Make accuracy")]
    if "detection" in data:
        stats.append((f"{data['detection']['map50_95']:.3f}", "Detection mAP@0.5:0.95"))
    return stats, f"{data['test_images']:,}" if "test_images" in data else None


examples = [[str(p)] for p in sorted((ROOT / "examples").glob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
import inspect
_THEME_KEYS = set(inspect.signature(gr.themes.Base.set).parameters)
THEME = gr.themes.Base(primary_hue="amber", neutral_hue="slate", radius_size="lg",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]).set(
    **{f"{k}{s}": v for k, v in {
        "body_background_fill": "#0a0e15", "body_text_color": "#f3efe6", "body_text_color_subdued": "#9aa4b2",
        "background_fill_primary": "#111824", "background_fill_secondary": "#162030",
        "block_background_fill": "#111824", "block_border_color": "#243044", "block_label_background_fill": "#162030",
        "block_label_text_color": "#9aa4b2", "block_title_text_color": "#f3efe6", "border_color_primary": "#243044",
        "input_background_fill": "#162030", "input_border_color": "#243044", "color_accent": "#e8b339",
        "color_accent_soft": "#2a2414", "button_secondary_background_fill": "#162030",
        "button_secondary_text_color": "#f3efe6", "button_secondary_border_color": "#243044",
        "table_even_background_fill": "#111824", "table_odd_background_fill": "#162030", "table_border_color": "#243044",
    }.items() for s in ("", "_dark") if f"{k}{s}" in _THEME_KEYS})
with gr.Blocks(title="Vehicle Intelligence · P5 Team T10", theme=THEME, css=ui.CSS, head=ui.head_html()) as demo:
    gr.HTML(ui.header_html())
    stats, test_images = hero_stats()
    gr.HTML(ui.hero_html(stats, test_images))
    if STARTUP_ERROR:
        gr.HTML(ui.empty_state(STARTUP_ERROR))
    with gr.Row(equal_height=False):
        with gr.Column(elem_classes="vi-panel", min_width=320):
            gr.HTML('<div class="vi-section-title">1 · Upload a photo</div>')
            upload = gr.Image(type="filepath", label="Vehicle photo", sources=["upload", "webcam", "clipboard"], height=320)
            multi = gr.Checkbox(False, label="Multi-vehicle detection", visible=MULTI_PIPELINE is not None,
                                info="Switches on automatically in busy scenes.")
            run = gr.Button("Identify vehicle", variant="primary", size="lg")
            if examples:
                gr.Examples(examples, inputs=upload, label="Try an example: cars, rooms without cars, and a synthetic test image",
                            examples_per_page=12)
        with gr.Column(elem_classes="vi-panel", min_width=320):
            gr.HTML('<div class="vi-section-title">2 · Result</div>')
            output = gr.Image(label="Detected vehicles", interactive=False, height=320)
            cards = gr.HTML(ui.PLACEHOLDER)
    with gr.Tabs():
        with gr.Tab("Processing steps"):
            details = gr.Markdown()
            gallery = gr.Gallery(label="Intermediate results", columns=4, height="auto")
        with gr.Tab("Compare approaches"):
            compare_button = gr.Button("Compare on this image")
            table = gr.Dataframe(headers=["Approach", "Prediction", "Confidence", "Note"], interactive=False, wrap=True)
            gr.Markdown(benchmark_markdown())
        with gr.Tab("Limitations"):
            gr.Markdown(LIMITATIONS)
    gr.HTML(ui.FOOTER)
    run.click(analyse, [upload, multi], [output, cards, gallery, details], api_name="identify")
    compare_button.click(compare, upload, table, api_name="compare")
demo.queue(default_concurrency_limit=2, max_size=32)
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), max_file_size="15mb")
