---
title: Vehicle Recognition T10
emoji: 🚗
colorFrom: yellow
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
python_version: "3.11"
---

# Fine-Grained Vehicle Recognition (P5, Team T10)

Identifies the make, model and year of a car in a photograph, across the 196 classes of the Stanford Cars dataset. A fine-tuned YOLO11n detector locates the vehicle, a ConvNeXt V2 classifier names it, and classical computer vision handles image quality, a detector-free fallback region and a set of handcrafted features that are fused with the deep model. Confidence is calibrated, and when the system is unsure it answers at a coarser level (make and model, or make only) instead of guessing the year.

Submission for Problem Statement P5, *Intelligent Fine-Grained Vehicle Recognition and Analysis System*, Computer Vision Hackathon 2026.

**Live demo:** the hosted link will be added here once deployment is complete. The app runs locally in two commands (see [Running the app](#running-the-app)).

## Results

Measured once on the 8,041 unseen images of the official test split. Accuracy is end to end: boxes come from the detector, not from the annotations. Full tables are in [`eval_P5_T10.md`](eval_P5_T10.md).

| Approach | Top-1 | Top-5 | Make |
|---|---|---|---|
| Classical features only | 7.04% | 19.18% | 15.86% |
| Deep learning only | 94.53% | 99.48% | 98.48% |
| Hybrid (deployed) | 94.48% | 99.50% | 98.46% |

| Metric | Result |
|---|---|
| Mean IoU / AP@0.5 / mAP@0.5:0.95 | 0.951 / 0.990 / 0.925 |
| Classifier size after int8 quantization | 89.3 MB (from 351.8 MB, same accuracy) |
| End-to-end CPU time per image | 3.7 s mean, 5.4 s P95 (Kaggle CPU) |
| Expected calibration error | 0.0064 |
| Full answers (make, model and year) | 93.28% of images, 97.44% correct |

The hybrid and deep-only systems differ by four test images, so the classical branch does not raise clean accuracy. Its measured effects (small gains under occlusion and low resolution, a loss in severe low light) are reported in the [ablation study](docs/ABLATION_AND_FAILURES.md) and the [technical report](docs/REPORT.md).

## How it works

![Architecture](docs/architecture.png)

1. **Input checks and adaptive enhancement.** Brightness, contrast and sharpness are measured; gamma correction, bilateral filtering, CLAHE and unsharp masking are applied only when a measurement calls for them.
2. **Detection.** A YOLO11n model fine-tuned on Stanford Cars boxes finds the vehicle. GrabCut tightens the box, and a Canny-and-contour region is used if the detector finds nothing.
3. **Classification.** ConvNeXt V2 Base at 384 px (int8 ONNX, flip averaging) and a classical model (HOG, LBP, colour and geometry features with softmax regression) score the crop. Their probabilities are combined by log-linear fusion with a weight chosen on validation data.
4. **Decision.** Temperature-scaled probabilities are rolled up from class to make-and-model to make, and the most specific level that clears its validation threshold is reported.

**Busy scenes.** Stanford Cars annotates one car per image, so the fine-tuned detector is built for one main vehicle. When a stock COCO YOLO11n detector sees three or more vehicles, or the main box is weak and covers several cars, the app boxes every vehicle and names the four largest. This route was added after the evaluation and is not part of the reported results. It needs `models_extra/detector_coco.onnx`; without that file the app runs in single-vehicle mode.

## Repository layout

| Path | Contents |
|---|---|
| `app.py`, `src/vehicle_id/ui.py` | Gradio application and its interface |
| `src/vehicle_id/` | Pipeline, classical vision, decision logic, metrics and evaluation code |
| `scripts/01_*` to `scripts/10_*` | Data preparation, training, export, calibration, evaluation, reports and figures |
| `notebooks/` | Kaggle notebooks used for the final run |
| `results/metrics.json` | Raw output of the final evaluation; every reported number is generated from it |
| `results/kaggle_log_excerpts.txt` | Log lines behind the validation-only numbers (detector choice, quantization, training) |
| `eval_P5_T10.md` | Comparative evaluation in the order required by the problem statement |
| `docs/DESIGN.md` | Design document |
| `docs/architecture.png` | System architecture diagram |
| `docs/ABLATION_AND_FAILURES.md` | Ablation study and failure-case analysis |
| `docs/REPORT.md`, `docs/figures/` | Technical report and its figures |
| `tests/` | Unit and end-to-end tests on synthetic images |
| `deploy/` | Hugging Face Space configuration |

## Running the app

Model weights are not stored in Git because of their size. Place the four files produced by the training pipeline (`classifier_int8_dynamic.onnx`, `detector_finetuned.onnx`, `classical.npz`, `meta.json`) in `models/`, or set `MODEL_REPO` to a Hugging Face model repository that holds them.

```bash
pip install -r requirements.txt
python app.py            # http://localhost:7860
```

Or with Docker:

```bash
docker build -t vehicle-id . && docker run -p 7860:7860 vehicle-id
```

The optional busy-scene detector is the stock YOLO11n checkpoint exported with the same settings as `scripts/05_export.py`. Export it in a separate environment so that PyTorch stays out of the app's dependencies:

```bash
pip install ultralytics==8.4.203
yolo export model=yolo11n.pt format=onnx imgsz=640 opset=17 simplify=False
mkdir -p models_extra && mv yolo11n.onnx models_extra/detector_coco.onnx
```

## Reproducing the results

The dataset is the Kaggle copy recommended in the [torchvision issue tracker](https://github.com/pytorch/vision/issues/7545), `rickyyyyyyy/torchvision-stanford-cars`, which includes the labelled test annotations (8,144 training and 8,041 test images). The original Stanford download is no longer available.

Import `notebooks/p5_kaggle_train.ipynb` into a Kaggle notebook with a GPU and follow its first cell, or run:

```bash
bash scripts/run_kaggle.sh /kaggle/input/<dataset-folder>
python scripts/08_report.py --team-id T10 --usns <USN1,USN2,...>
```

| Step | Script | Data used |
|---|---|---|
| Manifest and 10% stratified validation split (seed 42) | `01_prepare_data.py` | Training annotations |
| Classical baseline | `02_train_classical.py` | Training set, tuned on validation |
| Classifier fine-tuning (resumable) | `03_train_classifier.py` | Training set, selected on validation |
| Detector fine-tuning | `04_train_detector.py` | Training set, selected on validation |
| ONNX export and int8 quantization | `05_export.py` | Validation images for calibration |
| Detector, temperature, fusion weight, thresholds | `06_calibrate.py` | Validation only |
| Final evaluation, run once | `07_evaluate.py` | Test set |
| Reports and figures | `08_report.py`, `10_figures.py` | `results/metrics.json` |

The test set was not used for any choice. Before a submission, run `python scripts/preflight.py`.

## Tests

The tests use synthetic images and need no GPU or model download:

```bash
pip install pytest scikit-learn
python -m pytest tests
```

## Limitations

- The model knows 196 classes, mostly US-market vehicles up to 2012. Newer or regional cars cannot be named exactly; the app falls back to a make-only answer marked as low confidence, and that make can still be wrong.
- The make-level threshold chosen on validation is 0.0, so any object the detector picks up receives at least a make. When the detector finds nothing, the contour fallback is reported only if it reaches full confidence; otherwise the app returns "No vehicle found".
- Severe low light is the weakest condition: the enhancement step amplifies noise and the hybrid loses 11 points against the deep model.
- Viewpoint robustness is measured with a synthetic perspective warp, because the dataset has no viewpoint labels.
- Files that are not images are rejected by the upload component with a general error rather than a custom message.

## Responsible use and licences

This is a research demonstration. It must not be used to identify people, for surveillance, or for law-enforcement or insurance decisions. Uploaded images are processed only to return a result, and the app does not store them.

Check the licences before any reuse: Stanford Cars is provided for research purposes, the ConvNeXt V2 pre-trained weights are released under CC BY-NC 4.0, and Ultralytics YOLO is licensed under AGPL-3.0.

## Team

Team T10, School of Computer Science and Engineering, RV University.

| USN | Name |
|---|---|
| 1RUA24CSE0262 | Monish R |
| 1RUA24CSE0306 | |
| 1RUA24CSE0291 | |
| 1RUA24CSE0300 | |
| 1RUA24CSE0296 | |

## Citation

J. Krause, M. Stark, J. Deng and L. Fei-Fei. "3D Object Representations for Fine-Grained Categorization." 4th IEEE Workshop on 3D Representation and Recognition, ICCV 2013.

<!-- Refresh Space after classifier compatibility conversion -->
