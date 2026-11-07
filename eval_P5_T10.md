1RUA24CSE0306,1RUA24CSE0291,1RUA24CSE0262,1RUA24CSE0300,1RUA24CSE0296

# Evaluation: P5 Fine-Grained Vehicle Recognition

Measured on 8041 unseen test images. Model: `convnextv2_base.fcmae_ft_in22k_in1k_384`, detector: `detector_finetuned.onnx`.

## 1. Classification performance

| Approach | Top-1 | Top-5 | Mean per class | Make | Make + model |
|---|---|---|---|---|---|
| Classical only | 7.04% | 19.18% | 7.03% | 15.86% | 7.19% |
| Deep learning only | 94.53% | 99.48% | 94.38% | 98.48% | 94.96% |
| Hybrid | 94.48% | 99.50% | 94.36% | 98.46% | 94.93% |

These are end-to-end results: boxes come from the model, not from the annotations.

## 2. Object localization (IoU)

- Mean IoU: 0.9508
- Median IoU: 0.9647
- Images with IoU of at least 0.5: 99.81%
- Images with no usable detection: 0.02%

## 3. Object detection (mAP)

- AP@0.5: 0.9897
- mAP@0.5:0.95: 0.9246

Stanford Cars annotates one car per image, so correct detections of background cars count as false positives.

## 4. Model efficiency

| Model file | Size | Mean CPU latency | P95 CPU latency |
|---|---|---|---|
| classifier_fp32.onnx | 351.8 MB | 1,246 ms | 1,276 ms |
| classifier_int8_dynamic.onnx | 89.3 MB | 1,550 ms | 1,601 ms |
| classifier_int8_static.onnx | 90.0 MB | 653 ms | 679 ms |
| detector_finetuned.onnx | 10.6 MB | 55 ms | 58 ms |

Deployed classifier: `classifier_int8_dynamic.onnx`. End-to-end CPU time per image: mean 3,689 ms, P95 5,379 ms.

On the same 500 test images, top-1 was 94.20% before optimisation and 94.20% for the deployed model.

## 5. Robustness (300 test images)

Clean: deep top-1 95.33%, IoU 0.952; hybrid top-1 94.67%, IoU 0.909.

| Condition / severity | Deep top-1 | Hybrid top-1 | Deep IoU | Hybrid IoU |
|---|---|---|---|---|
| low_light/1 | 95.00% | 94.67% | 0.952 | 0.896 |
| low_light/2 | 93.67% | 92.33% | 0.943 | 0.861 |
| low_light/3 | 82.00% | 71.00% | 0.779 | 0.608 |
| overexposure/1 | 95.00% | 94.00% | 0.949 | 0.901 |
| overexposure/2 | 94.00% | 92.67% | 0.945 | 0.884 |
| overexposure/3 | 91.33% | 90.67% | 0.933 | 0.866 |
| motion_blur/1 | 94.33% | 94.33% | 0.950 | 0.902 |
| motion_blur/2 | 92.67% | 91.67% | 0.927 | 0.887 |
| motion_blur/3 | 85.67% | 85.00% | 0.899 | 0.869 |
| low_resolution/1 | 93.33% | 93.33% | 0.950 | 0.906 |
| low_resolution/2 | 82.00% | 83.00% | 0.942 | 0.905 |
| low_resolution/3 | 50.67% | 48.67% | 0.896 | 0.873 |
| occlusion/1 | 87.00% | 88.00% | 0.912 | 0.880 |
| occlusion/2 | 74.33% | 76.00% | 0.824 | 0.813 |
| occlusion/3 | 39.00% | 39.00% | 0.688 | 0.700 |
| jpeg/1 | 94.67% | 94.00% | 0.952 | 0.903 |
| jpeg/2 | 92.67% | 91.67% | 0.949 | 0.896 |
| jpeg/3 | 87.67% | 87.67% | 0.936 | 0.888 |
| viewpoint_proxy/1 | 95.00% | 93.67% | n/a | n/a |
| viewpoint_proxy/2 | 95.00% | 93.67% | n/a | n/a |
| viewpoint_proxy/3 | 94.33% | 93.33% | n/a | n/a |

`viewpoint_proxy` is a synthetic perspective warp, because the dataset has no viewpoint labels. It moves the car, so IoU is not measured for it.

Hybrid top-1 on real, uncorrupted test images, by group. Box shape is only a rough proxy for viewing angle.

| Group | Value | Images | Top-1 |
|---|---|---|---|
| brightness | bright (above 0.75) | 465 | 92.47% |
| brightness | dark (below 0.25) | 700 | 92.71% |
| brightness | normal | 6876 | 94.79% |
| car_size | large (over 300 px) | 3255 | 95.08% |
| car_size | medium (150-300 px) | 3343 | 94.14% |
| car_size | small (under 150 px) | 1443 | 93.90% |
| box_shape | medium box (1.4-1.9) | 3467 | 95.39% |
| box_shape | narrow box (under 1.4) | 835 | 94.25% |
| box_shape | wide box (over 1.9) | 3739 | 93.69% |

## 6. Confidence and threshold analysis

Expected calibration error: 0.0064. Thresholds were chosen on validation data: {'full': 0.73, 'model': 0.7, 'make': 0.0}.

| Answer level | Share of images | Reported answer correct |
|---|---|---|
| full | 93.28% | 97.44% |
| model | 1.01% | 77.78% |
| make | 5.71% | 80.39% |
| abstain | 0.00% | n/a |

| Confidence threshold | Images answered | Accuracy on answered |
|---|---|---|
| 0.00 | 100.00% | 94.48% |
| 0.05 | 100.00% | 94.48% |
| 0.10 | 99.96% | 94.51% |
| 0.15 | 99.94% | 94.54% |
| 0.20 | 99.85% | 94.62% |
| 0.25 | 99.74% | 94.71% |
| 0.30 | 99.56% | 94.80% |
| 0.35 | 99.29% | 94.91% |
| 0.40 | 98.91% | 95.16% |
| 0.45 | 98.51% | 95.42% |
| 0.50 | 97.82% | 95.82% |
| 0.55 | 96.93% | 96.27% |
| 0.60 | 96.11% | 96.61% |
| 0.65 | 95.00% | 96.95% |
| 0.70 | 94.06% | 97.22% |
| 0.75 | 92.82% | 97.52% |
| 0.80 | 91.39% | 97.81% |
| 0.85 | 89.54% | 98.17% |
| 0.90 | 86.94% | 98.40% |
| 0.95 | 81.59% | 98.87% |

## Measured change

- Hybrid top-1 against deep only: -0.05 points.
- Hybrid top-1 against classical only: +87.44 points.
- Top-1 after optimisation, same images: +0.00 points.
- Classifier size after optimisation: 351.8 MB to 89.3 MB.
- Mean classifier CPU latency after optimisation: 1,246 ms to 1,550 ms.
