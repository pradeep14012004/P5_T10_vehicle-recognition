# Fine-Grained Vehicle Recognition: Technical Report

Nov 6, 2026 · @design

## Summary

This system detects vehicles in photos and identifies make, model and year across the 196 Stanford Cars classes. a fine-tuned YOLO11n finds the vehicle, ConvNeXt V2 classifies it, and classical computer vision handles image quality, fallback regions and comparison features. When confidence is low, the app gives a less specific answer and labels it as low confidence.

**Final results:** the hybrid system reached **94.48% top-1 and 99.50% top-5 accuracy** on the 8,041 unseen test images, end to end, with detection mAP@0.5:0.95 of **0.925**. Quantization cut the classifier from 351.8 MB to 89.3 MB with no accuracy loss, and the deployed pipeline takes about 3.7 seconds per image on a CPU. Classical components did not raise headline accuracy; their contribution is described honestly in sections 6 and 7.

| Project detail | Value |
| --- | --- |
| Problem statement | P5: Intelligent Fine-Grained Vehicle Recognition and Analysis System |
| Team ID | T10 |
| USNs | 1RUA24CSE0306, 1RUA24CSE0291, 1RUA24CSE0262, 1RUA24CSE0300, 1RUA24CSE0296 |
| Live application | https://huggingface.co/spaces/lonewolfpradeep/vehicle-recognition-system |
| GitHub repository | https://github.com/pradeep14012004/P5_T10_vehicle-recognition |

**Key test results (8,041 unseen images):**

| Metric | Result |
| --- | --- |
| Hybrid top-1 / top-5 accuracy | 94.48% / 99.50% |
| Make accuracy | 98.46% |
| Mean IoU | 0.951 |
| AP@0.5 / mAP@0.5:0.95 | 0.990 / 0.925 |
| Deployed classifier size | 89.3 MB (from 351.8 MB) |
| End-to-end CPU time per image | 3.7 s mean, 5.4 s P95 |
| Answers given at full confidence | 93.3% of images, 97.4% correct |

## 1. Problem and objectives

Fine-grained recognition is hard because different classes can look almost the same. Two model years of one car may differ only in small details such as lights or trim. Real photos add poor lighting, occlusion, low resolution and unusual angles.

The system has five objectives:

1. Detect vehicles and draw bounding boxes.
2. Predict make, model, year and confidence across 196 classes.
3. Combine classical computer vision with deep learning in ways that can each be measured.
4. Respond clearly to low-confidence, unfamiliar or invalid inputs.
5. Run on a CPU within about 10 seconds per image in a public web application.

As the brief states, number-plate recognition, vehicle tracking and traffic-event analysis are out of scope.

## 2. Dataset

The Stanford Cars dataset contains 16,185 images across 196 classes, each with one annotated bounding box \[1\]. The official download is no longer available, so we used the [TorchVision-compatible Kaggle copy](https://www.kaggle.com/datasets/rickyyyyyyy/torchvision-stanford-cars). The pipeline confirmed the image and class counts before training.

| Split | Images | Used for |
| --- | --- | --- |
| Train | 7,329 | Fitting all models |
| Validation | 815 | Choosing epochs, settings, the detector, fusion weight and thresholds |
| Test | 8,041 | Final evaluation only |

The validation set is a stratified 10% split of the training images, using seed 42. **Test images were not used for training or for any model choice.**

The data has three limits that affect the results:

- Each image has one annotated box, so correct detections of background cars count as false positives when calculating mAP.
- There are no viewpoint labels, so viewing angle can only be tested through proxies.
- The classes are mostly older models, so newer and regional vehicles are outside the known classes.

## 3. System architecture

An uploaded image passes through checks and enhancement, detection, classification, fusion and a confidence decision.

![Inference pipeline, as implemented in pipeline.py](architecture.png)

## 4. Methods

The system uses classical vision to prepare and support decisions, and deep learning for the main detection and classification work. Each hybrid step can be switched off, which makes its effect measurable.

### 4.1 Classical computer vision

| Brief requirement | Implementation | Purpose |
| --- | --- | --- |
| Preprocessing and normalization | Brightness, contrast and sharpness measurement; gamma correction, bilateral filtering, CLAHE and unsharp masking; classifier input normalization | Corrects measured problems and leaves good images unchanged |
| Edge detection and contour analysis | Automatic-threshold Canny, morphological closing, contour scoring | Finds a candidate region without the detector |
| Colour and texture | 8x4x4 HSV histogram; uniform LBP histograms on a 3x3 grid | Describes appearance |
| Shape | HOG with 9 orientations on a 128 x 128 crop | Describes vehicle outline and structure |
| Region of interest | Contour region; GrabCut box refinement | Reduces background; refinement is rejected if IoU with the detector box is below 0.6 |
| Geometry | Aspect ratio, share of frame, edge density, solidity, extent, Hu moments | Describes vehicle proportions |

The classical-only model scales these features, reduces them to 256 PCA components, and applies softmax regression. It reached **19.63% validation top-1** on annotated crops, far above the 0.51% expected from random guessing. Its regularization setting, C = 0.01, was chosen on validation data.

### 4.2 Deep learning

**Detector:** YOLO11n at 640 px, chosen for its small size and CPU speed. We fine-tuned a one-class "car" model for 10 epochs. On the 815 validation images, it reached AP@0.5 of 0.989, mAP@0.5:0.95 of 0.919 and mean IoU of 0.947, against 0.851, 0.774 and 0.939 for stock COCO weights, so calibration selected the fine-tuned model. These are validation numbers from the Kaggle run logs (`results/kaggle_log_excerpts.txt`), not test results.

**Classifier:** ConvNeXt V2 Base at 384 px \[2\], fine-tuned from ImageNet-22k and ImageNet-1k weights. High input resolution helps preserve small details such as badges and lights. Training crops have random box jitter from -5% to +12% per side, because the live classifier receives imperfect detector boxes.

### 4.3 Hybrid integration

1. **Adaptive enhancement** runs before detection when the quality measurements call for it.
2. **Detected regions** are cropped for classification. The pipeline accepts up to three detections, but the fine-tuned detector was trained with one annotated car per image, so in practice it usually finds only the main car. For busy scenes the app switches to the stock COCO YOLO11n detector (score at least 0.5): it boxes every vehicle and names the four largest. This route was added after the evaluation and is not part of the reported results. On validation, the stock detector reached mean IoU 0.939 and mAP@0.5:0.95 of 0.774 (validation numbers from `results/kaggle_log_excerpts.txt`).
3. **GrabCut** refines each detector box when it agrees with the original. If the detector finds nothing, a contour region is used as a fallback.
4. **Log-linear fusion** combines the deep and classical probabilities, with the weight beta chosen on validation data. It stays 0 if classical features don't help:

```latex
\log p(c) = \log p_{\mathrm{deep}}(c) + \beta \log p_{\mathrm{classical}}(c) + \mathrm{const}
```

5. **Low-confidence handling** applies temperature scaling, then adds probabilities across years and models. The app reports the most specific answer that clears its validation-derived threshold:

| Level | Reported answer | Example situation |
| --- | --- | --- |
| Full | Make, model and year | One class is clearly likely |
| Model | Make and model, year uncertain | Adjacent model years compete |
| Make | Make only | Several models of one make compete |
| Abstain | Not recognised | The vehicle may be outside the 196 classes |

A contour fallback result is shown only if it reaches full-answer confidence. Otherwise, the app reports that no vehicle was found. With the thresholds chosen on validation, the make-level threshold is 0.0, so in practice the abstain level is never reached for a detected object: the least specific answer is a make, marked as low confidence.

## 5. Training

The final classifier was trained for 15 epochs on a Kaggle T4 GPU, at about 8 minutes per epoch. Its best epoch was chosen by validation top-1, without using test images. A separate 30-epoch run peaked at 94.72% validation top-1 at epoch 14 and then levelled off, which supports the shorter schedule. Training times, memory use and the 30-epoch run come from the Kaggle run logs (`results/kaggle_log_excerpts.txt`).

![Validation top-1 and top-5 by epoch for the final 15-epoch run, best epoch 12 marked](figures/training_curve.png)

| Training setting | Value |
| --- | --- |
| Epochs | 15, including 2 warm-up epochs, then a cosine learning-rate schedule |
| Batch size | 16 physical, 32 effective with 2-step gradient accumulation |
| Optimizer | AdamW, weight decay 0.05 |
| Learning rate | 5e-4 at the head, multiplied by 0.7 per earlier stage |
| Regularization | Mixup, CutMix, label smoothing 0.1, RandAugment, random erasing, stochastic depth 0.2 |
| Stability | Automatic mixed precision, gradient clipping at 1.0, EMA weights with decay 0.999 |

The first attempt at batch size 20 ran out of GPU memory, so the training script now tests batch sizes and keeps the largest that stays under 80% of GPU memory. In the final run's test steps, batch 20 used 92% and batch 16 used 74%, so batch 16 was used.

**Best epoch:** 12, with 95.09% validation top-1 and 99.02% top-5. Training took about 2 hours.

## 6. Evaluation

All final figures come from one run on the 8,041 unseen test images. Accuracy is end-to-end, using boxes from the system rather than annotations. A ground-truth-box result appears only as a labelled upper bound in the ablation table. Validation figures quoted in sections 4, 5 and 6.3 were used only to choose settings, and come from the Kaggle run logs (`results/kaggle_log_excerpts.txt`).

### 6.1 Classification: top-1 and top-5

| Approach | Top-1 | Top-5 | Mean per class | Make | Make + model |
| --- | --- | --- | --- | --- | --- |
| Classical only | 7.04% | 19.18% | 7.03% | 15.86% | 7.19% |
| Deep learning only | 94.53% | 99.48% | 94.38% | 98.48% | 94.96% |
| Hybrid | 94.48% | 99.50% | 94.36% | 98.46% | 94.93% |

The hybrid and deep-only results differ by 0.05 points, or 4 images, so they are effectively equal. Make accuracy of 98.5% shows that most remaining errors are between models of the same make. The classical-only model reached 19.63% on validation crops from annotated boxes, but only 7.04% here, because its contour-based region often misses the car.

![Top-1 and top-5 test accuracy for classical only, deep learning only and hybrid](figures/comparison.png)

### 6.2 Localization and detection: IoU and mAP

| Metric | Result |
| --- | --- |
| Mean IoU | 0.951 |
| Median IoU | 0.965 |
| Images with IoU of at least 0.5 | 99.81% |
| Images with no usable detection | 0.02% |
| AP@0.5 | 0.990 |
| mAP@0.5:0.95 | 0.925 |

### 6.3 Efficiency and optimization

We applied int8 quantization in ONNX Runtime, trying dynamic and static methods. The smallest model losing no more than 0.5 validation top-1 points is deployed. Dynamic int8 kept full accuracy and cut size by 3.9 times, but it ran 24% slower than 32-bit on the CPU. Static int8 was almost twice as fast, but its validation accuracy collapsed to 0.4%, so the selection rule rejected it. The deployed app uses ONNX Runtime without PyTorch.

| Model file | Size | Mean CPU time | P95 CPU time | Top-1, 500 validation crops |
| --- | --- | --- | --- | --- |
| Classifier, 32-bit | 351.8 MB | 1,246 ms | 1,276 ms | 95.60% |
| Classifier, dynamic int8 (deployed) | 89.3 MB | 1,550 ms | 1,601 ms | 95.60% |
| Classifier, static int8 (rejected) | 90.0 MB | 653 ms | 679 ms | 0.40% |
| Detector | 10.6 MB | 55 ms | 58 ms | Not applicable |

Classifier times are for one pass on one crop; the app makes two passes per vehicle for flip averaging.

| Comparison | Before optimization | Deployed |
| --- | --- | --- |
| Top-1 on the same 500 test images | 94.20% | 94.20% |
| Classifier size | 351.8 MB | 89.3 MB |
| End-to-end CPU time per image | Not measured | 3.7 s mean, 5.4 s P95 |

### 6.4 Robustness

Stress tests use a fixed random set of 300 test images, seven conditions and three severity levels. One image is 0.33 points here. Perspective warp stands in for viewing angle because the dataset has no viewpoint labels.

| Condition | Deep top-1, level 2 | Hybrid top-1, level 2 | Deep top-1, level 3 | Hybrid top-1, level 3 |
| --- | --- | --- | --- | --- |
| Clean images | 95.33% | 94.67% | 95.33% | 94.67% |
| Low light | 93.67% | 92.33% | 82.00% | 71.00% |
| Overexposure | 94.00% | 92.67% | 91.33% | 90.67% |
| Motion blur | 92.67% | 91.67% | 85.67% | 85.00% |
| Low resolution | 82.00% | 83.00% | 50.67% | 48.67% |
| Occlusion | 74.33% | 76.00% | 39.00% | 39.00% |
| JPEG damage | 92.67% | 91.67% | 87.67% | 87.67% |
| Perspective warp | 95.00% | 93.67% | 94.33% | 93.33% |

**Findings:**

- Severe low resolution and heavy occlusion cause the largest drops, to about 50% and 39%.
- Perspective warp barely changes accuracy, though it is only a proxy for real viewpoint change.
- The hybrid helps slightly with moderate occlusion and low resolution, by 1.7 and 1.0 points.
- The hybrid is 11 points worse in severe low light. There, its mean IoU fell to 0.608 against 0.779 for deep only, suggesting that enhancement and GrabCut moved the box.
- On clean images, hybrid IoU is 0.909 against 0.952, because GrabCut tightens boxes away from the annotated ones.

![Deep-only and hybrid top-1 under seven conditions, from clean images to severity level 3](figures/robustness.png)

Results for all three severities are in `eval_P5_T10.md`.

We also grouped real, uncorrupted test images by brightness, vehicle size and box shape. Box shape is only a rough proxy for viewing angle.

| Group | Value | Images | Hybrid top-1 |
| --- | --- | --- | --- |
| Brightness | Dark (below 0.25) | 700 | 92.71% |
| Brightness | Normal | 6,876 | 94.79% |
| Brightness | Bright (above 0.75) | 465 | 92.47% |
| Vehicle size | Small (under 150 px) | 1,443 | 93.90% |
| Vehicle size | Medium (150–300 px) | 3,343 | 94.14% |
| Vehicle size | Large (over 300 px) | 3,255 | 95.08% |
| Box shape | Narrow (under 1.4) | 835 | 94.25% |
| Box shape | Medium (1.4–1.9) | 3,467 | 95.39% |
| Box shape | Wide (over 1.9) | 3,739 | 93.69% |

On real images, dark and bright photos lose about 2 points against normal ones, and small vehicles about 1 point against large ones.

### 6.5 Confidence and thresholds

Temperature (0.756), fusion weight (0.02) and thresholds were all chosen on validation data. Thresholds target 97% accuracy for each answer level: 0.73 for a full answer and 0.70 for make and model. The expected calibration error on the test set is **0.0064**, so confidence closely matches accuracy.

| Answer level | Share of test images | Reported answer correct |
| --- | --- | --- |
| Make, model and year | 93.28% | 97.44% |
| Make and model | 1.01% | 77.78% |
| Make only | 5.71% | 80.39% |
| Abstain | 0.00% | Not applicable |

| Confidence threshold | Images answered | Accuracy on answered images |
| --- | --- | --- |
| 0.00 | 100.00% | 94.48% |
| 0.50 | 97.82% | 95.82% |
| 0.73 (used) | 93.28% | 97.44% |
| 0.90 | 86.94% | 98.40% |
| 0.95 | 81.59% | 98.87% |

![Images answered at full confidence against accuracy, for thresholds 0.00 to 0.95, with the 0.73 threshold marked](figures/confidence.png)

The make-level threshold came out at 0.0, so a detected vehicle always gets at least a make. Images with no detected vehicle get a no-vehicle message instead.

## 7. Ablation study and failure analysis

The ablation adds each component to the detector-crop baseline, and removes each from the full hybrid. It uses a fixed random set of 2,000 test images, where one image is 0.05 points.

| Pipeline | Top-1 | Top-5 | Change from deep-only |
| --- | --- | --- | --- |
| Full image, no detector | 92.80% | 99.30% | −1.20 pts |
| Ground-truth box (upper bound, not deployable) | 94.10% | 99.65% | +0.10 pts |
| Detector crop (deep-only baseline) | 94.00% | 99.55% | Baseline |
| Detector + adaptive enhancement | 94.15% | 99.60% | +0.15 pts |
| Detector + GrabCut refinement | 93.80% | 99.50% | −0.20 pts |
| Detector + flip averaging | 93.90% | 99.45% | −0.10 pts |
| Detector + classical fusion | 94.00% | 99.55% | 0.00 pts |
| Full hybrid | 93.75% | 99.55% | −0.25 pts |
| Hybrid without enhancement | 93.80% | 99.45% | −0.20 pts |
| Hybrid without refinement | 94.15% | 99.50% | +0.15 pts |
| Hybrid without flip averaging | 93.95% | 99.55% | −0.05 pts |
| Hybrid without classical fusion | 94.00% | 99.50% | 0.00 pts |

**Main findings:**

- **Detection matters most:** cropping with the detector adds 1.2 points over the full image. Detector boxes come within 0.1 points of annotated boxes.
- **The classical components don't improve clean accuracy:** every effect is within ±0.25 points, or 5 images, which is too small to separate from chance. GrabCut refinement is slightly negative, and enhancement slightly positive.
- **Their value lies elsewhere:** a detector-free fallback region, interpretable processing steps, and modest gains under occlusion and low resolution.

We did not change the configuration after seeing these test results, so the reported numbers stay unbiased.

### Failure cases

The system misclassified 444 of 8,041 test images. Most errors are between closely related vehicles:

| Error type | Errors |
| --- | --- |
| Same make, wrong model | 284 |
| Wrong make | 125 |
| Difficult exposure | 77 |
| Wrong year only | 35 |
| Small vehicle | 8 |
| Wrong or missed box | 2 |
| Blurred | 1 |

Types can overlap. The most frequent confusions were Audi TTS Coupe 2012 → Audi TT Hatchback 2011 (18 images), Audi S5 Coupe → A5 Coupe (11), and the near-identical Chevrolet Express and GMC Savana vans, confused 8 times in each direction. We examined three failures in detail:

| Case | True class | Predicted class | Confidence | Cause found | Possible fix |
| --- | --- | --- | --- | --- | --- |
| 1. Unusual viewpoint (`00353.jpg`) | Aston Martin V8 Vantage Convertible 2012 | Bugatti Veyron 16.4 Convertible 2009 | 0.20 | Overhead aerial view, rare in training data. The car fills a small part of the frame, and the detector box overlapped the annotation by only 0.18 IoU. Low confidence flagged the uncertainty, but the make-level answer was still wrong | Overhead-view augmentation; classify several candidate boxes |
| 2. Probable label error (`07470.jpg`) | GMC Savana Van 2012 | Chevrolet Express Van 2007 | 0.98 | The van clearly shows a Chevrolet badge, so the dataset label is probably wrong. The Express and Savana also share one body design | Clean noisy labels; group badge-engineered twins |
| 3. Wrong year (`02012.jpg`) | Bentley Continental GT Coupe 2012 | Bentley Continental GT Coupe 2007 | 1.00 | Rear view only, and model years differ in small styling details. The model was fully confident, so thresholds could not catch it | Year-specific training signal; more rear-view examples |

![The three analysed failure cases, with true class, predicted class and confidence](figures/failure_cases.png)

## 8. Application, limitations and ethics

The Gradio application is built to work on phones and laptops, and to run its models on a CPU.

| Brief output | Application feature |
| --- | --- |
| Accept an uploaded image | Upload, webcam and clipboard input |
| Detect and draw a box | Annotated image with a box on the main vehicle; in busy scenes, a box on every vehicle and labels on the four largest |
| Predict make, model and year | Result card, with closest matches |
| Display confidence | Result card, and classifier and detector confidence |
| Show intermediate results | Processing steps: enhancement, edges, contours, HOG, LBP, colour and geometry |
| Compare the three approaches | Compare tab, per image and from final test metrics |

The app gives a clear message for very small images and images with no detected vehicle, and a general error for files that aren't images. It also applies phone photo rotation and limits very large images.

**Limitations:**

- The system knows only 196 classes, so it can't name a newer or regional vehicle exactly. It can still misidentify one with high confidence, because confidence measures similarity to known classes. A non-car object that the detector picks up still gets a make-level answer; in our tests, two real indoor photos without cars got "No vehicle found"; a synthetic shapes image that the detector picked up got two "Chevrolet (model uncertain)" answers, at 12% and 10% confidence.
- Mostly older training data may reduce accuracy on photos from different places or eras.
- Viewing angle and occlusion results partly rely on synthetic tests.
- Accuracy may fall further on real conditions that the tests don't reproduce. The busy-scene route names only the four largest vehicles and was not part of the evaluation; cars outside the 196 classes usually get only a make, which can be wrong.

**Ethics and responsible use:** this is a research demo. It doesn't identify people, and it shouldn't be used for surveillance, law enforcement or insurance decisions. The app code doesn't save uploaded images. Dataset licences and model licences must be checked before any other use.

## 9. Conclusion

The system meets every mandatory P5 component, and reaches **94.48% top-1, 99.50% top-5 and 98.46% make accuracy** on 8,041 unseen test images, end to end. Its fine-tuned detector reaches mAP@0.5:0.95 of 0.925 and mean IoU of 0.951. Quantization made the classifier 3.9 times smaller without losing accuracy, and the full pipeline runs in about 3.7 seconds per image on a CPU.

The deep classifier carries almost all the accuracy. Classical components didn't measurably improve clean accuracy, and we report that directly. They still provide a detector-free fallback, visible processing steps, small gains under occlusion and low resolution, and the confidence back-off, which answers 93% of images fully at 97.4% accuracy.

The main weaknesses are heavy occlusion, very low resolution, near-identical vehicle twins and model years, and some label noise in the dataset. Next steps would be static quantization that keeps accuracy, a separate signal for model year, and cleaning mislabelled training images.

## 10. Team contributions

| USN | Member name | Part owned |
| --- | --- | --- |
| 1RUA24CSE0262 | Monish R | PENDING |
| 1RUA24CSE0306 | PENDING | PENDING |
| 1RUA24CSE0291 | PENDING | PENDING |
| 1RUA24CSE0300 | PENDING | PENDING |
| 1RUA24CSE0296 | PENDING | PENDING |

## References

1. J. Krause, M. Stark, J. Deng and L. Fei-Fei, "3D Object Representations for Fine-Grained Categorization," 4th IEEE Workshop on 3D Representation and Recognition, ICCV, 2013.
2. S. Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders," CVPR, 2023.
3. Ultralytics, YOLO11 documentation and software.
4. R. Wightman, PyTorch Image Models (timm) software.
