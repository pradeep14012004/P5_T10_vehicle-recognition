# Hackathon Technical Solution Report

*Intelligent Fine-Grained Vehicle Recognition and Analysis System*

> **Conversion note:** This Markdown preserves the extractable text from all 11 pages of the supplied PDF, including headings, paragraphs, tables, code commands, checklist, and AI declaration. Page boundaries are marked below. Original graphical figures and charts are not embedded as images in Markdown; their captions and surrounding explanatory text are retained where present.

<!-- Page 1 -->

Hackathon                   Technical               Solution              Report
                Intelligent Fine-Grained            Vehicle Recognition         and    Analysis System

 Field                                                          Details
 Problem ID                                                     P5
 Problem Title                                                  Intelligent Fine-Grained   Vehicle Recognition   and
                                                                Analysis System
 Team Name                                                      T10  (Parakram)
 Team Members                                                   Monish   R (1RUA24CSE0262); Pradeep         M
                                                                Doddakaragi (1RUA24CSE0306); Omkar Suresh             Naik
                                                                (1RUA24CSE0291); Piyush        Poonacha
                                                                (1RUA24CSE0300); Pasumarty         Krishna Tanish
                                                                (1RUA24CSE0296)
 GitHub   Repository                                            https://github.com/pradeep14012004/P5_T10_vehicle-r
                                                                ecognition
 Live Demo    / Deployment                                      https://huggingface.co/spaces/lonewolfpradeep/vehic
                                                                le-recognition-system

Where    the  numbers     come    from.   Every   figur e  in this  report   is taken   from  results/metrics.json,
eval_P5_T10.md or        results/kaggle_log_excerpts.txt in                 the repository  .
                    1. Problem Understanding                         & Solution Strategy
CORE TECHNICAL CHALLENGE
Naming    a car model from one photo           is hard  because cars look      similar. Every     car has wheels, a grille
and  tail lights;  what makes one model different             from   another is a detail, like the shape of a lamp          a
badge or the     model    year. Some     classes are the     same   vehicle. The Chevrolet Express and            the GMC
Savana are one van        sold  under   two   badges. At the      time photos of a single class change a lot, with
viewpoint, light and      background. So      the system has to      ignore that kind     of change and      still notice the
first. The  job   is to find  the car   and   then  give its make, model and         year from the 196      Stanford    Cars
classes. Four things worried       us most:
●   Sibling    confusion: the same model in          different years, different body      styles, and   badge-engineered
    twins.
●   Poor localisation: a small car in         a big  frame, clutter, or other vehicles in      view.
●   Poor     image     quality:    low   light,   over-exposure,      blur,  low    resolution,   occlusion     and   heavy
    compression      all remove fine detail.
●   Overconfident       answers: a model that is wrong            but certain   is worse than     one that admits doubt,
    especially    for cars outside the 196     classes.
We also   worked    under real limits. Training      had   to fit on one free Kaggle GPU, the demo           had   to run  on
a free CPU-only      host, and   the whole thing     had  to be built in  about 24    hours.
PROPOSED METHODOLOGY
The system has three stages. First the system measures the photo                  and  corrects the errors that are truly
present.   Then   the  system    uses   a  fine‑tuned    YOLO11n       detector to    locate the   car. The    system uses
GrabCut     to tighten   the  bounding     box. Second      the  system    applies  a ConvNeXt        V2  classifier   and  a


                                                              1

---

<!-- Page 2 -->

classical model that relies on        features; the system blends the outputs from these two                 models. Third      the
system    adds the    calibrated    probabilities, along      the make, model and          year hierarchy. When         the exact
answer    is  uncertain     the  system     supplies    a   broader    answer.     The   system     uses   fixes  by   design.    A
correction    that saves a photo      can  damage a well‑lit photo        so  a static pipeline would       fail.
                       2. Technical Architecture & Implementation

No  raw   code  is included    in this  report.  The  modules    live in src/vehicle_id/ (classical.py for                  quality
metrics, enhancement,      GrabCut     and  handcrafted     featur es; decision.py for        temperatur  e scaling,    fusion  and
thresholds;   pipeline.py for        end-to-end    ONNX inference;       corruptions.py and           evaluation.py for          the
robustness   and  metric code),    with number  ed   scripts  scripts/01–09 and         the  Gradio   interface  in app.py.






























                         Figure 1:   Architecture    of the proposed    hybrid  pipeline   (three stages).

We   used   the  Stanford     Cars dataset, which        has   16,185    images across       196  classes. The      official   split
gives   us  8,144    training   images     and   8,041    test  images.    From     the  training   set  we   took   10%    of  the
images—815        in  total—as     a validation     set.  We made sure        the validation     set  was balanced       by   class
using   a seed  of 42. Every     decision, including       the number of epochs the detector used, the temperature,
the fusion    weight and     the thresholds was based          on  performance, on       this validation     set. The 8,041     test
images were never used          during   training   or tuning. They      were only    used   once for the evaluation.
2.1  CLASSICAL ALGORITHMIC BASELINE
Algorithms utilized. For each            photo   we calculate the brightness the contrast, the sharpness using                  the
variance of the Laplacian         and   the amount of the image that is very             dark  or very   bright. We can      apply
fixes:   gamma      correction,    a  bilateral    filter, CLAHE        on   the  LAB      lightness    channel    and    unsharp
masking.     After that GrabCut helps tighten           the detector box. If the detector doesn’t find              anything    we


                                                                 2

---

<!-- Page 3 -->

use a Canny      edge detection     and   contour search     to  get a fallback    region. For the classifier we use HOG
to capture shape, uniform LBP for texture a hue-saturation-value histogram and                          simple box     geometry.
Then   PCA reduces        these   features to    256   dimensions. A       softmax     regression     with   C = 0.01. Chosen
based   on  validation. Scores them. This approach             reached    19.63% top-1      accuracy, on     validation.
Adaptive     selection logic. I      notice that each      fix switches on      when    a measurement asks for a fix           and
the   strength    of   a  fix   follows     the   photo.A     sharp    well-exposed       contrasty     image     goes    through
unchanged.
                                         Table   1: Rules   for adaptive   enhancement

 Condition measured                                                Action taken
 Median   brightness b   < 0.25                                    Gamma with      γ = ln(0.45)/ln(b), limited  to 0.3–1.0;
                                                                   bilateral filter (diameter 5, σ colour 30, σ space 5);
                                                                   CLAHE
 b > 0.80, or over 40% of pixels above 240                         Gamma with      γ = 1.6; CLAHE
 Contrast below 0.14                                               CLAHE on      the L channel (clip   limit 2.0, 8 × 8 tiles)
 Laplacian   variance below 40                                     Unsharp   mask   (Gaussian   σ = 1.5, weights 1.5   and  −0.5)
 None of the above                                                 Image left unchanged

I choose the gamma value so           that the median      brightness lands near 0.45.I keep          the GrabCut box        when
the  GrabCut     box   overlaps the detector box         with   an  IoU of at least 0.6.This stops a segmentation               for
example     one   that  swallows      a  shadow from pulling          the GrabCut box        off  the  car.Before     cropping    I
widen   the box    by  eight percent, on     every   side.

2.2  DEEP LEARNING PIPELINE
Model     architecture.The         classifier    is  ConvNeXt        V2    Base.    This    model     has   about    89   million
parameters.     It  takes   an   input   of  384    by   384   pixels.   It  was   pre-trained     using   FCMAE        and   then
fine-tuned    on  ImageNet-22k        and   ImageNet-1k.       We chose      the   classifier  for  three   reasons.    The high
resolution   keeps badges and        lamp    clusters visible. The classifier is purely          convolutional. This means
it exports cleanly      to ONNX for use on          CPU. The large-scale pre-training            helps when      each   class has
about 37    training   images. The detector is YOLO11n. This model has about 2.6                        million   parameters. It
was fine-tuned      on  the training   boxes with     one class the class called       "car". On    validation    this improved
AP50    from    0.851    (stock   COCO)      to  0.989. It   also   improved     mAP50-95        from   0.774,    to 0.919. The
detector also    removed     the 0.37% of images where the COCO model found                      no  car.
Loss functions and training. We trained                the model using       cross‑entropy     loss on   targets and    we added
label  smoothing      of  0.1,   Mixup     and   CutMix     to  the   training   process.    The    training   strategy    used   a
layer‑wise decay      of 0.7   which    means that the early       layers of the network       changed     more slowly     during
training.   We   also  kept   a running     average    of the weights with        a moving     average decay       of 0.999    and
used  that averaged      model for validation        and  for exporting     the final model. During        test time the image
crops   come from      detector boxes. Those boxes are never exactly                 the same as the annotated          boxes so
we jittered   each   training   box   by  minus five percent to       plus twelve percent to       make training      and  testing
more    similar.  The training      lasted  for fifteen    epochs.    We used      AdamW       with   a learning    rate   of  five
times   ten  to  the   negative    fourth   a  weight decay       of 0.05,    a batch   size   of  sixteen   and   two   gradient
accumulation      steps. We ran      training, with    precision    and   clipped   gradients to     a maximum of one. The
validation   top ‑one accuracy      reached    its value, 95.09     percent in   epoch    twelve.




                                                                 3

---

<!-- Page 4 -->

Figure 2:   Training   loss and  validation   accuracy   of the classifier over   15 epochs   (best  epoch  12).

However, our first attempt for 30            epochs stopped       working    at the evaluation      stage, therefore our last
model was obtained         using   15  epochs on     the account of the other team member. We chose batch                     size
16  (74% GPU memory) with             validation   for actual forward      and   back   propagation     passes.
2.3  HYBRID INTEGRATION & ADAPTIVE MECHANISM
Fusion     strategy.     The    classical    and   deep    components        interact    in  three   aspects.    The    adaptive
enhancement       chooses     what    the   detector    and   classifier    observe.    GrabCut      and   the   contour-based
fallback   control the bounding        box. In   the end, the probabilities from the two             systems are fused      using
log-linear method, log        p = log   pdeep    + β · log    pclassical. The best value of the parameter was found
through    validation: the     choice   of β = 0.02     improved      validation   results from 94.60% to          94.48%, and
any  larger values decreased         performance (93.37%           for  β  =  0.3). The    classical model is correct only
once out of five, so     it can  be used    merely   as a tie-breaker.
Decision logic.      The classifier logits are divided          using   the temperature T = 0.756         (fit to minimise the
negative log-likelihood        in validation). The 196        fused   probabilities are added       up   for make-and-model
and  make probabilities, and        the response is scored       based   on:
●    ≥ 0.73   for the highest level: make, model, and            year;
●    otherwise, probability       for make and      model ≥ 0.70: make and         model, but year uncertain;
●    otherwise: make alone, considered            low probability.
The make threshold        turned    out to  be 0.0, therefore, the thresholds will not cause the system to                abstain
from making       a prediction. For      any   detections made by        the model, at least one make will always be
made. In    the absence of a detection         by  the detector where the fallback           zone is also    not confident, the
application    displays the output “No        vehicle found”. The application           also  validates what is provided. It
discards   images with      sizes less than     32  pixels using     our custom message (and           other non-image files
with   a generic    message),     resizes images with        very   large sizes    to  at most    1,600   pixels, and     applies
phone (EXIF) rotation.
There   is  also  the busy    scene path     in  the  last application.     This path     is capable of recognizing          busy
scenes on    its own, drawing       a bounding      box   around   each   vehicle and     identifying    up   to three vehicles
by  their   name,    based    on  the   general    COCO      detector.   This   path   was    not  evaluated;     therefore,    no
accuracy    metrics listed    below are related      to it.







                                                                4

---

<!-- Page 5 -->

3. Experimental Benchmarks & Evaluation
3.1  QUANTITATIVE BENCHMARKS
The measures used         by   the template for the enhancement quality                 (PSNR, SSIM, and           LPIPS) have no
significance in      a recognition      task,   hence    the reason     we opted     for measures suitable for this specific
task. The test results include all 8,041           test images, with       boxes provided       by  our own     detector.
                                  Table    2: Classification    results on  the  8,041  test images

 Approach /           Top-1   (%)         Top-5   (%)         Mean per           Make (%)              Make +             Latency
 model                                                        class (%)                              model (%)             (CPU)
 variant
 Classical                7.04               19.18               7.03               15.86                7.19          Not measured
 baseline                                                                                                                separately
 Deep   learning         94.53               99.48               94.38              98.48               94.96             1,550  ms
 only                                                                                                                    (classifier
                                                                                                                            alone)
 Proposed                94.48               99.50               94.36              98.46               94.93             3,689  ms
 hybrid                                                                                                                    (whole
 system                                                                                                                   pipeline)















                   Figure 3:    Accuracy     of the  classical, deep   and  hybrid   approaches     on  the test set.

                                         Table   3:  Localisation,   model    size and  speed

 Detector / data                  Mean IoU                  IoU ≥ 0.5   (%)                 AP50                    mAP50-95
 COCO YOLO11n,                       0.939                       99.39                      0.851                       0.774
 validation
 Fine-tuned                          0.947                       99.63                      0.989                       0.919
 YOLO11n,
 validation
 Fine-tuned                          0.951                       99.81                      0.990                       0.925
 YOLO11n, test

 Model file                        Size (MB)                 Mean (ms)                    P95   (ms)                 Val. top-1
 Classifier, fp32                    351.8                       1,246                      1,276                     95.60%
 Classifier, dynamic                  89.3                       1,550                      1,601                     95.60%
 int8  (deployed)
 Classifier, static int8              90.0                        653                        679                       0.40%
 (rejected)
 Detector, fine-tuned                 10.6                         55                         58                         —




                                                                   5

---

<!-- Page 6 -->

Analysis. These two         approaches are effectively         coupled. The hybrid         approach     has 0.05    less in  top-1
(only  four   images     out  of  8,041) and     0.02   more in    top-5   results, and    both  are 87    points ahead     of the
classical one. This model achieved            19.63% in      validation    and  7.04% in     test sets. The reason       is yet to
be found; however, it might have been              trained   using   annotated-box       crops while the test pipeline uses
enhanced     detector    crops.   All   the  mistakes     fall into   the  correct brand      category    (accuracy     98.46%).
Performance      is  actually   an  issue   here. The dynamic int8          reduced     the  classifier   in  four times (from
351.8   to 89.3   MB), while maintaining           the same accuracy        but running     24% slower compared           to  fp32
on  Kaggle     CPU. It has been       retained   because of the importance of the smaller file on                 a free hosting
service.   The   static   int8  is  2.4  times    faster  than   the   previous    one   but   had   0.40%     accuracy,    likely
because of fixed      activation   range being     inappropriate for ConvNeXt normalisations.
3.2  VISUAL OUTPUTS & INTERMEDIATE REPRESENTATIONS
Fig. 4  illustrates the process that the implemented              pipeline applies on       a test image, just like it would
be  shown      in  the  "Processing      Steps"    window       of  the  application.     In   this  example,     the   image    is
letterboxed    with   black   bars, resulting    in a low average image brightness of only              0.19. Thus, the image
was   marked     as  too  dark   by  the   quality   check, with     the applied     gamma correction         of 0.48, bilateral
filtering,  and   CLAHE. This is one of our own               weaknesses: since the brightness check               considers the
entire image, the padding         could   have activated      a process that was actually         unnecessary      for the car. It
can  be   fixed   by  checking     the  brightness inside the        detected    bounding     box.   Fig. 4(f)-(h)     depict the
results  of   classical   feature   extraction.    HOG      highlights    the   body    contours    and   wheel    arches, LBP
identifies surface texture, while the hue histogram captures the sky, vegetation                      and  car paint.



















                      Figure 4:   Intermediate    outputs  of the  pipeline  for a letterboxed   test image.
3.3  ABLATION STUDY
We measured       the effect of adding      or removing      each  component on        a fixed  random subset of 2,000         test
images. The baseline is the deep          model on     the detector crop.
                                       Table   4: Ablation   study  on  2,000  test images

 Pipeline variant                          Top-1   (%)                      Top-5   (%)                   Change (points)
 Full image, no   detector                    92.80                            99.30                            −1.20
 Ground-truth    box  (upper                  94.10                            99.65                            +0.10
 bound, not deployable)



                                                                 6

---

<!-- Page 7 -->

Detector crop (deep-only                       94.00                              99.55                              0.00
 baseline)
 Detector + adaptive                            94.15                              99.60                             +0.15
 enhancement
 Detector + GrabCut                             93.80                              99.50                             −0.20
 refinement
 Detector + flip   TTA                          93.90                              99.45                             −0.10
 Detector + classical fusion                    94.00                              99.55                              0.00
 Full hybrid                                    93.75                              99.55                             −0.25
 Hybrid   without                               93.80                              99.45                             −0.20
 enhancement
 Hybrid   without refinement                    94.15                              99.50                             +0.15
 Hybrid   without TTA                           93.95                              99.55                             −0.05
 Hybrid   without classical                     94.00                              99.50                              0.00
 fusion

Localisation     is the one element which            makes a difference. The use of classifying                 the crop    rather than
the  whole     image     gives   a  1.2   point    improvement,        and   the   crop    itself  is 0.1   points away       from the
annotation     box, meaning       there is not much        room for improving          the detector. All other elements affect
accuracy     in the range of -0.25        to  +0.15    points. This is equivalent to          three to    five images from 2,000
images, which       is too   insignificant to      be measured. On         the undegraded        images, the classical elements
do  not make any       difference.
3.4  ROBUSTNESS AND CONFIDENCE
For each    of these 300       test images, we applied          six  types of degradation        to  varying    degrees, along      with
an  artificial perspective distortion          in lieu   of view-point variation, which            is absent from the database.
For the clean     images, the deep        classifier achieved       a 95.33% accuracy, while the hybrid               got 94.67%.





















                Figure 5:    Top-1   accuracy    under   six degradations    at three  severities   (300  test images).

                               Table   5:  Robustness     at the highest   severity  (300   test images)

 Condition, severity           Deep top-1     (%)         Hybrid top-1     (%)            Deep IoU                   Hybrid IoU
 3
 Low light                            82.00                      71.00                       0.779                       0.608
 Over-exposure                        91.33                      90.67                       0.933                       0.866

                                                                    7

---

<!-- Page 8 -->

Motion    blur                      85.67                      85.00                     0.899                      0.869
 Low resolution                      50.67                      48.67                     0.896                      0.873
 Occlusion                           39.00                      39.00                     0.688                      0.700
 JPEG compression                    87.67                      87.67                     0.936                      0.888
 Viewpoint                           94.33                      93.33                       —                          —
 (perspective warp)

These    results   vary,   and   we   report    them    as  is. The    hybrid    performs     better   slightly    in  the  case   of
moderate occlusions (+1.0           and   +1.7   points at severities 1      and  2) and   moderate low resolution          (+1.0   at
severity   2). It remains within        1.4   points from      the deep    model     when     there is blur or over-exposure.
Obviously, the hybrid         suffers in   the case of severe low light, dropping              by  11  points. In   addition, both
models perform badly          if two-thirds of the car are occluded            or if an   image becomes very          small, with    a
size of just a few dozen         pixels, because class discriminative features are not present any                   more in    such
cases. Synthetic viewpoint shift leads to             a drop    of 1.0   point for the deep      model and      1.3  points for the
hybrid    at  the   highest    severity    level.   Real-world      un-degraded       images     are   more    difficult   for  both
models in     the case of dark      and   bright photographs (by         approximately       two   points compared        to normal
ones) and    small cars (by      approximately      one point compared         to  large ones).
Confidence.       After    calibration    through     temperature      scaling,    the   estimated     calibration    error   in  the
testing  set is 0.0064. In     simpler language, if the system claims a confidence of 90%, the probability                          of
that being    correct is almost 90%. The graded              answer subsequently         distinguishes between         reliable and
unreliable answers.
                                   Table   6:  Distribution   of answer   levels  on  the test set

 Answer level                                         Share of images (%)                           Answer correct (%)
 Full: make, model and      year                                93.28                                        97.44
 Make and    model, year uncertain                              1.01                                         77.78
 Make only, low confidence                                      5.71                                         80.39
 Abstain                                                        0.00                                          —
















Figure 6:    Coverage    and  accuracy    as the confidence    threshold   rises (at 0.9:  86.94%    answered,    98.40%    correct).
                                    4. Failure Modes & Limitations
OBSERVED FAILURE CASES
Hybrid    has   failed   to  classify   444   images out of the         8,041   images.     Errors   have    been   automatically
classified   by  us   (multiple tagging       is possible).     From those,      284   images were correctly          classified   as
make    but  incorrectly     as  model (64.0%),        125   images incorrectly        classified    as make (28.2%) and           35

                                                                  8

---

<!-- Page 9 -->

images    correctly    as  model     but  incorrectly     as  year   (7.9%).    Harsh    lighting    has   happened     77   times
(17.3%), small car        – 8  times, wrong      or no   box   –  2 times, blur –     1  time. Most frequent mistakes are
made    between     the  most   similar: between        Audi TTS      Coupe and       S5   Coupe for     A5   Coupe 11      times,
between     Chevrolet     Express     Van   and   GMC      Savana     16   times   (8  for   both   directions)    and   between
Dodge    Caliber    Wagon     2012    for  2007    version   11   times. Figure 7      illustrates three mistakes made for
different reasons.














 Figure 7:   Three   failure cases from    the test set: (a) overhead    view, (b)  twin  vans  and  a probable   label  error, (c)
                                                       wrong   model   year.

●    (a)  Unusual      point-of-view.       The    top  view    of   an  Aston     Martin    V8   Vantage     Convertible      was
     misclassified    as Bugatti Veyron        with   a confidence of 0.20. This top          view is rarely    seen   during   the
     training   process, the     vehicle itself is tiny     and   is surrounded     by   a blurry   background      picture, and
     there is a detection     box   overlap   with   the annotation     box   with  an  IoU of 0.18.
●    (b)  Most likely      a  label mistake. The vehicle was classified               as a GMC Savana 2012, but it was
     predicted    as a Chevrolet Express 2007           with   a confidence level of 0.98. From the image, it is clear
     that the car is sporting     a Chevrolet logo       and  therefore must be an        error in  labeling.
●    (c)  Model     year   mistake.     A   2012    Bentley     Continental     GT    Coupe     seen   from behind      has   been
     mistakenly     classified   as the 2007     model with      certainty   1.00. There are slight differences between
     these models in      their lights and    bumpers, and      hence there can      be no   threshold    that would     help.

ROOT CAUSE & MITIGATION
The main     weakness of our robustness tests is very             low light. In    the beginning      of an  image captured       in
very  dark   conditions, the      signal   of interest    is concentrated      in  low   intensity   values, and     the noise is
about the same order of magnitude. Gamma correction                      with   γ much    less than    1 extends the range of
intensity   values to   include all the possible ones, thus amplifying               both   signal and    noise. Bilateral filter
with  very   small fixed     kernel size     cannot    remove     this noise, and     CLAHE further amplifies the local
contrast. This leads to      a noisy   image. In    such   an  image, the detector creates a box          of poor quality     (IoU
= 0.608    vs. IoU = 0.779        when    no   enhancements are used).          The    classifier trained    on   other types of
images does      not  detect   any   objects because there are no          examples in      the training    data. The possible
solutions are as follows: tune the strength             of denoising     and  gamma correction         based   on  the estimated
amount of noise; create         boxes in     both  original and     enhanced     images and      choose the better one; and
use artificial low-light images for classifier training. As for the rest of the failure cases, we would                        add
overhead     view    augmentation, clean        or  merge    the   label-twin    classes, and     provide    a year output      for
year-specific features.



                                                                 9

---

<!-- Page 10 -->

OTHER LIMITATIONS AND INFORMAL REAL-WORLD TESTS
●   Thresholds of confidence never lead           the algorithm to    give up  on  classification. It provides at least
    a  rough   classification   of any  objects detected     by  the algorithm, even     those that do    not belong    to
    196   categories (cars); pictures where nothing        is detected   receive the “No    vehicle found” label.
●   The most pronounced         car is labeled   and  recognized    in  the pipeline used    for evaluation. Route for
    busy   scenes in   the final application   was not used     during  the evaluation    procedure.
●   Coverage     stops   at the  2012   year of production;      newer vehicles are       classified  according    to the
    nearest available category. On        the Kaggle CPU it takes approximately           3.7  seconds to   analyze one
    image. This number highly         depends on     the environment; on       the team’s laptop    it takes about 2–3
    seconds, but on     the server where the application        is hosted  it takes approximately     15  seconds since
    the free tier provides only      a fraction   of CPU power. Accuracy         remains unchanged; the only        thing
    that changes is time.
Our   app  was   tested   on  some    photographs of our own. This          was an    unmeasured     test, not  a result.
Photographs     that  had  no  car gave    a “no   vehicle   found”   response, and     an  Indian   street  photograph
yielded   the  correct   model    for  the  Toyota    and   the  Suzuki.    The   cars  which    fall  outside   the  196
categories receive the closest match        available, hence the absence of open-set rejection           means that the
user gets low confidence.
Licences    and   responsible     use.As   the  files  for  the  trained  model are large,     therefore they     are not
stored  on  GitHub;    however, the     files can   be found    in the models     folder  or  from the Hugging       Face
model   repository   pointed   out by   the MODEL_REPO environment variable. There are 48                     automated
tests (unit tests and  end-to-end    tests on  simulated   images) in    the repository   that pass.
                              5. Deployment & Quickstart Guide
QUICKSTART COMMANDS

# 1. Clone the repository and install dependencies
git clone https://github.com/pradeep14012004/P5_T10_vehicle-recognition.git
cd P5_T10_vehicle-recognition
pip install -r requirements.txt
# (place the model files in ./models first; they are not stored in Git)

# 2. Start the web application (models in ./models)
python app.py            # then open http://localhost:7860

# Or build and run the container
docker build -t vehicle-id .
docker run -p 7860:7860 vehicle-id

# 3. Reproduce training and evaluation in a Kaggle GPU notebook
bash scripts/run_kaggle.sh /kaggle/input/<dataset-folder>
python scripts/08_report.py --team-id T10 --usns <USN1,USN2,...>
The  trained   model files are     not  uploaded    to GitHub    due to   their size, but they    reside in  the models
directory  or are   pulled   from   the Hugging     Face   model   repository    referenced    in the MODEL_REPO
environment variable. The repository          also  contains 48   automated     tests (unit tests and  end-to-end    tests
on  synthetic images), which      all pass.
DEPLOYMENT FORMAT
A Gradio    web   interface provided     by  the Python    server with    the int8 ONNX classifier and       the ONNX
detector using    ONNX Runtime running           on  the CPU, packaged       into  a Docker container and       deployed
to the Hugging     Face Spaces. The same inference pipeline is used              by  the evaluation    scripts, and  thus


                                                           10

---

<!-- Page 11 -->

the  reported   results   pertain  to  the deployed     models,    apart from the busy       scene (multiple      vehicles)
route that was not tested. The free-tier CPU-based             server is being     used  for the deployment, and        so  a
single photo    will take around    15  seconds there as compared         to 2–3  seconds on    a laptop.
                                   Student Submission Checklist

 Item                                                                                     Status
 Working   GitHub   repository link included  with an                                      Done
 updated  README.md
 No  raw code pasted   into this report                                                    Done
 All table metrics match   test runs reproducible via the                                  Done
 repository
 All team member details accurately     recorded                                           Done

AI Tool Usage & Declaration Policy
The    team    used    ChatGPT        and    Claude     Code     as   AI-assisted      tools    for   code    generation,
debugging, brainstorming, and            documentation.
AI-generated      outputs were reviewed, modified, and               validated    by  the team.
System     design,    implementation,        integration,     testing,   experimentation,        and   evaluation     were
carried   out by   the team members.
The team takes full responsibility           for the submitted       work   and   its results.




































                                                             11