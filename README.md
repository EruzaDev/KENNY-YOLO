# Kenny Gesture Lab

YOLOv8 transfer learning and a FastAPI / local Starlette browser dashboard for **Open Palm, Thumbs Up and Closed Fist**. This is an image classifier because the supplied dataset has image-level labels and no bounding boxes. The browser hand-presence gate supplies one hand crop to the actual trained checkpoint. Robot functions are displayed as text only; no hardware integration exists.

## Closed Fist displayed as Thumbs Up: camera diagnosis

The reported symptom is a wrong-class prediction. The on-disk checkpoint includes all three gestures; this does not prove the user's already-running server has loaded it. Model Details now shows loaded classes and a checkpoint fingerprint. The app warns when a legacy model lacks Closed Fist or weights on disk differ from the loaded model. Classifier scores for all classes are visible below the current prediction. These changes diagnose the failure; they do not establish a model-accuracy improvement.

Stop the existing server, then run `GESTURE_MODEL=models/best.pt ./run.sh --port 8001` from this project and hard-refresh the page. Expected current checkpoint prefix: `cd90f0b11011`, classes `closed_fist`, `open_palm`, `thumbs_up`. If a fist still shows Thumbs Up, select **Closed Fist** under **Correct gesture** and click **Download labeled frame** while the error is visible. It now saves the exact JPEG crop from the latest completed inference rather than a newly captured pose. Share that image for diagnosis/retraining; the source dataset validation views do not reproduce this error. Downloads remain local and human-labeled, never automatically trained. When no classified frame exists, the original snapshot capture remains available. Saved frame buffers are cleared on camera stop.

Current verification: **37 Python tests passed, 6 FastAPI-specific skipped; 8 browser-logic tests passed**. Actual-model upload/WebSocket tests still cover all three classes. Checkpoint identity/stale-state and request-to-saved-frame pairing have isolated regression coverage. No additional training or test-set tuning was performed for this diagnostic follow-up; deployed weights are unchanged. Physical-camera recognition remains unverified here.

## Latest fist/thumb investigation

Targeted retraining completed for 16 epochs with hand-focused crop augmentation and an auxiliary fist/thumb discrimination loss, preserving existing background, resolution and Open Palm training techniques. **The candidate was rejected and deployed weights remain unchanged:** accepted correctness on 176 correlated validation diagnostics decreased from 175 to 174 due to another Open Palm abstention. Neither model reproduces the reported fist/thumb confusion on these supplied validation images. There are only seven distinct fist captures, not 1,000 independent examples; no real camera failure examples were available. No test data was used for this candidate selection and no improvement in live recognition is claimed.

Details, per-class precision/recall/F1, classification AP, actual training curves, confusion matrices and rejection evidence: [investigation report](outputs/fist_confusion/REPORT.md). Configuration: `configs/fist_thumb.yaml`; candidate is retained in `models/fist_thumb_crops/`, prior deployed backup in `models/pre_fist_pair/`.

The dashboard now includes an **Experimental thumb consistency check**, disabled by default. It can abstain when MediaPipe 3D thumb evidence contradicts a fist/thumb prediction, never assign a replacement label. Its physical-camera accuracy is unverified; leave it off to preserve prior inference behavior. **Download labeled frame** lets you select the actual gesture and save a local example with a session ID. This does not upload or train automatically. Keep examples from one person/capture session together when preparing future splits; collect independent people/sessions for evaluation.

Current verification: **36 Python tests passed, 6 FastAPI-specific tests skipped; 7 browser-logic tests passed**, including actual trained-model uploads, all three function labels, WebSocket abstention and stabilization reset. The local Starlette runner is verified in process; physical webcam use and live thumb-check accuracy remain unverified. Stop the old server, run `./run.sh --port 8001`, and hard-refresh http://127.0.0.1:8001 to load the updated frontend/backend. All functions remain display-only.

## Current three-class release

The new `closedfist/` folder has 1,000 readable PNGs but only seven distinct captures. They were assigned the canonical class `closed_fist`, deduplicated, grouped and integrated without changing original files or the existing gesture splits. The current `models/best.pt` is a genuinely retrained three-class model. The prior version remains in `models/pre_three_class/`.

| Class ID | Gesture | Displayed function |
| --- | --- | --- |
| 0 | Closed Fist ✊ (`closed_fist`) | RETURN TO BASE |
| 1 | Open Palm ✋ (`open_palm`) | STOP ROBOT |
| 2 | Thumbs Up 👍 (`thumbs_up`) | DONE / RESUME |

Use `class_name` or the declared current mapping; IDs differ from the old two-class model. Predictions now include `display_name`, `function`, `confidence` and the existing output fields. `/api/model` exposes the mapping and `robot_controls_enabled: false`. No network/serial/ROS commands are issued for a gesture.

Prepared dataset: `data/three_class/`, with 53 train files (24 palm, 24 thumb, 5 fist), 13 validation files and 11 test files. Existing original/isolated views remain in the same inherited capture groups. Training-only sampling balances 24 samples per class per epoch; validation/test are not oversampled. The selected model retains prior background augmentation, adds resolution variation, emphasizes Open Palm loss and preserves frozen backbone weights and BatchNorm statistics during the final pass. Early stopping, dropout, weight decay and cosine scheduling remain enabled.

Measured at confidence 0.80: **13/13 unmodified validation images and 11/11 reserved test images accepted correctly**. Those 11 test files represent only six capture groups, with just one new fist capture. On correlated synthetic test variants, accepted accuracy improved **78/90 -> 89/90**, with no per-condition legacy precision/recall regression. The previous model did not support fist, accounting for much of that overall increase; legacy Thumbs Up recall also improved 46/48 -> 47/48 while Open Palm recall stayed 32/32 and its precision improved. These are small-dataset results, not proof of perfect recognition or natural-cluttered-scene generalization.

Per-class precision/recall/F1, classification one-vs-rest average precision, confusion matrices, training curves, all candidate decisions and limitations are in **`outputs/three_class/REPORT.md`** and **`outputs/three_class/final_test/`**. Detection mAP at IoU thresholds is not applicable without boxes; classification mean AP is labeled separately. AP measures ranking and is not the same as fixed-threshold accuracy.

Release-time verification: **31 Python tests passed, 6 FastAPI-specific tests skipped; 5 browser-logic tests passed** (see latest investigation above for current checks). Real-model uploads and WebSocket frames for all three gesture/function pairs passed through the current local ASGI runner. The warmed CPU classifier averaged about 45 ms per prediction; this excludes hand localization, network and rendering. Physical webcam and the external hand-detector execution remain unverified in the sandbox.

Stop the old server with Ctrl+C, then launch:

```bash
./run.sh --port 8001
```

Open http://127.0.0.1:8001 and hard-refresh. An existing process keeps its previous in-memory model until restarted. The hand detector still needs the documented browser download access. All upload, webcam, stabilization, settings and model-metadata features remain available.

## Three-class training reproduction

The following were executed in this workspace. Use fresh output directories/experiment names for any new run; immutable split and evaluation artifacts are not overwritten. Preserve the paired `models/pre_three_class/best.pt` and metadata when reproducing from the two-class source.

```bash
python -m training.prepare_three_class
python -m training.train --config configs/three_class_a.yaml --output-dir models/three_class_a
python -m training.train --config configs/three_class_b.yaml --output-dir models/three_class_b
python -m training.train --config configs/three_class_resolution.yaml --output-dir models/three_class_resolution
python -m training.train --config configs/three_class_stable.yaml --output-dir models/three_class_stable
```

Each refinement starts from its declared earlier trained checkpoint. For interrupted training, provide the matching `--config` as well as `--resume <run>/weights/last.pt` to retain the correct augmentation profile.

Evaluate candidates with the fixed validation protocol before promotion:

```bash
python -m training.robustness --model models/three_class_stable/best.pt --manifest data/three_class/manifest.json --distance --output outputs/three_class/candidate_stable_validation.json
```

Run the same validation command for A, B and resolution candidates with their corresponding checkpoint/output names, and for `models/pre_three_class/best.pt` with output `baseline_validation_distance.json`. Then:

```bash
python -m training.select_three_class --candidates outputs/three_class/candidate_a_validation.json outputs/three_class/candidate_b_validation.json outputs/three_class/candidate_resolution_validation.json outputs/three_class/candidate_stable_validation.json --promote
python -m training.evaluate --model models/best.pt --data data/three_class --split test --output outputs/three_class/final_test
.venv-runtime/bin/python -m unittest discover -s tests -v
node tests/test_browser.mjs
```

The selector refuses candidates with measured legacy-class regressions. Confidence remains 0.80 during comparison; test images are not used for hyperparameter or checkpoint selection. The report includes all four trained candidates, including rejected runs, to avoid overstating selection evidence. More independent fist captures, varied people/sessions and real negative/cluttered-scene examples are still necessary to assess deployment reliability.

## Historical two-class experiments and base setup

The sections below retain earlier setup, behavior and experimental evidence. Their model scores belong to their specified archived checkpoints, not the current three-class release.

## Camera false positives and open-palm misses

The original classifier reproduced the reported issue: a blank white image scored thumbs_up=0.90069 and a gray image scored thumbs_up=0.88096. These were diagnostic fixtures only, never training examples. The dataset has only two positive classes and 35 unique images; a two-class confidence threshold cannot establish hand presence. Training used isolated hands on white backgrounds, which also differs from webcam scenes.

The browser now uses an independent [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker?tab=Web) before gesture inference. It loads pinned `@mediapipe/tasks-vision@0.10.21` JavaScript/WASM from jsDelivr and the official hand_landmarker model from Google storage. This auxiliary model locates hands only; the actual fine-tuned YOLO checkpoint supplies the gesture class. Browser images stay local during hand localization; the cropped hand is sent to this project's backend. The script and model downloads require internet access.

For camera and upload testing, no detected hand means no YOLO request. Multiple hands produce “Show one hand at a time.” One detected hand is cropped with padding before classification, reducing irrelevant background. If the detector cannot load, inference pauses with an error instead of reverting to whole-frame guesses. This is inference preprocessing, not fabricated training annotations. The server also rejects near-uniform images and clears smoothing on `no_hand` control messages. Stabilization requires a full selected window before displaying a label, and immediately clears on rejected evidence.

**Restart `./run.sh` and hard-refresh the browser after this update.** Show one hand fully in view, fingers extended for open_palm; allow the hand-detector downloads. The raw `/api/predict` endpoint still classifies the submitted image and does not independently validate arbitrary hand presence; the hand gate operates in the browser. Its blank-image check is not an empty-scene or out-of-distribution detector. Other hand poses can still be falsely classified by the two-class model. Actual real-scene negative examples and additional independent gesture data remain necessary for reliable deployment.

Saved evaluation metrics describe the original full-image classifier only, not this new browser-gated pipeline. Live-camera improvement and the external hand detector could not be measured in this sandbox. The updated behavior passed backend regression tests and isolated browser-flow tests, which do not establish real-camera accuracy. The trained weights and held-out evaluation artifacts were not changed or retuned.

## Current improved model

The deployed `models/best.pt` is now a newly fine-tuned model trained for **34 epochs**, with **320px input, a partially frozen backbone and dropout**. It uses the provided original JPG photos as well as isolated images. Both views of each capture stay in their inherited split. This gives 48 training views and 12 validation views, but still only 35 source captures across the full dataset; the view copies do not create additional independence.

On the fixed validation comparison, accepted correct predictions improved from **11/12 to 12/12** at confidence 0.80. Original-background views improved from **5/6 to 6/6**; isolated views remained 6/6. The previously rejected open-palm image's score increased from 0.65 to 0.97. These are validation results across only four perceptual groups, not proof of perfect detection. No new final test was used for model selection. The earlier five-image test result below belongs to the original baseline, now preserved in `models/baseline/`, and must not be attributed to the new checkpoint.

See `outputs/improvement/REPORT.md`, `outputs/improvement/validation_comparison.json` and `outputs/improvement/candidate_validation/`. The app exposes validation results separately and suppresses stale test metrics whose checkpoint hash no longer matches. Restart `./run.sh` and hard-refresh the browser to use the promoted model. Existing no-hand gating and blank rejection remain enabled.

To reproduce the experiment from a fresh prepared checkout:

```bash
python -m training.prepare_originals
python -m training.train --config configs/improved.yaml --output-dir models/candidate_originals
python -m training.compare_models --promote
```

The baseline snapshot `models/baseline/best.pt` and its metadata must be present for comparison. Existing prepared datasets, experiment names and comparison artifacts are intentionally preserved; use new output directories/config experiment names for any separately justified run. Never retrain into the live checkpoint before validation comparison. Do not tune new runs against the previously inspected test set.

## Original baseline delivery history and runtime evidence

Dataset inspection, preparation, training, and held-out evaluation have now been executed. **Training completed for 46 CPU epochs**, with early stopping after 20 epochs without validation-fitness improvement. The real fine-tuned checkpoint is `models/best.pt`, paired with `models/metadata.json`. Validation: **6/6 correct**. Held-out test: **5/5 correct**, at confidence 0.80; mean measured image-inference latency approximately **19.35 ms** on the training CPU. These tiny target-only samples do not establish perfect recognition or real-webcam generalization.

**The local application runtime is repaired.** Run `./run.sh` from this folder in your normal terminal, then open http://127.0.0.1:8000. The launcher prefers FastAPI; the current offline environment uses a clearly logged Starlette compatibility runner with the same trained checkpoint, lifecycle and inference handlers. The original FastAPI entrypoint remains `app.main:app`. The launcher does not install packages or download weights.

A `.venv-runtime` environment was assembled from compatible packages already on this machine. Its pure Python web packages were copied with distribution metadata, and its ML packages are linked to the existing compatible Python 3.14 environment. This environment is specific to this machine; use requirements.txt for a fresh installation elsewhere. Training provenance still records the separate Python 3.10 training runtime.

Current checks: **18 Python tests passed, 6 skipped; 5 browser-logic tests passed** (FastAPI-specific). Real-model image uploads, WebSocket frame inference, dashboard assets, health/model information, invalid files and size/threshold limits passed through the in-process local ASGI runner. The HTTP health response reports `ready`. Physical browser-camera access has not been exercised. The sandbox blocks localhost socket binding and local HTTP connections, and automatic approval rejected escalation, so there is no reachable server URL from this session. Run the launcher outside that sandbox to listen on the port.

Inspection artifacts:

- `outputs/inspection.json`: all 2,072 source images inspected, including masters; dimensions, exact pixel duplicate groups, and corrupt-file checks.
- `outputs/contact_sheet.jpg`: all 35 distinct labeled class images, visually reviewed.
- `outputs/dataset_report.json`: class mapping, pixel hashes, perceptual duplicate pairs, group assignments, split counts, and provenance.
- `data/processed/manifest.json`: identical preparation manifest accompanying the training copy.
- `outputs/STATUS.md`: concise inspection findings and execution limits.

## Dataset findings

| Source folder | Files | Unique decoded RGB images | Redundant copies |
| --- | ---: | ---: | ---: |
| open_palm | 1,000 | 17 | 983 |
| thumbs_up | 1,000 | 18 | 982 |
| master_images (excluded from training) | 72 | 70 | 2 |

The supplied summary describes 500 class images, but the actual dataset contains 2,000. All images decoded; the labeled PNGs are 640 × 640. There are no YOLO boxes, COCO annotations, Pascal VOC annotations, or an existing split. Alphabetic class mapping is **0=open_palm, 1=thumbs_up**. Repeated copies do not add independent training information. Classes are almost balanced after deduplication.

The contact sheet broadly agrees with the folder labels, but several open-palm examples have bent or nearly closed fingers. These need human review against the requested extended-finger definition; no semantic labels were automatically changed. These captures show limited lighting, backgrounds and apparent subjects. There are no negative gestures, empty scenes, annotated multiple-hand frames, or condition metadata. Background removal introduces a substantial gap from ordinary webcam images. Skin-tone, rotation, occlusion, distance, and real-background robustness cannot be established from this data.

Original data is untouched. Preparation uses only the class folders; masters contain alternate copies of source captures; the baseline excluded them. The improved experiment adds the original JPG views only with their inherited capture groups, as described above. No new training images or fabricated boxes were created.

Exact pixel duplicates are reduced to one image. A 63-bit DCT perceptual hash groups likely near duplicates at Hamming distance ≤8; connected groups are kept together. Seed 42 yields 24 groups and:

| Split | open_palm | thumbs_up | Total |
| --- | ---: | ---: | ---: |
| train | 12 | 12 | 24 |
| val | 3 | 3 | 6 |
| test | 2 | 3 | 5 |

Integer counts and whole-group assignment approximate 70/15/15. Cross-split exact and detected near duplicates are checked by tests. Perceptual similarity is a heuristic, not a person identifier. The dataset owner confirmed no person/session IDs are available. **Person/session leakage cannot be ruled out.** Five test images are much too few to support strong deployment claims; one error changes accuracy by 20 percentage points. Do not tune on this test set or repeatedly compare runs against it.

## Setup

Use Python 3.11 or 3.12 for broad PyTorch wheel compatibility. The project .venv uses Python 3.14 without dependencies; completed training used a separately discovered Python 3.10.22 ML environment. Commands below require a machine with package-index and pretrained-checkpoint download access, or an offline wheelhouse and a trusted local YOLOv8 classification checkpoint.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Requirements now pin Ultralytics 8.4.189 to match the version that completed actual training and inference. Exact runtime versions are in models/metadata.json. After a successful setup, capture the resolved environment with `python -m pip freeze > requirements-lock.txt`. For CUDA, install compatible PyTorch and torchvision builds using the official PyTorch installation instructions before installing remaining requirements. CPU operation is supported. Inspect Ultralytics licensing before commercial deployment.

## Preparation

Expected source structure:

```text
hand_gesture_dataset/
  open_palm/*.png
  thumbs_up/*.png
  master_images/          # inspected, never included in model splits
  dataset_summary.txt
```

Preparation has already been executed in this workspace. To reproduce in a new output directory:

```bash
python -m training.prepare_dataset --output data/processed_reproduced --report outputs/reproduced_report.json
```

For a fresh checkout without a prepared copy:

```bash
python -m training.prepare_dataset
```

An existing output directory is never overwritten. Configure `data` in `configs/training.yaml` to select a different prepared copy. Classification uses `train/<class>`, `val/<class>`, and `test/<class>` directories; it does **not** require a detection `dataset.yaml` or label text files. Corrupt files and conflicting exact-duplicate labels stop preparation rather than silently excluding data.

If identity metadata becomes available, pass `--sessions sessions.json`, a JSON object mapping **every known source filename relative to the raw root** to a shared person/session ID, e.g. `{"open_palm/openpalm_1.png": "person_01"}`. Include both gesture classes for the same identity and all duplicate filenames. Missing IDs remain unknown. Session IDs merge groups even across classes. Insufficient independent groups cause preparation to stop; additional independent captures are then necessary. Review perceptual groups manually before making a fresh, final split.

## Training

```bash
python -m training.train
# Optional CUDA override:
python -m training.train --device 0
# Resume an INTERRUPTED run from last.pt (use the actual experiment directory):
python -m training.train --resume outputs/training/gesture/weights/last.pt
```

`configs/training.yaml` controls model, image size, device, epochs, batch, learning rate, weight decay, and early stopping. YOLOv8n-cls at 224 px is the initial choice for the tiny independent dataset and low inference cost. Maximum 100 epochs, batch 8, AdamW at 0.0003, cosine scheduling, weight decay 0.001, patience 20, and seed 42 are starting points. AMP is enabled only for supported CUDA runs. Do not assume these settings are optimal.

The trainer uses full-frame white letterboxing, avoiding default classification center crops that may cut off fingers. Training-only augmentations: ±8° affine rotations, up to 4% translation, scale 0.94–1.04, horizontal flips, mild color jitter, occasional mild blur and tensor noise. Validation/test do not receive random augmentation. Mosaic, mixup and random erasing are not applied. No synthetic background replacement is used. The custom validator preserves preprocessing during final validation too. Inference follows the same letterboxing and input scaling.

Ultralytics saves per-epoch CSV metrics, plots, `last.pt`, and `best.pt` in its experiment directory under `outputs/training/` (later runs get new names). Early stopping and best selection use validation fitness. Upon successful completion, the pipeline copies the actual best checkpoint to `models/best.pt` and writes `models/metadata.json`, including checkpoint and dataset-manifest hashes. An interrupted training run does not install a model. The web app requires both files and verifies their hash match. Checkpoints are trusted local artifacts; never load weights from an untrusted upload.

Review training/validation loss and accuracy curves before final testing. Classification top-5 accuracy is uninformative with two classes; use top-1 accuracy and per-class precision/recall/F1. With this tiny validation set, metric jumps are large and early stopping is noisy. These safeguards reduce risk but cannot demonstrate that overfitting was eliminated. Prefer acquiring independently captured data and negative classes over increasing model size.

## Validation and final evaluation

Select model and confidence threshold on validation only. Default 0.80 is an uncalibrated starting threshold, not a measured optimum. Generate a separate validation report without modifying the manifest or accessing test images:

```bash
python -m training.evaluate --split val --confidence 0.80
```

It writes to `outputs/validation/`. Use distinct `--output` directories for deliberately chosen validation comparisons. Freeze the threshold before final testing.

After fixing the model and threshold, run final testing once:

```bash
python -m training.evaluate --confidence 0.80
```

The evaluator uses only manifest test records, warms the model outside timing, and writes:

- `outputs/evaluation/metrics.json`: accuracy including abstentions as errors, coverage, per-class precision/recall/F1/support, confusion matrix, false positives/negatives, mean and p95 inference latency, checkpoint hash and limitations.
- `predictions.json`: each test file, label, scores, prediction and latency.
- `confusion_matrix.png`, `training_curves.png` (when training CSV is available).
- `samples/correct/` and `samples/incorrect/`: actual test examples grouped by outcome.

Existing final metrics are not overwritten. `unknown` means abstention when confidence is low, not a learned negative class. Detection mAP and bounding-box metrics do not apply. The last-10-epoch divergence heuristic flags rising validation loss against falling training loss; inspect actual curves and mistakes before drawing conclusions. If a category has no mistakes, no sample files are invented. Empty scenes and non-target gesture false-positive rates cannot be evaluated without such examples.

## Launch and test the app

On this machine, start the prepared offline runtime:

```bash
./run.sh
# Optional alternate port:
./run.sh --port 8001
```

It prefers FastAPI when importable; otherwise it logs use of the local Starlette runner. Shared inference handlers and the actual trained model are used in either case. `/docs` is supplied by the FastAPI entrypoint only. Do not expose the compatibility runner as a production service without the deployment checks below.

With a fresh, complete requirements installation, the original FastAPI command is also available:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --ws-max-size 7200000
```

Open **http://127.0.0.1:8000**. `/api/health` reports readiness separately from HTTP server availability. `/api/model` shows actual model metadata and only displays metrics whose checkpoint hash matches the loaded model. Without weights, the dashboard serves an explicit unavailable state and inference returns HTTP 503; it never downloads or substitutes a generic classifier.

- **Camera:** allow camera permission, select Start camera, hold one gesture in view, then Stop. Sliders update confidence, majority stabilization window, and request frequency (1–10 FPS). Only one frame per browser is in flight; the next is scheduled after the response. Server inference is serialized for safe shared-model access. A 30-second client timeout stops stalled streams. Camera tracks stop on tab hiding, mode switching, page exit, stop, errors or disconnect.
- **Upload:** select JPEG/PNG (≤5 MB, ≤16 megapixels), view the source image with a prediction overlay, then reset or choose another. Changing confidence reruns the upload. Browser and backend validate inputs; backend decodes actual image content rather than trusting extensions.
- **Uncertainty:** no accepted prediction displays “No gesture detected.” Stabilization never retains an old label over a weak current frame. A two-class softmax can still assign high confidence to unrelated images; it is not an out-of-distribution detector.
- **Model settings:** `GESTURE_MODEL` overrides the trusted checkpoint path; `GESTURE_DEVICE` selects inference hardware. Restart after replacing weights. Threshold changes do not retrain the model. IoU is explicitly unavailable for classification.

API documentation is at `/docs`. Webcam protocol is same-origin `/api/webcam`; JSON requests contain base64 JPEG/PNG in `image`, numeric `confidence`, and integer `smoothing`. Responses contain `task`, `predictions`, `scores`, original dimensions, latency, message and `stable_gesture`. Classification prediction `bbox` is always null. Malformed frames receive a JSON error. The server caps accepted frame frequency and WebSocket size.

## Tests

```bash
.venv-runtime/bin/python -m unittest discover -s tests -v
python -m compileall -q app training tests
node --input-type=module --check < frontend/js/app.js
node --input-type=module --check < frontend/js/camera.js
node --input-type=module --check < frontend/js/api.js
node tests/test_browser.mjs
```

Tests cover actual prepared-data integrity and class mapping, split leakage checks, malformed images, RGB decoding, missing-model behavior, abstention and stabilization. FastAPI tests cover routes, upload limits, invalid settings, missing-model HTTP 503, malformed WebSocket requests and frame processing with isolated mocks. The application has no mock mode. When trained weights exist, real-model prediction and upload tests run against an actual test image. Multiple detections are inapplicable to whole-frame classification. Physically verify browser permission denial, camera disconnection, slow inference, stop/restart and multiple tabs once dependencies and weights are available.

## Organization

`training/` owns audit, preparation, transforms, training, evaluation and export. `app/` owns configuration, real model inference, image validation and API routes. `frontend/` is buildless HTML/CSS/JavaScript. `configs/` holds training parameters, `models/` holds successfully trained deployment artifacts, and `outputs/` holds evidence. Original source images remain in `hand_gesture_dataset/`; prepared copies live in `data/processed/`.

## Optimization and deployment limits

Begin with one process and YOLOv8n-cls. CPU is usable for functional testing; real FPS and memory requirements must be measured on the target hardware. A CUDA GPU with roughly 4 GB or more VRAM and 8 GB host RAM is a practical starting point for this small classifier, not a guarantee. Reduce batch size if memory is exhausted. Larger backbones need independent validation benefits to justify cost.

Optional export (requires the export format's extra dependencies):

```bash
python -m training.export_model --format onnx
```

The application continues to load the actual PyTorch `best.pt`; exported runtimes need preprocessing parity, numerical/accuracy checks on validation, and target-device latency measurements before integration. No unmeasured speedup is claimed.

This is a localhost application. For shared hosting, add authentication, HTTPS, reverse-proxy body/request limits, connection limits and service monitoring before exposing it. The built-in inference semaphore prevents simultaneous model calls but is not a distributed queue or multi-user capacity guarantee. Camera permissions require localhost or HTTPS. Avoid multiple Uvicorn workers unless you intend multiple model copies in RAM/VRAM. No uploaded images or webcam frames are persisted by the app.

Troubleshooting:

- **Package download/DNS error:** run setup on a network-enabled host or install from a compatible offline wheelhouse. Do not interpret this as a dataset failure.
- **Missing or mismatched model:** complete training and keep its paired `best.pt` and `metadata.json`; do not rename generic weights into place.
- **Camera denied/unavailable:** enable browser permission, use localhost/HTTPS, close other camera users, or test image upload.
- **Backend disconnect:** confirm `/api/health`, inspect structured server logs, and restart the camera.
- **No prediction:** check threshold, framing and lighting. Lowering confidence may increase false positives.
- **Poor real-scene results:** acquire diverse, independently captured and reviewed images, negative examples, and identity/session metadata. For boxes or multiple hands, collect genuine object-detection annotations and implement a separately evaluated detection pipeline.

Implementation references: [Ultralytics classification](https://docs.ultralytics.com/tasks/classify/), [YOLOv8 model family](https://docs.ultralytics.com/models/yolov8/), [pinned classification trainer source](https://github.com/ultralytics/ultralytics/blob/v8.3.203/ultralytics/models/yolo/classify/train.py). The custom trainer and actual classifier have run successfully with Ultralytics 8.4.189. The local Starlette ASGI API is verified with actual weights; FastAPI-specific tests still await compatible dependencies.
