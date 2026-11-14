# Kenny Gesture Lab

Local hand-gesture recognition with a camera/upload dashboard, an independent browser hand localizer and a trained YOLOv8 classifier. Gesture functions are display-only; no robot commands are sent.

## Run the app

On this Windows machine:

```powershell
.\run.ps1
```

Open **http://127.0.0.1:8001**. Refresh with **Ctrl+F5** after frontend updates. The default checkpoint is now `models/webcam_fist/best.pt`.

Or run:

```powershell
py -3.11 -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

The existing Python 3.11 environment has the required ML/web dependencies. For another machine, see `requirements.txt`; the installed training runtime was Python 3.11.9, PyTorch 2.14.0+cpu and Ultralytics 8.4.154, which differs from the requirements file's Ultralytics pin.

## Current model and actual results

**Webcam fist correction**, trained for **13 CPU epochs**. Checkpoint fingerprint: `48620f85cce1`.

| Evidence | Accepted correctly at confidence 0.80 | Scope |
| --- | ---: | --- |
| Original validation views | 13/13 | Existing small validation set |
| Expanded validation | 54/55 (98.2%) | Includes correlated synthetic hand crops |
| Existing test regression | 11/11 | Previously inspected test files; not a fresh test |
| Uploaded fist frames | 5/5 | **Training-frame fit, not camera accuracy** |

Expanded validation has 100% raw top-1 accuracy, but one synthetic Open Palm crop scores below 0.80 and is rejected. The threshold was not lowered. The training loss-divergence heuristic flagged possible overfitting. Fresh camera examples of **all three gestures** are needed to verify performance.

The dashboard's **Results** section shows checkpoint-matched metrics, per-gesture precision/recall/F1, dataset distribution, actual training curves, validation confusion matrix, training examples and limitations. Results/images are served only when linked to the loaded checkpoint; stale results are not substituted.

- [Current training/evaluation report](outputs/webcam_fist/REPORT.md)
- [Consolidated current results](outputs/webcam_fist/results.json)
- [Training curves](outputs/webcam_fist/validation/training_curves.png)
- [Validation confusion matrix](outputs/webcam_fist/validation/confusion_matrix.png)
- [Earlier photo-background experiment](outputs/photo_backgrounds/REPORT.md) — a different checkpoint
- [Historical project notes](HISTORY.md)

## Camera and upload controls

1. Select **Live camera**.
2. Click **Refresh cameras** and allow browser camera access.
3. Choose **Astra Pro HD Camera** for the Orbbec RGB/color stream, or **Integrated Camera** for the laptop webcam.
4. Click **Start camera**, and show one hand fully in view.

Switching cameras releases the old stream and starts the selected device. Astra depth/IR and Orbbec SDK data are not used. Physical Astra capture still needs user verification. The browser needs access to the pinned MediaPipe scripts/model on jsDelivr and Google storage; recognition pauses if the localizer cannot load.

Accepted predictions show a **hand bounding box, gesture label and confidence** on the camera/upload image. The box comes from the browser hand localizer, not YOLO detection. Coordinates account for mirrored camera video and display resizing. No-hand/rejected results and camera stop clear the box. A confidence percentage is **not measured accuracy**.

| Gesture | Displayed function |
| --- | --- |
| Closed Fist | RETURN TO BASE |
| Open Palm | STOP ROBOT |
| Thumbs Up | DONE / RESUME |

Use **Upload image** for JPEG/PNG up to 5 MB. For incorrect live predictions, expand **Collect a labeled example**, choose the actual gesture and download the frame. Downloads are local; training never happens automatically.

## Image organization

```text
assets/images/
  backgrounds/                     # five supplied scene photos
  webcam/closed_fist/
    <session-id>/hand/              # five human-labeled hand crops in total
    <session-id>/full/              # one diagnostic full frame, excluded from training
  catalog.json                     # original path → new path, role and file hash
hand_gesture_dataset/
  closedfist/                      # 1,000 supplied files; seven distinct images
  open_palm/
  thumbs_up/
  master_images/
data/
  photo_backgrounds/               # previous prepared experiment
  webcam_fist/                     # current prepared dataset and manifest
outputs/
  photo_backgrounds/               # earlier experiment's evidence
  webcam_fist/                     # current results, plots and diagnostics
```

All 11 loose source photos were moved without changing their bytes. Original class folders, prepared datasets, split assignments, model weights and historical result files were preserved. The catalog and `training/image_sources.py` resolve historical root-level source paths without rewriting old manifests or invalidating training hashes.

Current dataset files:

| Split | Closed Fist | Open Palm | Thumbs Up | Total |
| --- | ---: | ---: | ---: | ---: |
| Training | 65 | 156 | 156 | 377 |
| Validation | 7 | 24 | 24 | 55 |
| Test | 1 | 4 | 6 | 11 |

Training-only balancing samples 156 views per class each epoch. Five uploaded webcam crops from two sessions are kept in training only. Synthetic crops and background replacements do not create independent captures. The test files cover only six capture groups; identity/session independence is unknown, and no negative-scene or new-person accuracy is established.

## Training and evidence maintenance

The original source preparation must exist before reproducing later stages. Use fresh experiment/output paths rather than overwriting completed evidence.

```powershell
py -3.11 -m training.prepare_webcam_fist
py -3.11 -m training.train --config configs/webcam_fist.yaml --output-dir models/webcam_fist
```

The completed run's configuration, runtime and checkpoint/dataset hashes are in `models/webcam_fist/metadata.json`. Earlier weights remain in `models/photo_backgrounds/`.

To rebuild the dashboard summary from the already-recorded evidence, without retraining or reevaluating:

```powershell
py -3.11 -m training.publish_results
```

Restart the server after changing weights or metadata. `GESTURE_MODEL` overrides the default model; paired provenance metadata is required.

## Checks

Latest run: **53 Python tests and 15 browser-logic tests passed**, including actual-model upload/WebSocket inference and the new results/image-organization checks.

```powershell
node --test tests/test_browser.mjs tests/test_camera_selection.mjs tests/test_hand_overlay.mjs
py -3.11 -m unittest discover -s tests -p test_results.py -v
py -3.11 -m unittest discover -s tests -p test_image_sources.py -v
py -3.11 -m unittest discover -s tests -p test_webcam_fist.py -v
```

Keep this application bound to localhost. It is not a production or robot-control service.
