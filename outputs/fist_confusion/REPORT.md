# Fist versus Thumbs Up investigation

## Outcome

The retraining run completed, but the candidate was rejected. Deployed `models/best.pt` and its paired metadata remain unchanged. There is **no measured reduction of the reported live-camera fist/thumb confusion**. No real failure examples were available; the supplied validation views did not reproduce any cross-confusion in either model.

## Dataset and training

The 1,000 Closed Fist files contain only seven distinct captures: five training, one validation, one test. Training retains all 53 supplied training views and balances sampling to 24 per class. Existing split groups and original source files are unchanged. Identity/session independence remains unknown. No test images were used in this investigation or candidate selection.

Executed config: `configs/fist_thumb.yaml`; initial weights `models/pre_fist_pair/best.pt`; candidate `models/fist_thumb_crops/best.pt`. Training: CPU, 320 px, batch 8, AdamW LR 0.000075, cosine scheduling, weight decay 0.002, dropout 0.2, seed 42, patience 15; 16 epochs completed, Ultralytics selected epoch 1 on top-1 validation accuracy. AMP was disabled on CPU.

Added training-only hand-focused crops of supplied near-white silhouettes to reduce the forearm/crop mismatch with browser localization. This is classification augmentation, never fabricated bounding-box annotation. Existing mild geometry, lighting, noise, blur, background replacement, resolution variation and Open Palm loss weighting remain. A 0.3-weight auxiliary binary loss emphasizes Fist versus Thumb while preserving three-class supervision. The first eight model blocks are frozen; inherited backbone BatchNorm statistics remain fixed. Candidate artifacts preserve actual run configuration and hashes.

## Validation comparison (confidence 0.80)

The same fixed protocol covers clean images, brightness, ±18° rotations, reduced-resolution proxy, synthetic dark/blue/clutter backgrounds, plus tighter crops of isolated views. These are **176 correlated diagnostics derived from 13 files, five heuristic capture groups**, including only one distinct validation fist capture. They are not 176 independent samples, natural backgrounds, measured distances or webcam frames.

| Gesture | Diagnostic views | Deployed precision / recall / F1 | Candidate precision / recall / F1 |
| --- | ---: | --- | --- |
| closed_fist | 20 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 |
| open_palm | 78 | 1.0000 / 0.9872 / 0.9935 | 1.0000 / 0.9744 / 0.9870 |
| thumbs_up | 78 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 |

Deployed: **175/176** accepted correctly (99.43%); candidate: **174/176** (98.86%). Both accept all 13 clean validation files correctly and have zero fist/thumb cross-confusions on these data. Both have classification one-vs-rest mean AP 1.0; AP ranking does not imply perfect thresholded recognition. Detection mAP@0.50 / @0.50:0.95 is inapplicable because labels have no boxes.

The deployed model abstains on one rotated Open Palm. The candidate additionally abstains on one reduced-resolution Open Palm. It was not promoted because this regresses Open Palm recall and there is no evidence of reduced fist/thumb confusion. Group-balanced validation negative log-likelihood worsened from 0.00738 to 0.01174. `selection.json` records the enforceable validation-only rejection.

## Overfitting review

Clean top-1 validation accuracy stays at 1.0 throughout, so it cannot distinguish checkpoints. Validation loss reaches 0.00088 at epoch 2 then increases to 0.00447 at epoch 16; training loss changes from 0.00738 to 0.00474 over that interval. This is a warning of overfitting/confidence degradation despite augmentation and regularization. Early stopping halted the run; preserved weights avoid deploying the observed regression. No claim that overfitting has been eliminated is supported.

## Application enhancements

An optional **Experimental thumb consistency check**, disabled by default, uses MediaPipe world landmarks to reject a clear thumb-position contradiction with the actual YOLO class. It never creates or substitutes another class. Missing/ambiguous evidence leaves normal inference unchanged; Open Palm is unaffected. Rejected frames clear stabilization history. Geometry fixtures and real-model API/WebSocket tests verify implementation, not live recognition accuracy. The thresholds are not calibrated against natural camera failures and may increase false negatives, so this is not enabled automatically.

The **Download labeled frame** button requires selecting the actual gesture manually and saves a local JPEG with a capture-session ID. A separate drawing buffer protects active camera inference. Images are not automatically uploaded, added to the dataset or used for training. If localization misses the hand, the saved example is marked `full`; recognition still has no whole-scene fallback. Camera samples from the same person/session must stay in one split when later integrated. Keep an independent new-person/session test set untouched.

For the next improvement cycle, collect independent natural-background fist/thumb examples with varied thumb placement, rotations, lighting and distance, plus non-target gestures. Retain full frames when localization itself fails. Review human labels and group by person/session before training. More duplicate copies of the same seven fists cannot establish generalization.

MediaPipe world-coordinate output is documented at https://developers.google.com/edge/api/mediapipe/js/tasks-vision.handlandmarkerresult .

## Artifacts and reproduction

- `baseline_val.json`, `candidate_val.json`: immutable per-image scores, precision/recall/F1, AP, conditions and synthetic sequences.
- `validation_confusion.png`: side-by-side diagnostic confusion matrices.
- `selection.json`: no promotion, no test use and rejection reasons.
- `../training/gesture_fist_thumb_crops/results.csv`, `results.png`, `weights/`: genuine logs, curves and checkpoints.
- `crop_preview.jpg`, `crop_boxes.json`, `crop_baseline.json`: exploratory train/validation crop inspection; no test use.
- Existing deployed-model test evidence remains under `../three_class/final_test/`; this investigation does not present it as a fresh test.

```bash
python -m training.train --config configs/fist_thumb.yaml --output-dir models/fist_thumb_crops
python -m training.robustness --model models/pre_fist_pair/best.pt --manifest data/three_class/manifest.json --output outputs/fist_confusion/baseline_val.json --distance --hand-crops
python -m training.robustness --model models/fist_thumb_crops/best.pt --manifest data/three_class/manifest.json --output outputs/fist_confusion/candidate_val.json --distance --hand-crops
python -m training.select_three_class --baseline outputs/fist_confusion/baseline_val.json --candidates outputs/fist_confusion/candidate_val.json --fist-thumb --output outputs/fist_confusion/selection.json
```

These commands were executed. Use fresh experiment names/output paths when repeating; recorded evaluation/selection artifacts refuse overwrites. Selection exits nonzero when no candidate qualifies. No trained-model improvement is claimed here.

## Verification

36 Python tests passed; six FastAPI-specific tests skipped because the local environment lacks FastAPI. Seven Node browser-logic tests passed. Tests use the actual deployed weights for uploads and WebSocket messages, including all three labels, invalid evidence, conflict abstention and smoothing reset. Landmark fixtures test only geometric logic. JavaScript syntax checks passed. A physical camera and browser model-download execution were not available for verification here.
