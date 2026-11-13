# Current camera misclassification follow-up

User reports Closed Fist displays Thumbs Up. On-disk checkpoint remains cd90f0b11011 and includes all three classes. The user's running server and physical camera are inaccessible here; its loaded version is not confirmed. Added loaded-class/fingerprint and stale-weight warnings, per-class score display, and exact classified-frame JPEG download instead of a new-pose snapshot. No weights changed, no further training or test tuning, and no claimed recognition fix. Need an actual failing crop to reproduce this camera-specific misclassification. Restart with GESTURE_MODEL=models/best.pt ./run.sh --port 8001, hard-refresh, select actual Closed Fist, and download the labeled frame while Thumbs Up appears.

Verification: 37 Python passed, 6 FastAPI-specific skipped; 8 browser-logic tests passed. Actual weights cover HTTP uploads and WebSocket processing for all classes; stale model identity and exact saved-frame pairing are tested. Physical webcam unavailable here.

# Latest fist/thumb investigation

Retraining completed (16 epochs), but the candidate failed preservation checks: 174/176 correlated validation diagnostics correct versus 175/176 for the deployed model, with one additional reduced-resolution Open Palm abstention. Neither reproduces the reported fist/thumb cross-confusion on available validation captures. Current weights and paired metadata are unchanged; no new test-set evaluation was used for tuning or selection. See outputs/fist_confusion/REPORT.md and selection.json. Only seven unique fist captures exist; no natural camera error examples were available.

Added optional thumb consistency abstention (experimental, disabled by default) and manually labeled local camera-frame download with session IDs. No robot integration. Verification: 36 Python passed, 6 FastAPI-specific skipped, 7 browser tests passed. Actual-model HTTP uploads and WebSocket abstention/reset verified through local Starlette ASGI; physical camera use remains unverified. Restart ./run.sh --port 8001 and hard-refresh.

# Current three-class checkpoint

Closed Fist was integrated from 1,000 files / seven unique captures into preserved capture splits. Actual training and validation selection completed. The current models/best.pt supports closed_fist, open_palm and thumbs_up and returns display-only function labels. No robot controls were added.

Unmodified reserved test: 11/11 correct across six source capture groups, including one fist capture. Synthetic reserved stress: 78/90 -> 89/90 accepted correct, with no per-condition legacy precision/recall regression. Classification mean AP=1.0 on the 11 images; detection IoU mAP is not applicable. Natural-clutter/new-person webcam reliability and unknown-gesture false-positive rates remain unverified.

Verification: 31 Python tests passed, 6 FastAPI-specific tests skipped; 5 browser-logic tests passed. Actual-model upload and WebSocket checks cover all three function labels. Existing two-class checkpoints and reports are preserved. Current details and caveats: outputs/three_class/REPORT.md. Stop the old process and restart ./run.sh --port 8001, then hard-refresh to load the current weights.

## Historical execution record

# Current improved checkpoint

A new model was trained for 34 epochs using provided original photos plus their isolated views. Validation accepted-correct predictions improved from 11/12 to 12/12 at confidence 0.80; the original-background open-palm abstention was corrected. The model was promoted to models/best.pt after validation-only selection. No new independent test-set accuracy or perfect recognition is claimed. Detailed evidence: outputs/improvement/REPORT.md and validation_comparison.json. Current tests: 18 Python passed, 6 FastAPI-specific skipped; 5 browser-logic tests passed. Restart the server to use the new weights.

## Previous baseline history

# Completed training and evaluation

YOLOv8n-cls was actually fine-tuned on this project's 24 unique grouped training images. 46 CPU epochs completed; early stopping used patience 20. The early-stopping monitor first achieved its best validation fitness at epoch 26. Ultralytics saves later checkpoints on tied best fitness too, so the deployed best.pt is the final tied-best checkpoint, not necessarily the epoch-26 weights. No hyperparameters or confidence thresholds were changed after test evaluation.

Runtime: Python 3.10.22; PyTorch 2.14.1+cpu; Ultralytics 8.4.189. This differs from the initial requirements pin and is recorded in model metadata. The local ImageNet YOLOv8n-cls initialization was verified to have 1,000 classes before the head was replaced and fine-tuned for two gestures.

Validation: 6 images, accuracy 100.0%, coverage 100.0%.
Test: 5 images, accuracy 100.0%, coverage 100.0%; target-class macro precision/recall/F1 all 1.0. Confidence was fixed at 0.80. Mean measured CPU image-inference latency: 19.35 ms, p95 20.34 ms, after warmup. This is not end-to-end webcam latency.

Training loss fell from 0.68794 to 0.02361; validation loss fell from 0.81958 to 0.032. The final-ten-epoch divergence flag is false. These observations do not establish absence of overfitting. There are only 35 independent pixel-unique images, no negative examples, and no person/session IDs. Five correct test predictions do not demonstrate perfect recognition, diverse-background generalization, or robust rejection of unrelated gestures.

Both gesture classes have zero observed false positives and false negatives in these five target-only test images. No empty-scene or non-target false-positive rate can be calculated. No incorrect sample artifacts were fabricated. Review outputs/evaluation/predictions.json, confusion_matrix.png, training_curves.png and samples/correct/.

Verification: 13 tests passed, including actual trained-model loading, image upload inference, WebSocket frame inference, dashboard/static assets, ready health/model endpoints, and invalid-upload/size/threshold checks; 6 FastAPI-specific tests skipped. Python compilation, JavaScript syntax checks and launcher shell syntax checks passed. The local runner uses Starlette and the same lifecycle/inference handlers as FastAPI; there are no dummy predictions. Physical camera/browser interaction remains unverified.

Runtime repaired: .venv-runtime uses Python 3.14, compatible local ML packages, copied pure Python web packages and cached python-multipart. The launcher ./run.sh prefers FastAPI and explicitly logs the current Starlette compatibility backend. The original FastAPI entrypoint remains available for fully installed environments. No user environments outside this workspace were changed.

Server startup reached Application startup complete and loaded the actual model, but localhost binding failed. Local HTTP checks returned Operation not permitted. A request to start outside the sandbox was automatically rejected because sandbox escalation is disabled. Therefore no localhost port is currently listening from this session. Run ./run.sh in the user's normal terminal to serve http://127.0.0.1:8000. No physical-webcam success or externally reachable URL is claimed.

Original baseline (now models/baseline/best.pt and paired metadata.json), SHA256 241be7c04c97dabdd6c75e23030e26a72fcf6a101e463c276a38624e3fe2478c. Full training CSV and checkpoint artifacts: outputs/training/gesture/.


Camera regression update: blank white and gray inputs reproduced high-confidence thumbs_up false positives. Added a browser-side MediaPipe hand-presence gate and padded hand crops for upload/webcam inference, explicit failure when assets cannot load, no-hand smoothing reset, full-window stabilization, and server uniform-image rejection. No weights or training/test images were changed. Current verification: 14 Python tests passed, 6 FastAPI-specific tests skipped, 5 isolated browser-flow tests passed. Actual MediaPipe execution and physical webcam performance remain unverified; the external assets require browser internet access. Raw API callers still receive classifier-only semantics beyond the uniform-image check. See regression_report.json and README.
