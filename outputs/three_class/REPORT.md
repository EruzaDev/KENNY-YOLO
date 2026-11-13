# Three-gesture integration, training and evaluation

The deployed models/best.pt is an actually fine-tuned YOLOv8n classification model with three classes. The prior two-class model and metadata are preserved in models/pre_three_class/. Model selection was frozen on validation before reserved test evaluation. No robot controls or hardware integration were added.

## Canonical labels and display-only meanings

| YOLO class ID | Canonical label | Function displayed |
| --- | --- | --- |
| 0 | closed_fist | RETURN TO BASE |
| 1 | open_palm | STOP ROBOT |
| 2 | thumbs_up | DONE / RESUME |

The original folder closedfist is an input alias for closed_fist. Original files were not renamed or modified. Numeric IDs have changed from the old two-class model: API consumers should use class_name or the declared mapping. Function values are strings shown in the UI only. /api/model reports robot_controls_enabled=false. There are no actuator, ROS, serial, or robot command endpoints.

## Inspection and split integrity

All 1,000 new PNG files decoded successfully at 640x640. There are only **seven pixel-unique Closed Fist images**, repeated 993 times. Their contact sheet is outputs/closed_fist_contact_sheet.jpg; the complete duplicate inventory is outputs/closed_fist_inspection.json. Visual review is consistent with the supplied Closed Fist label. No bounding boxes exist; classification remains the appropriate task.

| Split | closed_fist | open_palm | thumbs_up | Total files |
| --- | ---: | ---: | ---: | ---: |
| train | 5 | 24 | 24 | 53 |
| validation | 1 | 6 | 6 | 13 |
| test | 1 | 4 | 6 | 11 |

The existing original/isolated image pairs keep their inherited split and capture-group IDs. New fist images are deduplicated and connected by a 63-bit perceptual hash threshold of 8 before grouped 70/15/15 assignment, seed 42. The seven new groups produce a 5/1/1 split. Automated checks found no exact or detected perceptual near duplicates across splits. There are 42 source captures in 31 heuristic groups across the whole dataset, represented by 77 view files. Training has 29 captures / 20 groups; validation 7 captures / 5 groups; test 6 captures / 6 groups. These are not proof of independent people or sessions: identity metadata remains unavailable.

The final test contains just **one distinct fist capture** and correlated original/isolated views of five earlier captures. The old five capture groups had been evaluated in prior project work. They were not used to tune this three-class run, and the new fist test image was first inferred only after selection. Do not interpret this as a newly collected, person-independent field trial.

## Training and preservation

The two existing output-class weight/bias rows were copied by class name when expanding the head, avoiding incorrect numeric-index reuse. A unit test checks the reindexing, and models/three_class_a/metadata.json records both transferred classes. The inherited backbone weights come from the latest background-improved model, not a generic inference fallback.

Training-only sampling balances all three classes at 24 samples each per epoch, 72 total. This oversamples the five unique training fists and does not create extra files or independent data. Real original-background views remain included. Existing near-white border-connected background replacement, horizontal flips, moderate affine transforms, color jitter, mild blur/noise, weight decay, cosine scheduling, reproducible seeds, dropout and early stopping are retained. Mosaic, mixup and erasing remain disabled.

Four validation-assessed candidates were actually trained:

| Candidate | Completed epochs | Final training loss | Final validation loss |
| --- | ---: | ---: | ---: |
| a | 24 | 0.01537 | 0.01357 |
| b | 25 | 0.1231 | 0.09683 |
| resolution | 16 | 0.02393 | 0.01189 |
| stable | 16 | 0.00107 | 0.0012 |

A/B compare learning rates 0.0003/0.00015 and dropout 0.15/0.20. A initially missed one reduced-resolution palm that the prior model accepted. A subsequent pass added training-only resolution variation and Open Palm loss weight 1.25. The selected final pass uses Open Palm loss weight 2.0 and freezes both parameters and BatchNorm state of modules 0-8 while fine-tuning the classifier at LR 0.0005. Losses across differently weighted profiles are not directly comparable. The final-pass checkpoint has all **150/150 inherited extractor tensors identical** to its immediate source checkpoint. The selected ancestry is A -> resolution -> stable; B was rejected. All configuration files, CSV logs and candidate checkpoints are retained.

The selection rule rejects any candidate that reduces Open Palm or Thumbs Up precision/recall in any predefined validation condition, worsens clean views or synthetic sequence stabilization, or fails Closed Fist validation thresholds. The selected candidate passed; the other three were rejected. The entire decision is in selection.json. Repeated selection on only five validation groups creates selection-bias risk; dropout/freeze/early stopping do not establish that overfitting is eliminated.

## Quantitative results at fixed confidence 0.80

Unmodified validation: **13/13 accepted correct**. Unmodified reserved test: **11/11 accepted correct**, from six capture groups. Test per-class precision, recall and F1 are each 1.0 for closed_fist (support 1), open_palm (support 4) and thumbs_up (support 6). For example, Open Palm has TP=4, FP=0, FN=0, so P=4/(4+0)=1, R=4/(4+0)=1 and F1=2PR/(P+R)=1. These tiny supports do not demonstrate universal accuracy.

Synthetic stress variants preserve source labels and test fixed dim/bright exposure, +/-18-degree rotation, dark/blue/patterned backgrounds, patterned-plus-dim backgrounds, and reduced resolution as a distance proxy. They are **correlated derived views**, not additional independent subjects or measured camera distances.

Validation stress: **94/106 -> 105/106 accepted correct**. Open Palm recall stayed at 47/48 while its precision improved from 0.839 to 1.000 because fists are now distinguished. Thumbs Up recall improved from 47/48 to 48/48. Closed Fist was unsupported before and is accepted in all 10 variants derived from its single validation capture.

Reserved stress: **78/90 -> 89/90 accepted correct**. Most of the overall increase is the newly supported Closed Fist class; the additional legacy improvement is one Thumbs Up variant. Detailed per-class comparison:

| Class | Previous precision | New precision | Previous recall | New recall | New F1 | Derived variants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| closed_fist | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 10 |
| open_palm | 0.762 | 1.000 | 1.000 | 1.000 | 1.000 | 32 |
| thumbs_up | 1.000 | 1.000 | 0.958 | 0.979 | 0.989 | 48 |

Closed Fist's previous zeros mean unsupported, not a trained class failing evaluation. The current model maintained or improved legacy precision/recall in every reserved condition. Open Palm test-stress recall remained 32/32, while Thumbs Up recall improved 46/48 -> 47/48. The one remaining stress error is an abstention on a rotated thumbs-up example: {"closed_fist": 0.20864683389663696, "open_palm": 0.0003998900356236845, "thumbs_up": 0.7909533381462097}. No tuning followed the final test.

One-vs-rest classification average precision is 1.0 for each class on the unmodified reserved images, mean AP=1.0. **Detection mAP@0.50 or mAP@0.50:0.95 is not applicable**, because the images have no boxes and the model is a classifier. AP is a ranking metric and can be 1 even when some fixed-threshold stress frames abstain. The report exposes target_macro precision/recall/F1 separately from the confusion-matrix unknown rejection column.

## Stability, latency and app verification

Existing one-frame-at-a-time webcam requests, hand-presence gating, bounded crop preparation, adjustable confidence/frequency/smoothing, full-window stabilization, immediate no-hand reset, blank-image rejection, upload limits, connection errors and resource cleanup are preserved. The UI adds gesture labels, corresponding function text and confidence for all three classes. It sends no robot command.

The five reserved original-photo synthetic sequences had zero raw label changes and correct stabilization after warmup, matching the prior model. A separate seven-frame sequence from the single validation fist capture stabilized to closed_fist for 5/5 frames after the three-frame warmup. These are photo-derived simulations, not physical webcam measurements or evidence of real-video flicker improvement.

Isolated warmed CPU classification (20 runs, 640x640 input, two Torch threads): previous mean 45.31 ms, new mean 44.95 ms; new p95 48.34 ms. Browser hand localization, JPEG encoding, transport and rendering are excluded. This supports maintaining the existing classifier cost; it is not a measured camera FPS claim. Timing recorded during parallel final evaluation is explicitly marked separately.

**31 Python tests passed, 6 FastAPI-specific tests skipped; 5 browser-logic tests passed.** Actual new-model HTTP upload and ASGI WebSocket tests passed for all three gesture/function pairs. Tests also cover no-hand clearing, invalid requests, source-label integrity, split grouping, training balance including RAM-cache alignment, head-row transfer, AP score ties and validation-only regression guards. Python/JavaScript/shell syntax checks passed. The available runtime still uses the documented Starlette compatibility entrypoint; the original FastAPI entrypoint is preserved. Physical webcam operation and execution accuracy of the external MediaPipe detector could not be exercised in this sandbox.

## Artifacts and next use

- models/best.pt and metadata.json: deployed three-class checkpoint, SHA256 cd90f0b11011ab3efc0198a9353c906fb0eb1aed86095cc8730e78ea59a803e4.
- models/pre_three_class/: intact previous checkpoint and metadata.
- data/three_class/manifest.json: labels, aliases, duplicates, source lineage and inherited splits.
- final_test/: per-class metrics, classification AP, confusion matrix, correct examples and training curves.
- final_test_stress.json / previous_test_stress.json: every scored synthetic variant, condition, group and prediction.
- candidate_*_validation.json / selection.json: genuine candidate comparisons and selection reasons.
- condition_comparison.png / per_class_comparison.png: before/after reserved-variant figures.
- fist_stability.json / latency.json: scoped stability/timing evidence.

Stop the currently running server, restart with `./run.sh --port 8001`, and hard-refresh the browser at http://127.0.0.1:8001. Browser hand detection requires its existing model-download access. An already running process retains old in-memory weights until restarted.

The dataset still lacks non-target hand poses and natural empty/cluttered scenes with labels. False-positive rates on such inputs, new-person generalization, partial-occlusion robustness and long-distance recognition are unavailable. Several supplied Open Palm examples contain bent fingers and still need semantic review. More independently captured examples, especially Closed Fist and challenging Open Palm scenes, with person/session IDs and a fresh final test set are required before claiming deployment reliability. No perfect-recognition or eliminated-overfitting claim is made.
