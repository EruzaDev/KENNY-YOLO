# Model improvement experiment

A new YOLOv8n-cls checkpoint was actually trained and promoted after a validation-only comparison. It replaces models/best.pt; models/baseline preserves the original checkpoint and metadata.

Changes: include the 35 provided original JPG photos alongside their corresponding isolated views, at inherited splits; use 320px input, freeze the first five backbone modules, dropout 0.15, AdamW at 0.0002, weight decay 0.001, cosine scheduling and patience 20. No new images, synthetic negatives, bounding-box labels or external training datasets were introduced. PNG training copies preserve aspect ratio and downsample oversized original photos. There remain only 35 source captures in 24 perceptual groups, with unknown person/session overlap.

Split inventory: train=48 views of 24 captures in 15 groups; validation=12 views of 6 captures in 4 groups; test inventory=10 views of 5 captures in 5 groups. Original/isolated versions share the same split and group. No exact or detected perceptual duplicates cross splits. Original source files and the initial processed dataset remain untouched.

Training completed 34 epochs in approximately 136.2 seconds on CPU. Early stopping reached best fitness at epoch 14 and stopped after 20 further epochs; Ultralytics may save later tied-best weights. Final training loss=0.03968, validation loss=0.02252. These curves do not establish absence of overfitting.

Comparison at the fixed 0.80 threshold:

| Validation view | Previous accepted correct | Candidate accepted correct |
| --- | ---: | ---: |
| Isolated | 6/6 | 6/6 |
| Original background | 5/6 | 6/6 |
| Combined | 11/12 | 12/12 |

The previous model rejected one original-background open palm (open_palm probability 0.64989); the candidate correctly accepted the same validation image at 0.97372. It was an abstention, not an incorrect thumbs-up classification. Group-balanced negative log likelihood improved from 0.13158 to 0.02720. Class precision/recall and correctness by view/group did not decrease, satisfying the predeclared promotion rule. Confidence improvement on a small positive-only validation set does not establish calibration or unknown-gesture rejection. Timing in the comparison ran during other local evaluation work and is not a controlled latency benchmark.

No test images were used for comparison or promotion. The original five test captures were evaluated previously for the baseline; they were not rerun to choose this model. There is no newly independent held-out test result for the promoted candidate. Existing outputs/evaluation statistics belong exclusively to the preserved baseline hash, and the app suppresses them for the new checkpoint. The current checkpoint's validation summary is exposed separately through /api/model.

Current SHA256: 4b38fbeb97cf6bb3c8d3a86ed2b26d4ec02756481749f5ef2489d443cc7ec16b.

Artifacts: validation_comparison.json contains every validation score, group, view and selection decision. candidate_validation/ contains classifier metrics, confusion matrix, sample predictions and training curves. The trained model continues to require the browser hand-presence gate to reduce no-hand predictions; a two-class classifier does not learn a negative class from this dataset.

Verification: 18 Python tests passed, 6 FastAPI-specific tests skipped, and 5 isolated browser-logic tests passed. Actual new-model image upload and WebSocket frame inference passed through the local ASGI runner; data integrity/group-lineage checks and validation-only selection checks passed. Python compilation and shell syntax checks passed. Physical-camera and MediaPipe accuracy remain unverified.

Perfect detection is not established or guaranteed. A useful next evaluation needs newly captured thumbs_up/open_palm examples plus empty scenes and other hand poses, with multiple people, lights, backgrounds, distances and identifiable sessions. Keep separate sessions/people for training, validation and final testing. Do not substitute more duplicate copies for independent data.

Restart ./run.sh and hard-refresh the browser to load this checkpoint; an already-running process retains its old in-memory weights.
