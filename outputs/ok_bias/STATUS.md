# Camera bias repair

The reported live error was Open Palm / Thumbs Up appearing as OK. Exact failing
user crops were not supplied in this session; live correction remains unverified.

The former checkpoint passed 128/128 clean localized validation crops, but only
20/30 derived background views of six legacy validation captures. Two views were
incorrectly accepted as OK, and eight abstained. The selected replacement passed
128/128 clean crops and 30/30 background views, with no false OK predictions in
that diagnostic. Selection used validation only; `selection.json` records hashes.
These background photos are used in training augmentation too, so this is a
controlled shortcut check, not evidence of new-room or camera generalization.

Training used 608 localized training captures, inverse-frequency sampling (384
draws per epoch), varied real photo backgrounds and macro-recall checkpoint
selection. The 35 legacy capture splits were preserved. MediaPipe excluded 167
OK sources; no whole-frame fallback was used. Only 17 Open Palm and 18 Thumbs Up
source captures are available. Identity/session independence is unverified.

The 10-epoch checkpoint passed 132/132 reserved localized test images at 0.80.
Legacy test images have been inspected in older experiments; this is regression
evidence rather than a fresh physical-camera trial.

The browser guard never supplies a new gesture label. Broad fingertip-distance
and open-hand vetoes were rejected because they also vetoed valid oblique OK
poses. The retained clear-thumbs-up veto rejected zero of 122 OK validation
captures, and detected contradictory anatomy in two of three Thumbs Up captures.
This guard diagnostic uses predicted landmarks, not manually annotated anatomy.

To verify the actual user failure, hard-refresh the running app, reproduce both
gestures and attach the exact human-labeled crops from Collect a labeled example.
