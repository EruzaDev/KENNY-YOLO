# Organized source images

- `backgrounds/`: five scene photos used for synthetic training backgrounds.
- `webcam/closed_fist/<session>/hand/`: five human-labeled hand crops used for corrective training.
- `webcam/closed_fist/<session>/full/`: one diagnostic frame, excluded from training.
- `catalog.json`: each image's original path, new path, role and original file SHA-256.

Files were moved, not recompressed. No image was deleted. Original gesture class folders and prepared train/validation/test directories were not reorganized.

Historical experiment manifests/reports retain the paths recorded at training/evaluation time. `training.image_sources.resolve_source` uses the catalog to find moved originals without changing those manifests or their hashes. Raw webcam images are not published by the dashboard's result-image endpoint.
