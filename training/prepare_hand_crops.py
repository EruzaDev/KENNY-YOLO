"""Match browser landmark crops; retain capture splits and report localization failures."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def hand_bounds(points, width, height):
    if len(points) != 21:
        return None
    xs, ys = [p.x * width for p in points], [p.y * height for p in points]
    if not all(math.isfinite(x) for x in xs + ys):
        return None
    if max(xs) - min(xs) < 24 or max(ys) - min(ys) < 24:
        return None
    padding = max(max(xs) - min(xs), max(ys) - min(ys)) * .25
    return (max(0, math.floor(min(xs) - padding)), max(0, math.floor(min(ys) - padding)),
            min(width, math.ceil(max(xs) + padding)), min(height, math.ceil(max(ys) + padding)))


def prepare(manifest, model_path, output):
    import mediapipe as mp
    if output.exists():
        raise ValueError('Choose a fresh output directory.')
    source = json.loads(manifest.read_text())
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path),
                                         delegate=mp.tasks.BaseOptions.Delegate.CPU),
        num_hands=2, min_hand_detection_confidence=.6, min_hand_presence_confidence=.6)
    rows, skipped = [], []
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        for i, row in enumerate(source['manifest']):
            with Image.open(row['processed']) as im:
                image = ImageOps.exif_transpose(im).convert('RGB')
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(image)))
            box = hand_bounds(result.hand_landmarks[0], *image.size) if len(result.hand_landmarks) == 1 else None
            if box is None:
                skipped.append(dict(source=row['processed'], label=row['label'], split=row['split'],
                                    reason=f'{len(result.hand_landmarks)} hands or invalid bounds'))
                continue
            crop = image.crop(box)
            scale = min(1, 640 / max(crop.size))
            crop = crop.resize(tuple(max(1, math.floor(v * scale + .5)) for v in crop.size), Image.Resampling.BILINEAR)
            path = output / row['split'] / row['label'] / f'{row["sha256"]}.jpg'
            path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(path, quality=90)
            rows.append(dict(row, source_image=row['processed'], processed=str(path),
                             crop_box=box, capture_sha256=row['sha256']))
            if (i + 1) % 100 == 0:
                print(f'Localized {i+1}/{len(source["manifest"])}; saved {len(rows)}', flush=True)
    splits = {s: dict(Counter(r['label'] for r in rows if r['split'] == s)) for s in ['train', 'val', 'test']}
    if any(set(counts) != set(source['classes']) for counts in splits.values()):
        raise ValueError(f'Localization left a class missing from a split: {splits}')
    report = dict(source, manifest=rows, splits=splits, localization_skipped=skipped,
                  source_manifest=str(manifest), source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  localizer_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  mediapipe_version=mp.__version__, crop_policy='21 landmarks, 25% padding, max 640px, JPEG quality 90',
                  limitations=source['limitations'] + ['Localization failures are excluded and reported, never replaced by whole-frame inputs.'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(splits=splits, skipped=len(skipped)), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('data/ok_gestures_inherited/manifest.json'))
    parser.add_argument('--model', type=Path, default=Path('.runtime/hand_landmarker.task'))
    parser.add_argument('--output', type=Path, default=Path('data/ok_hand_crops'))
    args = parser.parse_args()
    prepare(args.manifest, args.model, args.output)
