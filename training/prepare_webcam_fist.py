"""Add human-labeled webcam crops and hand-focused views without session leakage."""
import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from training.background_augmentation import replace_background
from training.hand_crop import hand_focused_crop
from training.prepare_photo_backgrounds import photo_crop
from training.image_sources import resolve_source, source_images


def pixel_hash(image):
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def prepare(base, examples, output, preview):
    if output.exists():
        raise ValueError('Destination exists; use a fresh directory.')
    parent = json.loads(base.read_text())
    frames = []
    # Full-frame downloads are not classifier crops and are deliberately excluded.
    for path in sorted(examples):
        match = re.fullmatch(r'closed_fist__session_(.+)__\d+__hand', path.stem)
        if not match:
            raise ValueError(f'Expected a human-labeled hand crop: {path}')
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            frames.append((path, match[1], image.copy()))
    if not frames:
        raise ValueError('No labeled webcam hand crops were supplied.')
    photos = []
    for item in parent['background_photos']:
        path = resolve_source(item['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Background photo changed after dataset preparation.')
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            image.thumbnail((1600, 1600))
            photos.append(image.copy())
    rng, rows, tiles = random.Random(20261115), [], []

    def save(image, row, name):
        path = output / row['split'] / row['label'] / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        rows.append(dict(row, processed=str(path), processed_sha256=pixel_hash(image)))

    for record in parent['manifest']:
        with Image.open(record['processed']) as image:
            image = image.convert('RGB')
        save(image, record, Path(record['processed']).name)
        if record['split'] not in {'train', 'val'} or record['view'] != 'isolated':
            continue
        # Remove much of the forearm to resemble the browser's tight hand crop.
        crop = hand_focused_crop(image, .55)
        for index in range(-1, len(photos)):
            composite = (crop if index == -1 else
                         replace_background(crop, photo_crop(photos[index], crop.size, rng)))
            save(composite, dict(record, view='hand_crop', crop_cutoff=.55,
                                 source_image=record['processed'], background_index=index),
                 Path(record['processed']).stem + f'_handcrop_{index}.png')
            if record['split'] == 'train' and index in {-1, 0} and not any(
                    label == record['label'] and bg == index for label, bg, _ in tiles):
                tiles.append((record['label'], index, composite.copy()))
    # Every uploaded session stays exclusively in train. No upload is a test sample.
    session_groups = {}
    next_group = max(row['group'] for row in rows) + 1
    for path, session, image in frames:
        group = session_groups.setdefault(session, next_group + len(session_groups))
        digest = pixel_hash(image)
        if any(row['sha256'] == digest for row in rows if row['split'] != 'train'):
            raise ValueError('A webcam training frame duplicates held-out data.')
        save(image, dict(label='closed_fist', split='train', group=group,
                         view='webcam_hand', session=session, path=str(path),
                         source_image=str(path), sha256=digest, capture_sha256=digest,
                         file_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
             f'webcam_{digest}.png')
    result = dict(parent, manifest=rows, source_manifest=str(base),
                  source_manifest_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                  webcam_training_frames=[str(path) for path, _, _ in frames],
                  webcam_sessions_train_only=sorted(session_groups),
                  cropped_validation_policy='Derived crops from existing validation groups only; no webcam frames.',
                  splits={split: dict(Counter(row['label'] for row in rows if row['split'] == split))
                          for split in ['train', 'val', 'test']},
                  limitations=parent.get('limitations', []) + [
                      'Five human-labeled webcam crops from two sessions are training data, not an independent test.',
                      'Both sessions appear to share a subject; no new-person independence is established.',
                      'Inherited test images were previously inspected; they are regression checks, not a fresh test.'])
    (output / 'manifest.json').write_text(json.dumps(result, indent=2))
    sheet = Image.new('RGB', (6 * 200, 250), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, (label, _, image) in enumerate(tiles):
        sheet.paste(ImageOps.fit(image, (200, 200)), (index * 200, 0))
        draw.text((index * 200 + 5, 210), label, fill='black')
    preview.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(preview)
    print(json.dumps(dict(splits=result['splits'], webcam_frames=len(frames),
                         webcam_sessions=len(session_groups)), indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('data/photo_backgrounds/manifest.json'))
    parser.add_argument('--output', type=Path, default=Path('data/webcam_fist'))
    parser.add_argument('--preview', type=Path, default=Path('outputs/webcam_fist/crop_preview.jpg'))
    args = parser.parse_args()
    prepare(args.base, source_images('webcam_hand'),
            args.output, args.preview)


if __name__ == '__main__':
    main()
