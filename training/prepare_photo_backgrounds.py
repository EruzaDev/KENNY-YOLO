"""Create reproducible training-only composites using supplied background photos."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from training.background_augmentation import replace_background, white_background_mask
from training.image_sources import resolve_source, source_images


def photo_crop(image, size, rng):
    """Random aspect-preserving crop, with no distortion of the background."""
    scale = rng.uniform(.65, 1.)
    width, height = max(1, int(image.width * scale)), max(1, int(image.height * scale))
    left = rng.randint(0, image.width - width)
    top = rng.randint(0, image.height - height)
    return ImageOps.fit(image.crop((left, top, left + width, top + height)), size,
                        method=Image.Resampling.LANCZOS,
                        centering=(rng.random(), rng.random()))


def prepare(manifest, backgrounds, output, preview, seed=42):
    if output.exists():
        raise ValueError('Output exists; choose a fresh directory.')
    source = json.loads(manifest.read_text())
    rng = random.Random(seed)
    photos, provenance = [], []
    for path in backgrounds:
        path = resolve_source(path)
        with Image.open(path) as image:
            photo = ImageOps.exif_transpose(image).convert('RGB')
            photo.thumbnail((1600, 1600))
            photos.append(photo.copy())
        provenance.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if not photos:
        raise ValueError('At least one background photo is required.')
    rows, tiles = [], []
    for record in source['manifest']:
        with Image.open(record['processed']) as image:
            image = image.convert('RGB')
        destination = output / record['split'] / record['label'] / Path(record['processed']).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination)
        rows.append(dict(record, processed=str(destination)))
        if record['split'] != 'train' or record['view'] != 'isolated':
            continue
        if white_background_mask(image) is None:
            raise ValueError(f'Isolated image has no replaceable background: {record["processed"]}')
        order = list(range(len(photos)))
        rng.shuffle(order)
        for index in order:
            composite = replace_background(image, photo_crop(photos[index], image.size, rng))
            path = destination.with_stem(destination.stem + f'_photo_{index}')
            composite.save(path)
            rows.append(dict(record, processed=str(path), view='photo_composite',
                             processed_sha256=hashlib.sha256(
                                 str(composite.size).encode() + composite.tobytes()).hexdigest(),
                             background_index=index, source_image=record['processed']))
            if sum(label == record['label'] for label, _ in tiles) < 5:
                tiles.append((record['label'], composite.copy()))
    result = dict(source, manifest=rows, source_manifest=str(manifest),
                  source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  background_photos=provenance, background_seed=seed,
                  splits={s: dict(Counter(r['label'] for r in rows if r['split'] == s))
                          for s in ['train', 'val', 'test']},
                  photo_background_policy='Five random photo crops per isolated training image; original views retained. Val/test unchanged.')
    (output / 'manifest.json').write_text(json.dumps(result, indent=2))
    sheet = Image.new('RGB', (5 * 240, 3 * 270), 'white')
    draw = ImageDraw.Draw(sheet)
    for label_index, label in enumerate(source['classes']):
        for column, (_, image) in enumerate(t for t in tiles if t[0] == label):
            sheet.paste(ImageOps.fit(image, (240, 240)), (column * 240, label_index * 270))
            draw.text((column * 240 + 8, label_index * 270 + 245), label, fill='black')
    preview.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(preview)
    print(json.dumps(result['splits'], indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('data/three_class/manifest.json'))
    parser.add_argument('--backgrounds', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, default=Path('data/photo_backgrounds'))
    parser.add_argument('--preview', type=Path, default=Path('outputs/photo_backgrounds/preview.jpg'))
    args = parser.parse_args()
    prepare(args.manifest, args.backgrounds or source_images('background'),
            args.output, args.preview)


if __name__ == '__main__':
    main()
