"""Evaluate held-out hand captures composited on known training background photos."""
import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from app.services.detector import Detector
from training.background_augmentation import replace_background
from training.prepare_photo_backgrounds import photo_crop
from training.image_sources import resolve_source


def evaluate(model, data, output):
    if output.exists():
        raise ValueError('Use a fresh output directory.')
    manifest = data / 'manifest.json'
    source = json.loads(manifest.read_text())
    detector = Detector(model)
    detector.load()
    if detector.metadata['dataset_manifest_sha256'] != hashlib.sha256(manifest.read_bytes()).hexdigest():
        raise ValueError('Evaluation manifest differs from training provenance.')
    photos = []
    for row in source['background_photos']:
        path = resolve_source(row['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Background photo changed after training.')
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            image.thumbnail((1600, 1600))
            photos.append(image.copy())
    records = [r for r in source['manifest'] if r['split'] == 'test' and r['view'] == 'isolated']
    rng, rows = random.Random(20261114), []
    sheet = Image.new('RGB', (len(photos) * 240, len(records) * 290), 'white')
    draw = ImageDraw.Draw(sheet)
    output.mkdir(parents=True)
    for index, record in enumerate(records):
        with Image.open(record['processed']) as image:
            image = image.convert('RGB')
        for bg_index, photo in enumerate(photos):
            frame = replace_background(image, photo_crop(photo, image.size, rng))
            result = detector.predict(frame, .8)
            raw = max(result['scores'], key=result['scores'].get)
            prediction = result['predictions'][0]['class_name'] if result['predictions'] else 'unknown'
            rows.append(dict(source=record['processed'], group=record['group'], actual=record['label'],
                             background_index=bg_index, predicted=prediction, raw_top1=raw,
                             scores=result['scores']))
            x, y = bg_index * 240, index * 290
            sheet.paste(ImageOps.fit(frame, (240, 240)), (x, y))
            color = 'darkgreen' if prediction == record['label'] else 'darkred'
            draw.text((x + 5, y + 243), f"True: {record['label']}", fill='black')
            draw.text((x + 5, y + 260), f'{prediction}: {result["scores"][raw]:.1%}', fill=color)
    report = dict(checkpoint_sha256=detector.metadata['checkpoint_sha256'], split='test',
                  confidence=.8, count=len(rows), source_images=len(records),
                  capture_groups=len({r['group'] for r in rows}),
                  accepted_correct=sum(r['actual'] == r['predicted'] for r in rows),
                  raw_top1_correct=sum(r['actual'] == r['raw_top1'] for r in rows),
                  accuracy=sum(r['actual'] == r['predicted'] for r in rows) / len(rows),
                  limitations=['Held-out hand captures, but background photos were used for training.',
                               'Synthetic correlated composites, not independent real-camera images.',
                               'No new-background, new-person, negative-scene or physical webcam accuracy established.'],
                  predictions=rows)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    sheet.save(output / 'predictions.jpg')
    print(json.dumps({k: v for k, v in report.items() if k != 'predictions'}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('models/photo_backgrounds/best.pt'))
    parser.add_argument('--data', type=Path, default=Path('data/photo_backgrounds'))
    parser.add_argument('--output', type=Path, default=Path('outputs/photo_backgrounds/photo_test'))
    args = parser.parse_args()
    evaluate(args.model, args.data, args.output)


if __name__ == '__main__':
    main()
