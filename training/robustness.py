"""Validation stress selection and fixed-protocol reserved-group evaluation."""
import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from app.services.detector import Detector
from app.services.inference import Stabilizer
from training.background_augmentation import patterned_background, replace_background
from training.hand_crop import hand_focused_crop
from training.metrics import classification_metrics, classification_average_precision


def variants(image: Image.Image, view: str, distance: bool = False):
    yield 'clean', image
    yield 'dim', ImageEnhance.Brightness(image).enhance(.6)
    yield 'bright', ImageEnhance.Brightness(image).enhance(1.3)
    yield 'rotate_left', image.rotate(18, expand=True, fillcolor='white')
    yield 'rotate_right', image.rotate(-18, expand=True, fillcolor='white')
    if distance:
        reduced = image.resize((max(1, image.width//3), max(1, image.height//3)), Image.Resampling.BILINEAR)
        yield 'reduced_resolution', reduced.resize(image.size, Image.Resampling.BILINEAR)
    if view == 'isolated':
        yield 'background_dark', replace_background(image, Image.new('RGB', image.size, (35, 40, 48)))
        yield 'background_blue', replace_background(image, Image.new('RGB', image.size, (45, 90, 160)))
        clutter = patterned_background(image.size, random.Random(20261113), 'clutter')
        composite = replace_background(image, clutter)
        yield 'background_clutter', composite
        yield 'background_clutter_dim', ImageEnhance.Brightness(composite).enhance(.7)


def summarize(rows: list[dict], classes=None) -> dict:
    classes = classes or sorted({r['label'] for r in rows})
    labels = list(classes) + ['unknown']
    _, report = classification_metrics([r['label'] for r in rows], [r['prediction'] for r in rows], labels)
    groups = defaultdict(list)
    for row in rows:
        groups[row['group']].append(row)
    return dict(count=len(rows), group_count=len(groups), accuracy=report['accuracy'],
                per_class={c: report[c] for c in classes},
                classification_ap=classification_average_precision([r['label'] for r in rows], [r['scores'] for r in rows], classes),
                group_nll=float(np.mean([np.mean([r['nll'] for r in group]) for group in groups.values()])))


def evaluate(model_path: Path, manifest: Path, output: Path, confidence: float = .8, split: str = 'val', distance: bool = False, hand_crops: bool = False):
    if output.exists():
        raise ValueError('Stress report already exists; keep recorded comparison evidence immutable.')
    source = json.loads(manifest.read_text())
    if split not in {'val', 'test'}:
        raise ValueError('Only validation and reserved test splits are supported.')
    records = [r for r in source['manifest'] if r['split'] == split]
    if not records:
        raise ValueError('Selected split is empty.')
    detector = Detector(model_path)
    detector.load()
    allowed_lineage = {hashlib.sha256(manifest.read_bytes()).hexdigest(), source.get('source_manifest_sha256')}
    if detector.metadata['dataset_manifest_sha256'] not in allowed_lineage:
        raise ValueError('Model and validation manifest lineage differ.')
    rows, sequences = [], []
    with Image.open(records[0]['processed']) as image:
        detector.predict(image.convert('RGB'), confidence)
    for record in records:
        with Image.open(record['processed']) as source_image:
            image = source_image.convert('RGB')
        frames = list(variants(image, record['view'], distance))
        if hand_crops and record['view'] == 'isolated':
            crop = hand_focused_crop(image)
            frames.extend(('hand_crop_' + name, frame) for name, frame in variants(crop, 'isolated', distance))
        for condition, frame in frames:
            result = detector.predict(frame, confidence)
            guess = result['predictions'][0]['class_name'] if result['predictions'] else 'unknown'
            rows.append(dict(path=record['processed'], group=record['group'], label=record['label'],
                             view=record['view'], condition=condition, prediction=guess,
                             scores=result['scores'], nll=-math.log(max(result['scores'].get(record['label'], 0), 1e-12)),
                             latency_ms=result['latency_ms']))
        if record['view'] == 'original':
            stabilizer, raw, stable = Stabilizer(), [], []
            for angle, light in zip([-8, -4, 0, 4, 8, 4, 0], [.75, .9, 1, .85, 1.1, .8, 1]):
                frame = ImageEnhance.Brightness(image.rotate(angle, expand=True, fillcolor='white')).enhance(light)
                prediction = detector.predict(frame, confidence)
                raw.append(prediction['predictions'][0]['class_name'] if prediction['predictions'] else None)
                stable.append(stabilizer.apply(prediction, 3)['stable_gesture'])
            sequences.append(dict(path=record['processed'], label=record['label'], raw=raw, stabilized=stable,
                                  raw_label_changes=sum(a != b for a, b in zip(raw, raw[1:])),
                                  stable_correct_after_warmup=sum(x == record['label'] for x in stable[2:]) / len(stable[2:])))
    classes = source['classes']
    conditions = {c: summarize([r for r in rows if r['condition'] == c], classes)
                  for c in sorted({r['condition'] for r in rows})}
    views = {v: summarize([r for r in rows if r['view'] == v and r['condition'] == 'clean'], classes)
             for v in ['original', 'isolated']}
    report = dict(checkpoint=str(model_path), checkpoint_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), split=split, test_used=split == 'test',
                  confidence=confidence, original_image_count=len(records),
                  reduced_resolution_distance_proxy=distance, hand_crop_diagnostics=hand_crops,
                  capture_group_count=len({r['group'] for r in records}),
                  classes=classes, unsupported_classes=[c for c in classes if c not in detector.classes],
                  overall=summarize(rows, classes), conditions=conditions, clean_views=views,
                  stability=dict(sequence_count=len(sequences),
                                 mean_raw_label_changes=float(np.mean([s['raw_label_changes'] for s in sequences])),
                                 stable_correct_after_warmup=float(np.mean([s['stable_correct_after_warmup'] for s in sequences]))),
                  predictions=rows, sequences=sequences,
                  limitations=[f'Lighting/orientation/background changes are synthetic diagnostic variants of {split} data.',
                               'Variants are correlated, not additional independent test examples.',
                               'No natural cluttered-background or new-person camera test set exists.',
                               'No negative-class false-positive rate or physical-video stability is established.',
                               'Latency recorded during local CPU activity is not a controlled device benchmark.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ['predictions', 'sequences']}, indent=2))
    return report


def qualifies(baseline: dict, candidate: dict) -> tuple[bool, list[str]]:
    failures = []
    if (baseline.get('manifest_sha256') != candidate.get('manifest_sha256')
            or baseline.get('confidence') != candidate.get('confidence')
            or baseline.get('split') != 'val' or candidate.get('split') != 'val'):
        return False, ['Reports do not describe the same validation protocol.']
    for condition, reference in baseline['conditions'].items():
        current = candidate['conditions'][condition]
        for label in ['open_palm', 'thumbs_up']:
            for metric in ['precision', 'recall']:
                if current['per_class'][label][metric] + 1e-12 < reference['per_class'][label][metric]:
                    failures.append(f'{condition}: {label} {metric} decreased')
    for view, reference in baseline['clean_views'].items():
        if candidate['clean_views'][view]['accuracy'] < reference['accuracy']:
            failures.append(f'Clean {view} accuracy decreased')
    for key in ['stable_correct_after_warmup']:
        if candidate['stability'][key] < baseline['stability'][key]:
            failures.append('Synthetic stabilized-sequence correctness decreased')
    if candidate['stability']['mean_raw_label_changes'] > baseline['stability']['mean_raw_label_changes']:
        failures.append('Synthetic sequence label changes increased')
    improved_background = any(
        candidate['conditions'][c]['per_class']['open_palm']['recall'] > baseline['conditions'][c]['per_class']['open_palm']['recall']
        for c in baseline['conditions'] if c.startswith('background_'))
    if not improved_background:
        failures.append('No measured open-palm background recall improvement')
    return not failures, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('data/processed_with_originals/manifest.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--confidence', type=float, default=.8)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--distance', action='store_true', help='Include reduced-resolution distance proxy (not measured camera distance).')
    parser.add_argument('--hand-crops', action='store_true', help='Additional isolated-hand crop diagnostics.')
    args = parser.parse_args()
    evaluate(args.model, args.manifest, args.output, args.confidence, args.split, args.distance, args.hand_crops)


if __name__ == '__main__':
    main()
