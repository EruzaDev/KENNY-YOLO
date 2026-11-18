"""Audit immutable class folders, deduplicate and group before stratified splitting."""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

CLASSES = ('ok', 'open_palm', 'thumbs_up')  # Ultralytics alphabetic folder order.
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
_PIL_OPEN = Image.open  # Keep the standard decoder before optional Ultralytics monkey patches.


def fingerprint(path: Path) -> tuple[str, int]:
    try:
        image = _PIL_OPEN(path)
    except ImportError as exc:
        raise ValueError('Image could not be decoded; no optional format fallback is used.') from exc
    with image:
        image.load()
        image = ImageOps.exif_transpose(image).convert('RGB')
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        gray = np.array(image.convert('L').resize((32, 32))) / 255.0
        dct = cv2.dct(gray.astype(np.float32))[:8, :8].flatten()[1:]
        bits = dct > np.median(dct)
        return digest, sum(int(bit) << i for i, bit in enumerate(bits))


def audit(raw: Path) -> dict:
    records, errors = [], []
    for label in CLASSES:
        if not (raw / label).is_dir():
            raise ValueError(f'Missing class directory: {raw / label}')
        for path in sorted((raw / label).rglob('*')):
            if not path.is_file():
                continue
            if path.suffix.lower() not in EXTENSIONS:
                errors.append({'path': str(path), 'error': 'Unsupported file'})
                continue
            try:
                sha, phash = fingerprint(path)
                records.append(dict(path=str(path), label=label, sha256=sha, phash=phash))
            except (OSError, ValueError) as exc:
                errors.append(dict(path=str(path), error=str(exc)))
    by_hash = defaultdict(list)
    for record in records:
        by_hash[record['sha256']].append(record)
    conflicts = [items for items in by_hash.values() if len({r['label'] for r in items}) > 1]
    return dict(format='image-level classification', classes=list(CLASSES),
                counts=dict(Counter(r['label'] for r in records)),
                unique_counts=dict(Counter(v[0]['label'] for v in by_hash.values())),
                duplicate_files=len(records) - len(by_hash), errors=errors,
                label_conflicts=conflicts, records=records)


def prepare(raw: Path, destination: Path, report_path: Path, seed: int = 42,
            distance: int = 8, sessions: dict | None = None,
            inherit_splits: Path | None = None) -> dict:
    if inherit_splits is not None and report_path.resolve() == inherit_splits.resolve():
        raise ValueError('Choose a separate report path to preserve the inherited split manifest.')
    report = audit(raw)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    if report['errors'] or report['label_conflicts']:
        raise ValueError('Dataset audit failed; inspect report before proceeding.')
    if destination.exists():
        raise ValueError('Output already exists; choose a new directory to preserve splits.')
    if destination.resolve().is_relative_to(raw.resolve()):
        raise ValueError('Processed output must be outside the raw dataset.')
    unique = {r['sha256']: r for r in report['records']}
    records = list(unique.values())
    parent = list(range(len(records)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    for i, a in enumerate(records):
        for j in range(i):
            b = records[j]
            sa = (sessions or {}).get(str(Path(a['path']).relative_to(raw)))
            sb = (sessions or {}).get(str(Path(b['path']).relative_to(raw)))
            delta = (a['phash'] ^ b['phash']).bit_count()
            if delta <= distance or (sa is not None and sa == sb):
                parent[root(i)] = root(j)
                pairs.append(dict(a=a['path'], b=b['path'], hamming=delta))
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[root(i)].append(record)
    rng = random.Random(seed)
    # Assign entire connected groups while balancing each class toward 70/15/15.
    ordered = list(groups.values())
    rng.shuffle(ordered)
    ordered.sort(key=len, reverse=True)
    targets = {s: {c: report['unique_counts'][c] * ratio for c in CLASSES}
               for s, ratio in [('train', .70), ('val', .15), ('test', .15)]}
    counts = {s: Counter() for s in targets}
    assignment = []
    inherited = {}
    if inherit_splits is not None:
        previous = json.loads(inherit_splits.read_text())
        inherited = {r['sha256']: r['split'] for r in previous['manifest']}
        if not inherited.keys() <= unique.keys():
            raise ValueError('Inherited captures are missing from the current dataset.')
    pending = []
    for group in ordered:
        fixed = {inherited[r['sha256']] for r in group if r['sha256'] in inherited}
        if len(fixed) > 1:
            raise ValueError('A similarity group spans conflicting inherited splits.')
        if fixed:
            split = fixed.pop()
            counts[split].update(r['label'] for r in group)
            assignment.extend(dict(r, split=split, group=root(records.index(r))) for r in group)
        else:
            pending.append(group)
    for group in pending:
        gc = Counter(r['label'] for r in group)
        def cost(split):
            return sum(((counts[split][c] + gc[c] - targets[split][c]) ** 2
                        - (counts[split][c] - targets[split][c]) ** 2)
                       / max(targets[split][c], 1) for c in CLASSES)
        split = min(targets, key=cost)
        counts[split].update(gc)
        assignment.extend(dict(r, split=split, group=root(records.index(r))) for r in group)
    if any(counts[s][c] == 0 for s in counts for c in CLASSES):
        raise ValueError('Too few independent groups for three class-complete splits. Collect more data.')
    for record in assignment:
        out = destination / record['split'] / record['label'] / (record['sha256'] + '.png')
        out.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(record['path']) as image:
            ImageOps.exif_transpose(image).convert('RGB').save(out)
        record['processed'] = str(out)
    report.update(seed=seed, near_duplicate_distance=distance, near_duplicate_pairs=pairs,
                  inherited_split_manifest=str(inherit_splits) if inherit_splits else None,
                  inherited_capture_count=len(inherited),
                  group_count=len(groups), splits={s: dict(c) for s, c in counts.items()},
                  manifest=assignment, session_mapping_supplied=bool(sessions),
                  limitations=['Perceptual grouping is heuristic; unknown person/session identity leakage cannot be ruled out.',
                               'Target gesture images only; no negatives or annotated boxes.',
                               'Image-level labels require manual semantic review.'])
    report_path.write_text(json.dumps(report, indent=2))
    (destination / 'manifest.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('hand_gesture_dataset'))
    parser.add_argument('--output', type=Path, default=Path('data/processed'))
    parser.add_argument('--report', type=Path, default=Path('outputs/dataset_report.json'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--near-distance', type=int, default=8)
    parser.add_argument('--sessions', type=Path, help='JSON: relative filename -> person/session ID')
    parser.add_argument('--inherit-splits', type=Path, help='Preserve earlier capture splits when extending a trained dataset')
    args = parser.parse_args()
    result = prepare(args.raw, args.output, args.report, args.seed, args.near_distance,
                     json.loads(args.sessions.read_text()) if args.sessions else None, args.inherit_splits)
    print(json.dumps({k: result[k] for k in ['counts', 'unique_counts', 'group_count', 'splits']}, indent=2))


if __name__ == '__main__':
    main()
