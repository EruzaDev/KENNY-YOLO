"""Integrate audited Closed Fist captures, preserving all previous capture splits."""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps
from app.core.gestures import CLASSES
from training.prepare_dataset import fingerprint


def prepare(base: Path, raw_fists: Path, output: Path, seed: int = 42) -> dict:
    if output.exists():
        raise ValueError('Destination exists; choose a new directory to preserve prior splits.')
    parent = json.loads(base.read_text())
    groups, errors = defaultdict(list), []
    for path in sorted(raw_fists.rglob('*')):
        if not path.is_file():
            continue
        try:
            sha, phash = fingerprint(path)
            groups[sha].append(dict(path=str(path), sha256=sha, phash=phash))
        except (OSError, ValueError) as exc:
            errors.append(dict(path=str(path), error=str(exc)))
    if errors:
        raise ValueError(f'Closed Fist integrity errors: {errors}')
    unique = [items[0] for items in groups.values()]
    roots = list(range(len(unique)))
    def root(i):
        while roots[i] != i:
            roots[i] = roots[roots[i]]
            i = roots[i]
        return i
    for i, row in enumerate(unique):
        for j, other in enumerate(unique[:i]):
            if (row['phash'] ^ other['phash']).bit_count() <= 8:
                roots[root(i)] = root(j)
    clusters = defaultdict(list)
    for i, row in enumerate(unique):
        clusters[root(i)].append(row)
    if len(clusters) < 3:
        raise ValueError('Too few independent perceptual groups for train/validation/test.')
    inherited = [dict(r, source=r['processed']) for r in parent['manifest']]
    for row in unique:
        for other in inherited:
            if row['sha256'] == other['sha256']:
                raise ValueError('Closed Fist duplicates a different gesture label; semantic review is required.')
            if (row['phash'] ^ other['phash']).bit_count() <= 8:
                raise ValueError('New class is perceptually close to an existing split; review grouping before integration.')
    ordered = list(clusters.values())
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=len, reverse=True)
    targets = {'train':len(unique)*.7, 'val':len(unique)*.15, 'test':len(unique)*.15}
    counts = Counter()
    next_group = max(r['group'] for r in inherited) + 1
    rows = list(inherited)
    for i, cluster in enumerate(ordered):
        split = min(targets, key=lambda s: ((counts[s]+len(cluster)-targets[s])**2
                                          -(counts[s]-targets[s])**2)/max(targets[s],1))
        counts[split] += len(cluster)
        for row in cluster:
            rows.append(dict(row, source=row['path'], label='closed_fist', split=split,
                             group=next_group+i, view='isolated', capture_sha256=row['sha256']))
    if any(counts[s] == 0 for s in targets):
        raise ValueError('Cannot create all three splits from whole Closed Fist groups.')
    # Existing originals and their isolated views retain all inherited IDs/splits.
    for row in rows:
        destination = output/row['split']/row['label']/(row['sha256']+'.png')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(row['source']) as image:
            ImageOps.exif_transpose(image).convert('RGB').save(destination)
        row['processed'] = str(destination)
        row['processed_sha256'] = fingerprint(destination)[0]
    distribution = {s:dict(Counter(r['label'] for r in rows if r['split']==s)) for s in targets}
    result = dict(classes=list(CLASSES), class_mapping=dict(enumerate(CLASSES)),
                  source_aliases={'closedfist':'closed_fist'}, format='classification', seed=seed,
                  source_manifest=str(base), source_manifest_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                  closed_fist_source_files=sum(map(len,groups.values())), closed_fist_unique_captures=len(unique),
                  closed_fist_duplicate_files=sum(len(v)-1 for v in groups.values()),
                  closed_fist_duplicate_groups={sha:[r['path'] for r in items] for sha,items in groups.items()},
                  splits=distribution, manifest=rows,
                  limitations=['Only seven unique new fist captures; one validation and one test capture.',
                               'Person/session identity is unknown; heuristic grouping cannot prove identity independence.',
                               'Original/isolated views are correlated; no negative class or bounding boxes.'])
    (output/'manifest.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=Path('data/processed_with_originals/manifest.json'))
    parser.add_argument('--fists',type=Path,default=Path('hand_gesture_dataset/closedfist'))
    parser.add_argument('--output',type=Path,default=Path('data/three_class'))
    args=parser.parse_args()
    result=prepare(args.base,args.fists,args.output)
    print(json.dumps({k:result[k] for k in ['classes','splits','closed_fist_unique_captures']},indent=2))


if __name__=='__main__':
    main()
