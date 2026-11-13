"""Add provided original photos to their existing capture groups without resplitting."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from PIL import Image, ImageOps
from training.prepare_dataset import fingerprint


def original_number(master_index: int, label: str) -> int | None:
    if master_index == 1:
        return None
    boundary = 13 if label == 'thumbs_up' else 14
    return master_index - 1 if master_index < boundary else 300 + master_index - boundary


def prepare(base: Path, originals: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError('Destination exists. Choose a new directory; splits are immutable.')
    source = json.loads(base.read_text())
    inherited = source['manifest']
    by_pixels = {row['sha256']:row for row in inherited}
    original_paths = {}
    for path in sorted(originals.glob('*.jpg')):
        match = re.fullmatch(r'(open_palm|thumbs_up)(?: \((\d+)\))?', path.stem, flags=re.I)
        if not match:
            raise ValueError(f'Unrecognized original name: {path.name}')
        original_paths[(match[1].lower(), int(match[2]) if match[2] else None)] = path
    records = []
    for row in inherited:
        records.append(dict(row, view='isolated', source=row['processed']))
    used = set()
    for label, prefix, total in [('open_palm','openpalm',17), ('thumbs_up','thumbsup',18)]:
        for index in range(1, total+1):
            master = originals / f'{prefix}_master_{index}.png'
            digest, _ = fingerprint(master)
            if digest not in by_pixels or by_pixels[digest]['label'] != label:
                raise ValueError(f'Processed master lacks a consistent inherited class/split: {master}')
            inherited_row = by_pixels[digest]
            original = original_paths[(label, original_number(index, label))]
            if original in used:
                raise ValueError('Duplicate original mapping')
            used.add(original)
            sha, phash = fingerprint(original)
            records.append(dict(path=str(original), source=str(original), label=label,
                                split=inherited_row['split'], group=inherited_row['group'],
                                capture_sha256=digest, sha256=sha, phash=phash, view='original'))
    if used != set(original_paths.values()):
        raise ValueError('Not all originals are mapped to existing capture groups.')
    # Validate exact duplicates and inherited groups before writing any files.
    hashes = {}
    groups = {}
    for row in records:
        hashes.setdefault(row['sha256'], set()).add(row['split'])
        groups.setdefault(row['group'], set()).add(row['split'])
    if any(len(splits)>1 for splits in [*hashes.values(),*groups.values()]):
        raise ValueError('A duplicate/capture group crosses splits.')
    for i, first in enumerate(records):
        for second in records[:i]:
            if (first['split'] != second['split']
                    and (first['phash'] ^ second['phash']).bit_count() <= source['near_duplicate_distance']):
                raise ValueError('A new perceptual near duplicate crosses inherited splits; review the source groups.')
    counts = {split:{label:0 for label in source['classes']} for split in ['train','val','test']}
    for row in records:
        destination = output / row['split'] / row['label'] / f'{row["sha256"]}.png'
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Preserve aspect ratio while downsampling large originals; save the result as PNG.
        with Image.open(row['source']) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            image.thumbnail((1280,1280), Image.Resampling.LANCZOS)
            image.save(destination)
        row['processed'] = str(destination)
        row['processed_sha256'] = fingerprint(destination)[0]
        counts[row['split']][row['label']] += 1
    report = dict(classes=source['classes'], manifest=records, splits=counts,
                  source_manifest=str(base), source_manifest_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                  independent_capture_count=len(inherited), independent_group_count=len(groups),
                  format='classification',
                  limitations=['Original and isolated views are correlated, not additional independent captures.',
                               'Inherited test groups are untouched and excluded from model selection.',
                               'Person/session independence is unknown; no negative examples are provided.'])
    (output/'manifest.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=Path('data/processed/manifest.json'))
    parser.add_argument('--originals',type=Path,default=Path('hand_gesture_dataset/master_images'))
    parser.add_argument('--output',type=Path,default=Path('data/processed_with_originals'))
    args=parser.parse_args()
    result=prepare(args.base,args.originals,args.output)
    print(json.dumps({k:result[k] for k in ['splits','independent_capture_count','independent_group_count']},indent=2))


if __name__=='__main__':
    main()
