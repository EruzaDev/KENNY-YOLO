"""Locate organized sources while preserving historical manifest paths and hashes."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve_source(path, root=ROOT):
    path = Path(path)
    candidate = path if path.is_absolute() else root / path
    if candidate.is_file():
        return candidate
    catalog = root / 'assets/images/catalog.json'
    if catalog.is_file():
        for row in json.loads(catalog.read_text())['images']:
            if str(path).replace('\\', '/') == row['original_path'] or candidate == root / row['original_path']:
                target = root / row['path']
                if target.is_file():
                    return target
    raise FileNotFoundError(f'Image source not found: {path}')


def source_images(role, root=ROOT):
    catalog = root / 'assets/images/catalog.json'
    if catalog.is_file():
        return [root / row['path'] for row in json.loads(catalog.read_text())['images'] if row['role'] == role]
    patterns = {'background': 'IMG_*.jpg', 'webcam_hand': 'closed_fist__session_*__hand.jpg'}
    return sorted(root.glob(patterns[role]))
