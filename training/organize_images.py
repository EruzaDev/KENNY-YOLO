"""Move loose source photos into categorized folders and record hash-verified relocations."""
import hashlib
import json
import re
from pathlib import Path

from training.image_sources import ROOT


def organize(root=ROOT):
    catalog_path = root / 'assets/images/catalog.json'
    catalog = json.loads(catalog_path.read_text()) if catalog_path.is_file() else {'images': []}
    plans = []
    for path in sorted(root.glob('*.jpg')):
        if path.name.startswith('IMG_'):
            role, target = 'background', Path('assets/images/backgrounds') / path.name
            extra = {}
        else:
            match = re.fullmatch(r'closed_fist__session_(.+)__\d+__(hand|full)\.jpg', path.name)
            if not match:
                continue
            session, kind = match.groups()
            role = f'webcam_{kind}'
            target = Path('assets/images/webcam/closed_fist') / session / kind / path.name
            extra = dict(label='closed_fist', session=session,
                         usage='training example' if kind == 'hand' else 'diagnostic full frame; not used for training')
        if (root / target).exists():
            raise ValueError(f'Destination already exists; refusing to overwrite: {target}')
        row = dict(original_path=path.name, path=target.as_posix(), role=role,
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **extra)
        plans.append((path, root / target, row))
    # Preflight all destinations before moving; never rewrite dataset manifests.
    for path, target, row in plans:
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == row['sha256']
        catalog['images'].append(row)
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_text(json.dumps(catalog, indent=2))
    return catalog


if __name__ == '__main__':
    result = organize()
    print(json.dumps({'organized_images': len(result['images']), 'catalog': 'assets/images/catalog.json'}, indent=2))
