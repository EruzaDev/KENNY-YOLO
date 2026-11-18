"""Validation-only crop/background diagnostics; derived views are not camera trials."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from app.services.detector import Detector
from training.background_augmentation import replace_background


def evaluate(model_path, manifest_path):
    detector = Detector(model_path)
    detector.load()
    manifest = json.loads(manifest_path.read_text())
    photos = []
    for path in sorted(Path('assets/images/backgrounds').glob('*.jpg')):
        with Image.open(path) as im:
            photos.append(im.convert('RGB'))
    rows = []
    for record in manifest['manifest']:
        if record['split'] != 'val':
            continue
        with Image.open(record['processed']) as im:
            image = im.convert('RGB')
        views = [('clean', image)]
        if record['label'] != 'ok':
            views += [(f'photo_{i}', replace_background(image, photo)) for i, photo in enumerate(photos)]
        for kind, pixels in views:
            result = detector.predict(pixels, .8)
            guess = result['predictions'][0]['class_name'] if result['predictions'] else 'unknown'
            rows.append(dict(source=record['processed'], group=record['group'], actual=record['label'],
                             view=kind, predicted=guess, scores=result['scores']))
    def summarize(items):
        return dict(count=len(items), accepted_correct=sum(r['predicted'] == r['actual'] for r in items),
                    false_ok=sum(r['predicted'] == 'ok' and r['actual'] != 'ok' for r in items),
                    abstentions=sum(r['predicted'] == 'unknown' for r in items),
                    predictions=dict(Counter(r['predicted'] for r in items)),
                    per_class={c:dict(count=sum(r['actual'] == c for r in items),
                                      accepted_correct=sum(r['actual'] == c and r['predicted'] == c for r in items))
                               for c in manifest['classes']})
    return dict(checkpoint_sha256=detector.metadata['checkpoint_sha256'], checkpoint=str(model_path),
                manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(), split='val', confidence=.8,
                kind='Existing capture crops and derived background diagnostics; not live camera accuracy.',
                clean=summarize([r for r in rows if r['view'] == 'clean']),
                photo_backgrounds=summarize([r for r in rows if r['view'] != 'clean']),
                predictions=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('data/ok_hand_crops/manifest.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Choose a fresh diagnostic output.')
    result = evaluate(args.model, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:result[k] for k in ['clean', 'photo_backgrounds']}, indent=2))
