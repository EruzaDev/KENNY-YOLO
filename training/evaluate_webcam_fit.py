"""Training-frame fit diagnostic only; never report this as held-out camera accuracy."""
import argparse
import json
from pathlib import Path

from PIL import Image

from app.services.detector import Detector
from training.image_sources import source_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('models/webcam_fist/best.pt'))
    parser.add_argument('--output', type=Path, default=Path('outputs/webcam_fist/training_frame_fit.json'))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Report exists; choose a fresh output.')
    paths = source_images('webcam_hand')
    if not paths:
        raise ValueError('No webcam training frames found.')
    detector = Detector(args.model)
    detector.load()
    rows = []
    for path in paths:
        with Image.open(path) as image:
            result = detector.predict(image.convert('RGB'), .8)
        rows.append(dict(path=str(path), actual='closed_fist', scores=result['scores'],
                         predicted=result['predictions'][0]['class_name'] if result['predictions'] else 'unknown'))
    report = dict(checkpoint_sha256=detector.metadata['checkpoint_sha256'], split='train',
                  kind='Training-frame fit diagnostic, NOT held-out accuracy',
                  confidence=.8, count=len(rows),
                  accepted_correct=sum(r['predicted'] == r['actual'] for r in rows), predictions=rows,
                  limitations=['All these frames were used for corrective training.',
                               'Fresh independent camera frames are needed to evaluate live improvement.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
