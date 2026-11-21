"""Evaluate the captured-gesture model on its untouched chronological test split."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image
from app.services.detector import Detector


def evaluate(model_path: Path, data: Path, output: Path) -> dict:
    detector = Detector(model_path)
    detector.load()
    labels = detector.classes
    paths = [(path, path.parent.name) for label in labels
             for path in sorted((data / 'test' / label).glob('*')) if path.is_file()]
    matrix = {actual: Counter() for actual in labels}
    for path, actual in paths:
        with Image.open(path) as image:
            result = detector.predict(image.convert('RGB'), confidence=0)
        predicted = max(result['scores'], key=result['scores'].get)
        matrix[actual][predicted] += 1
    count = len(paths)
    correct = sum(matrix[label][label] for label in labels)
    report = {}
    for label in labels:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[actual][label] for actual in labels)
        precision = tp / predicted if predicted else 0
        recall = tp / support if support else 0
        report[label] = {'precision': precision, 'recall': recall,
                         'f1-score': 2*precision*recall/(precision+recall) if precision+recall else 0,
                         'support': support}
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    result = {'task': 'classify', 'split': 'test', 'checkpoint_sha256': digest,
              'confidence': 0, 'count': count, 'accuracy': correct/count if count else 0,
              'raw_top1_accuracy': correct/count if count else 0, 'coverage': 1.0,
              'classification_report': report,
              'confusion_matrix': [[matrix[actual][pred] for pred in labels] for actual in labels],
              'labels': labels,
              'limitations': ['Chronological test frames come from the same single-person gesture capture sessions as training.',
                              'This does not measure accuracy on new people, cameras, rooms, or lighting.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    metadata_path = model_path.with_name('metadata.json')
    metadata = json.loads(metadata_path.read_text())
    metadata['test_metrics_path'] = str(output)
    metadata['test_metrics_summary'] = {'accuracy': result['accuracy'], 'count': count,
                                        'split': 'chronological test frames from captured sessions'}
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('models/gesture_captures/best.pt'))
    parser.add_argument('--data', type=Path, default=Path('data/gesture_captures'))
    parser.add_argument('--output', type=Path, default=Path('outputs/gesture_captures/test_metrics.json'))
    args = parser.parse_args()
    result = evaluate(args.model, args.data, args.output)
    print(json.dumps({key: result[key] for key in ('count', 'accuracy', 'classification_report')}, indent=2))


if __name__ == '__main__':
    main()
