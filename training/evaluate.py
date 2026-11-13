"""One final held-out evaluation, with full-frame preprocessing identical to deployment."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('models/best.pt'))
    parser.add_argument('--data', type=Path, default=Path('data/processed'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--confidence', type=float, default=.8,
                        help='Select using validation ONLY, before final test.')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.output is None:
        args.output = Path('outputs/evaluation' if args.split == 'test' else 'outputs/validation')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from training.metrics import classification_metrics, classification_average_precision, plot_confusion
    from app.services.detector import Detector

    if not 0 <= args.confidence <= 1:
        parser.error('Confidence must be in [0,1].')
    if (args.output / 'metrics.json').exists():
        raise SystemExit('Report already exists. Choose a fresh output only for an independently justified run.')
    detector = Detector(args.model, args.device)
    detector.load()
    manifest_path = args.data / 'manifest.json'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != detector.metadata.get('dataset_manifest_sha256'):
        raise SystemExit('Dataset manifest differs from training provenance; evaluation stopped.')
    manifest = json.loads(manifest_path.read_text())
    test = [r for r in manifest['manifest'] if r['split'] == args.split]
    if not test:
        raise SystemExit('Selected split is empty.')
    args.output.mkdir(parents=True, exist_ok=True)
    labels = detector.classes
    actual, predicted, rows, elapsed = [], [], [], []
    # Warm up outside latency measurement.
    with Image.open(args.data / args.split / test[0]['label'] / Path(test[0]['processed']).name) as image:
        detector.predict(image.convert('RGB'), args.confidence)
    for i, record in enumerate(test):
        path = args.data / args.split / record['label'] / Path(record['processed']).name
        with Image.open(path) as image:
            start = time.perf_counter()
            result = detector.predict(image.convert('RGB'), args.confidence)
            elapsed.append((time.perf_counter() - start) * 1000)
            guess = result['predictions'][0]['class_name'] if result['predictions'] else 'unknown'
            actual.append(record['label']); predicted.append(guess)
            rows.append(dict(path=str(path), actual=record['label'], predicted=guess,
                             scores=result['scores'], latency_ms=elapsed[-1]))
            kind = 'correct' if guess == record['label'] else 'incorrect'
            target = args.output / 'samples' / kind
            target.mkdir(parents=True, exist_ok=True)
            image.convert('RGB').save(target / f'{i}_{record["label"]}_as_{guess}.jpg')
    names = labels + ['unknown']
    cm, report = classification_metrics(actual, predicted, names)
    metrics = dict(task='classify', split=args.split, checkpoint_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                   capture_group_count=len({r['group'] for r in test}),
                   source_capture_count=len({r.get('capture_sha256', r['sha256']) for r in test}),
                   confidence=args.confidence, count=len(test), accuracy=float(np.mean(np.array(actual) == predicted)),
                   coverage=float(np.mean(np.array(predicted) != 'unknown')), classification_report=report,
                   raw_top1_accuracy=float(np.mean([max(row['scores'], key=row['scores'].get) == row['actual'] for row in rows])),
                   target_macro={key:float(np.mean([report[c][key] for c in labels]))
                                 for key in ['precision', 'recall', 'f1-score']},
                   classification_average_precision=classification_average_precision(actual, [r['scores'] for r in rows], labels),
                   detection_map=None,
                   confusion_matrix=cm.tolist(), labels=names,
                   latency_ms=dict(mean=float(np.mean(elapsed)), p95=float(np.percentile(elapsed, 95))),
                   limitations=['Tiny grouped test sample; no negative gestures or empty scenes.',
                                'Identity/session independence is unverified.',
                                'Unknown represents abstention, not a trained negative class.'],
                   error_analysis={c: {'false_negatives':int(cm[j].sum()-cm[j,j]),
                                       'false_positives':int(cm[:,j].sum()-cm[j,j])}
                                   for j,c in enumerate(names)})
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    (args.output / 'predictions.json').write_text(json.dumps(rows, indent=2))
    plot_confusion(cm, names, args.output / 'confusion_matrix.png')
    metadata = json.loads(args.model.with_name('metadata.json').read_text())
    curve_path = Path(metadata['source']).parent.parent / 'results.csv'
    if curve_path.exists():
        with curve_path.open() as handle:
            curves = list(csv.DictReader(handle, skipinitialspace=True))
        keys = [k for k in curves[0] if 'loss' in k or 'accuracy' in k]
        fig, axes = plt.subplots(2, 1, figsize=(9, 7))
        for key in keys:
            axes[0 if 'loss' in key else 1].plot([float(r[key]) for r in curves], label=key)
        for axis in axes:
            axis.legend(); axis.set_xlabel('Epoch')
        fig.tight_layout(); fig.savefig(args.output / 'training_curves.png'); plt.close(fig)
        train_key = next((k for k in keys if k.startswith('train/') and 'loss' in k), None)
        val_key = next((k for k in keys if k.startswith('val/') and 'loss' in k), None)
        if train_key and val_key and len(curves) >= 10:
            a, b = curves[-10], curves[-1]
            flag = float(b[train_key]) < float(a[train_key]) and float(b[val_key]) > float(a[val_key])
            metrics['overfitting_review'] = {'last_10_epochs_diverge':flag,
                'interpretation':'Review curves and validation errors; this heuristic cannot establish absence of overfitting.'}
            (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
