"""Publish checkpoint-matched OK gesture results for the dashboard."""
import csv
import argparse
import hashlib
import json
from pathlib import Path


def publish(model_dir=Path('models/ok_gestures'), data_dir=Path('data/ok_gestures_inherited'),
            output=Path('outputs/ok_gestures')):
    metadata_path = model_dir / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    checkpoint = hashlib.sha256((model_dir / 'best.pt').read_bytes()).hexdigest()
    if checkpoint != metadata['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from training metadata.')
    dataset = json.loads((data_dir / 'manifest.json').read_text())
    reports = {name: json.loads((output / name / 'metrics.json').read_text())
               for name in ['validation', 'test']}
    if any(report['checkpoint_sha256'] != checkpoint for report in reports.values()):
        raise ValueError('Evaluation differs from trained checkpoint.')
    curves = Path(metadata['source']).parent.parent / 'results.csv'
    with curves.open() as handle:
        epochs = list(csv.DictReader(handle))
    assets = {}
    for name, path in {
        'training-curves': output / 'validation/training_curves.png',
        'validation-confusion': output / 'validation/confusion_matrix.png',
        'test-confusion': output / 'test/confusion_matrix.png',
    }.items():
        assets[name] = dict(path=path.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    limitations = dataset['limitations'] + [
        'Only 17 unique Open Palm and 18 unique Thumbs Up images; repeated files add no independent evidence.',
        'OK photos have different backgrounds from the other classes; background shortcuts may affect camera use.',
        'The test split is reserved for this run but older gesture images have been used in prior experiments.',
        'Closed Fist is no longer a trained class. Unsupported gestures can still receive a supported label.',
        'No live camera accuracy or new-person performance has been established.',
    ]
    if dataset.get('crop_policy'):
        limitations += [
            f'{len(dataset["localization_skipped"])} source images failed the single-hand localizer and were excluded.',
            'The reported camera misclassification still requires verification on the exact user-provided failing crops.',
            'The browser OK fingertip check is a separate rejection guard; classifier metrics do not measure that guard.',
        ]
    result = dict(checkpoint_sha256=checkpoint, checkpoint=(model_dir / 'best.pt').as_posix(),
                  title='OK, Open Palm and Thumbs Up', classes=dataset['classes'],
                  training=dict(epochs=len(epochs), image_count=sum(dataset['splits']['train'].values()),
                                splits=dataset['splits'],
                                inherited_split_manifest=dataset['inherited_split_manifest'],
                                inherited_capture_count=dataset['inherited_capture_count']),
                  metrics=dict(validation=reports['validation'], regression=reports['test']),
                  summary=('Browser-style landmark crops with varied training backgrounds. '
                           'Validation fitness weights each gesture equally. '
                           if dataset.get('crop_policy') else '') +
                          'Exact duplicates removed; perceptually similar images kept in the same split. '
                          'Training samples balanced by class. Results describe supplied images, not live camera accuracy.',
                  assets=assets, limitations=limitations)
    (output / 'results.json').write_text(json.dumps(result, indent=2))
    metadata.update(results_report_path=(output / 'results.json').as_posix(),
                    test_metrics_path=(output / 'test/metrics.json').as_posix(),
                    validation_summary=reports['validation'])
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(dict(checkpoint_sha256=checkpoint, epochs=len(epochs), results=str(output / 'results.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path('models/ok_gestures'))
    parser.add_argument('--data', type=Path, default=Path('data/ok_gestures_inherited'))
    parser.add_argument('--output', type=Path, default=Path('outputs/ok_gestures'))
    args = parser.parse_args()
    publish(args.model_dir, args.data, args.output)
