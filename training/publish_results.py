"""Publish existing, checkpoint-matched evidence without retraining or reevaluating."""
import csv
import hashlib
import json
from pathlib import Path


def publish():
    model_dir, output = Path('models/webcam_fist'), Path('outputs/webcam_fist')
    metadata_path = model_dir / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    checkpoint = hashlib.sha256((model_dir / 'best.pt').read_bytes()).hexdigest()
    if metadata['checkpoint_sha256'] != checkpoint:
        raise ValueError('Checkpoint and metadata differ.')
    reports = {name: json.loads((output / path).read_text()) for name, path in {
        'validation': 'validation/metrics.json', 'regression': 'regression_test/metrics.json',
        'training_fit': 'training_frame_fit.json'}.items()}
    if any(report['checkpoint_sha256'] != checkpoint for report in reports.values()):
        raise ValueError('Results belong to a different checkpoint.')
    dataset = json.loads(Path('data/webcam_fist/manifest.json').read_text())
    if hashlib.sha256(Path('data/webcam_fist/manifest.json').read_bytes()).hexdigest() != metadata['dataset_manifest_sha256']:
        raise ValueError('Training dataset lineage differs.')
    curves_path = Path(metadata['source']).parent.parent / 'results.csv'
    with curves_path.open() as handle:
        epochs = list(csv.DictReader(handle, skipinitialspace=True))
    predictions = json.loads((output / 'validation/predictions.json').read_text())
    original = [row for row in predictions if '_handcrop_' not in Path(row['path']).stem]
    assets = {}
    for name, path in {
        'training-curves': output / 'validation/training_curves.png',
        'validation-confusion': output / 'validation/confusion_matrix.png',
        'regression-confusion': output / 'regression_test/confusion_matrix.png',
        'training-crops': output / 'crop_preview.jpg',
    }.items():
        assets[name] = dict(path=path.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    metrics = {}
    for name in ['validation', 'regression']:
        r = reports[name]
        metrics[name] = {key: r[key] for key in [
            'accuracy', 'raw_top1_accuracy', 'count', 'confidence', 'capture_group_count',
            'source_capture_count', 'coverage', 'target_macro', 'classification_report']}
    result = dict(checkpoint_sha256=checkpoint, title='Webcam fist correction',
                  checkpoint='models/webcam_fist/best.pt', training=dict(
                      epochs=len(epochs), source='Photo-background gesture checkpoint',
                      image_count=sum(dataset['splits']['train'].values()),
                      splits=dataset['splits'], webcam_frames=len(dataset['webcam_training_frames']),
                      webcam_sessions=len(dataset['webcam_sessions_train_only']),
                      background_photos=len(dataset['background_photos']),
                      early_stopping=True),
                  metrics=metrics, original_validation=dict(
                      correct=sum(row['actual'] == row['predicted'] for row in original), count=len(original)),
                  training_fit={key: reports['training_fit'][key] for key in ['count', 'accepted_correct', 'kind']},
                  overfitting_review=reports['validation'].get('overfitting_review'), assets=assets,
                  limitations=[
                      'Uploaded webcam crops were used for training. Their 5/5 result is training fit, not held-out accuracy.',
                      'Expanded validation contains correlated synthetic crops. One Open Palm view abstained below 80% confidence.',
                      'The 11 test files are a previously inspected regression set, not a fresh independent test.',
                      'Training/validation loss divergence was flagged. Fresh camera frames of all three gestures are required.',
                      'No Astra capture, new-person, empty-scene or physical-camera accuracy has been established.'])
    (output / 'results.json').write_text(json.dumps(result, indent=2))
    metadata['results_report_path'] = (output / 'results.json').as_posix()
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(dict(checkpoint_sha256=checkpoint, epochs=len(epochs), results=str(output / 'results.json')), indent=2))
    return result


if __name__ == '__main__':
    publish()
