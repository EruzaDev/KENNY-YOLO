"""Fine-tune YOLOv8n-cls; deploy only a completed run's best checkpoint."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/training.yaml'))
    parser.add_argument('--resume', type=Path, help='Trusted last.pt from an interrupted run')
    parser.add_argument('--device')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--output-dir', type=Path, default=Path('models'),
                        help='Keep candidates separate until validation comparison.')
    args = parser.parse_args()
    import torch
    import ultralytics
    from ultralytics import YOLO
    from training.transforms import GestureTrainer, BackgroundGestureTrainer, BalancedBackgroundGestureTrainer, ResolutionBalancedTrainer, StableBackbonePalmTrainer, FistThumbTrainer, PhotoBackgroundTrainer
    from training.camera_crops import CameraCropTrainer

    config = yaml.safe_load(args.config.read_text())
    profile = config.pop('augmentation_profile', 'standard')
    if profile not in {'standard', 'background', 'background_balanced', 'background_balanced_resolution', 'background_balanced_stable_backbone', 'background_balanced_fist_thumb', 'background_balanced_photos', 'balanced_camera_crops'}:
        raise ValueError('Unknown augmentation profile.')
    trainer = {'standard':GestureTrainer, 'background':BackgroundGestureTrainer,
               'background_balanced':BalancedBackgroundGestureTrainer,
               'background_balanced_resolution':ResolutionBalancedTrainer,
               'background_balanced_stable_backbone':StableBackbonePalmTrainer,
                'background_balanced_fist_thumb':FistThumbTrainer,
                'background_balanced_photos':PhotoBackgroundTrainer,
                'balanced_camera_crops':CameraCropTrainer}[profile]
    # Resolve against this checkout, independently of pretrained checkpoint settings.
    config['project'] = str(Path(config['project']).resolve())
    config['data'] = str(Path(config['data']).resolve())
    manifest = Path(config['data']) / 'manifest.json'
    if not manifest.is_file():
        raise SystemExit('Run python -m training.prepare_dataset first.')
    if args.device:
        config['device'] = args.device
    if args.epochs:
        config['epochs'] = args.epochs
    config['amp'] = config['amp'] and str(config['device']) != 'cpu' and torch.cuda.is_available()
    model_source = config.pop('model')
    if Path(model_source).is_file():
        model_source = str(Path(model_source).resolve())
    model = YOLO(str(args.resume or model_source))
    initial_hash = hashlib.sha256(Path(args.resume or model_source).read_bytes()).hexdigest()
    parent_metadata = Path(args.resume or model_source).with_name('metadata.json')
    architecture = Path(model_source).stem
    if parent_metadata.is_file():
        parent = json.loads(parent_metadata.read_text())
        if parent.get('checkpoint_sha256') != initial_hash:
            raise ValueError('Initial checkpoint differs from its metadata.')
        architecture = parent['architecture']
    if model.task != 'classify':
        raise ValueError('This image-level dataset requires a classification checkpoint.')
    if args.resume:
        model.train(trainer=trainer, resume=True)
    else:
        model.train(trainer=trainer, **config)
    best = Path(model.trainer.best)
    if not best.is_file():
        raise RuntimeError('Training produced no best checkpoint; nothing deployed.')
    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, destination / 'best.pt')
    actual_config = vars(model.trainer.args).copy()
    manifest = Path(actual_config['data']) / 'manifest.json'
    metadata = dict(task='classify', architecture=architecture, source=str(best),
                    transferred_class_rows=getattr(model.trainer, 'transferred_classes', []),
                    class_balancing=('inverse-frequency sampling, 384 draws per epoch; macro recall validation fitness'
                                     if profile == 'balanced_camera_crops' else
                                     'training-only oversampling to equal class counts' if profile.startswith('background_balanced') else None),
                    initial_checkpoint_sha256=initial_hash, augmentation_profile=profile,
                    checkpoint_sha256=hashlib.sha256(best.read_bytes()).hexdigest(),
                    dataset_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    classes=model.names, config=actual_config, transforms='full-frame letterbox',
                    torch_version=torch.__version__, ultralytics_version=ultralytics.__version__)
    (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str))
    print(f'Completed training. Best validation checkpoint: {destination / "best.pt"}')


if __name__ == '__main__':
    main()
