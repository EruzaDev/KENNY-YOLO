"""Real classification inference only; never download fallback weights."""
import hashlib
import json
import threading
import time
from pathlib import Path

from PIL import Image, ImageOps
from app.core.gestures import CLASSES, LEGACY_CLASSES, CAPTURE_CLASSES, GESTURE_FUNCTIONS, GESTURE_LABELS



class Detector:
    def __init__(self, path: Path, device: str = 'cpu'):
        self.path = path
        self.device = device
        self.model = None
        self.classes = list(CLASSES)
        self.error = None
        self.lock = threading.Lock()
        self.metadata = {}
        self.image_size = 224

    def load(self):
        if not self.path.is_file():
            raise FileNotFoundError('Trained model missing. Prepare the dataset and run training first.')
        metadata_path = self.path.with_name('metadata.json')
        if not metadata_path.is_file():
            raise ValueError('Training provenance metadata.json is missing beside the model.')
        metadata = json.loads(metadata_path.read_text())
        if metadata.get('checkpoint_sha256') != hashlib.sha256(self.path.read_bytes()).hexdigest():
            raise ValueError('Checkpoint does not match training metadata.')
        from ultralytics import YOLO
        model = YOLO(str(self.path))
        names = [model.names[k] for k in sorted(model.names)]
        if model.task != 'classify' or tuple(names) not in {CLASSES, LEGACY_CLASSES, CAPTURE_CLASSES}:
            raise ValueError('Checkpoint has an unsupported gesture class mapping.')
        declared = metadata['classes']
        declared_names = [declared[str(i)] if str(i) in declared else declared[i] for i in range(len(declared))]
        if names != declared_names:
            raise ValueError('Checkpoint class order differs from its training metadata.')
        self.classes = names
        self.model = model
        self.metadata = metadata
        self.image_size = int(metadata['config']['imgsz'])

    def deployment_status(self) -> dict:
        """Expose stale in-memory weights without silently reloading a running model."""
        loaded_hash = self.metadata.get('checkpoint_sha256')
        disk_hash = None
        try:
            disk_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        except OSError:
            pass
        expected = CAPTURE_CLASSES if tuple(self.classes) == CAPTURE_CLASSES else CLASSES
        return dict(checkpoint_sha256=loaded_hash,
                    checkpoint_on_disk_sha256=disk_hash,
                    restart_required=bool(self.model is not None and loaded_hash != disk_hash),
                    missing_classes=[name for name in expected if name not in self.classes])

    def predict(self, image: Image.Image, confidence: float = .8) -> dict:
        if not 0 <= confidence <= 1:
            raise ValueError('Confidence must be between 0 and 1.')
        if self.model is None:
            raise RuntimeError(self.error or 'Trained model is unavailable.')
        import torch
        import numpy as np
        start = time.perf_counter()
        pixels = np.asarray(image.convert('RGB'))
        # Contentless images cannot contain either gesture. This is a blank-image
        # check, not a replacement for an independent hand detector.
        if np.max(np.ptp(pixels, axis=(0, 1))) <= 2:
            return dict(task='classify', predictions=[], scores={},
                        message='No gesture detected', rejection_reason='No visual content',
                        width=image.width, height=image.height,
                        latency_ms=round((time.perf_counter() - start) * 1000, 2))
        # A tensor bypasses Ultralytics default center cropping; match custom training exactly.
        square = ImageOps.pad(image.convert('RGB'), (self.image_size, self.image_size), color='white',
                              method=Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.array(square).copy()).permute(2, 0, 1).float().unsqueeze(0) / 255
        with self.lock:
            result = self.model.predict(tensor, device=self.device, imgsz=self.image_size, verbose=False)[0]
        scores = {name: float(result.probs.data[i].item()) for i, name in enumerate(self.classes)}
        name = max(scores, key=scores.get)
        predictions = ([dict(class_name=name, display_name=GESTURE_LABELS.get(name, name.replace('_', ' ').title()), function=GESTURE_FUNCTIONS.get(name, ''), confidence=scores[name], bbox=None)]
                       if scores[name] >= confidence else [])
        return dict(task='classify', predictions=predictions, scores=scores,
                    message=name if predictions else 'No gesture detected',
                    width=image.width, height=image.height,
                    latency_ms=round((time.perf_counter() - start) * 1000, 2))
