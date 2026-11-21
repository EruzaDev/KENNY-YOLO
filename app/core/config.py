"""Environment-controlled local deployment settings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv('GESTURE_MODEL', str(ROOT / 'models/gesture_captures/best.pt')))
DEVICE = os.getenv('GESTURE_DEVICE', 'cpu')
MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 16_000_000
METRICS_PATH = ROOT / 'outputs/evaluation/metrics.json'
CONFIDENCE = .8
