"""Environment-controlled local deployment settings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv('GESTURE_MODEL', str(ROOT / 'models/ok_camera/best.pt')))
DEVICE = os.getenv('GESTURE_DEVICE', 'cpu')
MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 16_000_000
METRICS_PATH = ROOT / 'outputs/evaluation/metrics.json'
CONFIDENCE = .8
SERVO_ENABLED = os.getenv('SERVO_ENABLED', '1').lower() not in {'0', 'false', 'no'}
SERVO_PORT = os.getenv('SERVO_PORT') or None
SERVO_BAUD = int(os.getenv('SERVO_BAUD', '9600'))
SERVO_OPEN_ANGLE = int(os.getenv('SERVO_OPEN_ANGLE', '60'))
SERVO_CLOSE_ANGLE = int(os.getenv('SERVO_CLOSE_ANGLE', '0'))
