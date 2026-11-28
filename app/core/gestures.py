"""Canonical labels and servo actions for detected hand gestures."""
LEGACY_CLASSES = ('open_palm', 'thumbs_up')
CLASSES = ('ok', 'open_palm', 'thumbs_up')
CAPTURE_CLASSES = ('like', 'ok', 'open_palm')
GESTURE_FUNCTIONS = {
    'open_palm': 'OPEN SERVO (60°)',
    'like': 'CLOSE SERVO (0°)',
    'thumbs_up': 'CLOSE SERVO (0°)',
    'ok': 'HOLD POSITION',
}
GESTURE_LABELS = {
    'open_palm': 'Open Palm ✋',
    'like': 'Like 👍',
    'thumbs_up': 'Thumbs Up 👍',
    'ok': 'OK 👌',
}
