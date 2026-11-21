"""Canonical labels and display-only meanings. This module sends no commands."""
LEGACY_CLASSES = ('open_palm', 'thumbs_up')
CLASSES = ('closed_fist', 'open_palm', 'thumbs_up')
CAPTURE_CLASSES = ('like', 'ok', 'open_palm')
GESTURE_FUNCTIONS = {
    'open_palm': 'STOP ROBOT',
    'thumbs_up': 'DONE / RESUME',
    'closed_fist': 'RETURN TO BASE',
    'ok': 'OK GESTURE',
    'like': 'LIKE / THUMBS UP',
}
GESTURE_LABELS = {'open_palm':'Open Palm ✋', 'thumbs_up':'Thumbs Up 👍', 'closed_fist':'Closed Fist ✊',
                  'ok':'OK Sign 👌', 'like':'Like 👍'}
