"""Canonical labels and display-only meanings. This module sends no commands."""
LEGACY_CLASSES = ('open_palm', 'thumbs_up')
CLASSES = ('ok', 'open_palm', 'thumbs_up')
GESTURE_FUNCTIONS = {
    'open_palm': 'STOP ROBOT',
    'thumbs_up': 'DONE / RESUME',
    'ok': 'RETURN TO BASE',
}
GESTURE_LABELS = {'open_palm':'Open Palm ✋', 'thumbs_up':'Thumbs Up 👍', 'ok':'OK 👌'}
