"""Per-connection stabilization with immediate abstention on weak evidence."""
from collections import deque


class Stabilizer:
    def __init__(self):
        self.history = deque(maxlen=12)

    def reset(self):
        self.history.clear()

    def apply(self, result: dict, window: int) -> dict:
        if not 1 <= window <= 12:
            raise ValueError('Smoothing must be from 1 to 12 frames.')
        current = result['predictions']
        name = current[0]['class_name'] if current else None
        if name is None:
            self.reset()
        else:
            self.history.append(name)
        recent = list(self.history)[-window:]
        stable = name is not None and len(recent) >= window and recent.count(name) > window // 2
        return dict(result, stable_gesture=name if stable else None,
                    message=name if stable else 'No gesture detected')


def check_thumb_evidence(result: dict, pose_hint: str = 'unknown') -> dict:
    """Veto a fist/thumb contradiction; do not replace the model's predicted class."""
    if pose_hint not in {'unknown', 'thumb_folded', 'thumb_extended'}:
        raise ValueError('Invalid thumb-position evidence.')
    predictions = result['predictions']
    if not predictions:
        return result
    gesture = predictions[0]['class_name']
    conflict = ((gesture == 'thumbs_up' and pose_hint == 'thumb_folded')
                or (gesture == 'closed_fist' and pose_hint == 'thumb_extended'))
    if not conflict:
        return result
    return dict(result, predictions=[], message='No gesture detected',
                rejection_reason='Thumb position conflicts with the gesture score. Hold the pose clearly.')
