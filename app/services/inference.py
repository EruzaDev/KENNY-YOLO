"""Per-connection stabilization with immediate abstention on weak evidence."""
from collections import deque
from app.core.gestures import CAPTURE_CLASSES, GESTURE_FUNCTIONS, GESTURE_LABELS


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
    """Veto a folded-thumb Thumbs Up prediction without relabeling it."""
    if pose_hint not in {'unknown', 'thumb_folded', 'thumb_extended'}:
        raise ValueError('Invalid thumb-position evidence.')
    predictions = result['predictions']
    if not predictions:
        return result
    gesture = predictions[0]['class_name']
    conflict = gesture == 'thumbs_up' and pose_hint == 'thumb_folded'
    if not conflict:
        return result
    return dict(result, predictions=[], message='No gesture detected',
                rejection_reason='Thumb position conflicts with the gesture score. Hold the pose clearly.')


def check_gesture_evidence(result: dict, gesture_hint: str = 'unknown') -> dict:
    """Use a clear 21-joint pose to confirm or correct the image-model label."""
    if gesture_hint not in {'unknown', *CAPTURE_CLASSES}:
        raise ValueError('Invalid MediaPipe gesture evidence.')
    if gesture_hint == 'unknown' or not result.get('scores'):
        return result
    predicted = max(result['scores'], key=result['scores'].get)
    if predicted not in CAPTURE_CLASSES:
        return result
    confidence = result['scores'].get(gesture_hint, 0.0)
    prediction = dict(class_name=gesture_hint,
                      display_name=GESTURE_LABELS.get(gesture_hint, gesture_hint),
                      function=GESTURE_FUNCTIONS.get(gesture_hint, ''),
                      confidence=confidence, bbox=None)
    return dict(result, predictions=[prediction], joint_verified=True,
                joint_corrected=predicted != gesture_hint,
                joint_gesture=gesture_hint, model_top_class=predicted,
                model_top_confidence=result['scores'][predicted])
