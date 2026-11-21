"""Shared real inference handlers for the FastAPI and portable ASGI runners."""
import asyncio
import base64
import binascii
import json
import logging
import time
from pathlib import Path

from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.websockets import WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool
from app.core.config import MAX_BYTES, METRICS_PATH, ROOT
from app.core.gestures import GESTURE_FUNCTIONS, GESTURE_LABELS
from app.services.inference import Stabilizer, check_thumb_evidence, check_gesture_evidence
from app.utils.image_utils import decode_image

logger = logging.getLogger(__name__)


def health(request: Request):
    detector = request.app.state.detector
    return {'status': 'ready' if detector.model else 'model_unavailable', 'detail': detector.error}


def model_info(request: Request):
    detector = request.app.state.detector
    metrics = None
    metrics_path = METRICS_PATH
    recorded_path = detector.metadata.get('test_metrics_path')
    if recorded_path:
        metrics_path = ROOT / Path(recorded_path)
    if metrics_path.is_file():
        candidate = json.loads(metrics_path.read_text())
        if (candidate.get('split') == 'test'
                and candidate.get('checkpoint_sha256') == detector.metadata.get('checkpoint_sha256')):
            metrics = candidate
    return dict(**detector.deployment_status(),
                ready=detector.model is not None, architecture=detector.metadata.get('architecture'),
                task='classify', classes=detector.classes, checkpoint=detector.path.name,
                device=detector.device, confidence=.8, metrics=metrics, error=detector.error,
                validation_metrics=detector.metadata.get('validation_summary'),
                gesture_functions={name: GESTURE_FUNCTIONS.get(name, '') for name in detector.classes},
                gesture_labels={name: GESTURE_LABELS.get(name, name.replace('_', ' ').title()) for name in detector.classes},
                robot_controls_enabled=False,
                limitations='Browser hand-presence check + YOLO classification of one hand crop. '
                            'No negative training examples; real-scene accuracy is unverified.',
                metrics_scope='Saved metrics evaluate the classifier on reserved dataset views, not the browser hand-gated pipeline.')


async def predict(request: Request, file, confidence: float = .8, pose_hint: str = 'unknown',
                  gesture_hint: str = 'unknown'):
    try:
        if file.content_type not in {'image/jpeg', 'image/png'}:
            raise HTTPException(415, 'Upload a JPEG or PNG image.')
        data = await file.read(MAX_BYTES + 1)
        try:
            image = await run_in_threadpool(decode_image, data)
        except ValueError as exc:
            raise HTTPException(413 if len(data) > MAX_BYTES else 422, str(exc)) from exc
        detector = request.app.state.detector
        if detector.model is None:
            raise HTTPException(503, detector.error or 'Trained model unavailable.')
        async with request.app.state.inference_gate:
            result = await run_in_threadpool(detector.predict, image, confidence)
            return check_gesture_evidence(check_thumb_evidence(result, pose_hint), gesture_hint)
    finally:
        await file.close()


async def webcam(socket: WebSocket):
    # Reject cross-origin browser access while allowing same-origin localhost deployment.
    origin = socket.headers.get('origin')
    if origin and origin.split('://', 1)[-1] != socket.headers.get('host'):
        await socket.close(code=1008)
        return
    await socket.accept()
    stabilizer = Stabilizer()
    last = 0.0
    try:
        while True:
            message = await asyncio.wait_for(socket.receive_text(), timeout=60)
            try:
                if len(message) > MAX_BYTES * 4 // 3 + 2048:
                    raise ValueError('Frame exceeds the 5 MB limit.')
                payload = json.loads(message)
                if not isinstance(payload, dict):
                    raise ValueError('Frame request must be a JSON object.')
                confidence = float(payload.get('confidence', .8))
                window = int(payload.get('smoothing', 3))
                if not 0 <= confidence <= 1 or not 1 <= window <= 12:
                    raise ValueError('Invalid confidence or smoothing setting.')
                if time.monotonic() - last < .08:
                    raise ValueError('Frame rate exceeds the server limit of 12 FPS.')
                last = time.monotonic()
                if payload.get('type') == 'no_hand':
                    stabilizer.reset()
                    await socket.send_json(dict(task='classify', predictions=[], scores={},
                        stable_gesture=None, message='No gesture detected',
                        rejection_reason=payload.get('reason', 'No hand detected'), latency_ms=0))
                    continue
                data = base64.b64decode(payload['image'], validate=True)
                image = await run_in_threadpool(decode_image, data)
                detector = socket.app.state.detector
                if detector.model is None:
                    await socket.send_json({'error': detector.error or 'Trained model unavailable.'})
                    continue
                async with socket.app.state.inference_gate:
                    result = await run_in_threadpool(detector.predict, image, confidence)
                result = check_thumb_evidence(result, payload.get('pose_hint', 'unknown'))
                result = check_gesture_evidence(result, payload.get('gesture_hint', 'unknown'))
                await socket.send_json(stabilizer.apply(result, window))
            except (ValueError, KeyError, TypeError, binascii.Error) as exc:
                await socket.send_json({'error': str(exc)})
    except (WebSocketDisconnect, asyncio.TimeoutError):
        return
    except Exception:
        logger.exception('Webcam inference failed')
        await socket.close(code=1011, reason='Inference failed; inspect server logs.')
