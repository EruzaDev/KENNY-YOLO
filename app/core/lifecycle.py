"""Load and verify the actual trained checkpoint and optional Arduino servo for either ASGI entrypoint."""
import asyncio
import logging
from contextlib import asynccontextmanager
from app.core.config import (
    MODEL_PATH,
    DEVICE,
    SERVO_ENABLED,
    SERVO_PORT,
    SERVO_BAUD,
    SERVO_OPEN_ANGLE,
    SERVO_CLOSE_ANGLE,
)
from app.services.detector import Detector
from arduino_servo.gesture_servo import GestureServoMapper, ServoController


@asynccontextmanager
async def lifespan(application):
    logger = logging.getLogger(__name__)
    detector = Detector(MODEL_PATH, DEVICE)
    try:
        detector.load()
    except Exception as exc:
        detector.error = str(exc)
        logger.exception('Model startup failed')
    application.state.detector = detector
    application.state.inference_gate = asyncio.Semaphore(1)

    servo = None
    mapper = GestureServoMapper(open_angle=SERVO_OPEN_ANGLE, close_angle=SERVO_CLOSE_ANGLE)
    if SERVO_ENABLED:
        candidate = ServoController(port=SERVO_PORT, baud=SERVO_BAUD)
        try:
            candidate.connect()
            candidate.set_angle(SERVO_CLOSE_ANGLE, force=True)
            servo = candidate
        except Exception as exc:
            logger.warning('Arduino servo unavailable (%s); continuing without hardware output.', exc)
    application.state.servo = servo
    application.state.servo_mapper = mapper
    try:
        yield
    finally:
        if servo is not None:
            try:
                servo.set_angle(SERVO_CLOSE_ANGLE)
            finally:
                servo.close()

