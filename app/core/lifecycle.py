"""Load and verify the actual trained checkpoint for either ASGI entrypoint."""
import asyncio
import logging
from contextlib import asynccontextmanager
from app.core.config import MODEL_PATH, DEVICE
from app.services.detector import Detector


@asynccontextmanager
async def lifespan(application):
    detector = Detector(MODEL_PATH, DEVICE)
    try:
        detector.load()
    except Exception as exc:
        detector.error = str(exc)
        logging.getLogger(__name__).exception('Model startup failed')
    application.state.detector = detector
    application.state.inference_gate = asyncio.Semaphore(1)
    yield

