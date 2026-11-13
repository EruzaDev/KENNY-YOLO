"""Local Starlette runner when FastAPI's compiled dependencies are unavailable.

Uses exactly the same checkpoint lifecycle and inference handlers as FastAPI.
"""
from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles

from app.api import handlers
from app.core.config import ROOT, MAX_BYTES
from app.core.lifecycle import lifespan
from app.core.logging_config import configure_logging

configure_logging()


async def index(request):
    return FileResponse(ROOT / 'frontend/index.html')


async def health(request):
    return JSONResponse(handlers.health(request))


async def model_info(request):
    return JSONResponse(dict(handlers.model_info(request), backend='starlette-local'))


async def predict(request):
    try:
        confidence = float(request.query_params.get('confidence', '.8'))
    except ValueError as exc:
        raise HTTPException(422, 'Confidence must be a number from 0 to 1.') from exc
    if not 0 <= confidence <= 1:
        raise HTTPException(422, 'Confidence must be a number from 0 to 1.')
    pose_hint = request.query_params.get('pose_hint', 'unknown')
    if pose_hint not in {'unknown', 'thumb_folded', 'thumb_extended'}:
        raise HTTPException(422, 'Invalid thumb-position evidence.')
    async with request.form(max_files=1, max_fields=0, max_part_size=MAX_BYTES) as form:
        file = form.get('file')
        if not isinstance(file, UploadFile):
            raise HTTPException(422, 'Upload a JPEG or PNG image in the file field.')
        result = await handlers.predict(request, file, confidence, pose_hint)
    return JSONResponse(result)


async def http_error(request, exception):
    return JSONResponse({'detail': exception.detail}, status_code=exception.status_code)


app = Starlette(
    lifespan=lifespan,
    routes=[
        Route('/', index),
        Route('/api/health', health),
        Route('/api/model', model_info),
        Route('/api/predict', predict, methods=['POST']),
        WebSocketRoute('/api/webcam', handlers.webcam),
        Mount('/static', StaticFiles(directory=ROOT / 'frontend'), name='static'),
    ],
    exception_handlers={HTTPException: http_error},
)
