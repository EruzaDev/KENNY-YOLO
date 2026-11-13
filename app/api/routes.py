"""FastAPI validation and routing over the shared inference handlers."""
from fastapi import APIRouter, File, Query, Request, UploadFile, WebSocket
from app.api import handlers

router = APIRouter()


@router.get('/health')
def health(request: Request):
    return handlers.health(request)


@router.get('/model')
def model_info(request: Request):
    return handlers.model_info(request)


@router.post('/predict')
async def predict(request: Request, file: UploadFile = File(...),
                  confidence: float = Query(.8, ge=0, le=1),
                  pose_hint: str = Query('unknown', pattern='^(unknown|thumb_folded|thumb_extended)$')):
    return await handlers.predict(request, file, confidence, pose_hint)


@router.websocket('/webcam')
async def webcam(socket: WebSocket):
    await handlers.webcam(socket)
