"""FastAPI validation and routing over the shared inference handlers."""
from fastapi import APIRouter, File, Form, Query, Request, UploadFile, WebSocket, HTTPException
import json
import time
from app.core.config import ROOT
from app.api import handlers
from app.services import results

router = APIRouter()


@router.post('/dataset/capture')
async def dataset_capture(image: UploadFile = File(...), label: str = Form(...),
                          annotation: str = Form(...)):
    """Persist one locally captured, MediaPipe-annotated dataset frame."""
    if label not in {'ok', 'open_palm', 'like'}:
        raise HTTPException(400, 'Unsupported dataset label')
    data = await image.read()
    if not data or len(data) > 5 * 1024 * 1024:
        raise HTTPException(400, 'Image must be between 1 byte and 5 MB')
    try:
        details = json.loads(annotation)
    except (ValueError, TypeError):
        raise HTTPException(400, 'Invalid annotation JSON')
    if not isinstance(details, dict) or len(details.get('landmarks', [])) != 21:
        raise HTTPException(400, 'Expected 21 hand landmarks')
    folder = ROOT / 'data' / 'dataset_creator' / label
    folder.mkdir(parents=True, exist_ok=True)
    stem = f'{label}_{time.time_ns()}'
    (folder / f'{stem}.jpg').write_bytes(data)
    (folder / f'{stem}.json').write_text(json.dumps(details, separators=(',', ':')))
    return {'saved': True, 'label': label, 'filename': f'{stem}.jpg'}


@router.get('/health')
def health(request: Request):
    return handlers.health(request)


@router.get('/model')
def model_info(request: Request):
    return handlers.model_info(request)


@router.post('/predict')
async def predict(request: Request, file: UploadFile = File(...),
                  confidence: float = Query(.8, ge=0, le=1),
                  pose_hint: str = Query('unknown', pattern='^(unknown|thumb_folded|thumb_extended)$'),
                  gesture_hint: str = Query('unknown', pattern='^(unknown|ok|open_palm|like)$')):
    return await handlers.predict(request, file, confidence, pose_hint, gesture_hint)


@router.get('/results')
def results_info(request: Request):
    return results.results_info(request.app.state.detector)


@router.get('/results/assets/{name}')
def results_image(request: Request, name: str):
    return results.results_asset(request.app.state.detector, name)


@router.websocket('/webcam')
async def webcam(socket: WebSocket):
    await handlers.webcam(socket)
