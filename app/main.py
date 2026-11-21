"""Run with: uvicorn app.main:app --host 127.0.0.1 --port 8000"""

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.api.routes import router
from app.core.config import ROOT
from app.core.lifecycle import lifespan
from app.core.logging_config import configure_logging

configure_logging()


app = FastAPI(title='Kenny Gesture Lab', lifespan=lifespan)
app.include_router(router, prefix='/api')
app.mount('/static', StaticFiles(directory=ROOT / 'frontend'), name='static')


@app.get('/', include_in_schema=False)
def index():
    return FileResponse(ROOT / 'frontend/index.html')


@app.get('/dataset-creator', include_in_schema=False)
def dataset_creator():
    return FileResponse(ROOT / 'frontend/dataset-creator.html')
