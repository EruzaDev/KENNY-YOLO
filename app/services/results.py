"""Expose only verified results and allowlisted plot files for the loaded checkpoint."""
import hashlib
import json
from pathlib import Path

from starlette.exceptions import HTTPException
from starlette.responses import FileResponse

from app.core.config import ROOT


def results_report(detector):
    recorded = detector.metadata.get('results_report_path')
    if detector.model is None or not recorded:
        return None
    path = (ROOT / recorded).resolve()
    if not path.is_relative_to((ROOT / 'outputs').resolve()) or not path.is_file():
        return None
    report = json.loads(path.read_text())
    if report.get('checkpoint_sha256') != detector.metadata.get('checkpoint_sha256'):
        return None
    return report


def results_info(detector):
    report = results_report(detector)
    if report is None:
        return dict(available=False, reason='No verified results are linked to the loaded checkpoint.')
    return dict(report, available=True, assets={
        name: f'/api/results/assets/{name}' for name in report.get('assets', {})})


def results_asset(detector, name):
    report = results_report(detector)
    asset = report.get('assets', {}).get(name) if report else None
    if not asset:
        raise HTTPException(404, 'Result image unavailable.')
    path = (ROOT / asset['path']).resolve()
    if (not path.is_relative_to((ROOT / 'outputs').resolve()) or path.suffix not in {'.png', '.jpg'}
            or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != asset['sha256']):
        raise HTTPException(404, 'Result image does not match verified evidence.')
    return FileResponse(path, headers={'Cache-Control': 'no-cache'})
