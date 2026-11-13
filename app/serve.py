"""Launch the local dashboard, preferring FastAPI when its runtime is usable."""
import argparse
import logging
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--backend', choices=['auto', 'fastapi', 'starlette'], default='auto')
    args = parser.parse_args()
    # Keep runtime configuration in this checkout, including read-only home setups.
    from app.core.config import ROOT
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.runtime/yolo'))
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.runtime/matplotlib'))
    os.environ.setdefault('YOLO_OFFLINE', 'true')
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    backend = args.backend
    if backend == 'auto':
        try:
            import fastapi  # noqa: F401
            backend = 'fastapi'
        except (ImportError, ModuleNotFoundError) as exc:
            logging.warning('FastAPI unavailable (%s); using the local Starlette runner.', exc)
            backend = 'starlette'
    import uvicorn
    target = 'app.main:app' if backend == 'fastapi' else 'app.portable:app'
    uvicorn.run(target, host=args.host, port=args.port, loop='asyncio', http='h11',
                ws='websockets-sansio', ws_max_size=7_200_000)


if __name__ == '__main__':
    main()
