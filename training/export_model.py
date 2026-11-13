"""Export an actual fine-tuned checkpoint; benchmark and verify parity before deployment."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('models/best.pt'))
    parser.add_argument('--format', choices=['onnx', 'torchscript', 'openvino'], default='onnx')
    args = parser.parse_args()
    if not args.model.is_file():
        raise SystemExit('Trained checkpoint is missing.')
    from app.services.detector import Detector
    detector = Detector(args.model)
    detector.load()
    print(detector.model.export(format=args.format, imgsz=detector.image_size, batch=1))


if __name__ == '__main__':
    main()
