#!/usr/bin/env python3
"""Webcam hand-gesture controller for an Arduino Uno servo.

Detects hand gestures with the trained YOLO classifier and sends servo angles
to an Arduino Uno over USB serial:
  - Open hand (`open_palm`)       -> 60 degrees (open)
  - Like (`like` / `thumbs_up`)   -> 0 degrees (closed)
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_MODEL = ROOT / "models" / "gesture_captures_v2" / "best.pt"
FALLBACK_MODEL = ROOT / "models" / "ok_camera" / "best.pt"

OPEN_GESTURES = frozenset({"open_palm", "open_hand"})
CLOSE_GESTURES = frozenset({"like", "thumbs_up"})


class GestureStabilizer:
    """Temporal majority-vote smoother to prevent servo jitter on single frames."""

    def __init__(self, window: int = 5):
        if window < 1:
            raise ValueError("Smoothing window must be >= 1.")
        self.window = window
        self.history: deque[str] = deque(maxlen=window)

    def reset(self) -> None:
        self.history.clear()

    def update(self, gesture: str | None) -> str | None:
        if gesture is None:
            self.reset()
            return None
        self.history.append(gesture)
        if len(self.history) < self.window:
            return None
        if self.history.count(gesture) > self.window // 2:
            return gesture
        return None


class GestureServoMapper:
    """Maps canonical gesture labels to target servo angles."""

    def __init__(self, open_angle: int = 60, close_angle: int = 0):
        self.open_angle = max(0, min(180, int(open_angle)))
        self.close_angle = max(0, min(180, int(close_angle)))

    def target_angle(self, gesture: str | None) -> int | None:
        if gesture is None:
            return None
        normalized = gesture.strip().lower()
        if normalized in OPEN_GESTURES:
            return self.open_angle
        if normalized in CLOSE_GESTURES:
            return self.close_angle
        return None


def detect_arduino_port() -> str | None:
    """Auto-detect an attached Arduino Uno serial port."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return None

    ports = list(list_ports.comports())
    if not ports:
        return None

    # Prefer ports whose USB description or VID/device path matches Arduino Uno / USB-serial.
    arduino_keywords = ("arduino", "uno", "ch340", "cp210", "ftdi", "usb serial", "usbmodem", "ttyacm")
    for port in ports:
        haystack = f"{port.device} {port.description} {port.manufacturer or ''}".lower()
        if any(keyword in haystack for keyword in arduino_keywords) or port.vid in {0x2341, 0x2A03, 0x1A86}:
            return port.device
    return ports[0].device


class ServoController:
    """Manages USB serial communication with the Arduino Uno servo sketch."""

    def __init__(
        self,
        port: str | None = None,
        baud: int = 9600,
        dry_run: bool = False,
        reset_delay: float = 2.0,
        serial_factory: Any = None,
    ):
        self.port = port
        self.baud = baud
        self.dry_run = dry_run
        self.reset_delay = reset_delay
        self._serial_factory = serial_factory
        self._conn: Any = None
        self.current_angle: int | None = None
        self.last_command: str | None = None
        self.last_ack: str | None = None

    def connect(self) -> None:
        if self.dry_run:
            print("[Servo] Dry-run mode enabled (no serial commands will be sent).")
            return

        if self._serial_factory is None:
            try:
                import serial  # type: ignore[import-untyped]
            except ImportError as exc:
                raise RuntimeError(
                    "pyserial is not installed. Run `pip install pyserial` or use `--dry-run`."
                ) from exc
            self._serial_factory = serial.Serial

        if not self.port:
            self.port = detect_arduino_port()
        if not self.port:
            raise RuntimeError(
                "No Arduino serial port found. Connect the Arduino Uno, pass `--port /dev/ttyACM0` "
                "(or `COM3` on Windows), or test with `--dry-run`."
            )

        self._conn = self._serial_factory(self.port, self.baud, timeout=1.0)
        # Opening USB serial resets the Arduino Uno; wait for bootloader + setup().
        if self.reset_delay > 0:
            time.sleep(self.reset_delay)
        if hasattr(self._conn, "reset_input_buffer"):
            self._conn.reset_input_buffer()
        print(f"[Servo] Connected to Arduino on {self.port} @ {self.baud} baud.")

    def set_angle(self, angle: int, force: bool = False) -> bool:
        """Send target angle (0-180) to the Arduino if it changed. Returns True if sent."""
        clamped = max(0, min(180, int(angle)))
        if not force and self.current_angle == clamped:
            return False

        self.current_angle = clamped
        self.last_command = f"{clamped}"
        if self.dry_run or self._conn is None:
            self.last_ack = "DRY-RUN"
            print(f"[Servo][DRY-RUN] -> {clamped} deg")
            return True

        payload = f"{clamped}\n".encode("ascii")
        self._conn.write(payload)
        self._conn.flush()
        ack = ""
        if hasattr(self._conn, "readline"):
            try:
                ack = self._conn.readline().decode("ascii", errors="ignore").strip()
            except Exception:
                ack = ""
        self.last_ack = ack or None
        suffix = f" (Arduino: {ack})" if ack else ""
        print(f"[Servo] -> {clamped} deg{suffix}")
        return True

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None


class GestureClassifier:
    """Runs YOLO classification inference matching the repository's training transforms."""

    def __init__(self, model_path: Path, device: str = "cpu"):
        self.model_path = Path(model_path)
        self.device = device
        self.model: Any = None
        self.classes: list[str] = []
        self.image_size: int = 320

    def load(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        metadata_path = self.model_path.with_name("metadata.json")
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            self.image_size = int(metadata.get("config", {}).get("imgsz", 320))

        from ultralytics import YOLO

        model = YOLO(str(self.model_path))
        if model.task != "classify":
            raise ValueError(f"Expected a classification model, got task={model.task!r}.")
        self.model = model
        self.classes = [model.names[k] for k in sorted(model.names)]

    def predict(self, rgb_image: Image.Image, confidence: float = 0.80) -> tuple[str | None, float, dict[str, float]]:
        if self.model is None:
            raise RuntimeError("Classifier must be loaded before calling predict().")

        import torch

        pixels = np.asarray(rgb_image.convert("RGB"))
        if np.max(np.ptp(pixels, axis=(0, 1))) <= 2:
            return None, 0.0, {}

        # Match training preprocessing: square letterbox with white padding.
        square = ImageOps.pad(
            rgb_image.convert("RGB"),
            (self.image_size, self.image_size),
            color="white",
            method=Image.Resampling.BILINEAR,
        )
        tensor = torch.from_numpy(np.array(square).copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        result = self.model.predict(tensor, device=self.device, imgsz=self.image_size, verbose=False)[0]
        scores = {name: float(result.probs.data[i].item()) for i, name in enumerate(self.classes)}
        top_class = max(scores, key=scores.get)
        top_conf = scores[top_class]
        if top_conf < confidence:
            return None, top_conf, scores
        return top_class, top_conf, scores


def draw_hud(
    frame: np.ndarray,
    raw_gesture: str | None,
    confidence: float,
    stable_gesture: str | None,
    servo_angle: int | None,
    port_label: str,
    scores: dict[str, float],
) -> None:
    """Draw status overlay onto the live OpenCV camera frame."""
    import cv2

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (420, 165), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    angle_text = f"{servo_angle} deg" if servo_angle is not None else "--"
    if servo_angle == 60:
        state_text = f"SERVO: OPEN ({angle_text})"
        state_color = (80, 220, 100)
    elif servo_angle == 0:
        state_text = f"SERVO: CLOSED ({angle_text})"
        state_color = (80, 180, 255)
    else:
        state_text = f"SERVO: {angle_text}"
        state_color = (220, 220, 220)

    cv2.putText(frame, state_text, (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
    cv2.putText(
        frame,
        f"Stable: {stable_gesture or 'none'}  |  Raw: {raw_gesture or 'none'} ({confidence:.0%})",
        (22, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
    )
    score_line = "  ".join(f"{k}:{v:.2f}" for k, v in sorted(scores.items()))
    cv2.putText(frame, score_line or "Waiting for frame...", (22, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1)
    cv2.putText(frame, f"Arduino: {port_label}", (22, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1)
    cv2.putText(frame, "Keys: [o] Open 60  [c] Close 0  [q] Quit", (22, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1)


def resolve_default_model() -> Path:
    if DEFAULT_MODEL.is_file():
        return DEFAULT_MODEL
    return FALLBACK_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Control an Arduino Uno servo using webcam hand-gesture detection "
        "(open hand -> 60 deg, like -> 0 deg)."
    )
    parser.add_argument("--model", type=Path, default=resolve_default_model(), help="Path to YOLO .pt checkpoint.")
    parser.add_argument("--device", default="cpu", help="Inference device (default: cpu).")
    parser.add_argument("--port", default=None, help="Arduino serial port (e.g. /dev/ttyACM0 or COM3). Auto-detected if omitted.")
    parser.add_argument("--baud", type=int, default=9600, help="Serial baud rate (default: 9600).")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0).")
    parser.add_argument("--confidence", type=float, default=0.80, help="Minimum confidence threshold (default: 0.80).")
    parser.add_argument("--window", type=int, default=5, help="Consecutive smoothing frames required (default: 5).")
    parser.add_argument("--open-angle", type=int, default=60, help="Servo angle for open hand (default: 60).")
    parser.add_argument("--close-angle", type=int, default=0, help="Servo angle for like/thumbs up (default: 0).")
    parser.add_argument("--dry-run", action="store_true", help="Run without connecting to an Arduino.")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV preview window.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import cv2

    classifier = GestureClassifier(args.model, device=args.device)
    print(f"[Model] Loading {args.model}...")
    classifier.load()
    print(f"[Model] Ready. Classes: {classifier.classes}, imgsz={classifier.image_size}")

    mapper = GestureServoMapper(open_angle=args.open_angle, close_angle=args.close_angle)
    stabilizer = GestureStabilizer(window=args.window)
    servo = ServoController(port=args.port, baud=args.baud, dry_run=args.dry_run)
    servo.connect()
    # Initialize servo in closed position (0 deg).
    servo.set_angle(args.close_angle, force=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        servo.close()
        raise RuntimeError(f"Could not open webcam at index {args.camera}.")

    port_label = "DRY-RUN" if args.dry_run else (servo.port or "CONNECTED")
    print("[Camera] Running. Show OPEN HAND (60 deg) or LIKE (0 deg). Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[Camera] Failed to read frame; retrying...", file=sys.stderr)
                time.sleep(0.05)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            raw_gesture, conf, scores = classifier.predict(pil_image, confidence=args.confidence)
            stable_gesture = stabilizer.update(raw_gesture)

            target = mapper.target_angle(stable_gesture)
            if target is not None:
                servo.set_angle(target)

            if not args.no_display:
                draw_hud(frame, raw_gesture, conf, stable_gesture, servo.current_angle, port_label, scores)
                cv2.imshow("Gesture Servo Controller", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("o"):
                    servo.set_angle(args.open_angle, force=True)
                elif key == ord("c"):
                    servo.set_angle(args.close_angle, force=True)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        servo.set_angle(args.close_angle)
        servo.close()
        print("[System] Shutdown complete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
