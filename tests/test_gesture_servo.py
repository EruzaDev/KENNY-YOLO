"""Unit tests for the Arduino Uno gesture-to-servo controller."""

from pathlib import Path
import unittest

from PIL import Image

from arduino_servo.gesture_servo import (
    DEFAULT_MODEL,
    GestureClassifier,
    GestureServoMapper,
    GestureStabilizer,
    ServoController,
)


class FakeSerial:
    def __init__(self, port: str, baud: int, timeout: float = 1.0):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.written: list[bytes] = []
        self.closed = False
        self.flushed = False
        self.input_reset = False

    def write(self, data: bytes) -> int:
        self.written.append(data)
        return len(data)

    def flush(self) -> None:
        self.flushed = True

    def readline(self) -> bytes:
        last = self.written[-1].decode("ascii").strip()
        return f"OK:{last}\n".encode("ascii")

    def reset_input_buffer(self) -> None:
        self.input_reset = True

    def close(self) -> None:
        self.closed = True


class GestureServoTests(unittest.TestCase):
    def test_mapper_open_and_close_angles(self):
        mapper = GestureServoMapper(open_angle=60, close_angle=0)
        self.assertEqual(mapper.target_angle("open_palm"), 60)
        self.assertEqual(mapper.target_angle("open_hand"), 60)
        self.assertEqual(mapper.target_angle("like"), 0)
        self.assertEqual(mapper.target_angle("thumbs_up"), 0)
        self.assertIsNone(mapper.target_angle("ok"))
        self.assertIsNone(mapper.target_angle(None))

    def test_stabilizer_requires_window_and_resets_on_none(self):
        stabilizer = GestureStabilizer(window=3)
        self.assertIsNone(stabilizer.update("open_palm"))
        self.assertIsNone(stabilizer.update("open_palm"))
        self.assertEqual(stabilizer.update("open_palm"), "open_palm")
        # Weak/missing frame resets history immediately.
        self.assertIsNone(stabilizer.update(None))
        self.assertIsNone(stabilizer.update("like"))
        self.assertIsNone(stabilizer.update("like"))
        self.assertEqual(stabilizer.update("like"), "like")

    def test_servo_controller_deduplicates_serial_commands(self):
        created: list[FakeSerial] = []

        def factory(port, baud, timeout=1.0):
            conn = FakeSerial(port, baud, timeout=timeout)
            created.append(conn)
            return conn

        servo = ServoController(port="/dev/ttyACM0", baud=9600, reset_delay=0.0, serial_factory=factory)
        servo.connect()
        self.assertEqual(len(created), 1)
        conn = created[0]
        self.assertTrue(conn.input_reset)

        self.assertTrue(servo.set_angle(0))
        self.assertFalse(servo.set_angle(0))  # Duplicate angle suppressed
        self.assertTrue(servo.set_angle(60))
        self.assertFalse(servo.set_angle(60))  # Duplicate angle suppressed
        self.assertTrue(servo.set_angle(0))
        self.assertEqual(conn.written, [b"0\n", b"60\n", b"0\n"])

        servo.close()
        self.assertTrue(conn.closed)

    def test_real_model_inference_on_captured_samples(self):
        if not DEFAULT_MODEL.is_file():
            self.skipTest("Default gesture_captures_v2 model not found.")

        root = Path(__file__).resolve().parents[1]
        open_samples = sorted((root / "data" / "gesture_captures_v2" / "test" / "open_palm").glob("*.jpg"))
        like_samples = sorted((root / "data" / "gesture_captures_v2" / "test" / "like").glob("*.jpg"))
        if not open_samples or not like_samples:
            self.skipTest("Sample test split images not found.")

        classifier = GestureClassifier(DEFAULT_MODEL, device="cpu")
        classifier.load()
        mapper = GestureServoMapper(open_angle=60, close_angle=0)

        open_gesture, open_conf, _ = classifier.predict(Image.open(open_samples[0]), confidence=0.5)
        like_gesture, like_conf, _ = classifier.predict(Image.open(like_samples[0]), confidence=0.5)

        self.assertEqual(open_gesture, "open_palm", f"Unexpected prediction ({open_conf:.2f})")
        self.assertEqual(mapper.target_angle(open_gesture), 60)
        self.assertEqual(like_gesture, "like", f"Unexpected prediction ({like_conf:.2f})")
        self.assertEqual(mapper.target_angle(like_gesture), 0)


if __name__ == "__main__":
    unittest.main()
