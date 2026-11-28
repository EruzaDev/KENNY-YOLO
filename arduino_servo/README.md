# Hand Gesture Servo Controller (Arduino Uno + YOLO)

Controls a servo motor attached to an **Arduino Uno** using your laptop's webcam and the trained YOLO gesture classifier in this repository:

| Detected Gesture | Servo Action | Angle |
| --- | --- | --- |
| **Open Hand** (`open_palm`) | Open | **60°** |
| **Like / Thumbs Up** (`like`, `thumbs_up`) | Close | **0°** |
| Other / No gesture (`ok`, low confidence) | Hold current position | — |

## 1. Wiring (Arduino Uno + Servo)

| Servo Wire | Arduino Uno Pin |
| --- | --- |
| **Signal** (Orange / Yellow) | **Digital Pin 9** |
| **VCC** (Red) | **5V** *(use an external 5V supply for large/high-torque servos)* |
| **GND** (Brown / Black) | **GND** *(connect external supply GND to Arduino GND)* |

> For a small micro servo (e.g., SG90), the Arduino Uno's 5V pin is usually sufficient for light loads. For larger servos (e.g., MG996R), power the servo from a separate 5V supply and share GND with the Arduino.

## 2. Upload the Arduino Sketch

1. Open [servo_controller.ino](servo_controller/servo_controller.ino) in the **Arduino IDE**.
2. Select **Tools → Board → Arduino Uno**.
3. Select **Tools → Port →** your Arduino Uno port (e.g., `/dev/ttyACM0` on Linux, `COM3` on Windows, `/dev/cu.usbmodem*` on macOS).
4. Click **Upload**.

## 3. Install `pyserial` and Run

Install `pyserial` (the rest of the dependencies are already in the root `requirements.txt`):

```bash
pip install -r arduino_servo/requirements.txt
```

Run the webcam controller (auto-detects the Arduino Uno serial port):

```bash
python3 arduino_servo/gesture_servo.py
```

### Useful Options

```bash
# Specify the serial port explicitly (Linux / Windows / macOS)
python3 arduino_servo/gesture_servo.py --port /dev/ttyACM0
python3 arduino_servo/gesture_servo.py --port COM3

# Test webcam gesture detection without an Arduino plugged in
python3 arduino_servo/gesture_servo.py --dry-run

# Tune confidence threshold or smoothing frames
python3 arduino_servo/gesture_servo.py --confidence 0.85 --window 5

# Use a different trained checkpoint
python3 arduino_servo/gesture_servo.py --model models/ok_camera/best.pt
```

### Keyboard Controls (Preview Window)

- **`o`** — Manually open servo to 60°
- **`c`** — Manually close servo to 0°
- **`q`** or **`Esc`** — Quit (closes servo to 0° on exit)
