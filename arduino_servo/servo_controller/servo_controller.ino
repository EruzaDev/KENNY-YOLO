/*
 * Arduino Uno Servo Controller for Hand Gesture Recognition
 *
 * Wiring:
 *   Servo Signal (Orange/Yellow) -> Digital Pin 9
 *   Servo VCC    (Red)           -> 5V (or external 5V supply for high-torque servos)
 *   Servo GND    (Brown/Black)   -> GND (shared with Arduino GND if using external power)
 *
 * Serial Protocol (9600 baud, newline-terminated):
 *   "OPEN"  or "60"  -> Smoothly move servo to 60 degrees (open)
 *   "CLOSE" or "0"   -> Smoothly move servo to 0 degrees (closed)
 *   "<0-180>"        -> Smoothly move servo to any clamped angle in [0, 180]
 *   "PING"           -> Responds with "PONG"
 */

#include <Servo.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

const uint8_t SERVO_PIN = 9;
const uint16_t MIN_PULSE_US = 500;
const uint16_t MAX_PULSE_US = 2500;
const int CLOSE_ANGLE = 0;
const int OPEN_ANGLE = 60;
const unsigned long BAUD_RATE = 9600;
const unsigned long STEP_INTERVAL_MS = 15; // Smooth 1-deg steps to prevent USB 5V brownout resets

Servo handServo;
int currentAngle = CLOSE_ANGLE;
int targetAngle = CLOSE_ANGLE;
unsigned long lastStepTime = 0;

char rxBuffer[32];
uint8_t rxIndex = 0;

void setTargetAngle(int angle) {
  targetAngle = constrain(angle, 0, 180);
  digitalWrite(LED_BUILTIN, targetAngle > CLOSE_ANGLE ? HIGH : LOW);
  Serial.print("OK:");
  Serial.println(targetAngle);
}

bool isIntegerString(const char* str) {
  if (str == nullptr || str[0] == '\0') {
    return false;
  }
  for (uint8_t i = 0; str[i] != '\0'; i++) {
    if (!isdigit(static_cast<unsigned char>(str[i]))) {
      return false;
    }
  }
  return true;
}

void handleCommand(char* cmd) {
  while (*cmd == ' ' || *cmd == '\t') {
    cmd++;
  }
  size_t len = strlen(cmd);
  while (len > 0 && (cmd[len - 1] == ' ' || cmd[len - 1] == '\t')) {
    cmd[--len] = '\0';
  }
  if (len == 0) {
    return;
  }

  for (size_t i = 0; i < len; i++) {
    cmd[i] = static_cast<char>(toupper(static_cast<unsigned char>(cmd[i])));
  }

  if (strcmp(cmd, "OPEN") == 0) {
    setTargetAngle(OPEN_ANGLE);
  } else if (strcmp(cmd, "CLOSE") == 0) {
    setTargetAngle(CLOSE_ANGLE);
  } else if (strcmp(cmd, "PING") == 0) {
    Serial.println("PONG");
  } else if (isIntegerString(cmd)) {
    setTargetAngle(atoi(cmd));
  } else {
    Serial.println("ERR:INVALID_CMD");
  }
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  Serial.begin(BAUD_RATE);
  handServo.attach(SERVO_PIN, MIN_PULSE_US, MAX_PULSE_US);
  handServo.write(CLOSE_ANGLE);
  currentAngle = CLOSE_ANGLE;
  targetAngle = CLOSE_ANGLE;
  Serial.print("READY:");
  Serial.println(currentAngle);
}

void loop() {
  while (Serial.available() > 0) {
    char c = static_cast<char>(Serial.read());
    if (c == '\n' || c == '\r') {
      if (rxIndex > 0) {
        rxBuffer[rxIndex] = '\0';
        handleCommand(rxBuffer);
        rxIndex = 0;
      }
    } else if (rxIndex < sizeof(rxBuffer) - 1) {
      rxBuffer[rxIndex++] = c;
    } else {
      rxIndex = 0;
    }
  }

  // Non-blocking smooth movement (same 15ms/deg stepping as ../Servo/Servo.ino)
  if (currentAngle != targetAngle && (millis() - lastStepTime >= STEP_INTERVAL_MS)) {
    lastStepTime = millis();
    if (currentAngle < targetAngle) {
      currentAngle++;
    } else {
      currentAngle--;
    }
    handServo.write(currentAngle);
  }
}
