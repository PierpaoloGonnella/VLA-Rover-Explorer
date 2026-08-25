/**********************************************************************
  Product     : Freenove 4WD Car for UNO
  Based on    : Multifunctional_RF24_Remote_Car.ino (Freenove)
  Purpose     : BLE remote control with an always-on ultrasonic stop guard.

  IMPORTANT
  - The ultrasonic sensor NEVER commands motion.
  - It may only stop forward translation and veto new forward commands.
  - Reverse and in-place turns remain available to the remote controller so
    it can escape and plan a detour.
**********************************************************************/
#include "Freenove_WS2812B_RGBLED_Controller.h"
#include <EEPROM.h>
#include <Servo.h>

#define ACTION_MOVE 'A'
#define ACTION_RGB 'C'
#define ACTION_BUZZER 'D'
#define ACTION_ULTRASONIC 'E'
#define ACTION_CAR_MODE 'H'
#define ACTION_GET_VOLTAGE 'I'

#define MODE_NONE 0
#define MODE_GRAVITY 1
#define MODE_ULTRASONIC 2
#define MODE_TRACKING 3

#define PIN_SERVO 2
#define MOTOR_DIRECTION 0
#define PIN_DIRECTION_LEFT 4
#define PIN_DIRECTION_RIGHT 3
#define PIN_MOTOR_PWM_LEFT 6
#define PIN_MOTOR_PWM_RIGHT 5
#define PIN_SONIC_TRIG 7
#define PIN_SONIC_ECHO 8
#define PIN_BATTERY A0
#define PIN_BUZZER A0
#define MOTOR_PWM_DEAD 10

#define STRIP_I2C_ADDRESS 0x20
#define STRIP_LEDS_COUNT 10
#define SERVO_OFFSET_EEPROM_ADDRESS 0
#define SERVO_CENTER_DEGREES 90
#define SERVO_LEFT_DEGREES 140
#define SERVO_RIGHT_DEGREES 40

// Tune these two distances for the rover speed and physical stopping distance.
// Separate thresholds provide hysteresis, preventing rapid stop/clear chatter.
#define SONAR_STOP_DISTANCE_CM 25
#define SONAR_CLEAR_DISTANCE_CM 35
#define SONAR_MAX_DISTANCE_CM 300
#define SONAR_SAMPLE_INTERVAL_MS 60
#define SONAR_UPLOAD_INTERVAL_MS 200
#define SONAR_CLEAR_READINGS_REQUIRED 3
#define SONAR_TIMEOUT_US ((unsigned long)SONAR_MAX_DISTANCE_CM * 59UL)
#define SONAR_SERVO_SETTLE_MS 140
#define SONAR_SCAN_SAMPLES 3
#define SONAR_RESCAN_INTERVAL_MS 750

#define VOLTAGE_UPLOAD_INTERVAL_MS 3000
#define COMMAND_FIELDS_MAX 8

Freenove_WS2812B_Controller strip(STRIP_I2C_ADDRESS, STRIP_LEDS_COUNT, TYPE_GRB);
Servo servo;

String inputStringBLE;
bool stringComplete = false;
int currentLeftSpeed = 0;
int currentRightSpeed = 0;
int sonarDistanceCm = SONAR_MAX_DISTANCE_CM;
int sonarLeftCm = SONAR_MAX_DISTANCE_CM;
int sonarRightCm = SONAR_MAX_DISTANCE_CM;
bool obstacleBlocked = false;
uint8_t clearReadings = 0;
uint32_t lastSonarSampleTime = 0;
uint32_t lastSonarUploadTime = 0;
uint32_t lastVoltageUploadTime = 0;
uint32_t lastCompletedScanTime = 0;
uint32_t scanStateChangedTime = 0;
uint32_t lastScanSampleTime = 0;
uint16_t scanSequence = 0;
uint16_t scanSampleSum = 0;
uint8_t scanSampleCount = 0;

enum SonarScanState {
  SCAN_FRONT,
  SCAN_WAIT_LEFT,
  SCAN_SAMPLE_LEFT,
  SCAN_WAIT_CENTER,
  SCAN_SAMPLE_CENTER,
  SCAN_WAIT_RIGHT,
  SCAN_SAMPLE_RIGHT,
  SCAN_RETURN_CENTER
};
SonarScanState sonarScanState = SCAN_FRONT;

char servoOffset = 0;
float batteryVoltage = 0;
bool isBuzzered = false;
uint8_t stripDisplayMode = 1;
uint8_t rgbRed = 255;
uint8_t rgbGreen = 0;
uint8_t rgbBlue = 0;

void setup() {
  pinsSetup();
  Serial.begin(115200);
  servoSetup();
  strip.begin();
  strip.setAllLedsColor(0xFF0000);
  motorRunRaw(0, 0);
}

void loop() {
  // Run the local guard independently of BLE mode or incoming commands.
  updateUltrasonicSafety();

  if (stringComplete) {
    processBleCommand(inputStringBLE);
    inputStringBLE = "";
    stringComplete = false;
  }

  if (millis() - lastVoltageUploadTime >= VOLTAGE_UPLOAD_INTERVAL_MS) {
    uploadVoltageToApp();
    lastVoltageUploadTime = millis();
  }

  updateLeds();
}

void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\r') continue;
    inputStringBLE += inChar;
    if (inChar == '\n') stringComplete = true;
  }
}

void processBleCommand(String command) {
  String fields[COMMAND_FIELDS_MAX];
  int values[COMMAND_FIELDS_MAX] = {0};
  uint8_t fieldCount = 0;

  while (fieldCount < COMMAND_FIELDS_MAX) {
    int separator = command.indexOf('#');
    if (separator < 0) break;
    fields[fieldCount] = command.substring(0, separator);
    values[fieldCount] = fields[fieldCount].toInt();
    command = command.substring(separator + 1);
    fieldCount++;
  }
  if (fieldCount == 0 || fields[0].length() == 0) return;

  switch (fields[0].charAt(0)) {
    case ACTION_MOVE:
      if (fieldCount >= 3) guardedMotorRun(values[1], values[2]);
      break;

    case ACTION_CAR_MODE:
      // Modes are accepted for compatibility with the Freenove application,
      // but no autonomous mode is allowed to generate motor commands.
      motorRunRaw(0, 0);
      cancelSideScan();
      break;

    case ACTION_BUZZER:
      if (fieldCount >= 2) setBuzzer(values[1] != 0);
      break;

    case ACTION_RGB:
      if (fieldCount >= 5) {
        stripDisplayMode = constrain(values[1], 0, 3);
        rgbRed = constrain(values[2], 0, 255);
        rgbGreen = constrain(values[3], 0, 255);
        rgbBlue = constrain(values[4], 0, 255);
      }
      break;

    default:
      break;
  }
}

bool isForwardTranslation(int speedLeft, int speedRight) {
  // Forward and forward arcs have non-negative wheel speeds and at least one
  // driven wheel. Mixed signs are an in-place turn and remain remotely usable.
  return speedLeft >= 0 && speedRight >= 0 &&
         (speedLeft > MOTOR_PWM_DEAD || speedRight > MOTOR_PWM_DEAD);
}

void guardedMotorRun(int speedLeft, int speedRight) {
  if (obstacleBlocked && isForwardTranslation(speedLeft, speedRight)) {
    motorRunRaw(0, 0);
    uploadSonarToApp();
    return;
  }
  if (speedLeft != 0 || speedRight != 0) cancelSideScan();
  motorRunRaw(speedLeft, speedRight);
}

void motorRunRaw(int speedLeft, int speedRight) {
  currentLeftSpeed = constrain(speedLeft, -255, 255);
  currentRightSpeed = constrain(speedRight, -255, 255);

  int pwmLeft = abs(currentLeftSpeed);
  int pwmRight = abs(currentRightSpeed);
  int dirLeft = (currentLeftSpeed > 0 ? 0 : 1) ^ MOTOR_DIRECTION;
  int dirRight = (currentRightSpeed > 0 ? 1 : 0) ^ MOTOR_DIRECTION;

  if (pwmLeft < MOTOR_PWM_DEAD) pwmLeft = 0;
  if (pwmRight < MOTOR_PWM_DEAD) pwmRight = 0;
  digitalWrite(PIN_DIRECTION_LEFT, dirLeft);
  digitalWrite(PIN_DIRECTION_RIGHT, dirRight);
  analogWrite(PIN_MOTOR_PWM_LEFT, pwmLeft);
  analogWrite(PIN_MOTOR_PWM_RIGHT, pwmRight);
}

bool motorsAreStopped() {
  return abs(currentLeftSpeed) <= MOTOR_PWM_DEAD &&
         abs(currentRightSpeed) <= MOTOR_PWM_DEAD;
}

void applyFrontMeasurement(int distanceCm) {
  sonarDistanceCm = distanceCm;
  if (sonarDistanceCm <= SONAR_STOP_DISTANCE_CM) {
    obstacleBlocked = true;
    clearReadings = 0;
    if (isForwardTranslation(currentLeftSpeed, currentRightSpeed)) {
      // This is the only motor action the sonar is permitted to create.
      motorRunRaw(0, 0);
    }
  } else if (sonarDistanceCm >= SONAR_CLEAR_DISTANCE_CM) {
    if (clearReadings < SONAR_CLEAR_READINGS_REQUIRED) clearReadings++;
    if (clearReadings >= SONAR_CLEAR_READINGS_REQUIRED) obstacleBlocked = false;
  } else {
    clearReadings = 0;
  }
}

void beginScanSamples(uint8_t sampleState) {
  sonarScanState = (SonarScanState)sampleState;
  scanSampleSum = 0;
  scanSampleCount = 0;
  lastScanSampleTime = 0;
}

void moveScanServo(uint8_t degrees, uint8_t waitState) {
  writeServo(degrees);
  sonarScanState = (SonarScanState)waitState;
  scanStateChangedTime = millis();
}

void startSideScan() {
  if (!obstacleBlocked || !motorsAreStopped() || sonarScanState != SCAN_FRONT) return;
  moveScanServo(SERVO_LEFT_DEGREES, SCAN_WAIT_LEFT);
}

void cancelSideScan() {
  if (sonarScanState == SCAN_FRONT) return;
  writeServo(SERVO_CENTER_DEGREES);
  sonarScanState = SCAN_FRONT;
  lastSonarSampleTime = millis();
}

bool collectScanSample(int *result) {
  uint32_t now = millis();
  if (lastScanSampleTime != 0 &&
      now - lastScanSampleTime < SONAR_SAMPLE_INTERVAL_MS) return false;
  lastScanSampleTime = now;
  scanSampleSum += readSonarCm();
  scanSampleCount++;
  if (scanSampleCount < SONAR_SCAN_SAMPLES) return false;
  *result = scanSampleSum / SONAR_SCAN_SAMPLES;
  return true;
}

void updateSideScan() {
  uint32_t now = millis();
  if (!motorsAreStopped()) {
    cancelSideScan();
    return;
  }

  switch (sonarScanState) {
    case SCAN_WAIT_LEFT:
      if (now - scanStateChangedTime >= SONAR_SERVO_SETTLE_MS)
        beginScanSamples(SCAN_SAMPLE_LEFT);
      break;
    case SCAN_SAMPLE_LEFT:
      if (collectScanSample(&sonarLeftCm))
        moveScanServo(SERVO_CENTER_DEGREES, SCAN_WAIT_CENTER);
      break;
    case SCAN_WAIT_CENTER:
      if (now - scanStateChangedTime >= SONAR_SERVO_SETTLE_MS)
        beginScanSamples(SCAN_SAMPLE_CENTER);
      break;
    case SCAN_SAMPLE_CENTER: {
      int centre;
      if (collectScanSample(&centre)) {
        applyFrontMeasurement(centre);
        moveScanServo(SERVO_RIGHT_DEGREES, SCAN_WAIT_RIGHT);
      }
      break;
    }
    case SCAN_WAIT_RIGHT:
      if (now - scanStateChangedTime >= SONAR_SERVO_SETTLE_MS)
        beginScanSamples(SCAN_SAMPLE_RIGHT);
      break;
    case SCAN_SAMPLE_RIGHT:
      if (collectScanSample(&sonarRightCm))
        moveScanServo(SERVO_CENTER_DEGREES, SCAN_RETURN_CENTER);
      break;
    case SCAN_RETURN_CENTER:
      if (now - scanStateChangedTime >= SONAR_SERVO_SETTLE_MS) {
        sonarScanState = SCAN_FRONT;
        lastCompletedScanTime = now;
        lastSonarSampleTime = now;
        scanSequence++;
        uploadSonarToApp();
        lastSonarUploadTime = now;
      }
      break;
    default:
      break;
  }
}

void updateUltrasonicSafety() {
  uint32_t now = millis();
  if (sonarScanState != SCAN_FRONT) {
    updateSideScan();
    return;
  }
  if (now - lastSonarSampleTime < SONAR_SAMPLE_INTERVAL_MS) return;
  lastSonarSampleTime = now;

  bool wasBlocked = obstacleBlocked;
  applyFrontMeasurement(readSonarCm());
  if (wasBlocked != obstacleBlocked ||
      now - lastSonarUploadTime >= SONAR_UPLOAD_INTERVAL_MS) {
    uploadSonarToApp();
    lastSonarUploadTime = now;
  }
  if (obstacleBlocked && motorsAreStopped() &&
      now - lastCompletedScanTime >= SONAR_RESCAN_INTERVAL_MS) {
    startSideScan();
  }
}

int readSonarCm() {
  digitalWrite(PIN_SONIC_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_SONIC_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_SONIC_TRIG, LOW);

  unsigned long echoTime = pulseIn(PIN_SONIC_ECHO, HIGH, SONAR_TIMEOUT_US);
  if (echoTime == 0) return SONAR_MAX_DISTANCE_CM;
  int centimetres = (int)(echoTime * 0.0343f / 2.0f);
  return constrain(centimetres, 1, SONAR_MAX_DISTANCE_CM);
}

void uploadSonarToApp() {
  // E#front_cm#blocked#left_cm#right_cm#scan_sequence#
  // The first value remains compatible with the original Freenove telemetry.
  Serial.print(ACTION_ULTRASONIC);
  Serial.print('#');
  Serial.print(sonarDistanceCm);
  Serial.print('#');
  Serial.print(obstacleBlocked ? 1 : 0);
  Serial.print('#');
  Serial.print(sonarLeftCm);
  Serial.print('#');
  Serial.print(sonarRightCm);
  Serial.print('#');
  Serial.print(scanSequence);
  Serial.println('#');
}

void uploadVoltageToApp() {
  int millivolts = 0;
  if (getBatteryVoltage()) millivolts = (int)(batteryVoltage * 1000.0f);
  Serial.print(ACTION_GET_VOLTAGE);
  Serial.print('#');
  Serial.print(millivolts);
  Serial.println('#');
}

void servoSetup() {
  servoOffset = (char)EEPROM.read(SERVO_OFFSET_EEPROM_ADDRESS);
  servoOffset = constrain(servoOffset, -10, 10);
  servo.attach(PIN_SERVO);
  writeServo(SERVO_CENTER_DEGREES);
}

void writeServo(uint8_t degrees) {
  servo.write(constrain((int)degrees + servoOffset, 0, 180));
}

void updateLeds() {
  static uint8_t lastMode = 255;
  static uint8_t lastRed = 0, lastGreen = 0, lastBlue = 0;
  if (lastMode == stripDisplayMode && lastRed == rgbRed &&
      lastGreen == rgbGreen && lastBlue == rgbBlue) return;

  // For this safety firmware all app animation modes resolve to the requested
  // solid colour; LED animation must never delay sonar sampling.
  strip.setAllLedsColor(rgbRed, rgbGreen, rgbBlue);
  lastMode = stripDisplayMode;
  lastRed = rgbRed;
  lastGreen = rgbGreen;
  lastBlue = rgbBlue;
}

void pinsSetup() {
  pinMode(PIN_DIRECTION_LEFT, OUTPUT);
  pinMode(PIN_MOTOR_PWM_LEFT, OUTPUT);
  pinMode(PIN_DIRECTION_RIGHT, OUTPUT);
  pinMode(PIN_MOTOR_PWM_RIGHT, OUTPUT);
  pinMode(PIN_SONIC_TRIG, OUTPUT);
  pinMode(PIN_SONIC_ECHO, INPUT);
  digitalWrite(PIN_SONIC_TRIG, LOW);
  setBuzzer(false);
}

bool getBatteryVoltage() {
  if (isBuzzered) return false;
  pinMode(PIN_BATTERY, INPUT);
  int batteryAdc = analogRead(PIN_BATTERY);
  if (batteryAdc >= 614) return false;
  batteryVoltage = batteryAdc / 1023.0f * 5.0f * 4.0f;
  return true;
}

void setBuzzer(bool enabled) {
  isBuzzered = enabled;
  pinMode(PIN_BUZZER, OUTPUT);
  digitalWrite(PIN_BUZZER, enabled ? HIGH : LOW);
}
