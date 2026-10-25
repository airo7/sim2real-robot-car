/**
 * Elegoo Smart Robot Car V4 firmware: motors, IR + ultrasonic sensors, binary link to the PC.
 *
 * TB6612FNG with one direction pin per motor; IN2 is hardware-tied to the complement of IN1.
 * The link is the hardware UART (D0/D1), shared with USB: unplug the BLE module before flashing.
 * PC → car: 6-byte CMD (0xAA); car → PC: 8-byte SENS (0xBB). Must match hardware/protocol.py.
 */

// 1 = BLE UART module in the shield's ESP32 socket (3.3 V level-safe), 0 = USB cable.
#define USE_BLUETOOTH  1

#if USE_BLUETOOTH
  #define COMM_BAUD          9600       // the BLE module's UART rate
  #define SENS_INTERVAL_MS   40
#else
  #define COMM_BAUD          115200
  #define SENS_INTERVAL_MS   20
#endif

static Stream* comm = &Serial;

#define PIN_STBY    3   // TB6612 standby, HIGH = enabled
#define PIN_PWL     6
#define PIN_PWR     5
#define PIN_IL      8   // left direction, HIGH = forward
#define PIN_IR_MOT  7   // right direction, HIGH = forward

// A0 is physically the RIGHT sensor; SENS keeps A0,A1,A2 order and protocol.SensReader swaps them.
#define PIN_IR_L    A0
#define PIN_IR_C    A1
#define PIN_IR_R    A2

#define PIN_TRIG    13  // HC-SR04
#define PIN_ECHO    12

#define CMD_HEADER        0xAA
#define CMD_LEN           6
#define SENS_HEADER       0xBB
#define SENS_LEN          8
#define CMD_TIMEOUT_MS    500
#define MOTOR_MIN_PWM     55     // mode-0 dead-band offset
#define MOTOR_MAX_PWM     230    // below 255 to protect the gearbox
#define V_MAX             0.5f   // m/s, must match hardware/protocol.py

static unsigned long lastCmdMs  = 0;
static unsigned long lastSensMs = 0;
static byte          cmdBuf[CMD_LEN];
static int           cmdBufIdx  = 0;

// Signed PWM (positive = forward) with dead-band offset.
static int velToPWM(float v) {
    if (fabsf(v) < 0.02f) return 0;
    int mag = (int)(fabsf(v) / V_MAX * (MOTOR_MAX_PWM - MOTOR_MIN_PWM)) + MOTOR_MIN_PWM;
    mag = constrain(mag, 0, MOTOR_MAX_PWM);
    return (v > 0) ? mag : -mag;
}

static void setMotors(int pwm_l, int pwm_r) {
    digitalWrite(PIN_IL, (pwm_l >= 0) ? HIGH : LOW);
    analogWrite(PIN_PWL, abs(pwm_l));
    digitalWrite(PIN_IR_MOT, (pwm_r >= 0) ? HIGH : LOW);
    analogWrite(PIN_PWR, abs(pwm_r));
}

static void stopMotors() {
    analogWrite(PIN_PWL, 0);
    analogWrite(PIN_PWR, 0);
}

static uint16_t readUltrasonicMM() {
    digitalWrite(PIN_TRIG, LOW);  delayMicroseconds(2);
    digitalWrite(PIN_TRIG, HIGH); delayMicroseconds(10);
    digitalWrite(PIN_TRIG, LOW);
    long dur = pulseIn(PIN_ECHO, HIGH, 25000UL);  // 25 ms ≈ 4.25 m max
    if (dur == 0) return 9999;
    return (uint16_t)(dur * 0.17f);   // mm = µs × 340 m/s ÷ 2 ÷ 1000
}

/*
 * CMD: [0] 0xAA  [1] left  [2] right  [3] mode  [4] reserved  [5] [1]^[2]^[3]
 *   mode 0: byte 0..255 → -V_MAX..+V_MAX (128 ≈ stop), dead-band offset applied
 *   mode 1: raw PWM = (byte - 128) * 2, clamped to ±MOTOR_MAX_PWM
 * The checksum covers the mode byte, so mode-0 packets checksummed as [1]^[2] stay valid.
 */
#define CMD_MODE_VEL  0
#define CMD_MODE_PWM  1

static int byteToPWM(byte b) {
    return constrain(((int)b - 128) * 2, -MOTOR_MAX_PWM, MOTOR_MAX_PWM);
}

static void processCmd(const byte* buf) {
    if (buf[5] != (buf[1] ^ buf[2] ^ buf[3])) return;

    if (buf[3] == CMD_MODE_PWM) {
        setMotors(byteToPWM(buf[1]), byteToPWM(buf[2]));
    } else if (buf[3] == CMD_MODE_VEL) {
        float vl = ((float)buf[1] / 255.0f * 2.0f - 1.0f) * V_MAX;
        float vr = ((float)buf[2] / 255.0f * 2.0f - 1.0f) * V_MAX;
        setMotors(velToPWM(vl), velToPWM(vr));
    } else {
        return;
    }
    lastCmdMs = millis();
}

static void readSerial() {
    while (comm->available()) {
        byte b = comm->read();
        if (cmdBufIdx == 0 && b != CMD_HEADER) continue;
        cmdBuf[cmdBufIdx++] = b;
        if (cmdBufIdx == CMD_LEN) {
            processCmd(cmdBuf);
            cmdBufIdx = 0;
        }
    }
}

/*
 * SENS: [0] 0xBB  [1..3] IR A0,A1,A2 (analogRead >> 2)  [4..5] distance mm, big-endian
 *       [6] reserved  [7] [1]^[2]^[3]^[4]^[5]
 * IR is sent analog and thresholded on the PC: digitalRead rarely crosses the logic threshold.
 */
static void sendSensors() {
    byte ir_l = (byte)(analogRead(PIN_IR_L) >> 2);
    byte ir_c = (byte)(analogRead(PIN_IR_C) >> 2);
    byte ir_r = (byte)(analogRead(PIN_IR_R) >> 2);
    uint16_t dist = readUltrasonicMM();

    byte pkt[SENS_LEN] = {
        SENS_HEADER,
        ir_l, ir_c, ir_r,
        (byte)(dist >> 8), (byte)(dist & 0xFF),
        0x00,
        (byte)(ir_l ^ ir_c ^ ir_r ^ (byte)(dist >> 8) ^ (byte)(dist & 0xFF))
    };
    comm->write(pkt, SENS_LEN);
}

void setup() {
    Serial.begin(COMM_BAUD);

    pinMode(PIN_STBY,   OUTPUT); digitalWrite(PIN_STBY, HIGH);
    pinMode(PIN_IL,     OUTPUT);
    pinMode(PIN_IR_MOT, OUTPUT);
    pinMode(PIN_PWL,    OUTPUT);
    pinMode(PIN_PWR,    OUTPUT);

    pinMode(PIN_IR_L, INPUT);
    pinMode(PIN_IR_C, INPUT);
    pinMode(PIN_IR_R, INPUT);

    pinMode(PIN_TRIG, OUTPUT); digitalWrite(PIN_TRIG, LOW);
    pinMode(PIN_ECHO, INPUT);

    stopMotors();
    lastCmdMs = millis();
}

void loop() {
    unsigned long now = millis();

    if (now - lastCmdMs > CMD_TIMEOUT_MS) {
        stopMotors();
    }

    readSerial();

    if (now - lastSensMs >= SENS_INTERVAL_MS) {
        lastSensMs = now;
        sendSensors();
    }
}
