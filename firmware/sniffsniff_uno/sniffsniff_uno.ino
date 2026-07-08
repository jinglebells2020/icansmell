/*
 * sniffsniff_uno.ino — Arduino Uno firmware for the sniffsniff 6-sensor e-nose
 * ===========================================================================
 *
 * Approach A (thin firmware, fat Python): the Uno does ADC reads + CSV out, and
 * drives one airflow servo on command. All calibration and feature math lives in
 * the Python host, so calibration constants (R0, RL) can be tuned and features
 * re-extracted without reflashing. No floating point on-device — raw 10-bit ADC
 * counts only.
 *
 * WIRE FORMAT (one line per full-array scan, newline-terminated ASCII):
 *
 *     <millis>,<c0>,<c1>,<c2>,<c3>,<c4>,<c5>\n
 *
 *   - millis : unsigned millis() timestamp (wraps ~every 49 days; host handles it)
 *   - c0..c5 : integer raw ADC counts 0..1023 (Uno 10-bit), one per analog
 *              channel C0..C5, each already the average of N=16 samples on-device.
 *   - ~20 Hz (one line per ~50 ms). Serial @ 115200 8N1.
 *
 * HOST -> DEVICE COMMAND (airflow servo):
 *
 *     S<angle>\n      e.g. "S90\n"  -> move the servo to <angle> degrees (0..180)
 *
 *   The command is read non-blockingly between scans; anything else is ignored.
 *   The servo selects the airflow path: one angle opens the FRESH-AIR straw
 *   (baseline/purge), the other opens the SAMPLE straw (exposure). The two exact
 *   angles are mechanical — find them with `sniffsniff servo` and set them in
 *   sniffsniff.toml ([servo] fresh_air_angle / sample_angle).
 *
 * DIRECT ANALOG WIRING — each MQ module's AO (analog out) goes straight to an
 * Uno analog input pin. There is NO multiplexer. Channel order C0..C5 maps to:
 *
 *     A0 -> MQ3    (C0)
 *     A1 -> MQ135  (C1)
 *     A2 -> MQ2    (C2)
 *     A3 -> MQ4    (C3)
 *     A4 -> MQ8    (C4)
 *     A5 -> MQ7    (C5)
 *
 * So sensor order (C0..C5) is: MQ3, MQ135, MQ2, MQ4, MQ8, MQ7.
 *
 * ---------------------------------------------------------------------------
 * HARDWARE MUST-DOS (read before powering the array):
 *
 *   * HEATER SUPPLY: The 6 MQ heaters must be driven from an EXTERNAL 5 V
 *     supply rated >= 3 A (6 x ~150-180 mA ~= 1 A total). Do NOT power them
 *     from the Uno's 5 V pin / USB.
 *
 *   * SERVO POWER: Power the servo from that same EXTERNAL 5 V supply (its
 *     current spikes can brown out / reset the Uno if taken from the 5 V pin).
 *     Servo SIGNAL -> D12. Servo GND / supply GND / Uno GND all COMMON.
 *
 *   * COMMON GROUND: The external 5 V supply ground MUST be tied to Uno GND, or
 *     the A0..A5 readings (and the servo) are meaningless.
 *
 *   * BURN-IN & WARM-UP: MQ sensors need a 24-48 h first-power burn-in, and a
 *     3-5 min warm-up at the start of every session before readings stabilize.
 * ---------------------------------------------------------------------------
 */

#include <Servo.h>

// -------- Configuration --------------------------------------------------

const int NCH = 6;          // number of analog channels scanned: C0..C5
const int N_SAMPLES = 16;   // ADC reads averaged per channel (after 1 discard)
const int SERVO_PIN = 12;   // airflow servo signal pin
const int SERVO_BOOT_ANGLE = 90;  // neutral start; real angles come from the host

// Direct analog inputs, in channel order C0..C5 (see wiring map above).
const int PINS[NCH] = {A0, A1, A2, A3, A4, A5};

Servo airflow;              // airflow selector servo

// -------- Command parsing (non-blocking) ---------------------------------

// Read any pending serial bytes and act on complete "S<angle>\n" commands.
// Kept tiny and non-blocking so the scan loop stays at ~20 Hz.
void handleCommands() {
  static char buf[8];
  static uint8_t len = 0;

  while (Serial.available() > 0) {
    char ch = (char)Serial.read();
    if (ch == '\n' || ch == '\r') {
      buf[len] = '\0';
      if (len > 0 && (buf[0] == 'S' || buf[0] == 's')) {
        int angle = atoi(buf + 1);
        angle = constrain(angle, 0, 180);
        airflow.write(angle);
      }
      len = 0;
    } else if (len < sizeof(buf) - 1) {
      buf[len++] = ch;
    } else {
      len = 0;  // overflow -> drop the malformed command
    }
  }
}

// -------- Setup ----------------------------------------------------------

void setup() {
  Serial.begin(115200);
  airflow.attach(SERVO_PIN);
  airflow.write(SERVO_BOOT_ANGLE);
}

// -------- Main scan loop -------------------------------------------------

void loop() {
  handleCommands();  // service any pending servo command first

  // Timestamp for this whole scan (unsigned; wraps ~every 49 days — host copes).
  unsigned long t = millis();
  Serial.print(t);

  // Scan channels C0..C(NCH-1).
  for (int c = 0; c < NCH; c++) {
    // One throwaway read to settle the Uno ADC sample/hold after its internal
    // channel switch — important for the high source impedance of MQ dividers.
    analogRead(PINS[c]);

    // Average N_SAMPLES reads with an integer accumulator (no floating point).
    // Max sum = 16 * 1023 = 16368, fits comfortably in an unsigned long.
    unsigned long acc = 0;
    for (int i = 0; i < N_SAMPLES; i++) {
      acc += analogRead(PINS[c]);
    }
    int avg = (int)(acc / N_SAMPLES);  // integer mean, still 0..1023

    Serial.print(',');
    Serial.print(avg);
  }

  Serial.print('\n');

  delay(50);  // ~20 Hz full-array scan rate
}
