/*
 * sniffsniff_uno_b.ino — UNO 2 of the dual-Uno 9-sensor e-nose (3 sensors + SERVO)
 * ===========================================================================
 *
 * The array is split across two Arduino Unos. This is UNO 2: 3 sensors AND the
 * airflow servo (the servo is physically wired to THIS board's D12). Uno 1
 * (firmware/sniffsniff_uno) carries the other 6 sensors. Both boards plug into
 * the Mac over USB; the Python host reads both and MERGES them into one 9-channel
 * frame (Uno 1's 6 channels then Uno 2's 3, in board order), and sends servo
 * commands to whichever board has servo=true in sniffsniff.toml (this one).
 *
 * Thin-firmware contract: raw 10-bit ADC counts only, no floating point, CSV out.
 *
 * WIRE FORMAT (one line per full-array scan, newline-terminated ASCII):
 *
 *     <millis>,<c0>,<c1>,<c2>\n
 *
 *   - millis : unsigned millis() timestamp (this board's own clock)
 *   - c0..c2 : integer raw ADC counts 0..1023, each the average of N=16 samples.
 *   - ~20 Hz (one line per ~50 ms). Serial @ 115200 8N1.
 *
 * HOST -> DEVICE COMMAND (airflow servo):
 *
 *     S<angle>\n      e.g. "S105\n" -> move the servo to <angle> degrees (0..180)
 *
 * DIRECT ANALOG WIRING — each MQ module's AO goes straight to an Uno analog pin.
 * There is NO multiplexer. Channel order C0..C2 maps to:
 *
 *     A0 -> MQ2    (C0)   broad smoke / VOC
 *     A1 -> MQ4    (C1)   methane
 *     A2 -> MQ6    (C2)   LPG / propane / butane
 *
 * So sensor order (C0..C2) is: MQ2, MQ4, MQ6.  Servo SIGNAL -> D12.
 *
 * ---------------------------------------------------------------------------
 * HARDWARE MUST-DOS:
 *   * HEATER + SERVO SUPPLY: drive the MQ heaters AND the servo from an EXTERNAL
 *     5 V supply (>= 3 A for the full 9-sensor rig); NOT the Uno 5 V pin / USB
 *     (the servo's current spikes can brown out / reset the Uno).
 *   * COMMON GROUND: the external supply ground MUST tie to Uno 2 GND, Uno 1 GND,
 *     and the servo GND, or the readings and the servo are meaningless.
 *   * BURN-IN & WARM-UP: 24-48 h first-power burn-in; 3-5 min warm-up each session.
 * ---------------------------------------------------------------------------
 */

#include <Servo.h>

// -------- Configuration --------------------------------------------------

const int NCH = 3;          // number of analog channels scanned: C0..C2
const int N_SAMPLES = 16;   // ADC reads averaged per channel (after 1 discard)
const int SERVO_PIN = 12;   // airflow servo signal pin
const int SERVO_BOOT_ANGLE = 0;   // boot into the fresh-air position (clean air over sensors)

// Direct analog inputs, in channel order C0..C2 (see wiring map above).
const int PINS[NCH] = {A0, A1, A2};

Servo airflow;              // airflow selector servo

// -------- Command parsing (non-blocking) ---------------------------------

// Read any pending serial bytes and act on complete "S<angle>\n" commands.
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
