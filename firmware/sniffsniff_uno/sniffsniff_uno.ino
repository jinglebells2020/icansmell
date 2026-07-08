/*
 * sniffsniff_uno.ino — Arduino Uno firmware for the sniffsniff 6-sensor e-nose
 * ===========================================================================
 *
 * Approach A (thin firmware, fat Python): the Uno does ONLY mux-scan + ADC +
 * CSV out. All calibration and feature math lives in the Python host, so
 * calibration constants (R0, RL) can be tuned and features re-extracted without
 * reflashing. The Uno's ~2 KB SRAM cannot hold feature buffers or a model
 * anyway. No floating point on-device — raw 10-bit ADC counts only.
 *
 * WIRE FORMAT (one line per full-array scan, newline-terminated ASCII):
 *
 *     <millis>,<c0>,<c1>,<c2>,<c3>,<c4>,<c5>\n
 *
 *   - millis : unsigned millis() timestamp (wraps ~every 49 days; host handles it)
 *   - c0..c5 : integer raw ADC counts 0..1023 (Uno 10-bit), one per mux channel
 *              C0..C5, each already the average of N=16 samples on-device.
 *   - ~20 Hz (one line per ~50 ms). Serial @ 115200 8N1.
 *
 * SENSOR / CHANNEL MAP (see design doc; only affects host-side labels):
 *   C0 MQ-2  C1 MQ-3  C2 MQ-4  C3 MQ-7  C4 MQ-8  C5 MQ-135
 *
 * ---------------------------------------------------------------------------
 * HARDWARE MUST-DOS (read before powering the array):
 *
 *   * HEATER SUPPLY: The 6 MQ heaters must be driven from an EXTERNAL 5 V
 *     supply rated >= 3 A. Budget: 6 x ~150-180 mA ~= 0.9-1.1 A total, so a
 *     2 A+ supply is fine; a 3 A supply leaves comfortable headroom. Do NOT
 *     power the heaters from the Uno's 5 V pin / USB — it cannot source ~1 A.
 *
 *   * COMMON GROUND: The external 5 V supply ground MUST be tied to the Uno
 *     GND. Without a shared reference the A0 readings are meaningless.
 *
 *   * CD74HC4067 MUX:
 *       - S0..S3 (address)  -> Uno D4, D5, D6, D7   (S0 = LSB)
 *       - SIG (common I/O)  -> Uno A0
 *       - EN (active-LOW)   -> tied to GND (mux permanently ENABLED)
 *       - Only C0..C5 are used. Tie the UNUSED channel pins C6..C15 to GND so
 *         they never float into a selected read.
 *       - Place a 100 nF (0.1 uF) decoupling cap from the mux VCC to GND, close
 *         to the chip.
 *
 *   * BURN-IN & WARM-UP: MQ sensors need a 24-48 h first-power burn-in before
 *     first use, and a 3-5 min warm-up at the start of every session before
 *     readings stabilize. Data taken before warm-up is drifty and unreliable.
 * ---------------------------------------------------------------------------
 */

// -------- Configuration --------------------------------------------------

const int NCH = 6;          // number of mux channels actually scanned: C0..C5
const int N_SAMPLES = 16;   // ADC reads averaged per channel (after 1 discard)

// Mux address pins S0..S3 on D4..D7 (S0 = least-significant address bit).
const int MUX_S0 = 4;
const int MUX_S1 = 5;
const int MUX_S2 = 6;
const int MUX_S3 = 7;

// Mux common signal output goes to A0.
const int SIG_PIN = A0;
// Mux EN is hard-wired to GND (active-LOW) so it is always enabled — no MCU pin.

// -------- Mux control ----------------------------------------------------

// selectCh(c): drive the 4 address bits for channel c (0..15), LSB = S0.
void selectCh(int c) {
  digitalWrite(MUX_S0, (c >> 0) & 0x01);
  digitalWrite(MUX_S1, (c >> 1) & 0x01);
  digitalWrite(MUX_S2, (c >> 2) & 0x01);
  digitalWrite(MUX_S3, (c >> 3) & 0x01);
}

// -------- Setup ----------------------------------------------------------

void setup() {
  Serial.begin(115200);

  pinMode(MUX_S0, OUTPUT);
  pinMode(MUX_S1, OUTPUT);
  pinMode(MUX_S2, OUTPUT);
  pinMode(MUX_S3, OUTPUT);

  selectCh(0);  // deterministic start on channel C0
}

// -------- Main scan loop -------------------------------------------------

void loop() {
  // Timestamp for this whole scan (unsigned; wraps ~every 49 days — host copes).
  unsigned long t = millis();

  // Emit the timestamp field first.
  Serial.print(t);

  // Scan channels C0..C(NCH-1).
  for (int c = 0; c < NCH; c++) {
    selectCh(c);
    delayMicroseconds(100);   // let the mux + ADC input settle after switching

    analogRead(SIG_PIN);      // one throwaway read to flush the ADC sample cap

    // Average N_SAMPLES reads with an integer accumulator (no floating point).
    // Max sum = 16 * 1023 = 16368, fits comfortably in an unsigned int/long.
    unsigned long acc = 0;
    for (int i = 0; i < N_SAMPLES; i++) {
      acc += analogRead(SIG_PIN);
    }
    int avg = (int)(acc / N_SAMPLES);  // integer mean, still 0..1023

    Serial.print(',');
    Serial.print(avg);
  }

  Serial.print('\n');

  delay(50);  // ~20 Hz full-array scan rate
}
