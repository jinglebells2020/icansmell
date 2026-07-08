/*
 * sniffsniff_uno.ino — Arduino Uno firmware for the sniffsniff 6-sensor e-nose
 * ===========================================================================
 *
 * Approach A (thin firmware, fat Python): the Uno does ONLY ADC reads + CSV
 * out. All calibration and feature math lives in the Python host, so
 * calibration constants (R0, RL) can be tuned and features re-extracted without
 * reflashing. The Uno's ~2 KB SRAM cannot hold feature buffers or a model
 * anyway. No floating point on-device — raw 10-bit ADC counts only.
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
 *     supply rated >= 3 A. Budget: 6 x ~150-180 mA ~= 1 A total, so a 3 A
 *     supply leaves comfortable headroom. Do NOT power the heaters from the
 *     Uno's 5 V pin / USB — it cannot source ~1 A.
 *
 *   * COMMON GROUND: The external 5 V supply ground MUST be tied to the Uno
 *     GND. Without a shared reference the A0..A5 readings are meaningless.
 *
 *   * BURN-IN & WARM-UP: MQ sensors need a 24-48 h first-power burn-in before
 *     first use, and a 3-5 min warm-up at the start of every session before
 *     readings stabilize. Data taken before warm-up is drifty and unreliable.
 * ---------------------------------------------------------------------------
 */

// -------- Configuration --------------------------------------------------

const int NCH = 6;          // number of analog channels scanned: C0..C5
const int N_SAMPLES = 16;   // ADC reads averaged per channel (after 1 discard)

// Direct analog inputs, in channel order C0..C5 (see wiring map above).
const int PINS[NCH] = {A0, A1, A2, A3, A4, A5};

// -------- Setup ----------------------------------------------------------

void setup() {
  Serial.begin(115200);
}

// -------- Main scan loop -------------------------------------------------

void loop() {
  // Timestamp for this whole scan (unsigned; wraps ~every 49 days — host copes).
  unsigned long t = millis();

  // Emit the timestamp field first.
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
