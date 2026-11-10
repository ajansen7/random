// Pin map for the Tattle Phone Lite board (ESP32-S3-WROOM-1-N8R2).
// Source of truth: hardware/kicad_gen/design.py. Keep the two in step.
#pragma once

// Buttons: each shorts to GND when pressed. All are RTC GPIOs (0-21), so any of
// them can wake the chip from deep sleep.
#define PIN_BTN_TATTLE  4    // yellow star
#define PIN_BTN_CALL    0    // green circle. Also the BOOT strap (10k pull-up on the
                             // board): hold it while switching on = programming mode
#define PIN_BTN_HANGUP  5    // red square

// Button lights: two LEDs per button behind one 150-ohm resistor each, driven HIGH = on.
#define PIN_LED_YELLOW  6
#define PIN_LED_GREEN   7
#define PIN_LED_RED     15

// PDM MEMS microphone (LinkMems LMD3526B261, L/R tied to GND = left slot)
#define PIN_MIC_CLK     16
#define PIN_MIC_DATA    17

// MAX98357A I2S amplifier (runs straight off the batteries)
#define PIN_AMP_BCLK    10
#define PIN_AMP_LRC     11
#define PIN_AMP_DIN     12
#define PIN_AMP_SD      13   // HIGH = amp on. 100k pull-down keeps it off while asleep

// Battery voltage through a 1M/1M divider: ADC reads VBAT / 2
#define PIN_VBAT_ADC    1

// Native USB (programming): GPIO19 = D-, GPIO20 = D+. Debug UART on test pads: TX0/RX0.
