// Pin map for the ESP32-S3 (WROOM-1 N16R8). The same map is used on the
// breadboard prototype and the custom PCB, so firmware carries over unchanged.
//
// Avoided on purpose: GPIO0/3/45/46 (strapping), GPIO19/20 (native USB, used
// for flashing over the USB-C port), GPIO35-37 (octal PSRAM on R8 modules).
#pragma once

// I2S microphone (INMP441 / ICS-43434). Tie the mic's L/R pin to GND (= left).
#define PIN_MIC_BCLK  4
#define PIN_MIC_WS    5
#define PIN_MIC_SD    6

// I2S amplifier (MAX98357A)
#define PIN_AMP_BCLK  15
#define PIN_AMP_LRC   16
#define PIN_AMP_DIN   17
#define PIN_AMP_SD    18   // amp shutdown: LOW = off (saves ~2 mA and kills hiss)

// The big "CALL" button (to GND, internal pull-up). Must be an RTC GPIO (0-21)
// so it can wake the chip from deep sleep.
#define PIN_CALL_BTN  7

// Addressable LED chain (SK6812MINI). LED 0 = call button, 1-12 = keypad 1..9 * 0 #,
// 13 = status ("in use") light above the keypad.
#define PIN_LEDS      8
#define NUM_LEDS      14
#define LED_STATUS    13
// Drives a P-MOSFET high-side switch on the LED 5V/VBAT rail. WS2812s draw
// ~1 mA each even when "off", which would flatten the battery in deep sleep.
#define PIN_LED_PWR   11   // HIGH = LEDs powered

// Battery voltage via 2:1 divider (only needed if you skip the MAX17048 gauge)
#define PIN_VBAT_ADC  1

// I2C bus: fuel gauge (MAX17048 @0x36) + keypad expander (TCA9555 @0x20)
#define PIN_I2C_SDA   9
#define PIN_I2C_SCL   10

// Charger status (BQ24075 open-drain outputs, pulled up to 3V3; LOW = active)
#define PIN_CHG       12   // LOW while charging
#define PIN_PGOOD     13   // LOW when USB power is present
#define PIN_GAUGE_ALRT 14  // MAX17048 low-battery alert (optional)
#define PIN_KEYPAD_INT 21  // TCA9555 INT; RTC GPIO so a key press can wake the phone
#define PIN_MODE_SW   38   // TATTLE/PLAY slide switch: LOW = PLAY (stay reachable), HIGH = TATTLE only

// Backup serial on test pads: GPIO43 (TX) / GPIO44 (RX)
