#include "lights.h"
#include "pins.h"

namespace lights {

static const uint8_t PINS[3] = {PIN_LED_YELLOW, PIN_LED_GREEN, PIN_LED_RED};

void begin() {
  for (uint8_t p : PINS) {
    ledcAttach(p, 5000, 8);   // 5 kHz, 8-bit
    ledcWrite(p, 0);
  }
}

void set(Color c, uint8_t level) {
  uint32_t g = (uint32_t)level * level / 255;   // cheap gamma (^2) so fades look even
  if (level && !g) g = 1;
  ledcWrite(PINS[c], g);
}

void all(uint8_t y, uint8_t g, uint8_t r) {
  set(YELLOW, y);
  set(GREEN, g);
  set(RED, r);
}

void off() { all(0, 0, 0); }

void breathe(Color c, uint8_t lo, uint8_t hi, uint32_t periodMs) {
  float ph = (millis() % periodMs) / (float)periodMs;
  float s = 0.5f - 0.5f * cosf(ph * 2 * PI);
  set(c, lo + (uint8_t)((hi - lo) * s));
}

void blink(Color c, uint8_t level, uint32_t onMs, uint32_t offMs) {
  set(c, (millis() % (onMs + offMs)) < onMs ? level : 0);
}

void flash(Color c, int n, uint32_t onMs, uint32_t offMs) {
  for (int i = 0; i < n; i++) {
    set(c, 255);
    delay(onMs);
    set(c, 0);
    delay(offMs);
  }
}

void sparkle() {
  for (int k = 0; k < 3; k++)
    for (int c = 0; c < 3; c++) {
      off();
      set((Color)c, 255);
      delay(90);
    }
  off();
}

}  // namespace lights
