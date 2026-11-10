// The three button lights (each is two LEDs on one GPIO), dimmable with PWM.
#pragma once
#include <Arduino.h>

namespace lights {

enum Color : uint8_t { YELLOW, GREEN, RED };

void begin();
void set(Color c, uint8_t level);          // 0..255 (gamma-corrected)
void all(uint8_t y, uint8_t g, uint8_t r);
void off();
// Smooth breathing between lo and hi with the given period, based on millis().
void breathe(Color c, uint8_t lo, uint8_t hi, uint32_t periodMs);
// On/off blinking based on millis(): `on` ms lit, `off` ms dark.
void blink(Color c, uint8_t level, uint32_t onMs, uint32_t offMs);
// Blocking: flash a light n times (battery level, errors).
void flash(Color c, int n, uint32_t onMs = 150, uint32_t offMs = 200);
// Blocking: a quick yellow -> green -> red chase ("sent!").
void sparkle();

}  // namespace lights
