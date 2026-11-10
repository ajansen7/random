// Audio: PDM microphone in (I2S0, hardware PDM->PCM), MAX98357A speaker out (I2S1).
// Both run at 16 kHz, 16-bit mono.
#pragma once
#include <Arduino.h>

namespace audio {

constexpr uint32_t RATE = 16000;
constexpr size_t FRAME = RATE / 50;      // 20 ms = 320 samples: recording read size
constexpr size_t MAX_CHUNK = 512;        // largest call chunk (the echo canceller picks it)

void begin();

// ---- speaker ----
void ampOn();
void ampOff();                            // also plays a few ms of silence first (no pop)
// Blocking write of PCM samples to the speaker (thread-safe).
void play(const int16_t *pcm, size_t n);
// One or two sine tones (f2 = 0 for one), with 5 ms fades so nothing clicks.
void tone(float f1, float f2, uint32_t ms, uint8_t volumePct);
void silence(uint32_t ms);

// ---- microphone ----
void micStart();                          // also discards the first ~100 ms (settling)
void micStop();
// Read up to n samples (blocking up to timeoutMs). DC-blocked and scaled by `gain`.
size_t micRead(int16_t *out, size_t n, uint32_t timeoutMs, int gain);

// RMS of a block of samples (speech detection, LED level meter).
int32_t rms(const int16_t *pcm, size_t n);

}  // namespace audio
