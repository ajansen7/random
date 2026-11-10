// Phone-to-phone calls on the home Wi-Fi (UDP).
//
// Finding each other: no fixed IPs. RING goes out as a broadcast and carries both
// phone names; once a phone hears from its friend it remembers the friend's address
// and talks to it directly. Every packet carries a hash of INTERCOM_KEY, so a
// neighbour's phones (or anything else on the LAN) are ignored.
//
// Audio: 16 kHz 16-bit PCM in small chunks (the echo canceller's chunk size),
// ~260 kbit/s each way. Incoming audio waits in a short jitter buffer (~80 ms) that
// the call loop pulls from, one chunk per step, in lockstep with the microphone.
#pragma once
#include <Arduino.h>

namespace intercom {

enum Msg : uint8_t { NONE = 0, RING, RINGACK, ANSWER, HANGUP, BUSY, PING, AUDIO };

void begin();                         // start listening (Wi-Fi must be up)
void end();
bool friendKnown();                   // heard from the friend at least once this boot
void send(Msg m);
Msg poll();                           // next control message from the friend, or NONE
uint32_t msSinceHeard();              // since the last packet of any kind from the friend

// ---- during a call ----
void callStart();                     // start accepting audio (clears the jitter buffer)
void callEnd();
void sendAudio(const int16_t *pcm, size_t n);
// Next n samples of the friend's voice; silence while the jitter buffer (re)fills.
// Returns the RMS level of what it returned (for the LED).
int32_t readAudio(int16_t *out, size_t n);
void dropIncoming(bool drop);         // walkie mode: ignore incoming audio while we talk

}  // namespace intercom
