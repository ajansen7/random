// Phone-to-phone calls on the home Wi-Fi (UDP).
//
// Finding each other: no fixed IPs. RING goes out as a broadcast and carries both
// phone names; once a phone hears from its friend it remembers the friend's address
// and talks to it directly. Every packet carries a hash of INTERCOM_KEY, so a
// neighbour's phones (or anything else on the LAN) are ignored.
//
// Audio: 20 ms frames of 16 kHz 16-bit PCM (640 bytes, ~260 kbit/s), half duplex:
// the phone whose green button is held talks, the other listens. A short jitter
// buffer (~80 ms) smooths Wi-Fi hiccups.
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
void sendAudio(const int16_t *pcm, size_t n);
void startPlayback(uint8_t volumePct);   // speaker task plays incoming audio
void stopPlayback();
void setTalking(bool talking);        // while we talk, incoming audio is dropped (no echo)
int32_t incomingLevel();              // RMS of the last frame played (for the LED)

}  // namespace intercom
