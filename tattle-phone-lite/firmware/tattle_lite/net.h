// Wi-Fi, clock, and the tattle outbox (WAV files on the flash FAT partition that wait
// there until they've been uploaded to the Google Apps Script receiver).
#pragma once
#include <Arduino.h>

namespace net {

using Tick = void (*)();          // called every ~50 ms while waiting (LED animations)

bool begin();                     // mount the FAT partition, create /outbox
void wifiStart();                 // start connecting (returns immediately)
bool wifiConnected();
bool wifiWait(uint32_t ms, Tick tick);   // wait for the connection, up to ms
void wifiStop();
bool timeValid();                 // clock set (from NTP; survives deep sleep)

bool saveWav(const uint8_t *wav, size_t len);   // into /outbox, named PHONE_NAME + time
int outboxCount();
int flushOutbox(Tick tick);       // upload everything waiting; returns how many remain

}  // namespace net
