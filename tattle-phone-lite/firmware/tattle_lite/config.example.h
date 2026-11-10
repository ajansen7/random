// Copy this file to config.h (same folder) and fill it in. config.h is git-ignored,
// so your Wi-Fi password never gets committed. Each phone gets its own config.h.
#pragma once

// ---- Wi-Fi (the ESP32 is 2.4 GHz only) ----
#define WIFI_SSID      "your-network"
#define WIFI_PASSWORD  "your-password"

// ---- Tattle upload (cloud/drive_upload.gs, deployed as a web app) ----
#define UPLOAD_URL     "https://script.google.com/macros/s/XXXXXXXXXXXX/exec"
#define UPLOAD_TOKEN   "change-me-to-something-long-and-random"   // same as TOKEN in the script

// ---- Names: this phone, and the phone it calls ----
// Swap the two on the other phone. Max 15 characters, no spaces.
#define PHONE_NAME     "Kitchen"
#define FRIEND_NAME    "Bedroom"
// Shared secret for the two phones (same on both). Packets without it are ignored.
#define INTERCOM_KEY   "change-me-too"

// ---- Play hours: stay awake and reachable for calls (local time, 24 h clock) ----
// Outside these hours the phone deep-sleeps a few minutes after it was last used and
// only a button press wakes it. Set PLAY_START_HOUR to -1 to turn play hours off
// (best battery life: then a phone is only reachable for AWAKE_MINUTES after use).
// Staying awake on Wi-Fi costs roughly 20-30 mA: about 3 days of AAs at 12 h/day.
#define PLAY_START_HOUR  -1
#define PLAY_END_HOUR    -1
#define AWAKE_MINUTES     5

// ---- Calls ----
// Default: full duplex with Espressif's echo canceller. If a call echoes or howls and
// tuning doesn't fix it, uncomment this for walkie-talkie calls (hold green to talk).
// #define CALL_MODE_WALKIE
// Speaker-to-mic delay the echo canceller assumes, in ~16 ms chunks. Tune at bring-up
// (watch "call: mic=... after-aec=..." in the serial monitor): try 1, 2, 3.
// #define CALL_ECHO_DELAY_CHUNKS 2
// If the mic records only silence, uncomment (selects the PDM mic's other data slot):
// #define MIC_RIGHT_SLOT

// ---- Time zone for file names and play hours (POSIX TZ string) ----
// US Eastern: "EST5EDT,M3.2.0,M11.1.0"   US Central: "CST6CDT,M3.2.0,M11.1.0"
// US Mountain: "MST7MDT,M3.2.0,M11.1.0"  US Pacific: "PST8PDT,M3.2.0,M11.1.0"
#define TZ_INFO        "MST7MDT,M3.2.0,M11.1.0"
