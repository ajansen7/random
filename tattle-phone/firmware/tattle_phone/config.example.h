// Copy this file to config.h (same folder) and fill in your values.
// config.h is git-ignored so your Wi-Fi password never gets committed.
#pragma once

// ---- Wi-Fi (ESP32 is 2.4 GHz only) ----
#define WIFI_SSID     "your-network"
#define WIFI_PASSWORD "your-password"

// ---- Upload endpoint ----
// The /exec URL you get after deploying cloud/drive_upload.gs as a web app.
#define UPLOAD_URL   "https://script.google.com/macros/s/XXXXXXXXXXXX/exec"
// Must match TOKEN in drive_upload.gs. Any long random string.
#define UPLOAD_TOKEN "change-me-to-something-long-and-random"

// ---- Time zone for file names (POSIX TZ string) ----
// US Eastern: "EST5EDT,M3.2.0,M11.1.0"   US Pacific: "PST8PDT,M3.2.0,M11.1.0"
#define TZ_INFO "CST6CDT,M3.2.0,M11.1.0"
