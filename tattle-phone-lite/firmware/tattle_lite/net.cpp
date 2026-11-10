#include "net.h"
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <FFat.h>
#include <time.h>
#include <vector>
#include "mbedtls/base64.h"
#include "config.h"

namespace net {

static bool timeStarted = false;

bool begin() {
  if (!FFat.begin(true)) return false;   // formats on first boot
  FFat.mkdir("/outbox");
  return true;
}

void wifiStart() {
  if (WiFi.status() == WL_CONNECTED) return;
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(true);                   // modem sleep between beacons: saves a lot of mA
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
}

bool wifiConnected() {
  bool up = WiFi.status() == WL_CONNECTED;
  if (up && !timeStarted) {              // first time we're online: start NTP in the background
    configTzTime(TZ_INFO, "pool.ntp.org", "time.google.com");
    timeStarted = true;
  }
  return up;
}

bool wifiWait(uint32_t ms, Tick tick) {
  wifiStart();
  uint32_t t0 = millis();
  while (!wifiConnected() && millis() - t0 < ms) {
    if (tick) tick();
    delay(50);
  }
  return wifiConnected();
}

void wifiStop() {
  WiFi.disconnect(true);
  WiFi.mode(WIFI_OFF);
}

bool timeValid() { return time(nullptr) > 1700000000; }

// ------------------------------------------------------------------ outbox
static String makeName() {
  char name[64];
  time_t t = time(nullptr);
  if (timeValid()) {
    struct tm tm;
    localtime_r(&t, &tm);
    char ts[24];
    strftime(ts, sizeof(ts), "%Y-%m-%d_%H-%M-%S", &tm);
    snprintf(name, sizeof(name), "/outbox/%s_%s.wav", PHONE_NAME, ts);
  } else {
    snprintf(name, sizeof(name), "/outbox/%s_notime_%08lx.wav", PHONE_NAME, (unsigned long)esp_random());
  }
  return String(name);
}

bool saveWav(const uint8_t *wav, size_t len) {
  String path = makeName();
  File f = FFat.open(path, FILE_WRITE);
  if (!f) return false;
  size_t w = f.write(wav, len);
  f.close();
  Serial.printf("saved %s (%u bytes, %u KB free)\n", path.c_str(), (unsigned)len,
                (unsigned)((FFat.totalBytes() - FFat.usedBytes()) / 1024));
  if (w != len) FFat.remove(path);       // disk full: don't keep a broken file
  return w == len;
}

int outboxCount() {
  int c = 0;
  File dir = FFat.open("/outbox");
  for (File f = dir.openNextFile(); f; f = dir.openNextFile()) c++;
  return c;
}

// Streams a file as base64 without loading it: a 40 s tattle is 1.3 MB, its base64
// 1.7 MB, and the N8R2 module has only 2 MB of PSRAM. 768 bytes in = 1024 chars out,
// so every chunk but the last encodes without padding.
class B64FileStream : public Stream {
 public:
  explicit B64FileStream(File &f) : f_(f) {}
  int available() override { return fill() ? (int)(len_ - pos_) : 0; }
  int read() override { return fill() ? out_[pos_++] : -1; }
  int peek() override { return fill() ? out_[pos_] : -1; }
  size_t readBytes(char *buf, size_t n) override {
    size_t got = 0;
    while (got < n && fill()) {
      size_t k = min(n - got, len_ - pos_);
      memcpy(buf + got, out_ + pos_, k);
      pos_ += k;
      got += k;
    }
    return got;
  }
  size_t write(uint8_t) override { return 0; }
  void flush() override {}

 private:
  bool fill() {
    if (pos_ < len_) return true;
    uint8_t in[768];
    size_t n = f_.read(in, sizeof(in));
    if (!n) return false;
    size_t olen = 0;
    if (mbedtls_base64_encode(out_, sizeof(out_), &olen, in, n) != 0) return false;
    len_ = olen;
    pos_ = 0;
    return olen > 0;
  }
  File &f_;
  uint8_t out_[1025];    // 1024 chars + mbedtls's terminating NUL
  size_t len_ = 0, pos_ = 0;
};

// Apps Script answers a POST with a 302 to script.googleusercontent.com, where the
// script's reply waits. Follow it by hand, and count the upload as done only if the
// script says {"ok":true}.
static bool uploadFile(const String &path) {
  File f = FFat.open(path);
  if (!f) return false;
  size_t len = f.size();
  size_t b64Len = 4 * ((len + 2) / 3);
  String name = path.substring(path.lastIndexOf('/') + 1);
  String url = String(UPLOAD_URL) + "?token=" + UPLOAD_TOKEN + "&name=" + name;

  WiFiClientSecure tls;
  tls.setInsecure();                     // fine for a toy; see README to pin Google's CA
  HTTPClient http;
  http.setTimeout(30000);
  http.setFollowRedirects(HTTPC_DISABLE_FOLLOW_REDIRECTS);
  const char *keep[] = {"Location"};
  http.collectHeaders(keep, 1);
  http.begin(tls, url);
  http.addHeader("Content-Type", "text/plain");
  B64FileStream body64(f);
  int code = http.sendRequest("POST", &body64, b64Len);
  String body = http.getString();
  String loc = http.header("Location");
  http.end();
  f.close();
  Serial.printf("POST %s (%u bytes) -> %d\n", name.c_str(), (unsigned)len, code);

  if (code == 302 && loc.indexOf("script.googleusercontent.com") >= 0) {
    HTTPClient get;
    get.setTimeout(15000);
    get.begin(tls, loc);
    code = get.GET();
    body = get.getString();
    get.end();
  }
  bool ok = (code == 200 && body.indexOf("\"ok\":true") >= 0);
  if (!ok) Serial.printf("upload failed: %d %s\n", code, body.substring(0, 200).c_str());
  return ok;
}

int flushOutbox(Tick tick) {
  if (outboxCount() == 0) return 0;
  if (!wifiWait(15000, tick)) {
    Serial.println("no wifi; tattles stay in the outbox");
    return outboxCount();
  }
  std::vector<String> paths;
  File dir = FFat.open("/outbox");
  for (File f = dir.openNextFile(); f; f = dir.openNextFile())
    paths.push_back(String("/outbox/") + f.name());
  dir.close();
  for (auto &p : paths) {
    if (tick) tick();
    if (uploadFile(p)) FFat.remove(p);
  }
  return outboxCount();
}

}  // namespace net
