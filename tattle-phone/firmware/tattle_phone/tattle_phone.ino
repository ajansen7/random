// Tattle Phone — phase 1 firmware (ESP32-S3, Arduino core 3.x)
//
// Flow: press CALL -> "ring ring" -> beep -> record (auto-stops on silence,
// a second press, or MAX_RECORD_S) -> "goodbye" tones -> save WAV to the
// on-board flash outbox -> Wi-Fi up -> upload every file in the outbox to the
// Google Apps Script endpoint -> delete what uploaded -> deep sleep.
// Anything that fails to upload stays in the outbox and is retried next wake.
//
// Arduino IDE settings:
//   Board: "ESP32S3 Dev Module"   USB CDC On Boot: Enabled
//   Flash Size: 16MB              PSRAM: OPI PSRAM
//   Partition Scheme: "16M Flash (3MB APP/9.9MB FATFS)"
// Library: Adafruit NeoPixel (Library Manager)

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <FFat.h>
#include <Adafruit_NeoPixel.h>
#include <time.h>
#include <vector>
#include "driver/i2s_std.h"
#include "mbedtls/base64.h"
#include "esp_sleep.h"

#include "pins.h"
#if __has_include("config.h")
#include "config.h"
#else
#error "Copy config.example.h to config.h and fill in your Wi-Fi + upload settings"
#endif

// ---------------- Tunables ----------------
static const uint32_t SAMPLE_RATE      = 16000;  // 16 kHz mono = 32 KB/s, plenty for speech
static const uint32_t MAX_RECORD_S     = 60;     // hard cap per tattle
static const uint32_t MIN_RECORD_MS    = 1500;   // ignore 2nd press before this
static const uint32_t SILENCE_STOP_MS  = 3000;   // stop after this much quiet once talking began
static const uint32_t NO_SPEECH_MS     = 10000;  // give up if nobody talks at all
static const int32_t  SPEECH_RMS       = 700;    // tune with the serial monitor ("rms=")
static const int      MIC_SHIFT        = 13;     // 32-bit mic word -> 16-bit; smaller = louder
static const uint32_t IDLE_SLEEP_MS    = 45000;  // go to deep sleep after this much idle
static const uint8_t  VOLUME_PCT       = 35;     // tone volume, toddler ears!

// ---------------- Globals ----------------
static i2s_chan_handle_t micChan = nullptr;
static i2s_chan_handle_t ampChan = nullptr;
static Adafruit_NeoPixel leds(NUM_LEDS, PIN_LEDS, NEO_GRB + NEO_KHZ800);
static uint32_t lastActivity = 0;
static bool timeSynced = false;

// =================================================================
// LEDs
// =================================================================
static void ledsPower(bool on) {
  pinMode(PIN_LED_PWR, OUTPUT);
  digitalWrite(PIN_LED_PWR, on ? HIGH : LOW);
  if (on) { delay(2); leds.begin(); leds.setBrightness(60); }
}

static void setAll(uint32_t c) { leds.fill(c); leds.show(); }
static void setCall(uint32_t c) { leds.setPixelColor(0, c); leds.show(); }

static void chase(uint32_t c, int step) {   // keypad "busy" animation
  leds.clear();
  leds.setPixelColor(0, c);
  if (NUM_LEDS > 1) leds.setPixelColor(1 + (step % (NUM_LEDS - 1)), c);
  leds.show();
}

static void rainbowFlash() {
  for (int j = 0; j < 256; j += 8) {
    for (int i = 0; i < NUM_LEDS; i++)
      leds.setPixelColor(i, leds.gamma32(leds.ColorHSV((i * 65536 / NUM_LEDS) + j * 256)));
    leds.show();
    delay(15);
  }
}

// =================================================================
// Audio: I2S setup
// =================================================================
static void audioInit() {
  // --- Microphone: I2S0, RX, 32-bit slots (INMP441 sends 24-bit left-justified)
  i2s_chan_config_t rxc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  ESP_ERROR_CHECK(i2s_new_channel(&rxc, nullptr, &micChan));
  i2s_std_config_t rx = {};
  rx.clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE);
  rx.slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO);
  rx.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;   // mic L/R pin tied to GND
  rx.gpio_cfg.mclk = I2S_GPIO_UNUSED;
  rx.gpio_cfg.bclk = (gpio_num_t)PIN_MIC_BCLK;
  rx.gpio_cfg.ws   = (gpio_num_t)PIN_MIC_WS;
  rx.gpio_cfg.dout = I2S_GPIO_UNUSED;
  rx.gpio_cfg.din  = (gpio_num_t)PIN_MIC_SD;
  ESP_ERROR_CHECK(i2s_channel_init_std_mode(micChan, &rx));

  // --- Amplifier: I2S1, TX, 16-bit mono duplicated to both slots
  i2s_chan_config_t txc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_1, I2S_ROLE_MASTER);
  txc.auto_clear = true;  // send silence on underrun instead of repeating garbage
  ESP_ERROR_CHECK(i2s_new_channel(&txc, &ampChan, nullptr));
  i2s_std_config_t tx = {};
  tx.clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE);
  tx.slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO);
  tx.slot_cfg.slot_mask = I2S_STD_SLOT_BOTH;
  tx.gpio_cfg.mclk = I2S_GPIO_UNUSED;
  tx.gpio_cfg.bclk = (gpio_num_t)PIN_AMP_BCLK;
  tx.gpio_cfg.ws   = (gpio_num_t)PIN_AMP_LRC;
  tx.gpio_cfg.dout = (gpio_num_t)PIN_AMP_DIN;
  tx.gpio_cfg.din  = I2S_GPIO_UNUSED;
  ESP_ERROR_CHECK(i2s_channel_init_std_mode(ampChan, &tx));
  ESP_ERROR_CHECK(i2s_channel_enable(ampChan));

  pinMode(PIN_AMP_SD, OUTPUT);
  digitalWrite(PIN_AMP_SD, LOW);  // amp off until we need it
}

// =================================================================
// Audio: tones (also what the keypad will use in phase 2)
// =================================================================
static void playTones(float f1, float f2, uint32_t ms) {
  digitalWrite(PIN_AMP_SD, HIGH);
  const int16_t amp = (int16_t)(32767 * VOLUME_PCT / 100 / (f2 > 0 ? 2 : 1));
  const uint32_t total = SAMPLE_RATE * ms / 1000;
  const uint32_t fade = SAMPLE_RATE / 200;  // 5 ms ramps -> no clicks
  int16_t buf[256];
  float p1 = 0, p2 = 0;
  const float d1 = 2 * PI * f1 / SAMPLE_RATE, d2 = 2 * PI * f2 / SAMPLE_RATE;
  for (uint32_t n = 0; n < total;) {
    size_t cnt = min<uint32_t>(256, total - n);
    for (size_t i = 0; i < cnt; i++, n++) {
      float env = 1.0f;
      if (n < fade) env = (float)n / fade;
      else if (total - n < fade) env = (float)(total - n) / fade;
      float s = sinf(p1) + (f2 > 0 ? sinf(p2) : 0);
      buf[i] = (int16_t)(s * amp * env);
      p1 += d1; if (p1 > 2 * PI) p1 -= 2 * PI;
      p2 += d2; if (p2 > 2 * PI) p2 -= 2 * PI;
    }
    size_t w;
    i2s_channel_write(ampChan, buf, cnt * sizeof(int16_t), &w, portMAX_DELAY);
  }
}

static void quiet(uint32_t ms) { playTones(0, 0, ms); }  // sin(0)=0 -> silence
static void ampOff() { quiet(30); digitalWrite(PIN_AMP_SD, LOW); }

static void soundRing() {                // classic two-tone ring, twice
  for (int r = 0; r < 2; r++) {
    for (int k = 0; k < 8; k++) { setCall(k % 2 ? 0 : leds.Color(255, 255, 255)); playTones(440, 480, 100); }
    setCall(0);
    quiet(400);
  }
}
static void soundBeep()    { playTones(1000, 0, 250); ampOff(); }
static void soundGoodbye() { playTones(784, 0, 150); playTones(659, 0, 150); playTones(523, 0, 300); ampOff(); }
static void soundSent()    { playTones(523, 0, 90); playTones(659, 0, 90); playTones(784, 0, 90); playTones(1047, 0, 200); ampOff(); }
static void soundError()   { playTones(200, 0, 400); ampOff(); }

// =================================================================
// Recording
// =================================================================
static void writeWavHeader(uint8_t *h, uint32_t dataBytes) {
  auto u32 = [&](int o, uint32_t v) { h[o] = v; h[o+1] = v >> 8; h[o+2] = v >> 16; h[o+3] = v >> 24; };
  auto u16 = [&](int o, uint16_t v) { h[o] = v; h[o+1] = v >> 8; };
  memcpy(h, "RIFF", 4);      u32(4, 36 + dataBytes);
  memcpy(h + 8, "WAVEfmt ", 8); u32(16, 16); u16(20, 1); u16(22, 1);
  u32(24, SAMPLE_RATE);      u32(28, SAMPLE_RATE * 2); u16(32, 2); u16(34, 16);
  memcpy(h + 36, "data", 4); u32(40, dataBytes);
}

static bool callPressed() { return digitalRead(PIN_CALL_BTN) == LOW; }

// Records into PSRAM. Returns WAV buffer (caller frees) and its size, or nullptr.
static uint8_t *recordTattle(size_t *outLen) {
  const size_t maxSamples = SAMPLE_RATE * MAX_RECORD_S;
  uint8_t *wav = (uint8_t *)ps_malloc(44 + maxSamples * 2);
  if (!wav) { Serial.println("PSRAM alloc failed"); return nullptr; }
  int16_t *pcm = (int16_t *)(wav + 44);

  const size_t CHUNK = SAMPLE_RATE / 10;   // 100 ms
  static int32_t raw[SAMPLE_RATE / 10];
  size_t n = 0;
  bool heardSpeech = false, wasPressed = true;
  uint32_t start = millis(), lastLoud = millis();
  int32_t dc = 0;

  i2s_channel_enable(micChan);
  // Throw away the first ~200 ms: MEMS mics need a moment to settle.
  for (int i = 0; i < 2; i++) { size_t r; i2s_channel_read(micChan, raw, sizeof(raw), &r, portMAX_DELAY); }

  setCall(leds.Color(255, 0, 0));
  while (n < maxSamples) {
    size_t got = 0;
    i2s_channel_read(micChan, raw, sizeof(raw), &got, portMAX_DELAY);
    size_t cnt = min(got / 4, maxSamples - n);
    int64_t sumSq = 0;
    for (size_t i = 0; i < cnt; i++) {
      int32_t s = raw[i] >> MIC_SHIFT;
      dc += (s - dc) >> 6;                 // cheap DC-blocking high-pass
      s -= dc;
      s = constrain(s, -32768, 32767);
      pcm[n++] = (int16_t)s;
      sumSq += (int64_t)s * s;
    }
    int32_t rms = cnt ? (int32_t)sqrtf((float)sumSq / cnt) : 0;
    uint32_t now = millis();
    if (rms > SPEECH_RMS) { heardSpeech = true; lastLoud = now; }

    // Pulse the call LED with the voice level — kids love seeing it react.
    uint8_t lvl = (uint8_t)min<int32_t>(255, 40 + rms / 8);
    setCall(leds.Color(lvl, 0, 0));
    Serial.printf("rms=%ld\n", (long)rms);

    bool pressed = callPressed();
    bool newPress = pressed && !wasPressed;
    wasPressed = pressed;
    if (newPress && now - start > MIN_RECORD_MS) break;                 // hang up
    if (heardSpeech && now - lastLoud > SILENCE_STOP_MS) break;         // done talking
    if (!heardSpeech && now - start > NO_SPEECH_MS) { n = 0; break; }   // nobody spoke
  }
  i2s_channel_disable(micChan);

  if (n < SAMPLE_RATE) { free(wav); return nullptr; }  // < 1 s: not a tattle
  writeWavHeader(wav, n * 2);
  *outLen = 44 + n * 2;
  return wav;
}

// =================================================================
// Outbox (on-board flash, FAT partition)
// =================================================================
static String makeName() {
  char name[48];
  time_t t = time(nullptr);
  if (t > 1700000000) {                       // clock is set (survives deep sleep)
    struct tm tm; localtime_r(&t, &tm);
    strftime(name, sizeof(name), "/outbox/tattle_%Y-%m-%d_%H-%M-%S.wav", &tm);
  } else {
    snprintf(name, sizeof(name), "/outbox/tattle_notime_%08lx.wav", (unsigned long)esp_random());
  }
  return String(name);
}

static bool saveToOutbox(const uint8_t *data, size_t len) {
  String path = makeName();
  File f = FFat.open(path, FILE_WRITE);
  if (!f) return false;
  size_t w = f.write(data, len);
  f.close();
  Serial.printf("saved %s (%u bytes)\n", path.c_str(), (unsigned)len);
  return w == len;
}

static int outboxCount() {
  int c = 0;
  File dir = FFat.open("/outbox");
  for (File f = dir.openNextFile(); f; f = dir.openNextFile()) c++;
  return c;
}

// =================================================================
// Wi-Fi + upload
// =================================================================
static bool wifiUp() {
  if (WiFi.status() == WL_CONNECTED) return true;
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  for (int i = 0; i < 150 && WiFi.status() != WL_CONNECTED; i++) {
    chase(leds.Color(0, 0, 255), i);
    delay(100);
  }
  if (WiFi.status() != WL_CONNECTED) return false;
  if (!timeSynced) {
    configTzTime(TZ_INFO, "pool.ntp.org", "time.google.com");
    struct tm tm;
    timeSynced = getLocalTime(&tm, 5000);
  }
  return true;
}

static void wifiDown() { WiFi.disconnect(true); WiFi.mode(WIFI_OFF); }

// Apps Script web apps answer a POST with a 302 to script.googleusercontent.com,
// where the script's actual reply waits. We follow it by hand with a GET and
// only count the upload as done if the script says {"ok":true}.
static bool uploadFile(const String &path) {
  File f = FFat.open(path);
  if (!f) return false;
  size_t len = f.size();
  uint8_t *bin = (uint8_t *)ps_malloc(len);
  size_t b64Cap = 4 * ((len + 2) / 3) + 1, b64Len = 0;
  uint8_t *b64 = (uint8_t *)ps_malloc(b64Cap);
  bool ok = false;
  if (bin && b64 && f.read(bin, len) == len &&
      mbedtls_base64_encode(b64, b64Cap, &b64Len, bin, len) == 0) {
    free(bin); bin = nullptr;

    String name = path.substring(path.lastIndexOf('/') + 1);
    String url = String(UPLOAD_URL) + "?token=" + UPLOAD_TOKEN + "&name=" + name;

    WiFiClientSecure tls;
    tls.setInsecure();  // OK for a toy; see README to pin Google's root CA
    HTTPClient http;
    http.setTimeout(30000);
    http.setFollowRedirects(HTTPC_DISABLE_FOLLOW_REDIRECTS);
    const char *keep[] = {"Location"};
    http.collectHeaders(keep, 1);
    http.begin(tls, url);
    http.addHeader("Content-Type", "text/plain");
    int code = http.POST(b64, b64Len);
    String body = http.getString();
    String loc = http.header("Location");
    http.end();
    Serial.printf("POST %s -> %d\n", name.c_str(), code);

    if (code == 302 && loc.indexOf("script.googleusercontent.com") >= 0) {
      HTTPClient get;
      get.setTimeout(15000);
      get.begin(tls, loc);
      code = get.GET();
      body = get.getString();
      get.end();
    }
    ok = (code == 200 && body.indexOf("\"ok\":true") >= 0);
    if (!ok) Serial.printf("upload failed: %d %s\n", code, body.substring(0, 200).c_str());
  }
  f.close();
  free(bin); free(b64);
  return ok;
}

// Upload everything waiting. Returns number still pending.
static int flushOutbox() {
  if (outboxCount() == 0) return 0;
  if (!wifiUp()) { Serial.println("no wifi"); return outboxCount(); }
  std::vector<String> paths;
  File dir = FFat.open("/outbox");
  for (File f = dir.openNextFile(); f; f = dir.openNextFile())
    paths.push_back(String("/outbox/") + f.name());
  dir.close();
  int i = 0;
  for (auto &p : paths) {
    chase(leds.Color(0, 0, 255), i++);
    if (uploadFile(p)) FFat.remove(p);
  }
  wifiDown();
  return outboxCount();
}

// =================================================================
// The call
// =================================================================
static void doCall() {
  soundRing();
  soundBeep();
  size_t len = 0;
  uint8_t *wav = recordTattle(&len);
  if (!wav) { setCall(leds.Color(255, 120, 0)); soundError(); return; }
  setCall(leds.Color(0, 80, 0));
  soundGoodbye();
  bool saved = saveToOutbox(wav, len);
  free(wav);
  if (!saved) { soundError(); return; }

  int pending = flushOutbox();
  if (pending == 0) { rainbowFlash(); soundSent(); }
  else { setCall(leds.Color(255, 120, 0)); soundError(); }  // stays queued for next time
  setAll(0);
}

static void goToSleep() {
  Serial.println("sleeping");
  setAll(0);
  ledsPower(false);
  digitalWrite(PIN_AMP_SD, LOW);
  wifiDown();
  esp_sleep_enable_ext0_wakeup((gpio_num_t)PIN_CALL_BTN, 0);  // wake on press (LOW)
  esp_deep_sleep_start();
}

// =================================================================
void setup() {
  Serial.begin(115200);
  pinMode(PIN_CALL_BTN, INPUT_PULLUP);
  ledsPower(true);
  setAll(0);
  audioInit();

  if (!FFat.begin(true)) { Serial.println("FFat mount failed"); soundError(); }
  FFat.mkdir("/outbox");

  bool wokeByButton = esp_sleep_get_wakeup_cause() == ESP_SLEEP_WAKEUP_EXT0;
  if (wokeByButton) doCall();
  else flushOutbox();         // cold boot / reset: retry anything left over
  lastActivity = millis();
}

void loop() {
  if (callPressed()) {
    delay(30);                          // debounce
    if (callPressed()) {
      while (callPressed()) delay(10);  // act on release so the ring isn't cut short
      doCall();
      lastActivity = millis();
    }
  }
  // Gentle "breathing" green on the call button while awake.
  uint8_t b = (uint8_t)(20 + 20 * sinf(millis() / 600.0f));
  setCall(leds.Color(0, b, 0));

  if (millis() - lastActivity > IDLE_SLEEP_MS) goToSleep();
  delay(20);
}
