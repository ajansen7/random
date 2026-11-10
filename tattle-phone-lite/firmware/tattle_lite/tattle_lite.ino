// Tattle Phone Lite firmware (ESP32-S3-WROOM-1-N8R2, Arduino-ESP32 core 3.x)
//
// Three buttons:
//   YELLOW STAR  (TATTLE)   beep -> record what the toddler says -> send it to the
//                           parent (Google Drive + email via cloud/drive_upload.gs).
//                           Recording stops on silence, on a press of STAR or RED, or
//                           at MAX_RECORD_S. If Wi-Fi is down the WAV waits in the
//                           flash outbox and is retried later.
//   GREEN CIRCLE (CALL)     ring the other phone (FRIEND_NAME). When it rings here,
//                           green answers. During a call: HOLD green to talk, let go
//                           to listen (walkie-talkie style; no echo, no DSP needed).
//   RED SQUARE   (HANG UP)  ends a call or declines a ringing one. When idle, red
//                           shows the battery level and puts the phone to sleep.
//
// Power: deep sleep (microamps) until a button is pressed. After use the phone
// stays awake on Wi-Fi for AWAKE_MINUTES so the other phone can call back, or for
// the whole of PLAY_START_HOUR..PLAY_END_HOUR if those are set (config.h).
//
// Build: see ../README.md (arduino-cli, or Arduino IDE: "ESP32S3 Dev Module",
// Flash 8MB, PSRAM "QSPI PSRAM", USB CDC On Boot enabled; partitions.csv in this
// folder is picked up automatically).

#include <WiFi.h>
#include <time.h>
#include "esp_sleep.h"
#include "driver/rtc_io.h"

#include "pins.h"
#if __has_include("config.h")
#include "config.h"
#else
#error "Copy config.example.h to config.h and fill in your Wi-Fi, upload and phone names"
#endif
#include "audio.h"
#include "lights.h"
#include "net.h"
#include "intercom.h"

// ---------------- Tunables ----------------
static const uint32_t MAX_RECORD_S    = 40;     // per tattle (PSRAM holds ~60 s; outbox ~3 x 40 s)
static const uint32_t MIN_RECORD_MS   = 1500;   // ignore a stop press before this
static const uint32_t SILENCE_STOP_MS = 3000;   // stop after this much quiet once talking began
static const uint32_t NO_SPEECH_MS    = 10000;  // give up if nobody talks at all
static const int32_t  SPEECH_RMS      = 600;    // tune with the serial monitor ("rms=")
static const int      MIC_GAIN        = 4;      // PDM mic -> recording level
static const uint8_t  TONE_VOLUME     = 30;     // % : toddler ears!
static const uint8_t  RING_VOLUME     = 45;
static const uint8_t  CALL_VOLUME     = 80;     // % of the far end's voice
static const uint32_t RING_TIMEOUT_MS = 30000;  // unanswered call gives up
static const uint32_t FIND_FRIEND_MS  = 5000;   // no RINGACK by then = the other phone is asleep
static const uint32_t CALL_DROP_MS    = 15000;  // nothing heard from the friend = call dropped
static const float    VBAT_LOW        = 3.45f;  // ~1.15 V/cell: "change batteries soon"
static const float    VBAT_EMPTY      = 3.20f;  // the 3.3 V rail is about to sag

// ---------------- Buttons ----------------
struct Button {
  uint8_t pin;
  bool down = false, raw = false, edge = false;
  uint32_t changed = 0;
  void begin() { pinMode(pin, INPUT_PULLUP); }
  void update() {
    bool r = digitalRead(pin) == LOW;
    if (r != raw) { raw = r; changed = millis(); }
    edge = false;
    if (millis() - changed > 25 && raw != down) {
      down = raw;
      edge = down;   // edge = newly pressed
    }
  }
};
static Button btnTattle{PIN_BTN_TATTLE}, btnCall{PIN_BTN_CALL}, btnHangup{PIN_BTN_HANGUP};
static void updateButtons() { btnTattle.update(); btnCall.update(); btnHangup.update(); }
static void waitReleased(Button &b) { while (digitalRead(b.pin) == LOW) delay(10); b.update(); }

static uint32_t lastActivity = 0;
static void touched() { lastActivity = millis(); }

// ---------------- Battery ----------------
static float batteryVolts() {
  uint32_t mv = 0;
  for (int i = 0; i < 8; i++) mv += analogReadMilliVolts(PIN_VBAT_ADC);
  return mv / 8.0f * 2.0f / 1000.0f;    // 1M/1M divider
}

// ---------------- Sounds ----------------
static void soundBeep()     { audio::tone(1000, 0, 250, TONE_VOLUME); audio::ampOff(); }
static void soundGoodbye()  { audio::tone(784, 0, 150, TONE_VOLUME); audio::tone(659, 0, 150, TONE_VOLUME);
                              audio::tone(523, 0, 300, TONE_VOLUME); audio::ampOff(); }
static void soundSent()     { audio::tone(523, 0, 90, TONE_VOLUME); audio::tone(659, 0, 90, TONE_VOLUME);
                              audio::tone(784, 0, 90, TONE_VOLUME); audio::tone(1047, 0, 200, TONE_VOLUME); audio::ampOff(); }
static void soundError()    { audio::tone(200, 0, 400, TONE_VOLUME); audio::ampOff(); }
static void soundSleepy()   { audio::tone(392, 0, 300, TONE_VOLUME); audio::tone(330, 0, 300, TONE_VOLUME);
                              audio::tone(262, 0, 500, TONE_VOLUME); audio::ampOff(); }   // "friend's asleep"
static void soundBusy()     { for (int i = 0; i < 3; i++) { audio::tone(480, 620, 250, TONE_VOLUME); audio::silence(250); } audio::ampOff(); }
static void soundLowBatt()  { audio::tone(330, 0, 120, TONE_VOLUME); audio::silence(80); audio::tone(330, 0, 120, TONE_VOLUME); audio::ampOff(); }
static void soundAnswered() { audio::tone(660, 0, 80, TONE_VOLUME); audio::tone(880, 0, 120, TONE_VOLUME); audio::ampOff(); }

// ---------------- Tattle ----------------
static void writeWavHeader(uint8_t *h, uint32_t dataBytes) {
  auto u32 = [&](int o, uint32_t v) { h[o] = v; h[o + 1] = v >> 8; h[o + 2] = v >> 16; h[o + 3] = v >> 24; };
  auto u16 = [&](int o, uint16_t v) { h[o] = v; h[o + 1] = v >> 8; };
  memcpy(h, "RIFF", 4);          u32(4, 36 + dataBytes);
  memcpy(h + 8, "WAVEfmt ", 8);  u32(16, 16); u16(20, 1); u16(22, 1);
  u32(24, audio::RATE);          u32(28, audio::RATE * 2); u16(32, 2); u16(34, 16);
  memcpy(h + 36, "data", 4);     u32(40, dataBytes);
}

// Records into PSRAM. Returns a WAV buffer (caller frees) and its size, or nullptr.
static uint8_t *recordTattle(size_t *outLen) {
  const size_t maxSamples = audio::RATE * MAX_RECORD_S;
  uint8_t *wav = (uint8_t *)ps_malloc(44 + maxSamples * 2);
  if (!wav) { Serial.println("PSRAM alloc failed"); return nullptr; }
  int16_t *pcm = (int16_t *)(wav + 44);
  size_t n = 0;
  bool heardSpeech = false;
  uint32_t start = millis(), lastLoud = millis();

  audio::micStart();
  while (n < maxSamples) {
    size_t got = audio::micRead(pcm + n, min<size_t>(audio::FRAME * 5, maxSamples - n), 200, MIC_GAIN);  // 100 ms
    int32_t level = audio::rms(pcm + n, got);
    n += got;
    uint32_t now = millis();
    if (level > SPEECH_RMS) { heardSpeech = true; lastLoud = now; }
    lights::set(lights::YELLOW, (uint8_t)min<int32_t>(255, 40 + level / 6));   // glows with the voice
    Serial.printf("rms=%ld\n", (long)level);

    updateButtons();
    if ((btnTattle.edge || btnHangup.edge) && now - start > MIN_RECORD_MS) break;   // done
    if (heardSpeech && now - lastLoud > SILENCE_STOP_MS) break;                     // stopped talking
    if (!heardSpeech && now - start > NO_SPEECH_MS) { n = 0; break; }               // nobody spoke
  }
  audio::micStop();
  lights::off();

  if (n < audio::RATE) { free(wav); return nullptr; }   // < 1 s: not a tattle
  writeWavHeader(wav, n * 2);
  *outLen = 44 + n * 2;
  return wav;
}

static void uploadTick() { lights::blink(lights::YELLOW, 160, 120, 120); }

static void doTattle() {
  touched();
  waitReleased(btnTattle);
  net::wifiStart();                       // connect while the child talks
  lights::set(lights::YELLOW, 255);
  soundBeep();
  size_t len = 0;
  uint8_t *wav = recordTattle(&len);
  if (!wav) { lights::flash(lights::YELLOW, 2); soundError(); touched(); return; }
  soundGoodbye();
  bool saved = net::saveWav(wav, len);
  free(wav);
  if (!saved) { soundError(); touched(); return; }
  int pending = net::flushOutbox(uploadTick);
  lights::off();
  if (pending == 0) { lights::sparkle(); soundSent(); }
  else { lights::flash(lights::YELLOW, 3, 300, 300); soundError(); }   // stays queued
  touched();
}

// ---------------- Calls ----------------
// Half duplex: hold green to talk. Mic frames go out while held; otherwise the speaker
// task plays what the friend says.
static void inCall() {
  Serial.println("in call");
  soundAnswered();
  intercom::startPlayback(CALL_VOLUME);
  audio::micStart();
  int16_t frame[audio::FRAME];
  uint32_t lastPing = 0;
  bool ended = false, dropped = false;
  while (!ended) {
    updateButtons();
    bool talk = btnCall.down;
    intercom::setTalking(talk);
    size_t got = audio::micRead(frame, audio::FRAME, 40, MIC_GAIN);   // also paces the loop (20 ms)
    if (talk && got) intercom::sendAudio(frame, got);

    if (talk) lights::all(0, 255, 0);
    else lights::all(0, (uint8_t)min<int32_t>(200, 25 + intercom::incomingLevel() / 20), 0);

    if (millis() - lastPing > 2000) { intercom::send(intercom::PING); lastPing = millis(); }
    switch (intercom::poll()) {
      case intercom::HANGUP: ended = true; break;
      case intercom::RING: intercom::send(intercom::BUSY); break;
      default: break;
    }
    if (btnHangup.edge) { intercom::send(intercom::HANGUP); ended = true; }
    if (intercom::msSinceHeard() > CALL_DROP_MS) { ended = true; dropped = true; }
  }
  intercom::setTalking(false);
  audio::micStop();
  intercom::stopPlayback();
  lights::off();
  if (dropped) soundError(); else soundGoodbye();
  touched();
}

// Ring pattern in 100 ms slices so buttons and messages are checked in between.
static void ringSlice(uint32_t t, bool incoming) {
  uint32_t ph = t % (incoming ? 3000 : 4000);
  bool on = incoming ? (ph < 1600 && (ph / 200) % 2 == 0) : ph < 1500;   // brrring-brrring / purr
  if (on) audio::tone(440, 480, 100, incoming ? RING_VOLUME : TONE_VOLUME / 2);
  else { audio::silence(100); }
}

static void callTick() { lights::blink(lights::GREEN, 200, 150, 150); }

static void doCall() {
  touched();
  waitReleased(btnCall);
  lights::set(lights::GREEN, 255);
  if (!net::wifiWait(12000, callTick)) { lights::off(); soundError(); touched(); return; }
  intercom::begin();
  while (intercom::poll() != intercom::NONE) {}         // forget stale messages
  uint32_t t0 = millis(), lastRing = 0;
  bool acked = false;
  while (true) {
    uint32_t now = millis();
    if (now - lastRing > 1000) { intercom::send(intercom::RING); lastRing = now; }
    lights::blink(lights::GREEN, 255, 300, 300);
    ringSlice(now - t0, false);
    updateButtons();
    switch (intercom::poll()) {
      case intercom::RINGACK: acked = true; break;
      case intercom::ANSWER: audio::ampOff(); inCall(); return;
      case intercom::HANGUP:                          // declined
      case intercom::BUSY: audio::ampOff(); lights::off(); soundBusy(); touched(); return;
      case intercom::RING:                            // both called at once: just connect
        intercom::send(intercom::ANSWER); audio::ampOff(); inCall(); return;
      default: break;
    }
    if (btnHangup.edge || btnCall.edge) { intercom::send(intercom::HANGUP); break; }
    if (!acked && now - t0 > FIND_FRIEND_MS) {     // the other phone is asleep (or off)
      intercom::send(intercom::HANGUP);           // in case it wakes and hears a stale RING
      audio::ampOff(); lights::off(); soundSleepy(); touched(); return;
    }
    if (now - t0 > RING_TIMEOUT_MS) { intercom::send(intercom::HANGUP); break; }
  }
  audio::ampOff();
  lights::off();
  soundGoodbye();
  touched();
}

static void answerOrDecline() {
  Serial.println("incoming call");
  intercom::send(intercom::RINGACK);
  uint32_t t0 = millis();
  while (millis() - t0 < RING_TIMEOUT_MS + 2000) {
    lights::blink(lights::GREEN, 255, 120, 120);
    ringSlice(millis() - t0, true);
    updateButtons();
    if (btnCall.edge) {
      intercom::send(intercom::ANSWER);
      audio::ampOff();
      waitReleased(btnCall);            // don't start the call already "talking"
      inCall();
      return;
    }
    if (btnHangup.edge) { intercom::send(intercom::HANGUP); break; }
    intercom::Msg m = intercom::poll();
    if (m == intercom::HANGUP) break;                   // caller gave up
    if (m == intercom::RING) intercom::send(intercom::RINGACK);
  }
  audio::ampOff();
  lights::off();
  touched();
}

// ---------------- Sleep / wake ----------------
static bool inPlayHours() {
  if (PLAY_START_HOUR < 0 || PLAY_END_HOUR < 0 || !net::timeValid()) return false;
  time_t t = time(nullptr);
  struct tm tm;
  localtime_r(&t, &tm);
  if (PLAY_START_HOUR <= PLAY_END_HOUR) return tm.tm_hour >= PLAY_START_HOUR && tm.tm_hour < PLAY_END_HOUR;
  return tm.tm_hour >= PLAY_START_HOUR || tm.tm_hour < PLAY_END_HOUR;   // e.g. 20..2
}

static uint64_t secondsUntilPlayStart() {
  time_t t = time(nullptr);
  struct tm tm;
  localtime_r(&t, &tm);
  int64_t now = tm.tm_hour * 3600 + tm.tm_min * 60 + tm.tm_sec;
  int64_t start = PLAY_START_HOUR * 3600;
  int64_t d = start - now;
  if (d <= 0) d += 24 * 3600;
  return (uint64_t)d;
}

static const uint64_t WAKE_MASK = (1ULL << PIN_BTN_TATTLE) | (1ULL << PIN_BTN_CALL) | (1ULL << PIN_BTN_HANGUP);

static void goToSleep() {
  Serial.println("sleeping");
  intercom::end();
  net::wifiStop();
  lights::off();
  digitalWrite(PIN_AMP_SD, LOW);
  // Buttons pull to GND: keep the internal pull-ups on in deep sleep (IO0 also has 10k).
  esp_sleep_pd_config(ESP_PD_DOMAIN_RTC_PERIPH, ESP_PD_OPTION_ON);
  for (gpio_num_t p : {(gpio_num_t)PIN_BTN_TATTLE, (gpio_num_t)PIN_BTN_CALL, (gpio_num_t)PIN_BTN_HANGUP}) {
    rtc_gpio_pullup_en(p);
    rtc_gpio_pulldown_dis(p);
  }
  esp_sleep_enable_ext1_wakeup_io(WAKE_MASK, ESP_EXT1_WAKEUP_ANY_LOW);
  if (PLAY_START_HOUR >= 0 && net::timeValid())
    esp_sleep_enable_timer_wakeup(secondsUntilPlayStart() * 1000000ULL);
  Serial.flush();
  esp_deep_sleep_start();
}

static void showBattery(float v) {
  // 3 red blinks = full, 2 = OK, 1 = low (and a sad double beep)
  int n = v > 4.2f ? 3 : v > VBAT_LOW ? 2 : 1;
  lights::flash(lights::RED, n, 200, 250);
  if (n == 1) soundLowBatt();
}

// =================================================================
void setup() {
  Serial.begin(115200);
  btnTattle.begin(); btnCall.begin(); btnHangup.begin();
  lights::begin();
  audio::begin();
  if (!net::begin()) Serial.println("FAT mount failed");
  analogSetPinAttenuation(PIN_VBAT_ADC, ADC_11db);

  float v = batteryVolts();
  Serial.printf("%s awake, battery %.2f V, %d tattle(s) waiting\n", PHONE_NAME, v, net::outboxCount());
  if (v < VBAT_LOW && v > 1.0f) soundLowBatt();   // (v < 1 V: powered from USB-only bench setup)

  esp_sleep_wakeup_cause_t cause = esp_sleep_get_wakeup_cause();
  uint64_t which = cause == ESP_SLEEP_WAKEUP_EXT1 ? esp_sleep_get_ext1_wakeup_status() : 0;
  if (cause != ESP_SLEEP_WAKEUP_UNDEFINED)   // after deep sleep: give the pins back to normal GPIO
    for (gpio_num_t p : {(gpio_num_t)PIN_BTN_TATTLE, (gpio_num_t)PIN_BTN_CALL, (gpio_num_t)PIN_BTN_HANGUP})
      rtc_gpio_deinit(p);
  btnTattle.begin(); btnCall.begin(); btnHangup.begin();
  updateButtons();

  net::wifiStart();                        // connect in the background either way
  touched();
  if (which & (1ULL << PIN_BTN_TATTLE)) doTattle();
  else if (which & (1ULL << PIN_BTN_CALL)) doCall();
  else if (which & (1ULL << PIN_BTN_HANGUP)) { showBattery(v); waitReleased(btnHangup); }
  else if (cause != ESP_SLEEP_WAKEUP_TIMER) lights::sparkle();   // power switched on
}

void loop() {
  static uint32_t lastFlush = 0;
  updateButtons();

  if (btnTattle.edge) doTattle();
  else if (btnCall.edge) doCall();
  else if (btnHangup.edge) {               // red while idle: "goodnight"
    showBattery(batteryVolts());
    soundGoodbye();
    waitReleased(btnHangup);
    goToSleep();
  }

  if (net::wifiConnected()) {
    intercom::begin();
    // A RING that's been sitting in the queue (e.g. during a tattle) is stale: the
    // caller has moved on. Only answer one we heard in the last couple of seconds.
    if (intercom::poll() == intercom::RING && intercom::msSinceHeard() < 2500) answerOrDecline();
    // retry queued tattles now and then
    if (net::outboxCount() && millis() - lastFlush > 60000) {
      lastFlush = millis();
      net::flushOutbox(nullptr);
    }
    lights::breathe(lights::GREEN, 0, 18, 4000);   // faint green breath = "you can call me"
  } else {
    lights::off();
  }

  bool keepAwake = inPlayHours() && batteryVolts() > VBAT_EMPTY;
  if (!keepAwake && millis() - lastActivity > AWAKE_MINUTES * 60000UL) goToSleep();
  delay(20);
}
