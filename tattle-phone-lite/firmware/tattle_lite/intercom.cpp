#include "intercom.h"
#include <WiFi.h>
#include <AsyncUDP.h>
#include "freertos/queue.h"
#include "freertos/stream_buffer.h"
#include "audio.h"
#include "config.h"

namespace intercom {

static const uint16_t PORT = 47260;
static const size_t FRAME_BYTES = audio::FRAME * sizeof(int16_t);
static const size_t JITTER_FRAMES = 4;          // start playing once this much is queued

struct __attribute__((packed)) Header {
  char magic[4];          // "TPL1"
  uint32_t key;           // FNV-1a of INTERCOM_KEY
  uint8_t type;
  uint8_t reserved;
  uint16_t seq;
  char from[16];
  char to[16];
};

static AsyncUDP udp;
static bool listening = false;
static QueueHandle_t ctrlQ = nullptr;
static StreamBufferHandle_t audioBuf = nullptr;
static volatile uint32_t lastHeard = 0;
static volatile bool haveFriend = false;
static IPAddress friendIP;
static uint16_t seq = 0;
static uint32_t keyHash = 0;

static TaskHandle_t speakerTask = nullptr;
static volatile bool playing = false;
static volatile bool talking = false;
static volatile uint8_t volume = 60;
static volatile int32_t lastLevel = 0;

static uint32_t fnv1a(const char *s) {
  uint32_t h = 2166136261u;
  while (*s) { h ^= (uint8_t)*s++; h *= 16777619u; }
  return h;
}

static void fillHeader(Header &h, Msg m) {
  memcpy(h.magic, "TPL1", 4);
  h.key = keyHash;
  h.type = m;
  h.reserved = 0;
  h.seq = seq++;
  memset(h.from, 0, sizeof(h.from));
  memset(h.to, 0, sizeof(h.to));
  strncpy(h.from, PHONE_NAME, sizeof(h.from) - 1);
  strncpy(h.to, FRIEND_NAME, sizeof(h.to) - 1);
}

static void onPacket(AsyncUDPPacket &p) {
  if (p.length() < sizeof(Header)) return;
  Header h;
  memcpy(&h, p.data(), sizeof(Header));
  if (memcmp(h.magic, "TPL1", 4) != 0 || h.key != keyHash) return;
  if (strncmp(h.from, FRIEND_NAME, sizeof(h.from)) != 0) return;
  if (strncmp(h.to, PHONE_NAME, sizeof(h.to)) != 0) return;
  friendIP = p.remoteIP();
  haveFriend = true;
  lastHeard = millis();
  if (h.type == AUDIO) {
    if (playing && !talking)
      xStreamBufferSend(audioBuf, p.data() + sizeof(Header), p.length() - sizeof(Header), 0);
  } else if (h.type != PING) {
    Msg m = (Msg)h.type;
    xQueueSend(ctrlQ, &m, 0);
  }
}

void begin() {
  if (listening) return;
  keyHash = fnv1a(INTERCOM_KEY);
  if (!ctrlQ) ctrlQ = xQueueCreate(16, sizeof(Msg));
  if (!audioBuf) audioBuf = xStreamBufferCreate(FRAME_BYTES * 16, FRAME_BYTES);
  if (udp.listen(PORT)) {
    udp.onPacket(onPacket);
    listening = true;
  }
}

void end() {
  stopPlayback();
  if (listening) udp.close();
  listening = false;
}

bool friendKnown() { return haveFriend; }

static void sendRaw(const uint8_t *buf, size_t len, bool broadcast) {
  if (!listening) return;
  if (haveFriend) udp.writeTo(buf, len, friendIP, PORT);
  if (broadcast || !haveFriend) udp.broadcastTo((uint8_t *)buf, len, PORT);
}

void send(Msg m) {
  Header h;
  fillHeader(h, m);
  // RING always goes out as a broadcast too, in case the friend's address changed.
  sendRaw((const uint8_t *)&h, sizeof(h), m == RING);
}

Msg poll() {
  Msg m = NONE;
  if (ctrlQ) xQueueReceive(ctrlQ, &m, 0);
  return m;
}

uint32_t msSinceHeard() { return lastHeard ? millis() - lastHeard : UINT32_MAX; }

void sendAudio(const int16_t *pcm, size_t n) {
  static uint8_t buf[sizeof(Header) + FRAME_BYTES];
  n = min(n, audio::FRAME);
  Header h;
  fillHeader(h, AUDIO);
  memcpy(buf, &h, sizeof(h));
  memcpy(buf + sizeof(h), pcm, n * sizeof(int16_t));
  sendRaw(buf, sizeof(h) + n * sizeof(int16_t), false);
}

// ------------------------------------------------------------------ speaker task
static void speakerLoop(void *) {
  static int16_t frame[audio::FRAME];
  bool primed = false;
  for (;;) {
    if (!playing) {
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }
    size_t avail = xStreamBufferBytesAvailable(audioBuf);
    if (talking || (!primed && avail < JITTER_FRAMES * FRAME_BYTES)) {
      primed = primed && !talking;
      lastLevel = 0;
      audio::silence(20);                 // paces the loop at the I2S rate
      continue;
    }
    size_t got = xStreamBufferReceive(audioBuf, frame, FRAME_BYTES, 0);
    if (got == 0) {                       // ran dry: re-buffer before playing again
      primed = false;
      continue;
    }
    primed = true;
    size_t n = got / sizeof(int16_t);
    for (size_t i = 0; i < n; i++) frame[i] = (int16_t)((int32_t)frame[i] * volume / 100);
    lastLevel = audio::rms(frame, n);
    audio::play(frame, n);
  }
}

void startPlayback(uint8_t volumePct) {
  volume = volumePct;
  xStreamBufferReset(audioBuf);
  if (!speakerTask) xTaskCreatePinnedToCore(speakerLoop, "speaker", 4096, nullptr, 5, &speakerTask, 1);
  audio::ampOn();
  playing = true;
}

void stopPlayback() {
  if (!playing) return;
  playing = false;
  delay(40);                              // let the speaker task finish its frame
  audio::ampOff();
}

void setTalking(bool t) {
  if (t && !talking && audioBuf) xStreamBufferReset(audioBuf);
  talking = t;
}

int32_t incomingLevel() { return lastLevel; }

}  // namespace intercom
