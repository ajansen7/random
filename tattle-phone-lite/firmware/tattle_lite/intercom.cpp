#include "intercom.h"
#include <WiFi.h>
#include <AsyncUDP.h>
#include "freertos/queue.h"
#include "freertos/stream_buffer.h"
#include "audio.h"
#include "config.h"

namespace intercom {

static const uint16_t PORT = 47260;
static const size_t JITTER_BYTES = audio::RATE * 2 * 80 / 1000;   // start playing once 80 ms is queued
static const size_t BUFFER_BYTES = audio::RATE * 2 * 400 / 1000;   // drop audio beyond 400 ms backlog

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

static volatile bool inCall = false;
static volatile bool dropping = false;
static bool primed = false;

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
    if (inCall && !dropping)
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
  if (!audioBuf) audioBuf = xStreamBufferCreate(BUFFER_BYTES, 1);
  if (udp.listen(PORT)) {
    udp.onPacket(onPacket);
    listening = true;
  }
}

void end() {
  callEnd();
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
  static uint8_t buf[sizeof(Header) + audio::MAX_CHUNK * sizeof(int16_t)];
  n = min(n, audio::MAX_CHUNK);
  Header h;
  fillHeader(h, AUDIO);
  memcpy(buf, &h, sizeof(h));
  memcpy(buf + sizeof(h), pcm, n * sizeof(int16_t));
  sendRaw(buf, sizeof(h) + n * sizeof(int16_t), false);
}

void callStart() {
  xStreamBufferReset(audioBuf);
  primed = false;
  dropping = false;
  inCall = true;
}

void callEnd() { inCall = false; }

int32_t readAudio(int16_t *out, size_t n) {
  const size_t bytes = n * sizeof(int16_t);
  if (!primed) {                          // wait for a little cushion before playing
    if (xStreamBufferBytesAvailable(audioBuf) < JITTER_BYTES) {
      memset(out, 0, bytes);
      return 0;
    }
    primed = true;
  }
  size_t got = xStreamBufferReceive(audioBuf, out, bytes, 0);
  if (got < bytes) {
    memset((uint8_t *)out + got, 0, bytes - got);
    if (got == 0) primed = false;         // ran dry: re-buffer before playing again
  }
  return audio::rms(out, n);
}

void dropIncoming(bool drop) {
  if (drop && !dropping) xStreamBufferReset(audioBuf);
  dropping = drop;
}

}  // namespace intercom
