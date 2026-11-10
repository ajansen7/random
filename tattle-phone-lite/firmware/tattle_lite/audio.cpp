#include "audio.h"
#include "pins.h"
#if __has_include("config.h")
#include "config.h"            // for MIC_RIGHT_SLOT
#endif
#include "driver/i2s_pdm.h"
#include "driver/i2s_std.h"
#include "freertos/semphr.h"

namespace audio {

static i2s_chan_handle_t micChan = nullptr;
static i2s_chan_handle_t ampChan = nullptr;
static SemaphoreHandle_t ampLock = nullptr;
static bool micRunning = false;
static int32_t dc = 0;   // running DC estimate for the mic's high-pass

void begin() {
  // --- Microphone: I2S0 in PDM RX mode. The S3 converts PDM to 16-bit PCM in hardware.
  // Default down-sampling (x128) gives a 2.048 MHz mic clock at 16 kHz, inside the
  // LMD3526B261's "standard mode" range (1.3-4.8 MHz).
  // Small DMA rings (3 x 256 samples = 48 ms) on both sides: the echo canceller needs
  // the delay from "written to the speaker" to "heard by the mic" short and steady.
  i2s_chan_config_t rxc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  rxc.dma_desc_num = 3;
  rxc.dma_frame_num = 256;
  ESP_ERROR_CHECK(i2s_new_channel(&rxc, nullptr, &micChan));
  i2s_pdm_rx_config_t rx = {};
  rx.clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(RATE);
  rx.slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO);
#ifdef MIC_RIGHT_SLOT
  rx.slot_cfg.slot_mask = I2S_PDM_SLOT_RIGHT;   // if a build ever reads silence, try this
#endif
  rx.gpio_cfg.clk = (gpio_num_t)PIN_MIC_CLK;
  rx.gpio_cfg.din = (gpio_num_t)PIN_MIC_DATA;
  rx.gpio_cfg.invert_flags.clk_inv = false;
  ESP_ERROR_CHECK(i2s_channel_init_pdm_rx_mode(micChan, &rx));

  // --- Amplifier: I2S1 standard (Philips) TX, 16-bit mono copied to both slots.
  i2s_chan_config_t txc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_1, I2S_ROLE_MASTER);
  txc.auto_clear = true;    // underrun -> silence, not a repeated buffer
  txc.dma_desc_num = 3;
  txc.dma_frame_num = 256;
  ESP_ERROR_CHECK(i2s_new_channel(&txc, &ampChan, nullptr));
  i2s_std_config_t tx = {};
  tx.clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(RATE);
  tx.slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO);
  tx.slot_cfg.slot_mask = I2S_STD_SLOT_BOTH;
  tx.gpio_cfg.mclk = I2S_GPIO_UNUSED;
  tx.gpio_cfg.bclk = (gpio_num_t)PIN_AMP_BCLK;
  tx.gpio_cfg.ws = (gpio_num_t)PIN_AMP_LRC;
  tx.gpio_cfg.dout = (gpio_num_t)PIN_AMP_DIN;
  tx.gpio_cfg.din = I2S_GPIO_UNUSED;
  ESP_ERROR_CHECK(i2s_channel_init_std_mode(ampChan, &tx));
  ESP_ERROR_CHECK(i2s_channel_enable(ampChan));

  ampLock = xSemaphoreCreateMutex();
  pinMode(PIN_AMP_SD, OUTPUT);
  digitalWrite(PIN_AMP_SD, LOW);
}

// ------------------------------------------------------------------ speaker
void ampOn() { digitalWrite(PIN_AMP_SD, HIGH); }

void ampOff() {
  silence(30);
  digitalWrite(PIN_AMP_SD, LOW);
}

void play(const int16_t *pcm, size_t n) {
  xSemaphoreTake(ampLock, portMAX_DELAY);
  size_t written;
  i2s_channel_write(ampChan, pcm, n * sizeof(int16_t), &written, portMAX_DELAY);
  xSemaphoreGive(ampLock);
}

void tone(float f1, float f2, uint32_t ms, uint8_t volumePct) {
  ampOn();
  const float amp = 32767.0f * volumePct / 100.0f / (f2 > 0 ? 2 : 1);
  const uint32_t total = RATE * ms / 1000;
  const uint32_t fade = RATE / 200;   // 5 ms
  const float d1 = 2 * PI * f1 / RATE, d2 = 2 * PI * f2 / RATE;
  float p1 = 0, p2 = 0;
  int16_t buf[256];
  for (uint32_t n = 0; n < total;) {
    size_t cnt = min<uint32_t>(256, total - n);
    for (size_t i = 0; i < cnt; i++, n++) {
      float env = 1.0f;
      if (n < fade) env = (float)n / fade;
      else if (total - n < fade) env = (float)(total - n) / fade;
      float s = (f1 > 0 ? sinf(p1) : 0) + (f2 > 0 ? sinf(p2) : 0);
      buf[i] = (int16_t)(s * amp * env);
      p1 += d1; if (p1 > 2 * PI) p1 -= 2 * PI;
      p2 += d2; if (p2 > 2 * PI) p2 -= 2 * PI;
    }
    play(buf, cnt);
  }
}

void silence(uint32_t ms) {
  static const int16_t zeros[256] = {0};
  uint32_t total = RATE * ms / 1000;
  while (total) {
    size_t cnt = min<uint32_t>(256, total);
    play(zeros, cnt);
    total -= cnt;
  }
}

// ------------------------------------------------------------------ microphone
void micStart() {
  if (micRunning) return;
  ESP_ERROR_CHECK(i2s_channel_enable(micChan));
  micRunning = true;
  dc = 0;
  int16_t junk[FRAME];
  for (int i = 0; i < 5; i++) micRead(junk, FRAME, 100, 1);   // let the mic settle
}

void micStop() {
  if (!micRunning) return;
  i2s_channel_disable(micChan);
  micRunning = false;
}

size_t micRead(int16_t *out, size_t n, uint32_t timeoutMs, int gain) {
  size_t got = 0;
  if (i2s_channel_read(micChan, out, n * sizeof(int16_t), &got, pdMS_TO_TICKS(timeoutMs)) != ESP_OK && got == 0)
    return 0;
  size_t cnt = got / sizeof(int16_t);
  for (size_t i = 0; i < cnt; i++) {
    int32_t s = out[i];
    dc += (s * 256 - dc) >> 8;          // slow running average (fixed point, x256)
    s = (s - (dc >> 8)) * gain;          // high-pass + gain
    out[i] = (int16_t)constrain(s, -32768, 32767);
  }
  return cnt;
}

int32_t rms(const int16_t *pcm, size_t n) {
  if (!n) return 0;
  int64_t sum = 0;
  for (size_t i = 0; i < n; i++) sum += (int32_t)pcm[i] * pcm[i];
  return (int32_t)sqrtf((float)sum / n);
}

}  // namespace audio
