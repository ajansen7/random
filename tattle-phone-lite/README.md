# Tattle Phone Lite

A much cheaper version of [`../tattle-phone`](../tattle-phone): **3 big buttons, 3×AA batteries,
a 2-layer board with every factory part on one side.** Same idea: a toddler presses a button,
it records what they say and sends it to a parent (Google Drive + email), and two phones can
call each other on the home Wi-Fi.

| | Original | Lite |
|---|---|---|
| Buttons | 12-key keypad + CALL | ⭐ TATTLE (yellow star), 🟢 CALL / ANSWER (green circle), 🔴 HANG UP (red square) |
| Lights | 14 smart RGB LEDs | 6 plain LEDs (two per button, matching colours), shining up through holes in the board |
| Power | Li-ion + charger + fuel gauge + buck-boost | 3×AA (alkaline or NiMH) + one 3.3 V regulator; battery level from a resistor divider |
| Board | 4 layers, parts on both sides | 2 layers, factory parts on the back only; you solder the 3 buttons and a few wires |
| Parts per board | ≈ $23 | ≈ $7 |
| Case | 175 × 52–58 × 30 mm | 170 × 60–62 × 34.5 mm, battery door with one screw |

## Status

1. ✅ Parts chosen and checked against JLC stock and datasheets (list below).
2. ✅ Board + case geometry (`hardware/layout.py`), case model (`enclosure/`), STLs build.
3. ✅ KiCad board: 54.4 × 120.5 mm, 2 layers, routed, **DRC 0 errors / 0 unconnected**,
   schematic and board match on all 164 pins. Fab files with a full LCSC BOM in `hardware/fab/`.
4. ✅ Firmware written and **compiles cleanly** (ESP32 core 3.3.12, 0 warnings with `-Wall`). Not yet run on hardware.

## Parts (factory-placed, all on the back)

| Ref | Part | LCSC | Notes |
|---|---|---|---|
| U1 | ESP32-S3-WROOM-1-N8R2 | C2913204 | S3, not C3: the C3 can't decode a PDM mic in hardware and has only one I2S |
| MK1 | LinkMems LMD3526B261-OFA03 | C41347997 | PDM, **bottom port** over a hole in the board. Pins: 1 DATA, 2 L/R, 3 CLK, 4 VDD, 5 GND ring. Do not wash |
| U2 | MAX98357A | C910544 | speaker amp, runs from the batteries |
| U3 | RT9080-33GJ5 | C841192 | 3.3 V LDO, 2 µA idle; same pinout as AP2112K |
| Q1 | AO3401A | C15127 | reverse-battery protection |
| SW4 | Shouhan MSK12C02 | C431540 | power switch, drives the LDO's EN pin only (it's a 50 mA switch) |
| J1 | HRO TYPE-C-31-M-12 | C165948 | **programming only**, no power path: batteries must be in to flash |
| LED1–6 | Lite-On LTST-C230 reverse-mount (yellow C125108, green C125106, red C125107) | | lens fits KiCad's `LED_1206_3216Metric_ReverseMount_Hole1.8x2.4mm` |

**You buy and solder:** see **[SHOPPING.md](SHOPPING.md)** for every part number, the LCSC
order file, the mechanical bits and a from-scratch tool list. The case is sized for a
**Keystone 2465** 3×AA holder (58 × 48.3 × 17 mm); for another holder, measure it and edit
`BATT_HOLDER` in `hardware/layout.py`.

## Case

`enclosure/tattle_lite.scad` (open in OpenSCAD, F5). `cd enclosure && make` builds `stl/` and
`pcb_outline.dxf`; `make renders` makes the pictures in `renders/`.

- **Battery door:** the holder sticks to the inside of the door (foam tape or screws). Undo
  one screw, lift door + holder out on the leads, swap cells. The far end hooks under the back
  wall. A screwed door is the norm for toys for under-3s.
- **Power switch** is recessed in the right side: a fingernail or pen tip moves it, a toddler
  mostly can't.
- **Buttons:** print the caps in translucent filament in each colour. A flange under the
  front wall keeps each one captive once the case is closed.

## Board (`hardware/`)

Everything is generated from two files: `hardware/layout.py` (positions shared with the case)
and `hardware/kicad_gen/design.py` (every part and what each pin connects to).

```
hardware/kicad_gen/run_all.sh     # build -> route -> stitch -> check -> make_fab
```

| Script | Does |
|---|---|
| `build.py` | schematic + placed, unrouted board; writes the mic footprint (from its datasheet), a reverse-mount LED footprint with a slightly smaller lens hole, and a DRC rule for those holes |
| `route.py` | FreeRouting (Java 25 + `~/tools/freerouting/freerouting-2.4.1.jar`). GND is left to the pours; the pours' "plane" blocks are stripped from the DSN so FreeRouting doesn't see a board-sized obstacle. Locked items (the amp's thermal-pad via) are kept |
| `stitch.py` | GND stitching vias on a 4 mm grid wherever both layers have pour, then one via per leftover pour island. Run **once** after `route.py` |
| `check.py` | schematic-vs-board netlist check + grouped DRC |
| `make_fab.py` | Gerbers + drill zip, `bom_jlc.csv` (LCSC numbers from `lcsc.py`), `cpl_jlc.csv` |

`run_all.sh` overwrites the board. Once you've hand-edited it in KiCad, run only `check.py`
and `make_fab.py`. FreeRouting isn't deterministic, so each full run gives slightly different
traces; the checks are what matter.

**Ordering (JLCPCB):** 2 layers, 1.6 mm, 54.4 × 120.5 mm. Min via **0.3 mm** (every hole is
≥ 0.3 mm: KiCad's ESP32 footprint has 0.2 mm thermal vias, which `build.py` re-drills to 0.3 mm,
because 0.2 mm holes move a 2-layer board into a much pricier class). Via covering: **Plugged**
for factory assembly (there's a via in the amp's thermal pad and 12 in the ESP32's ground pad);
**Tented** (or Untented) is fine if you assemble it yourself; see the note on via-in-pad below.
**Assembly: bottom side only.** Upload `hardware/fab/tattle_lite_gerbers.zip`, then
`bom_jlc.csv` + `cpl_jlc.csv`. In the placement preview check the LEDs, U1–U3, Q1 and the mic.
Board options: no washing / no conformal coating (MEMS mic).

**Expected DRC warnings:** "footprint doesn't match library" for the generated/locked
footprints, and two small silkscreen overlaps.

**Assembling it yourself:** stencil + paste + hotplate for the back (one pass), then the three
buttons by hand (clip the leads flush). Hard parts: the mic (no leads, keep paste and air out of
its port, never wash), the amp (3×3 mm QFN), the ESP32's ground pad, USB-C. Vias inside pads
(ESP32 ground pad, amp thermal pad) wick a little solder unless the vias are plugged: use a
touch more paste there or Kapton tape over them on the far side. First power-up from a
current-limited bench supply (~4.5 V, 200 mA) after checking 3V3 and VBAT aren't shorted to GND.

## Firmware (`firmware/tattle_lite/`)

**What the buttons do**

| Button | Idle | Ringing (incoming) | In a call |
|---|---|---|---|
| ⭐ yellow star | record a tattle (beep, talk, it stops on silence or another press) and send it to Drive + email | | |
| 🟢 green circle | call the other phone | answer | (nothing: just talk) |
| 🔴 red square | show battery (3/2/1 red blinks) and go to sleep | decline | hang up (either phone) |

Lights: the button you're using glows (yellow follows your voice while recording; green
follows the other kid's voice in a call). A faint green "breath" when idle means the
phone is on Wi-Fi and can be called.

**Power:** deep sleep until a button is pressed. After any use the phone stays awake
and callable for `AWAKE_MINUTES` (default 5). Optional `PLAY_START_HOUR`/`PLAY_END_HOUR`
keep both phones callable all afternoon (they also wake themselves at the start time),
at the cost of battery: roughly 20-30 mA while awake.

**Calls:** both phones must be on the same Wi-Fi. They find each other by name with
UDP broadcasts (no IP addresses to configure); `INTERCOM_KEY` must match on both. If
your router has "client/AP isolation" (common on guest networks), turn it off for this
network. If the other phone is asleep, the caller hears a descending "sleepy" tune.
Calls are **full duplex** like a real phone: both kids can talk at once. Each phone runs
Espressif's acoustic echo canceller (ESP-SR, built into the ESP32-S3 core) so its own
earpiece isn't sent back to the other kid as an echo. If a call still echoes or howls
after tuning, `#define CALL_MODE_WALKIE` in `config.h` switches to walkie-talkie calls
(hold green to talk); the firmware also falls back to that by itself if the echo
canceller can't start.

**Set up and flash**

1. Deploy `cloud/drive_upload.gs` as a Google Apps Script web app (steps in its header).
2. `cp firmware/tattle_lite/config.example.h firmware/tattle_lite/config.h` and fill in
   Wi-Fi, the upload URL/token, and the two phone names (swapped on the other phone).
3. Batteries in, power ON, USB-C cable to the computer, then
   `firmware/build.sh /dev/cu.usbmodem*` (find the port with `arduino-cli board list`).
   Arduino IDE instead: board "ESP32S3 Dev Module", Flash Size 8MB, PSRAM "QSPI PSRAM",
   Partition Scheme "Custom" (uses `partitions.csv`), USB CDC On Boot "Enabled".
4. If it won't flash: hold 🟢 green while switching on (that's the BOOT strap), flash,
   switch off and on.

**First bring-up, and tuning** (serial monitor at 115200 shows what's happening):
- Recording: `rms=` lines show the mic level. Talking should be well above `SPEECH_RMS`
  (600) and silence well below. Adjust `SPEECH_RMS` / `MIC_GAIN` in `tattle_lite.ino`.
- If the mic reads all zeros, add `#define MIC_RIGHT_SLOT` to `config.h` (some PDM mics
  put "left" data on the other clock edge).
- Battery: the boot line prints the battery voltage; compare with a multimeter.
- Volumes: `TONE_VOLUME`, `RING_VOLUME`, `CALL_VOLUME`.
- Echo (needs both phones): during a call the monitor prints
  `call: mic=… after-aec=… far=…` every 2 s. Have only the *other* kid talk: `after-aec`
  should be far below `mic` (the canceller removing your own earpiece). If it isn't,
  try `CALL_ECHO_DELAY_CHUNKS` 1, 2 or 3 in `config.h`, and keep `CALL_VOLUME` at earpiece
  level. The canceller needs a second or two at the start of each call to lock on.
