# Tattle Phone: PCB

There's a first-pass KiCad project in [`kicad/`](kicad/). It has a schematic
with every part and connection, and a board with the outline, every footprint
placed to fit the printed case, every pad on its net, GND pours and the antenna
keep-out. **The board is not routed yet.** That's the next step, and it's
yours to do in KiCad (or ask me to take a pass).

> ⚠️ Component values come from datasheets and common reference designs
> (Adafruit's BQ24075 and MAX98357A boards are worth opening alongside). Check
> them, and the footprints, against current datasheets before ordering. This
> board hasn't been built.

## Files

| File | What it is |
|---|---|
| `kicad/tattle_phone.kicad_pro` | Open this in KiCad 10. It was generated with KiCad 7, so 10 will offer to upgrade it; say yes |
| `kicad/tattle_phone.kicad_sch` | One-sheet schematic. Every pin carries a **net label**, so it reads like a netlist. Matching labels are connected |
| `kicad/tattle_phone.kicad_pcb` | Board: outline, placement, nets, pours, keep-out, net classes |
| `kicad/tattle.kicad_sym` | Project symbol library (only the MAX17048; everything else is stock KiCad) |
| `kicad_gen/design.py` | **The circuit as code:** every part, footprint, pin→net and position. This is the master copy |
| `kicad_gen/build.py` | Regenerates the whole KiCad project from `design.py` + `layout.py` |
| `kicad_gen/check.py` | Compares schematic and board pin by pin, then runs DRC and summarises it |
| `layout.py` | Dimensions shared by the board **and** the 3D-printed case (keys, LEDs, holes, outline) |
| `place_footprints.py` | Scripting-console helper. Re-snaps the case-critical parts if you move things by hand |

### Two ways to work

- **Edit in KiCad (normal):** open the project, tidy the schematic, route the
  board. From then on the KiCad files are the master copy, and `design.py` becomes reference.
  Don't re-run `build.py`, because it would overwrite your routing.
- **Edit the code:** change `design.py` / `layout.py`, run
  `python3 kicad_gen/build.py && python3 kicad_gen/check.py` (needs KiCad's Python
  module), and you get a fresh project. This is good for big part swaps before routing.

## Current check results

- **Schematic vs board:** 377 pins, 71 nets, 0 mismatches. KiCad exported the
  schematic's netlist and it was compared pad by pad with the board.
- **DRC:** no clearance, edge, hole or keep-out errors. What's left is expected:
  - *Unconnected items:* the board isn't routed.
  - *Courtyard overlap LEDn / SWn (12×):* each key LED sits under its keycap, just
    beside the tact switch. The bodies clear by about 0.25 mm; the switch's outline is drawn
    generously.
  - *Library footprint issues:* footprints were placed from KiCad's install
    folder, not through a library table. KiCad 10 may offer to relink them.
  - *Silkscreen:* `tidy.py` places every reference label clear of pads, other labels
    and the edge, and drops artwork nobody can see (LED outlines under the keycaps). A few
    footprint outlines still touch their own pads; KiCad trims those automatically in the Gerbers.
  - *Starved thermals:* a few QFN centre pads have fewer than two ground spokes until routing
    adds vias under them.

## What's on the board

Front (faces the keypad): the 13 tact switches (through-hole) and the 14 RGB LEDs.
Back: everything else.

| Block | Parts | Notes |
|---|---|---|
| USB-C | J1 HRO TYPE-C-31-M-12 (back, mouth at the bottom edge), R1/R2 5.1k CC pull-downs (separate!), F1 polyfuse, D1 USBLC6 ESD, D2 SMF5.0A TVS | Same port charges and flashes the ESP32 |
| Charger | U2 **BQ24075** (power path + SYSOFF), R3–R8, C1–C3, J2 battery JST-PH, J4 NTC JST-PH | ISET 1.2k ≈ 750 mA charge; ILIM 1.5k ≈ 1 A in; R5 (TMR) left unfitted = default safety timers. BQ24075 instead of BQ24074 because it has the SYSOFF pin the power switch needs |
| Power switch | SW14 PCM12 slide, right edge | Drives SYSOFF: OFF disconnects the battery; charging still works |
| Mode switch | SW17 PCM12 slide, left edge → GPIO38 | **TATTLE** = deep sleep, CALL only; **PLAY** = stay on Wi-Fi for intercom |
| 3.3 V | U3 TPS63001 buck-boost, L1 1.5 µH XAL4020, C4–C7 | Footprint has 0.2 mm thermal vias (board min drill set to 0.2 mm) |
| MCU | U1 ESP32-S3-WROOM-1-N16R8 (back, antenna flush with top edge), C8–C11, R9/R10, SW15 RESET, SW16 BOOT | Antenna keep-out zone: no copper in the top ~6 mm |
| Mic | MK1 ICS-43434 (back) + C12, R11 | Port hole through the board lines up with the case's mic duct |
| Amp | U4 MAX98357A, C13/C14, R12, J3 speaker JST-PH | GAIN tied to VDD = 6 dB. J3 sits low on the right; run the speaker leads up the inside of the case |
| Gauge | U6 MAX17048 + C15, R13–R15 (I2C pull-ups) | I2C 0x36 |
| Keypad | SW1–SW12 + U7 **TCA9555RTWR** (QFN, fits between the switch legs), R16, C16; SW13 CALL + R17 | I2C 0x20, internal pull-ups |
| LEDs | LED1 CALL, LED2–13 keys, **LED14 status light**, all SK6812MINI on the front; Q1/Q2 power gate, U8 74AHCT1G125 level shifter, R18–R20, C17/C18, C20–C33 | Chain order matches the firmware: 0 = CALL, 1–12 = keys, 13 = status |
| Test | TP1 TX, TP2 RX, TP3 GND, TP4 VBAT, TP5 VSYS, TP6 3V3, TP7 spare IO1; FID1–3 | |

The pin map matches [`firmware/tattle_phone/pins.h`](../firmware/tattle_phone/pins.h).

## Ordering (JLCPCB or similar)

- 2-layer, 1.6 mm, any colour. Min drill 0.2 mm (thermal vias under U1/U3).
- **Assembly on both sides** (the LEDs are on the front). That costs a bit more
  than one side, but nothing on this board is novice-friendly to hand-solder
  except the through-hole parts.
- **You solder:** 13 tact switches and 3 JST-PH connectors, all big through-hole
  pins. JLC can do these too for a fee.
- Check stock for: ESP32-S3-WROOM-1-N16R8, BQ24075RGTR, TPS63001DRCR,
  MAX98357AETE+T, ICS-43434, MAX17048G+T10, TCA9555RTWR, SK6812MINI, HRO TYPE-C-31-M-12,
  C&K PCM12SMTR. Swap in equivalents in KiCad if any are out of stock.

## Routing tips for the next step

1. Start with the power loop around U3/L1/C4–C6: short, wide, on the back.
2. USB D+/D− as a pair from J1 → D1 → U1 (`USB` net class).
3. I2S lines to the mic and amp. Keep them away from L1 and the speaker traces.
4. LED chain: DIN→DOUT hops along the front between keys, with VLED and GND alongside.
5. Key switches → U7. 12 short runs on the back.
6. Re-pour, run DRC, then export a STEP (File → Export → STEP) and drop it
   into the enclosure model as a fit check.

## Bring-up checklist

1. With no battery, plug in USB. Check VIN, VSYS, PGOOD low, and 3V3.
2. Connect the battery. A USB meter should read ~750 mA charge, CHG low. Check that
   the power switch OFF kills VSYS on battery.
3. Flash over USB-C (hold BOOT, tap RESET the first time).
4. Firmware: check the `rms=` mic levels, a test tone, the LEDs, then an upload.
5. Measure deep-sleep current with the switch on TATTLE. Target < 100 µA.
