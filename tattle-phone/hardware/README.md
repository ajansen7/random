# Tattle Phone: PCB design guide

This is the schematic, block by block, plus how to lay the board out in KiCad so
it fits the printed case on the first try.

**Source of truth for anything mechanical:** [`layout.py`](layout.py).
Positions of keys, LEDs, the CALL switch, mic, screw holes and the board
outline live there. The case (`../enclosure/`) and the KiCad helper
(`place_footprints.py`) both read it.

> ⚠️ Values below are good starting points from datasheets and common reference
> designs (Adafruit's BQ24074 and MAX98357A boards are worth opening alongside).
> Check every value against the current datasheet before ordering. This hasn't
> been built yet.

---

## 1. Workflow

1. **KiCad 8 or 9.** Create the project in `hardware/kicad/`.
2. **Schematic.** Draw it from §2, using the reference designators in §3 (the
   placement script finds parts by these).
3. **Assign footprints** (§3), then *Update PCB from Schematic*.
4. **PCB Editor → Tools → Scripting Console**:
   ```python
   import sys; sys.path.insert(0, r"/path/to/tattle-phone/hardware")
   import place_footprints; place_footprints.run()
   ```
   This draws the Edge.Cuts outline and locks the 12 keys, 13 LEDs, CALL switch,
   mic and 4 mounting holes where the case needs them. (`enclosure/pcb_outline.dxf`
   is the same outline, if you'd rather import it.)
5. Place the rest by hand using §4. Route, run DRC, then export a STEP file and drop
   it into the OpenSCAD/Fusion model as a fit check.
6. Order 5 boards from JLCPCB with assembly on the **back side only** (§4 explains
   why). Hand-solder the through-hole switches and the JST connectors.

---

## 2. Schematic by block

### 2.1 USB-C input
| Net / pin | Connects to |
|---|---|
| VBUS (A4, A9, B4, B9) | → F1 PTC fuse (0.75–1 A hold) → `VIN` |
| CC1 (A5) | **5.1 kΩ → GND** (its own resistor) |
| CC2 (B5) | **5.1 kΩ → GND** (its own resistor, never shared with CC1) |
| D+ (A6, B6 tied) | → D1 USBLC6-2SC6 → ESP32 **GPIO20** |
| D− (A7, B7 tied) | → D1 USBLC6-2SC6 → ESP32 **GPIO19** |
| GND, shell | GND (shell through 1 MΩ ∥ 4.7 nF if you want to be fancy, or straight to GND) |

Add a TVS on `VIN` (e.g. SMF5.0A). Toddlers plug in mystery chargers.

### 2.2 Charger and power path: U2 BQ24074
| Pin | Connection | Note |
|---|---|---|
| IN | `VIN`, 1 µF + 10 µF to GND | |
| OUT | `VSYS`, 4.7–10 µF to GND | Powers everything. It runs from USB when plugged in, otherwise from the battery |
| BAT | `VBAT` (J2 battery +), 4.7 µF | |
| ISET | R → GND | I_chg ≈ 890 / R. **1.2 kΩ → ~750 mA** (0.5C for 1500 mAh) |
| ILIM | R → GND | I_in ≈ 1550 / R. **1.5 kΩ → ~1 A** |
| EN2 / EN1 | EN2 = high (to IN via 10 kΩ), EN1 = GND | Selects "use ILIM resistor" mode |
| CE | GND | Charging always enabled |
| TS | 10 kΩ NTC (103AT, β 3435) to GND | **Glue the NTC to the cell.** Charging stops if the cell is too hot or cold |
| TMR | per datasheet table | Sets the safety timer. Don't disable it |
| SYSOFF | **SW14 slide switch**: ON → GND, OFF → VBAT | OFF disconnects the battery from VSYS. That's the real "off" switch, and USB charging still works |
| CHG | 10 kΩ pull-up to 3V3 → ESP32 **GPIO12** | Low while charging |
| PGOOD | 10 kΩ pull-up to 3V3 → ESP32 **GPIO13** | Low when USB power is present |

### 2.3 3.3 V rail: U3 TPS63001 (fixed 3.3 V buck-boost)
| Pin | Connection |
|---|---|
| VIN, VINA | `VSYS`, 10 µF + 0.1 µF |
| L1, L2 | 1.5 µH, ≥ 2 A saturation, low DCR (e.g. XAL4020-152 class) |
| VOUT, FB | `3V3`, 2 × 22 µF. On the fixed version FB ties to VOUT |
| EN | `VSYS` |
| PS/SYNC | GND (power-save mode, best for battery life) |
| GND, PGND | GND, short and wide |

Follow the datasheet's layout figure closely. Buck-boost layout matters.

### 2.4 ESP32-S3: U1 ESP32-S3-WROOM-1-**N16R8**
| Pin | Connection |
|---|---|
| 3V3 | `3V3`: 22 µF + 0.1 µF right at the pin, plus 100 µF bulk nearby for Wi-Fi bursts |
| EN | 10 kΩ to 3V3 + 1 µF to GND; RESET button (SW15) to GND |
| GPIO0 | 10 kΩ to 3V3; BOOT button (SW16) to GND |
| GPIO19 / 20 | USB D− / D+ (§2.1). Flashing and serial over the same USB-C port |
| GPIO 4, 5, 6 | Mic BCLK, WS, SD |
| GPIO 15, 16, 17, 18 | Amp BCLK, LRC, DIN, SD_MODE |
| GPIO 7 | CALL switch SW13 to GND, plus an external 10 kΩ pull-up (for reliable deep-sleep wake) |
| GPIO 8 | LED data → level shifter (§2.7) |
| GPIO 11 | LED power enable (§2.7) |
| GPIO 9 / 10 | I2C SDA / SCL, 4.7 kΩ pull-ups to 3V3 |
| GPIO 12 / 13 | Charger CHG / PGOOD |
| GPIO 14 | Fuel-gauge ALRT (optional) |
| GPIO 21 | Keypad expander INT (an RTC pin, so a key press can wake the phone) |
| GPIO 1 | Spare ADC (battery divider if you skip the gauge) |
| Test pads | TX0 (GPIO43) / RX0 (GPIO44) + GND as a backup serial port |

Avoid GPIO 0/3/45/46 (strapping), 19/20 (USB) and 35–37 (used by the octal PSRAM).
This matches `firmware/tattle_phone/pins.h`.

### 2.5 Microphone: MK1 ICS-43434 (bottom port, mounted on the **back**)
- VDD → 3V3 with 0.1 µF. L/R → GND (left channel). SCK/WS/SD → GPIO 4/5/6.
- 100 kΩ pull-down on SD (check the datasheet note).
- **Port hole:** 0.8–1.0 mm plated or NPTH hole through the board under the
  sound port, so the mic hears through the front case. Most footprints include it; check.

### 2.6 Speaker amp: U4 MAX98357A
- VDD → **VSYS** (not 3V3: louder and keeps 3V3 clean), 10 µF + 0.1 µF.
- BCLK/LRCLK/DIN → GPIO 15/16/17. SD_MODE → GPIO18 (high = on, left channel).
- GAIN: tie to VDD for 6 dB (quiet, toddler-friendly). Leave it floating for 9 dB.
  Firmware also caps volume.
- OUTP/OUTN → J3 (JST-PH 2-pin) → 36 mm 8 Ω speaker. Keep the traces short.
  Optional ferrite beads + 1 nF if you see EMI.

### 2.7 LEDs: LED1–LED13 SK6812MINI-E (reverse-mount, back side)
- **Power gate:** Q1 AO3401 P-FET, source = VSYS, drain = `VLED`. Gate gets 100 kΩ
  to VSYS (off by default) and Q2 2N7002 from gate to GND, with Q2's gate on
  **GPIO11** (100 kΩ pull-down). GPIO11 high = LEDs on. Unpowered WS2812-type
  LEDs still draw ~1 mA each, which would drain the battery in sleep.
- **Level shift:** U8 SN74AHCT1G125 powered from VLED, input GPIO8, output → 330 Ω → LED1 DIN.
  At 4–5 V the LEDs don't reliably see 3.3 V as "high".
- Chain order: **LED1 = CALL**, then LED2…LED13 = keys 1 2 3 4 5 6 7 8 9 * 0 #
  (this matches the firmware's index 0..12).
- 0.1 µF per LED, plus 10 µF on VLED.

### 2.8 Keypad: SW1–SW12 + U7 PCA9555 (I2C 0x20)
- Each key goes between an expander pin and GND. The PCA9555 has internal pull-ups.
- A0–A2 → GND. INT → GPIO21 (10 kΩ pull-up). VDD → 3V3.
- SW1…SW12 = 1 2 3 4 5 6 7 8 9 * 0 #, on IO0_0…IO1_3 in that order.
- Spare expander pins: break them out to a test-point header for later ideas
  (vibration motor, hook sensor, second button).

### 2.9 Battery gauge: U6 MAX17048 (I2C 0x36)
- CELL and VDD → VBAT, GND, SDA/SCL on the shared bus, ALRT → GPIO14, QSTRT → GND.

### 2.10 Connectors and mechanical
- J2: JST-PH 2-pin battery (**check the polarity of your cell's lead**, since it varies by vendor).
- J3: JST-PH 2-pin speaker.
- H1–H4: M2.5 mounting holes, 2.7 mm. The case screws pass through these and clamp the board.
- Fiducials ×3 on the back side for assembly.

---

## 3. Reference designators and footprints

| Ref | Part | Footprint (KiCad lib) | Side |
|---|---|---|---|
| U1 | ESP32-S3-WROOM-1-N16R8 | `RF_Module:ESP32-S3-WROOM-1` | back |
| U2 | BQ24074RGT | `Package_DFN_QFN:VQFN-16-1EP_3x3mm_P0.5mm_EP1.6x1.6mm` | back |
| U3 | TPS63001DRC | `Package_SON:VSON-10-1EP_3x3mm_P0.5mm_EP1.65x2.4mm` | back |
| U4 | MAX98357AETE+ | `Package_DFN_QFN:TQFN-16-1EP_3x3mm_P0.5mm_EP1.23x1.23mm` | back |
| U6 | MAX17048G+ | `Package_DFN_QFN:TDFN-8-1EP_2x2mm_P0.5mm_EP0.8x1.2mm` | back |
| U7 | PCA9555PW | `Package_SO:TSSOP-24_4.4x7.8mm_P0.65mm` | back |
| U8 | SN74AHCT1G125 | `Package_TO_SOT_SMD:SOT-23-5` | back |
| MK1 | ICS-43434 | vendor footprint (SnapEDA/LCSC) with port hole | back |
| J1 | USB-C 16-pin **mid-mount** receptacle | vendor footprint (board-edge cutout) | back |
| SW1–12 | 6×6 mm THT tact, 4.3 mm tall | `Button_Switch_THT:SW_PUSH_6mm_H4.3mm` | front |
| SW13 | 12×12 mm THT tact, 4.3 mm body | `Button_Switch_THT:SW_PUSH_12mm` (check height) | front |
| SW14 | SPDT side-actuated slide (e.g. MSK-12C02) | vendor | back, right edge |
| SW15/16 | small SMD tact (RESET/BOOT), internal only | `Button_Switch_SMD:SW_SPST_TL3342` | back |
| LED1–13 | SK6812MINI-E | `LED_SMD:LED_SK6812MINI-E_...` (reverse-mount) | back |
| H1–4 | M2.5 hole | `MountingHole:MountingHole_2.7mm_M2.5` | — |

*Footprint names drift between KiCad versions. Search the library browser if one isn't found.*

---

## 4. Layout

```
   case y (mm)      FRONT of board                   BACK of board
   ───────────      ──────────────                   ─────────────
   124  ┌──────────────────────────┐   ← antenna end of U1 flush with this edge,
        │                          │     NO copper (either layer) under the antenna
   108  │ H3                    H4 │     U1 ESP32 module (y ≈ 98.5–124)
    94  │  [1]    [2]    [3]       │     U3 buck-boost, bulk caps beside U1
    81  │  [4]    [5]    [6]       │     LEDs (reverse-mount) +4 mm right of each key
    68  │  [7]    [8]    [9]       │     U7 expander, U6 gauge, U2 charger
    55  │  [*]    [0]    [#]       │     (battery sits under here: ≤ 6 mm parts only)
    34  │      ( CALL )            │     SW14 slide switch on right edge at y = 64
    22  │ H1                    H2 │     U4 amp + J3 speaker connector (wires run up the side)
    14  │          o mic           │     MK1 + port hole
   3.6  └──────[ USB-C ]───────────┘     J1 mid-mount, J2 battery connector nearby
```

- **Front side has only the 13 tact switches** (through-hole, hand-soldered).
  Everything SMD goes on the back, so you pay for single-sided assembly. The
  reverse-mount LEDs shine forward through holes in the board into the
  translucent keycaps.
- **Height limits on the back** (the case model assumes these): under the battery
  area (y ≈ 29–81) parts must be ≤ 6 mm. The ESP32 module (3.1 mm) and everything
  else should stay under 6 mm too. Put the JST connectors outside the battery footprint.
- **Switch orientation:** keys are rotated 90° so their legs straddle the LED
  in y. The placement script handles this.
- **Ground:** pour GND on both layers, stitch with vias, and keep the buck-boost loop
  tight. Keep I2S lines away from the inductor and the amp outputs.
- **Antenna:** keep the speaker magnet, battery and screws as far from the module's
  antenna end as the case allows (that's why the speaker sits above the board).

## 5. Bring-up checklist
1. Before populating the ESP32 (or with the 3V3 jumper open), plug in USB and check
   VBUS, VSYS ≈ 4.4–5 V and PGOOD low.
2. Connect the battery and check the charge current with a USB meter (~750 mA), that CHG is low,
   and that SW14 OFF kills VSYS on battery.
3. Check 3V3 ripple under a Wi-Fi load (scope if you have one).
4. Flash over USB-C (hold BOOT, tap RESET the first time).
5. Run the firmware and check the `rms=` mic levels, a test tone, then upload.
6. Measure deep-sleep current. The target is < 100 µA total.
