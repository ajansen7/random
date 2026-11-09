# Tattle Phone: next steps and ordering

## Where things stand

| Piece | Ready to order? | What's missing |
|---|---|---|
| **PCB** | **Not yet** | The board is placed and checked but **not routed** (no copper traces). The fab needs Gerbers, and Gerbers from an unrouted board would be a blank board. |
| **Enclosure** | **Yes, for a test print** | STLs are in `enclosure/stl/`. Print one in cheap PLA first to check fit and feel. Do the final PETG print once you have a real board to test-fit. |
| **Parts you buy yourself** | **Yes** | See the shopping list below. Order these now; they're needed for the breadboard prototype and the final build either way. |

## Recommended order of work

1. **Breadboard prototype (now, ~$45).** ESP32-S3 DevKit + mic + amp breakouts,
   wired per `firmware/tattle_phone/pins.h`. This proves the firmware, audio quality
   and Google Drive upload before any money goes into boards. See README §7.
2. **Test-print the case in PLA** (now). Check the overall size in a toddler's hands,
   how the keycaps feel, and whether the screws, inserts and the battery pocket fit.
3. **Route the PCB** in KiCad 10 (`hardware/kicad/`), or ask me to take a pass.
   Then export Gerbers + drill files, the BOM and the placement (CPL) file.
4. **Order 5 PCBs with assembly** (instructions below). Turnaround is about 1–2 weeks.
5. **Solder the through-hole parts** (13 key switches, 3 JST connectors), flash, and run
   the bring-up checklist in `hardware/README.md`.
6. **Final print** in PETG, assemble, and hand it over.

## Shopping list: things you order yourself

These aren't on the PCB order.

**For the breadboard prototype**
- ESP32-S3-DevKitC-1 **N16R8** (it must be the N16R8 variant)
- INMP441 I2S microphone breakout
- MAX98357A I2S amplifier breakout
- 36 mm 8 Ω 1 W speaker (or 40 mm for the breadboard)
- A few WS2812/SK6812 LEDs or a short strip, a big push button, jumper wires, breadboard

**For each finished phone**
- **LiPo battery:** 1S 3.7 V, **with a built-in protection circuit**, about 34 × 50 × 10 mm
  or smaller (e.g. "103450", 1500–2000 mAh; 803040 or similar is fine if you want smaller).
  It needs a **JST-PH 2.0 mm 2-pin** plug. ⚠️ Polarity varies by seller: check red/black
  against the + mark on the board before plugging in.
- **Speaker:** 36 mm, 8 Ω, 1 W, with leads (crimp a JST-PH 2-pin plug on, or buy one pre-wired).
- **10k NTC thermistor** (β 3435, e.g. "MF52 103 3435") with leads + a JST-PH 2-pin plug.
  Tape it to the battery so the charger can stop if the cell gets hot.
- **4× M2.5 × 18 mm screws** (Torx or tri-wing heads to keep small hands out) and **4× M2.5 heat-set
  inserts** (≈ 3.6 mm OD × 4 mm long).
- Scrap of **speaker cloth or fine mesh** for behind the speaker slots, a little **thin foam**
  for the battery pocket, and a small **felt or foam ring** for the mic duct.
- **Filament:** PETG for the shell (any colour), **translucent/natural PETG** for the keys
  and the status lens, green PETG for the CALL button. Use PLA for the test print.
- A soldering iron with a fine-ish tip (also used to press in the heat-set inserts), solder, flux.

## PCB order: what to choose and what to tell them

(Written for JLCPCB, but any PCB-plus-assembly service works the same way.)

**Board options**
- Layers: **2** · Thickness: **1.6 mm** · Material: FR-4 · Surface finish: **lead-free HASL or ENIG**
- Min hole size: **0.2 mm** (there are tiny thermal vias under the ESP32 and the 3.3 V regulator)
- Quantity: 5 (the minimum, and spares are handy)

**Assembly options**
- Assembly type: **Standard** (the economy service won't do both sides)
- Assemble **both sides**: LEDs on the top/front, everything else on the bottom/back
- Upload: Gerbers zip, **BOM** (`hardware/kicad/bom.csv`, after you fill in the LCSC part
  numbers in KiCad or on their site), and the **CPL/placement file** (KiCad: *File → Fabrication
  Outputs → Component Placement*)
- **Leave off:** R5 (marked "DNP" / "DO NOT FIT"), and the through-hole parts SW1–SW13 and
  J2/J3/J4, unless you pay them to fit those too
- **Check their 3D preview before paying:** confirm the rotation of the LEDs, the ESP32 module
  (antenna tab at the top edge, hanging over nothing), the USB-C port (opening at the bottom edge),
  and the two slide switches (levers sticking out of the left and right edges). Wrong rotations
  are the most common assembly mistake, and the preview is where you catch them.

**Parts that may need special handling**
- **ESP32-S3-WROOM-1-N16R8**, **ICS-43434 mic**, **MAX17048** and **BQ24075** are sometimes
  "extended" or out-of-stock parts. Extended parts cost a small fee each. If one is out of stock,
  use their global sourcing, or I can swap in an equivalent part and regenerate the board.
- Expect roughly **$60–120 for 5 assembled boards** plus shipping, mostly setup fees for
  two-sided assembly and the extended parts. Per-board cost drops fast at higher quantities.

## Enclosure print notes

| Part | Material | Orientation | Settings |
|---|---|---|---|
| `front.stl` | PETG | face-down (as exported) | 0.2 mm layers, 4 walls, 30 % infill, brim |
| `back.stl` | PETG | back-down | tree supports **only** under the antenna nub |
| `keypad.stl` (all 12 keys) | translucent PETG | flanges down | 0.12–0.16 mm layers so the numbers come out crisp |
| `call_cap.stl` | green PETG | flange down | |
| `status_lens.stl` | translucent PETG | as exported | tiny part; print it with the keypad |
| `speaker_cup.stl` | anything | open side up | |

Assembly order: press in the heat-set inserts, glue cloth behind the speaker slots,
seat the speaker and glue on its cup, drop the keycaps, CALL cap and lens into the front
shell, lay the board on top, plug in the battery and speaker, close it up with the back shell, then
fit the screws.
