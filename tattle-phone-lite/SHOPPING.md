# Tattle Phone Lite: shopping list (self-assembly, starting from nothing)

Prices are rough (US, late 2026). Ask your EE friend first: anything marked **borrow?**
he probably owns.

## 1. Electronic parts

| What | Part | Where | Notes |
|---|---|---|---|
| Everything on the board (+ the 3 big buttons) | `hardware/fab/lcsc_order_3_boards.csv` | LCSC BOM Tool | Map **Quantity** + **LCSC Part Number**. ≈ $28 for 3 boards + spares. Can ship with the JLCPCB order |
| 3×AA holder, open, wire leads | **Keystone 2465** (DigiKey 2465K-ND) | DigiKey / Mouser | 58 × 48.3 × 17 mm: the case is sized for it. Eagle 12BH331 or Pololu #142 also fit (±0.3 mm) |
| Speaker, 28 mm 8 Ω 0.5 W | **Adafruit 1890** (has wires) or Challenge Electronics CS28-02W220-05-1X | Adafruit / DigiKey | Anything 28 mm × ≤ 5 mm, 8 Ω works |
| Batteries | 3 × AA per phone | anywhere | Alkaline or NiMH. **Not** lithium AA (1.8 V/cell is too much for the amp) |

## 2. Mechanical

| What | Qty per phone | Notes |
|---|---|---|
| M2.5 heat-set inserts, **CNC Kitchen "M2.5 Standard"** (M2.5 × 4.0) | 5 | The case holes are sized for these (3.7 mm). Other brands: change `INSERT_D` in `enclosure/tattle_lite.scad` |
| M2.5 × 20 screws (socket or pan head) | 4 | Case: from the back into the front-shell inserts |
| M2.5 × 6 **countersunk** screw (DIN 7991 / ISO 10642) | 1 | Battery door |
| Double-sided foam tape | a strip | Battery holder → inside of the door |
| Thin foam (2–3 mm) | a scrap | Mic gasket between the board and the front duct |
| Speaker cloth / thin felt | a scrap | Behind the grille slots |
| 26 AWG silicone wire, heat-shrink | a little | Only if the speaker leads are too short to reach the SPK pads |
| PETG filament (shell) + **translucent** PETG in yellow, green, red (caps) | | Caps glow with the LEDs behind them |

## 3. Tools

### Soldering

| Tool | Suggestion | ≈ | |
|---|---|---|---|
| Soldering iron | **Pinecil V2** + a 65 W USB-C PD charger, or Hakko FX-888DX | $30–110 | Pinecil also does heat-set inserts with an insert tip |
| Heat-set insert tip | Pinecil/TS100 M2.5 insert tip (or CNC Kitchen's tip set) | $10–15 | Or press inserts with a regular tip, carefully |
| **Hotplate** (reflow) | 200 × 200 mm preheater/reflow plate, e.g. UYUE 946C | $50–80 | **borrow?** The 30 × 30 mm Miniware MHP30 is too small for this 54 × 120 mm board |
| Hot-air station (rework) | YIHUA 959D / 8786D class | $60–100 | **borrow?** Only for fixing bridges/moving a part; not needed if the hotplate run goes well |
| Solder paste | **Chip Quik SMD291AX** (leaded Sn63/Pb37, easier) or **SMD291SNL** (lead-free), syringe | $15–25 | Keep it in the fridge. Lead-free is the cleaner choice for a toddler toy |
| Tacky flux | Chip Quik SMD291 or Amtech NC-559-V2-TF (beware fakes) | $10–15 | No-clean, so you don't have to wash (the mic must never be washed) |
| Solder wire | Kester 0.5–0.8 mm (63/37 or lead-free to match the paste) | $15–25 | |
| Solder wick | Chemtronics Chem-Wik 2 mm | $5 | For pulling bridges off the amp/USB-C pins |
| Paste squeegee | an old gift card works | $0 | Tape the board down beside scrap boards of the same thickness so the stencil sits flat |

### Handling & seeing

| Tool | Suggestion | ≈ | |
|---|---|---|---|
| ESD-safe tweezers, fine + curved | Vetus ST-15 / ST-12 | $10 | 0402 parts are 1 × 0.5 mm |
| Magnifier | 10× loupe (minimum) or a digital/stereo microscope | $10–150 | **borrow?** You need to see bridges on 0.5 mm pins |
| Flush cutters | Hakko CHP-170 | $8 | Clip the button legs flush |
| PCB holder | small vise or "helping hands" | $15 | |
| ESD mat + wrist strap | any | $20 | The ESP32 and mic are static-sensitive |
| Kapton tape | 10–20 mm roll | $8 | Masking, and taping over in-pad vias if wanted |
| Isopropyl alcohol 99% + brush | | $10 | Cleaning, but **keep it away from the mic** |
| Fan / fume extractor, safety glasses | | $15+ | |

### Testing

| Tool | Suggestion | ≈ | |
|---|---|---|---|
| Multimeter with continuity beep | AstroAI / Uni-T UT61E+ / Fluke 101 | $25–60 | Check 3V3 and VBAT for shorts to GND before first power |
| Bench power supply, current-limited | Korad KA3005D / Riden RD6006 class | $60–100 | **borrow?** First power-up at ~4.5 V, 200 mA limit |
| USB-C data cable | | | For flashing the firmware |

## 4. PCB order (JLCPCB)

2 layers · 1.6 mm · 54.4 × 120.5 mm · min via **0.3 mm** · OSP (build within a few weeks) ·
via covering Tented/Untented (self-assembly) · **stencil: bottom only**, ~0.12 mm.
Upload `hardware/fab/tattle_lite_gerbers.zip`.
