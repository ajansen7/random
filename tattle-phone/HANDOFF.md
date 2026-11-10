# Tattle Phone: project handoff

Read this first when picking the project up in a new session. It covers what the
project is, every decision made so far and why, the exact state of each piece, how to
run the tooling, the traps that already cost time, and what to do next.

Last updated: 2026-11-10 (end of the first cloud session).

---

## 1. What this is

A toddler "tattle phone": a 3D-printed, 90s-cordless-style handset. The toddler presses
a big **CALL** button. It rings and beeps, records what they say, and uploads the
recording to Google Drive, which emails the parent. Later: keypad sounds and lights,
and **two phones that can call each other** on the home Wi-Fi.

The owner is a hobbyist: comfortable with light through-hole soldering, **new to
KiCad and PCB design**, and has OpenSCAD and KiCad 10 installed locally. Explain
tools step by step, and prefer doing things *for* them (scripts, generated files) over
asking them to learn deep tool features.

## 2. Decisions made (and why)

| Topic | Decision | Why / notes |
|---|---|---|
| MCU | ESP32-S3-WROOM-1-**N16R8** module | Wi-Fi + BLE built in (pre-certified, PCB antenna), 2× I2S, 8 MB PSRAM holds a 60 s recording, native USB for flashing over the same USB-C |
| Radios | Only the module's Wi-Fi (2.4 GHz b/g/n) + Bluetooth LE (shared antenna) | BLE unused for now (could do phone-based setup). ESP-NOW (router-free intercom) uses the same radio |
| Storage | On-board flash, FAT partition, as an **outbox** | Recordings queue there if Wi-Fi is down. No SD card |
| Mic | ICS-43434 I2S MEMS, **bottom of the phone** (mouthpiece), back of PCB, port hole through board | Separate from the speaker at the top (earpiece) |
| Speaker | 36 mm 8 Ω via MAX98357A (GAIN to VDD = 6 dB) | Firmware also caps the volume |
| Charger | **BQ24075** (not BQ24074) | The 75 has **SYSOFF**, used by the hard power switch |
| 3.3 V | TPS63001 buck-boost | |
| Gauge | MAX17048 (custom symbol in `hardware/kicad/tattle.kicad_sym`) | |
| Keypad | 12 tact switches + **TCA9555RTWR** (QFN) I2C expander | QFN because a TSSOP collided with the switch legs |
| LEDs | 14× SK6812MINI on the **front**: 0 = CALL, 1–12 = keys, **13 = status ("in use") light** | Status light requested so you can tell when a tattle or call is live |
| Switches | **POWER** slide (right edge, SYSOFF) + **TATTLE/PLAY** slide (left edge, GPIO38) | TATTLE = deep sleep, CALL only. PLAY = stay on Wi-Fi for intercom. Small battery is fine |
| Buttons | **One CALL button** (call/answer/hang up) | Owner's choice. No separate END key |
| Upload | Google Apps Script web app to Drive + email | Owner is happy with it. Fallback: their always-on home computer |
| Keycaps | **12 separate captive caps** (flange under the front wall) | Replaced the printed flex-hinge web: simpler and kid-proof |
| Speaker grille | 5 smooth flush slots, cloth glued behind | Replaced the hex holes + dish, which owner disliked ("nooks and crannies") |
| Assembly | Factory assembles **both sides**; owner hand-solders only the 13 switches + 3 JST connectors | Owner is a novice solderer |

## 3. Where everything is

```
tattle-phone/
├── HANDOFF.md            ← this file
├── CLAUDE.md             ← short pointer for Claude Code sessions
├── README.md             ← design doc (architecture, BOM, power, enclosure, intercom §8)
├── ORDERING.md           ← next steps, shopping list, fab/assembly instructions
├── firmware/tattle_phone/  Arduino sketch (ESP32 core 3.x). pins.h = pin map for breadboard AND PCB
├── cloud/drive_upload.gs   Apps Script receiver (Drive + email)
├── enclosure/            OpenSCAD model (tattle_phone.scad), Makefile, stl/, renders/
└── hardware/
    ├── README.md          PCB overview, check results, ordering notes
    ├── KICAD_GUIDE.md     KiCad for first-timers + the 6 open connections, one by one with pictures
    ├── layout.py          SHARED dimensions for board AND case (run it to regenerate enclosure/layout.scad)
    ├── place_footprints.py KiCad scripting-console helper (KiCad 8/9/10)
    ├── kicad/             THE KiCad project (open tattle_phone.kicad_pro in KiCad 10)
    ├── kicad_gen/         circuit-as-code + tools (see §5)
    └── renders/           pcb_front/back.png, fixes/*.png close-ups
```

Git: repo `ajansen7/random`. Work happened on branch `ccr-84675764-9oidoz`, and everything
is merged to **`main`**. Pull `main` locally.

Review page for a friend (claude.ai artifact, private until shared):
<https://claude.ai/artifact/QF1NFQog3PrdwUX1PF7Vfk>. ⚠️ It shows the **first** enclosure
design (flex keypad, hex grille, no status light or mode switch). Update its renders if it's
going to be shared again.

## 4. Status of each piece

| Piece | State | Next step |
|---|---|---|
| Firmware | Phase 1 written: ring → beep → record (silence auto-stop) → WAV outbox → upload with retry → deep sleep. Status light + TATTLE/PLAY mode added. **Never compiled or run.** | Breadboard bring-up on an ESP32-S3 DevKit (N16R8); tune `SPEECH_RMS` / `MIC_SHIFT` |
| Apps Script | Written, not deployed | Deploy as web app (steps in the file header) |
| Enclosure | v2 model + STLs. Overlap check clean between all parts | PLA test print. Check fit, key feel, battery pocket |
| PCB schematic | Generated, one sheet, net-label style (every pin labelled). Netlist matches board on all 377 pins | Optional: tidy into wires |
| PCB board | **Routed except 6 connections** (VBAT at U2 pin 2; GND at SW12 ×2, U8 pin 3, Q2 pin 2, D1 pin 2). DRC otherwise clean (only expected warnings: 12 LED/switch courtyard overlaps, silkscreen, lib links) | Owner finishing the 6 in KiCad 10 using `hardware/KICAD_GUIDE.md` §4b. Then: DRC to 0 → `python3 hardware/kicad_gen/make_fab.py` → order |
| Fab files | Generator written (`make_fab.py`: Gerbers+drill zip, JLC BOM/CPL), **not yet run on a finished board** | Run after routing is complete. LCSC part numbers are blank and need filling in |
| Intercom | Designed (README §8: mDNS discovery, UDP audio, half-duplex v1, play-hours power plan). Not implemented | Later phase |

## 5. Tooling: how it works and how to run it

### Enclosure (OpenSCAD)
- Edit dimensions in `hardware/layout.py`, run `python3 hardware/layout.py`. That writes
  `enclosure/layout.scad`, which both the case and the KiCad tools use.
- `cd enclosure && make` builds the STLs + `pcb_outline.dxf`. `make renders` builds the PNGs
  (uses `xvfb-run` if headless).
- Open `tattle_phone.scad`, then F5 (colored preview) or F6 (full render: correct
  geometry, but OpenSCAD's own yellow/teal colors). `part = "closed"` at the top picks the view.
- Engraved text looks speckled in F5 preview. That's a preview artifact, not a model problem.

### PCB pipeline (`hardware/kicad_gen/`), needs KiCad's Python module (`pcbnew`)
| Script | Does | ⚠️ |
|---|---|---|
| `design.py` | Every part: lib symbol, footprint, pin→net, position (case coords), side | Master for the *circuit* |
| `build.py` | Generates schematic + **unrouted** board + project from design.py/layout.py | **Overwrites the routed board.** Only use to start over |
| `check.py` | Exports schematic netlist (kicad-cli) and compares it pad by pad with the board, then a grouped DRC | Safe any time |
| `route.py` | FreeRouting autoroute (`FREEROUTING_CP`, `JAVA` env vars; `ROUTE_GND=1` to include ground; `--continue`) | Needs FreeRouting 2.4.1 + Java 25 |
| `fixup.py` | A* router for leftovers, GND island joining (raster connectivity), fiducial placement. `--cleanup` = remove duplicate/dangling **vias only** | |
| `ripup.py` | Rip-up-and-reroute experiment | **Made things worse** on this board (22 open). Kept for reference |
| `tidy.py` | Silkscreen label placement | Called by build.py |
| `bom.py`, `make_fab.py` | BOM CSV; Gerbers/drill zip + JLC BOM/CPL into `hardware/fab/` | |

Now that the board has been hand-edited in KiCad, **the KiCad files are the master copy**.
Don't re-run `build.py` / `route.py` on them unless deliberately starting over.

### Getting FreeRouting (if ever needed again)
It's on Maven Central as `app.freerouting:freerouting:2.4.1` (needs **Java 25**). Easiest
on a desktop: KiCad's **Plugin and Content Manager → Freerouting** plugin. For the scripts,
fetch it with Maven `dependency:copy-dependencies` into a `lib/` folder and set
`FREEROUTING_CP="/path/lib/*"`.

## 6. Traps that already cost time (read before scripting KiCad)

1. **KiCad 7 Python: `board.Remove()` corrupts the SWIG bindings** for the rest of the process.
   Later calls return raw `SwigPyObject`s ("has no attribute ..."). Do all removals **last**
   in a run, or in a separate process, then save. (Unknown whether KiCad 10 has the same issue.)
2. **Flipping footprints to the back:** `Flip()` mirrors top/bottom **and** turns 180°. Set
   orientation to `rot + 180` afterwards, or the part ends up upside down (it did:
   antenna pointing into the keypad). Fixed in `build.py` and `place_footprints.py`.
3. `pcbnew.ImportSpecctraSES` only works inside the GUI in KiCad 7. `route.py` has its own SES importer.
4. Load the board **with its .kicad_pro** next to it, or net classes (power 0.4 mm, USB) are
   silently ignored by DSN export.
5. Ground pours need **0.2 mm clearance, 0.15 mm min width, solid pad connections** to reach
   0.5 mm-pitch QFN pins. Thermal/0.3 mm left dozens of pins stranded.
6. Never auto-delete "dangling" **tracks**: FreeRouting leaves short segments that are real
   connections. Deleting them broke ILIM/ISET/KEY_6 once.
7. DRC reports pour islands at the zone's *corner*, not the island. Use
   `fixup.gnd_components()` (raster labelling) to find where they really are.
8. kicad-cli SVG exports: board coordinates map as `svg = kicad − (65.0, 60.975)` mm for this
   board (calibrated from the mounting holes). Used to make the fix close-ups.
9. OpenSCAD 2021: `--render` PNGs drop colors. An SVG embedded via `<image>` won't load
   inside another SVG (inline it instead).
10. The cloud environment's safety check blocks running downloaded code (FreeRouting)
    unless the owner explicitly allows it. Locally that's not an issue.

## 7. What was tried for the last 6 connections (so nobody repeats it)

| Attempt | Result |
|---|---|
| FreeRouting, GND excluded (pours) | 3 signals + ~70 GND pads open |
| Own A* router with neck-down | Fixed 2 signals; GND stuck |
| Via-stitching every pour island | Made via chains that only linked islands to each other. Abandoned |
| Raster GND connectivity + join to main piece | 73 → 6 open ✅ (the board on `main`) |
| Rip-up and reroute (1.0–1.5 mm) | 6 → 22 open ❌ |
| Spread charger resistors, move D1, full re-route **with GND as a net** | All signals OK but 10 GND open + new errors ❌ (not kept) |

Conclusion: the remaining 6 need shove-routing. KiCad 10's interactive router does it in
minutes. Guide: `hardware/KICAD_GUIDE.md` §4b.

## 8. Next steps, in order

1. **Owner:** finish the 6 connections in KiCad 10 (guide §4b), DRC → 0 unconnected, save, commit.
2. **Claude:** run `check.py`, then `make_fab.py`. Help fill LCSC part numbers (check stock:
   ESP32-S3-WROOM-1-N16R8, BQ24075RGTR, TPS63001DRCR, MAX98357AETE+T, ICS-43434, MAX17048G+T10,
   TCA9555RTWR, SK6812MINI, HRO TYPE-C-31-M-12, C&K PCM12SMTR). Export a STEP for a case fit check.
3. **Owner:** order breadboard parts + battery, speaker, NTC, screws, inserts (ORDERING.md), and
   test-print the case in PLA.
4. Breadboard firmware bring-up (it has never been compiled). Deploy the Apps Script.
5. Order PCBs with two-sided assembly (ORDERING.md lists what to tell the fab, and to check their 3D
   preview for rotations).
6. Later: keypad sounds, then intercom firmware (README §8).

## 9. Open questions / things to verify

- Component values (charger ISET/ILIM, TMR left open, amp gain) come from datasheets and memory.
  Verify them before ordering.
- BQ24075 OUT regulation level: confirm it's ≤ 5.5 V for TPS63001/MAX98357A.
- The SK6812MINI LED sits beside each switch, and the bodies clear by ~0.25 mm. Check against the
  real parts.
- USB-C (HRO, top-mount on the back) is ~2.5 mm above the case's midline. The case slot follows it.
- Battery polarity on JST leads varies by seller.
