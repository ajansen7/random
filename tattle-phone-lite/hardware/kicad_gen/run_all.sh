#!/bin/sh
# Regenerate the whole Lite board from design.py + ../layout.py:
#   build (schematic + placed board) -> route (FreeRouting) -> stitch (GND vias)
#   -> check (netlist + DRC) -> make_fab (Gerbers, drill, JLC BOM/CPL)
# This OVERWRITES ../kicad/tattle_lite.kicad_pcb. Once you hand-edit the board in KiCad,
# run only check.py and make_fab.py.
# macOS paths below; on Linux point K at your KiCad install (or use system python3 + kicad-cli).
set -e
cd "$(dirname "$0")"
K=${KICAD_APP:-/Applications/KiCad/KiCad.app/Contents}
export KICAD_SYMBOL_DIR=${KICAD_SYMBOL_DIR:-$K/SharedSupport/symbols}
export KICAD_FOOTPRINT_DIR=${KICAD_FOOTPRINT_DIR:-$K/SharedSupport/footprints}
export PATH="$K/MacOS:$PATH"
export FREEROUTING_CP=${FREEROUTING_CP:-$HOME/tools/freerouting/freerouting-2.4.1.jar}
export JAVA=${JAVA:-/opt/homebrew/opt/openjdk@25/bin/java}
PY=${PY:-$K/Frameworks/Python.framework/Versions/Current/bin/python3}
$PY build.py
$PY route.py 30
$PY stitch.py 4.0
$PY check.py
$PY make_fab.py
