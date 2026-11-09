"""
Autoroute the board with FreeRouting.

    FREEROUTING_CP="/path/to/freerouting/lib/*" JAVA=/path/to/java python3 route.py [passes]

Exports a Specctra DSN from ../kicad/tattle_phone.kicad_pcb, leaves GND out
(both copper layers are GND pours), runs FreeRouting headless, imports the
session back and refills the pours. The routed board is saved as
tattle_phone.kicad_pcb (the unrouted one is kept as tattle_phone.unrouted.kicad_pcb).
"""
import os
import re
import shutil
import subprocess
import sys

import pcbnew

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.join(HERE, "..", "kicad")
PCB = os.path.join(KDIR, "tattle_phone.kicad_pcb")
DSN = os.path.join(KDIR, "tattle_phone.dsn")
SES = os.path.join(KDIR, "tattle_phone.ses")
SKIP_NETS = ["GND"]


def strip_nets(dsn_path, nets):
    """Drop nets from the DSN network section so the router leaves them alone."""
    txt = open(dsn_path).read()
    for n in nets:
        txt = re.sub(r'\(net "?%s"?\s+\(pins[^)]*\)\s*\)' % re.escape(n), "", txt)
        txt = re.sub(r'(\(class [^\n]*?)\s"?%s"?(?=[\s)])' % re.escape(n), r"\1", txt)
    open(dsn_path, "w").write(txt)


def main():
    passes = sys.argv[1] if len(sys.argv) > 1 else "40"
    unrouted = PCB.replace(".kicad_pcb", ".unrouted.kicad_pcb")
    if not os.path.exists(unrouted):
        shutil.copy(PCB, unrouted)
    board = pcbnew.LoadBoard(unrouted)
    if not pcbnew.ExportSpecctraDSN(board, DSN):
        raise SystemExit("DSN export failed")
    strip_nets(DSN, SKIP_NETS)

    java = os.environ.get("JAVA", "java")
    cp = os.environ["FREEROUTING_CP"]
    cmd = [java, "-cp", cp, "app.freerouting.Freerouting", "-de", DSN, "-do", SES, "-mp", passes,
           "--gui.enabled=false", "--usage_and_diagnostic_data.disable_analytics=true"]
    print(" ".join(cmd[:1] + ["..."] + cmd[3:]))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    log = (r.stdout + r.stderr).splitlines()
    for line in log:
        if any(k in line for k in ("unrouted", "Auto-routing", "completed", "ERROR", "Saving", "pass #")):
            print("  ", line[-160:])
    if not os.path.exists(SES):
        raise SystemExit("FreeRouting produced no session file")

    board = pcbnew.LoadBoard(unrouted)
    if not pcbnew.ImportSpecctraSES(board, SES):
        raise SystemExit("SES import failed")
    board.BuildConnectivity()
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    board.Save(PCB)
    tracks = [t for t in board.GetTracks() if t.GetClass() == "PCB_TRACK"]
    vias = [t for t in board.GetTracks() if t.GetClass() == "PCB_VIA"]
    print(f"imported: {len(tracks)} track segments, {len(vias)} vias")


if __name__ == "__main__":
    main()
