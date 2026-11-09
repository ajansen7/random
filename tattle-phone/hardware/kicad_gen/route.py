"""
Autoroute the board with FreeRouting.

    FREEROUTING_CP="/path/to/freerouting/lib/*" JAVA=/path/to/java python3 route.py [passes]

Exports a Specctra DSN from ../kicad/tattle_phone.kicad_pcb, leaves GND out
(both copper layers are GND pours), runs FreeRouting headless, imports the
session back and refills the pours. Any existing tracks are removed first, so
re-running gives a fresh route. Re-run build.py first if design.py changed.
"""
import os
import re
import shutil
import subprocess
import sys

import pcbnew

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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


def import_ses(board, ses_path):
    """Add the routed wires and vias from a FreeRouting .ses file to the board.
    (KiCad 7's ImportSpecctraSES only works inside the GUI, so do it by hand.)"""
    from sexp import parse, find, find1
    root = parse(open(ses_path).read())
    routes = find1(root, "routes")
    res = find1(routes, "resolution")
    unit_nm = {"um": 1000, "mm": 1_000_000, "mil": 25400}[str(res[1])] / float(res[2])
    pt = lambda x, y: pcbnew.VECTOR2I(int(float(x) * unit_nm), int(-float(y) * unit_nm))
    vias = {}
    for ps in find(find1(routes, "library_out") or [], "padstack"):
        m = re.search(r"_(\d+):(\d+)_um", ps[1])
        vias[ps[1]] = (int(m.group(1)) * 1000, int(m.group(2)) * 1000) if m else (600_000, 300_000)
    layers = {"F.Cu": pcbnew.F_Cu, "B.Cu": pcbnew.B_Cu}
    nets = board.GetNetsByName()
    n_trk = n_via = 0
    for net in find(find1(routes, "network_out"), "net"):
        ni = nets[str(net[1])] if str(net[1]) in nets else None
        for w in find(net, "wire"):
            path = find1(w, "path")
            layer, width, coords = layers[str(path[1])], float(path[2]) * unit_nm, path[3:]
            pts = [pt(coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2)]
            for a, b in zip(pts, pts[1:]):
                t = pcbnew.PCB_TRACK(board)
                t.SetStart(a); t.SetEnd(b); t.SetWidth(int(width)); t.SetLayer(layer)
                if ni: t.SetNet(ni)
                board.Add(t); n_trk += 1
        for v in find(net, "via"):
            dia, drill = vias.get(v[1], (600_000, 300_000))
            via = pcbnew.PCB_VIA(board)
            via.SetPosition(pt(v[2], v[3])); via.SetWidth(dia); via.SetDrill(drill)
            via.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
            if ni: via.SetNet(ni)
            board.Add(via); n_via += 1
    return n_trk, n_via


def main():
    passes = sys.argv[1] if len(sys.argv) > 1 else "40"
    # Load the real board so its .kicad_pro net classes (power widths, USB pair) apply.
    board = pcbnew.LoadBoard(PCB)
    old = [t for t in board.GetTracks()]
    if old:
        print(f"removing {len(old)} existing tracks/vias for a fresh route")
        for t in old:
            board.Remove(t)
    if not pcbnew.ExportSpecctraDSN(board, DSN):
        raise SystemExit("DSN export failed")
    strip_nets(DSN, SKIP_NETS)

    java = os.environ.get("JAVA", "java")
    cp = os.environ["FREEROUTING_CP"]
    cmd = [java, "-cp", cp, "app.freerouting.Freerouting", "-de", DSN, "-do", SES, "-mp", passes,
           "--gui.enabled=false", "--usage_and_diagnostic_data.disable_analytics=true"]
    if os.path.exists(SES):
        os.remove(SES)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    for line in (r.stdout + r.stderr).splitlines():
        if "final score" in line or "unrouted connection" in line or "ERROR" in line:
            print("  ", line[-150:])
    if not os.path.exists(SES):
        raise SystemExit("FreeRouting produced no session file")

    n_trk, n_via = import_ses(board, SES)
    board.BuildConnectivity()
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    board.Save(PCB)
    print(f"imported: {n_trk} track segments, {n_via} vias -> {PCB}")


if __name__ == "__main__":
    main()
