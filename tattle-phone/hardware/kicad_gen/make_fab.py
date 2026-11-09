"""
Make the files a PCB fab + assembly service needs (JLCPCB format, works elsewhere too).

    python3 make_fab.py

Writes ../fab/:
  tattle_phone_gerbers.zip   Gerbers + drill files -> upload as the PCB
  bom_jlc.csv                Comment, Designator, Footprint, LCSC Part #  (fill in LCSC numbers)
  cpl_jlc.csv                Designator, Mid X, Mid Y, Layer, Rotation   (placement)
Through-hole parts (switches, JST connectors) and DNP parts are left out of the
assembly files: you solder those (or add them back if you pay the fab to).
"""
import csv
import os
import shutil
import subprocess
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PCB = os.path.join(HERE, "..", "kicad", "tattle_phone.kicad_pcb")
OUT = os.path.join(HERE, "..", "fab")
LAYERS = "F.Cu,B.Cu,F.Paste,B.Paste,F.Silkscreen,B.Silkscreen,F.Mask,B.Mask,Edge.Cuts"


def run(*args):
    r = subprocess.run(["kicad-cli", *args], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stdout + r.stderr)


def main():
    gdir = os.path.join(OUT, "gerbers")
    shutil.rmtree(gdir, ignore_errors=True)
    os.makedirs(gdir)
    run("pcb", "export", "gerbers", "--layers", LAYERS, "--subtract-soldermask", "--no-x2",
        "-o", gdir + "/", PCB)
    run("pcb", "export", "drill", "--format", "excellon", "--excellon-separate-th",
        "--generate-map", "--map-format", "gerberx2", "-o", gdir + "/", PCB)
    zpath = os.path.join(OUT, "tattle_phone_gerbers.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(gdir)):
            z.write(os.path.join(gdir, f), f)

    parts = {p["ref"]: p for p in design.PARTS}
    skip = {r for r, p in parts.items()
            if p["dnp"] or p["fp"].startswith(("Button_Switch_THT", "Connector_JST", "MountingHole",
                                                "Fiducial", "TestPoint"))}

    pos = os.path.join(OUT, "pos.csv")
    run("pcb", "export", "pos", "--format", "csv", "--units", "mm", "--side", "both", "-o", pos, PCB)
    rows = list(csv.DictReader(open(pos)))
    with open(os.path.join(OUT, "cpl_jlc.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Designator", "Mid X", "Mid Y", "Layer", "Rotation"])
        for r in rows:
            if r["Ref"] in skip:
                continue
            w.writerow([r["Ref"], r["PosX"] + "mm", r["PosY"] + "mm",
                        "Top" if r["Side"].lower().startswith("top") else "Bottom", r["Rot"]])
    os.remove(pos)

    groups = {}
    for r, p in parts.items():
        if r in skip:
            continue
        groups.setdefault((p["value"], p["fp"].split(":")[1]), []).append(r)
    with open(os.path.join(OUT, "bom_jlc.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Comment", "Designator", "Footprint", "LCSC Part #"])
        for (val, fp), refs in groups.items():
            w.writerow([val, ",".join(sorted(refs, key=lambda s: (len(s), s))), fp, ""])
    print(f"wrote {os.path.normpath(OUT)}: gerbers zip ({len(os.listdir(gdir))} files), "
          f"cpl_jlc.csv ({len(rows) - len([r for r in rows if r['Ref'] in skip])} parts), "
          f"bom_jlc.csv ({len(groups)} lines)")


if __name__ == "__main__":
    main()
