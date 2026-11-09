"""Write a grouped BOM (JLCPCB-style columns) from design.py: ../kicad/bom.csv"""
import csv, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design

groups = collections.OrderedDict()
for p in design.PARTS:
    if p["fp"].startswith(("MountingHole", "Fiducial", "TestPoint")):
        continue
    key = (p["value"], p["fp"], p["dnp"])
    groups.setdefault(key, []).append(p)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kicad", "bom.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Comment", "Designator", "Footprint", "Qty", "Side", "Assembly", "LCSC Part #"])
    for (value, fp, dnp), ps in groups.items():
        sides = sorted({"Front" if p["side"] == "F" else "Back" for p in ps})
        tht = fp.startswith(("Button_Switch_THT", "Connector_JST"))
        how = "DO NOT FIT" if dnp else ("hand-solder (through-hole)" if tht else "factory")
        w.writerow([value, ",".join(p["ref"] for p in ps), fp.split(":")[1], len(ps), "/".join(sides), how, ""])
print("wrote", os.path.normpath(out), len(groups), "lines")
