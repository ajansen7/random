"""
Sanity-check the generated KiCad project.

    python3 check.py [kicad_dir]

1. Exports the schematic netlist with kicad-cli and compares it, pin by pin,
   with the pad nets on the board.
2. Runs DRC and prints the violations grouped by type and by the parts
   involved. "Unrouted" and silkscreen items are counted but not listed, since the board
   is unrouted by design.
"""
import collections
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import pcbnew

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "kicad"))
SCH = os.path.join(KDIR, "tattle_phone.kicad_sch")
PCB = os.path.join(KDIR, "tattle_phone.kicad_pcb")
QUIET = {"unconnected_items", "silk_over_copper", "silk_overlap", "silk_edge_clearance",
         "lib_footprint_issues", "lib_footprint_mismatch", "starved_thermal"}


def netlist_check(board):
    tmp = tempfile.mktemp(suffix=".xml")
    subprocess.run(["kicad-cli", "sch", "export", "netlist", "--format", "kicadxml", "-o", tmp, SCH],
                   check=True, capture_output=True)
    root = ET.parse(tmp).getroot()
    sch = {}
    for n in root.find("nets"):
        name = n.get("name").lstrip("/")
        for node in n.findall("node"):
            if not node.get("ref").startswith("#"):
                sch[(node.get("ref"), node.get("pin"))] = None if name.startswith(("unconnected-", "Net-(")) else name
    pcb = {}
    for fp in board.GetFootprints():
        for p in fp.Pads():
            if p.GetNumber():
                pcb[(fp.GetReference(), p.GetNumber())] = p.GetNetname() or None
    bad = [(k, v, pcb.get(k)) for k, v in sch.items() if (v or None) != pcb.get(k)]
    bad += [(k, sch.get(k), v) for k, v in pcb.items() if v and k not in sch]
    refs_s = {k[0] for k in sch}
    refs_p = {fp.GetReference() for fp in board.GetFootprints() if any(p.GetNumber() for p in fp.Pads())}
    cnt = collections.Counter(v for v in sch.values() if v)
    print(f"netlist: {len(sch)} pins, {len(cnt)} nets, {len(bad)} schematic/board mismatches")
    for b in bad[:20]:
        print("   mismatch", b)
    if refs_s ^ refs_p:
        print("   parts only on one side:", sorted(refs_s ^ refs_p))
    singles = [n for n, c in cnt.items() if c == 1]
    if singles:
        print("   nets with a single pin:", singles)
    return not bad and not (refs_s ^ refs_p)


def drc(board):
    rpt = tempfile.mktemp(suffix=".txt")
    pcbnew.WriteDRCReport(board, rpt, pcbnew.EDA_UNITS_MILLIMETRES, True)
    txt = open(rpt).read()
    groups = collections.defaultdict(set)
    counts = collections.Counter()
    for blk in re.split(r"\n(?=\[)", txt):
        m = re.match(r"\[(\w+)\]", blk)
        if not m:
            continue
        kind = m.group(1)
        counts[kind] += 1
        refs = tuple(sorted(set(re.findall(r"(?:Footprint|of) (\w+)", blk)))) or \
            tuple(i[:50] for i in re.findall(r"@\([^)]*\): (.*)", blk))
        groups[kind].add(refs)
    print("DRC:", ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    for kind, refs in sorted(groups.items()):
        if kind in QUIET:
            continue
        print(f"  [{kind}]")
        for r in sorted(refs):
            print("     ", " / ".join(r))
    return counts


if __name__ == "__main__":
    b = pcbnew.LoadBoard(PCB)
    b.BuildConnectivity()
    pcbnew.ZONE_FILLER(b).Fill(b.Zones())
    ok = netlist_check(b)
    drc(b)
    sys.exit(0 if ok else 1)
