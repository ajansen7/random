"""
Stitch the two GND pours together with a grid of vias.

    python3 stitch.py [pitch_mm]      (default 4.0; edits ../kicad/tattle_lite.kicad_pcb)

A via goes at a grid point only where BOTH copper layers have GND pour around it with room
to spare, so it can never touch another net. It also stays off pads, holes, keep-out areas
(antenna, LED lens holes) and the undersides of parts. On a 2-layer board this joins the
little pour islands that routing leaves behind, and gives the ground a lower impedance.
Run it once, right after route.py (it refuses to run twice on the same routing).
"""
import math
import os
import sys

import pcbnew

HERE = os.path.dirname(os.path.abspath(__file__))
PCB = os.path.join(HERE, "..", "kicad", "tattle_lite.kicad_pcb")
VIA_D, VIA_DRILL = 0.6, 0.3
MARGIN = VIA_D / 2 + 0.1        # via edge to pour edge (the pour edge is already 0.2 off other nets)
HOLE_GAP = 0.4                  # drill edge to any other drill edge
TAG = "gnd-stitch"
mm, nm = pcbnew.ToMM, pcbnew.FromMM


def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L))
    return math.hypot(ax + t * dx - px, ay + t * dy - py)


class Pour:
    """One filled polygon (outline + holes) of the GND pour on one layer."""
    def __init__(self, ps, i):
        self.ps = ps.UnitSet(i)
        self.edges = []
        poly = [self.ps.Outline(0)] + [self.ps.Hole(0, h) for h in range(self.ps.HoleCount(0))]
        for ch in poly:
            pts = [(mm(ch.CPoint(k).x), mm(ch.CPoint(k).y)) for k in range(ch.PointCount())]
            self.edges += [(pts[k], pts[(k + 1) % len(pts)]) for k in range(len(pts))]
        bb = self.ps.BBox()
        self.bb = (mm(bb.GetLeft()), mm(bb.GetTop()), mm(bb.GetRight()), mm(bb.GetBottom()))

    def inside_with_margin(self, x, y, m):
        if not (self.bb[0] < x < self.bb[2] and self.bb[1] < y < self.bb[3]):
            return False
        if not self.ps.Contains(pcbnew.VECTOR2I(nm(x), nm(y))):
            return False
        return all(seg_dist(x, y, *a, *b) >= m for a, b in self.edges
                   if not (max(a[0], b[0]) < x - m or min(a[0], b[0]) > x + m or
                           max(a[1], b[1]) < y - m or min(a[1], b[1]) > y + m))


def main():
    pitch = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    board = pcbnew.LoadBoard(PCB)
    gnd = board.FindNet("GND")

    old = [t for t in board.GetTracks() if t.GetClass() == "PCB_VIA" and t.GetNetCode() == gnd.GetNetCode()
           and t.GetParentGroup() and t.GetParentGroup().GetName() == TAG]
    if old:
        # Removing items with pcbnew's Remove() crashes or corrupts the bindings (see the
        # HANDOFF traps), so don't try: route.py rebuilds every track and via from scratch.
        raise SystemExit(f"already stitched ({len(old)} vias). Re-run route.py, then stitch.py once.")

    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    pours = {L: [] for L in (pcbnew.F_Cu, pcbnew.B_Cu)}
    for z in board.Zones():
        if z.GetIsRuleArea() or z.GetNetCode() != gnd.GetNetCode():
            continue
        for L in pours:
            if z.IsOnLayer(L):
                ps = z.GetFilledPolysList(L)
                pours[L] += [Pour(ps, i) for i in range(ps.OutlineCount())]

    holes, blocks, soft = [], [], []
    for t in board.GetTracks():
        if t.GetClass() == "PCB_VIA":
            holes.append((mm(t.GetPosition().x), mm(t.GetPosition().y), mm(t.GetDrillValue()) / 2))
    for fp in board.GetFootprints():
        for p in fp.Pads():
            if p.HasHole():
                holes.append((mm(p.GetPosition().x), mm(p.GetPosition().y),
                              mm(max(p.GetDrillSizeX(), p.GetDrillSizeY())) / 2))
            bb = p.GetBoundingBox()
            blocks.append((mm(bb.GetLeft()) - 0.4, mm(bb.GetTop()) - 0.4, mm(bb.GetRight()) + 0.4, mm(bb.GetBottom()) + 0.4))
        if fp.GetReference().startswith(("H", "FID")):
            continue
        cy = fp.GetCourtyard(pcbnew.B_CrtYd if fp.IsFlipped() else pcbnew.F_CrtYd)
        if cy.OutlineCount():
            bb = cy.BBox()
            soft.append((mm(bb.GetLeft()), mm(bb.GetTop()), mm(bb.GetRight()), mm(bb.GetBottom())))
    keepouts = [z for z in list(board.Zones()) + [z for f in board.GetFootprints() for z in f.Zones()]
                if z.GetIsRuleArea() and z.GetDoNotAllowVias()]

    bb = board.GetBoardEdgesBoundingBox()
    x0, y0, x1, y1 = mm(bb.GetLeft()), mm(bb.GetTop()), mm(bb.GetRight()), mm(bb.GetBottom())
    group = pcbnew.PCB_GROUP(board)
    group.SetName(TAG)
    board.Add(group)
    added = 0
    y = y0 + pitch / 2
    while y < y1:
        x = x0 + pitch / 2
        while x < x1:
            pt = pcbnew.VECTOR2I(nm(x), nm(y))
            ok = not any(b[0] < x < b[2] and b[1] < y < b[3] for b in blocks + soft)
            ok = ok and all(math.hypot(x - hx, y - hy) - hr - VIA_DRILL / 2 >= HOLE_GAP for hx, hy, hr in holes)
            ok = ok and not any(z.Outline().Contains(pt) for z in keepouts)
            ok = ok and all(any(p.inside_with_margin(x, y, MARGIN) for p in pours[L]) for L in pours)
            if ok:
                v = pcbnew.PCB_VIA(board)
                v.SetPosition(pt)
                v.SetWidth(nm(VIA_D))
                v.SetDrill(nm(VIA_DRILL))
                v.SetNet(gnd)
                board.Add(v)
                group.AddItem(v)
                holes.append((x, y, VIA_DRILL / 2))
                added += 1
            x += pitch
        y += pitch
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())

    # second pass: any pour island still cut off (usually GND pads in a crowded spot) gets
    # one via at the free spot nearest its pads. Allowed under a part's outline, never on a
    # pad, a hole or a keep-out.
    fixed = 0
    for island_pour, layer, pads in islands(board, gnd):
        other = pcbnew.F_Cu if layer == pcbnew.B_Cu else pcbnew.B_Cu
        others = [Pour(z.GetFilledPolysList(other), i) for z in board.Zones()
                  if not z.GetIsRuleArea() and z.GetNetCode() == gnd.GetNetCode() and z.IsOnLayer(other)
                  for i in range(z.GetFilledPolysList(other).OutlineCount())]
        cx = sum(p[0] for p in pads) / len(pads)
        cy = sum(p[1] for p in pads) / len(pads)
        bx0, by0, bx1, by1 = island_pour.bb
        best = None
        yy = by0
        while yy <= by1:
            xx = bx0
            while xx <= bx1:
                d = math.hypot(xx - cx, yy - cy)
                if (best is None or d < best[0]) and island_pour.inside_with_margin(xx, yy, MARGIN) \
                        and not any(b[0] < xx < b[2] and b[1] < yy < b[3] for b in blocks) \
                        and all(math.hypot(xx - hx, yy - hy) - hr - VIA_DRILL / 2 >= HOLE_GAP for hx, hy, hr in holes) \
                        and not any(z.Outline().Contains(pcbnew.VECTOR2I(nm(xx), nm(yy))) for z in keepouts) \
                        and any(o.inside_with_margin(xx, yy, MARGIN) for o in others):
                    best = (d, xx, yy)
                xx += 0.1
            yy += 0.1
        if best:
            v = pcbnew.PCB_VIA(board)
            v.SetPosition(pcbnew.VECTOR2I(nm(best[1]), nm(best[2])))
            v.SetWidth(nm(VIA_D))
            v.SetDrill(nm(VIA_DRILL))
            v.SetNet(gnd)
            board.Add(v)
            group.AddItem(v)
            holes.append((best[1], best[2], VIA_DRILL / 2))
            fixed += 1
        else:
            print("  ! no via spot for the island holding", len(pads), "GND pad(s) near",
                  f"({cx - 100:.1f}, {200 - cy:.1f}) case mm")
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    board.Save(PCB)
    print(f"stitching: removed {len(old)} old, added {added} grid vias at {pitch} mm pitch "
          f"+ {fixed} island vias -> {os.path.normpath(PCB)}")


def islands(board, gnd):
    """[(Pour, layer, [(x, y) of its GND pads])] for every GND pour piece that is not joined
    to the main ground (union-find over pours, GND pads, GND vias and GND tracks)."""
    LAY = [pcbnew.F_Cu, pcbnew.B_Cu]
    polys = []
    for z in board.Zones():
        if z.GetIsRuleArea() or z.GetNetCode() != gnd.GetNetCode():
            continue
        for L in LAY:
            if z.IsOnLayer(L):
                ps = z.GetFilledPolysList(L)
                polys += [(L, Pour(ps, i)) for i in range(ps.OutlineCount())]
    items = []
    for t in board.GetTracks():
        if t.GetNetCode() != gnd.GetNetCode():
            continue
        if t.GetClass() == "PCB_VIA":
            items.append((None, [(L, t.GetPosition()) for L in LAY]))
        else:
            items.append((None, [(t.GetLayer(), t.GetStart()), (t.GetLayer(), t.GetEnd())]))
    for fp in board.GetFootprints():
        for p in fp.Pads():
            if p.GetNetCode() == gnd.GetNetCode():
                items.append(((mm(p.GetPosition().x), mm(p.GetPosition().y)),
                              [(L, p.GetPosition()) for L in LAY if p.IsOnLayer(L)]))
    par = list(range(len(polys) + len(items)))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    base = len(polys)
    for n, (_, pts) in enumerate(items):
        for L, pt in pts:
            for k, (PL, po) in enumerate(polys):
                if PL == L and po.ps.Contains(pt):
                    par[find(base + n)] = find(k)
    comps = {}
    for k in range(len(par)):
        comps.setdefault(find(k), []).append(k)
    main = max(comps.values(), key=len)
    out = []
    for c in comps.values():
        if c is main:
            continue
        pads = [items[k - base][0] for k in c if k >= base and items[k - base][0]]
        for k in c:
            if k < base and pads:
                out.append((polys[k][1], polys[k][0], pads))
    return out


if __name__ == "__main__":
    main()
