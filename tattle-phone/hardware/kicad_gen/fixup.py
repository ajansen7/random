"""
Post-autoroute fix-ups, using only KiCad's Python API and numpy.

    python3 fixup.py

1. Routes connections the autorouter left open, with a small grid A* router
   (0.25 mm tracks, so it can reach fine-pitch QFN pins; vias when needed).
2. GND stitching: every GND pad or track that the pours don't reach gets a
   short track to a via, which ties it into the pours on both layers.
3. Moves fiducials to spots with a clear 1 mm ring around them.
Repeats until DRC shows nothing unconnected (or it stops making progress).
"""
import heapq
import math
import os
import re
import sys
import tempfile

import numpy as np
import pcbnew

HERE = os.path.dirname(os.path.abspath(__file__))
PCB = os.path.join(HERE, "..", "kicad", "tattle_phone.kicad_pcb")

G = 0.05                 # grid pitch, mm
CLR = 0.2                # copper clearance, mm
EDGE = 0.5               # copper to board edge, mm
TRACK_W = 0.25           # fix-up track width, mm
VIA_D, VIA_DRILL = 0.6, 0.3
LAYERS = [pcbnew.F_Cu, pcbnew.B_Cu]
mm = pcbnew.ToMM
nm = pcbnew.FromMM


class Grid:
    def __init__(self, board):
        bb = board.GetBoardEdgesBoundingBox()
        self.x0, self.y0 = mm(bb.GetLeft()) - 1, mm(bb.GetTop()) - 1
        self.nx = int((mm(bb.GetWidth()) + 2) / G) + 1
        self.ny = int((mm(bb.GetHeight()) + 2) / G) + 1
        self.board = board
        xs = self.x0 + np.arange(self.nx) * G
        ys = self.y0 + np.arange(self.ny) * G
        self.X, self.Y = np.meshgrid(xs, ys)          # [j, i] indexing (row = y)
        self.inside = self._outline_mask(EDGE)
        self.keepout = self._keepout_mask()

    def ij(self, x, y):
        return int(round((y - self.y0) / G)), int(round((x - self.x0) / G))

    def xy(self, j, i):
        return self.x0 + i * G, self.y0 + j * G

    def _poly_mask(self, pts):
        """Even-odd point-in-polygon on the whole grid."""
        mask = np.zeros(self.X.shape, bool)
        n = len(pts)
        for k in range(n):
            (xa, ya), (xb, yb) = pts[k], pts[(k + 1) % n]
            if ya == yb:
                continue
            cond = (self.Y >= min(ya, yb)) & (self.Y < max(ya, yb))
            xint = xa + (self.Y - ya) * (xb - xa) / (yb - ya)
            mask ^= cond & (self.X < xint)
        return mask

    def _outline_mask(self, margin):
        poly = pcbnew.SHAPE_POLY_SET()
        self.board.GetBoardPolygonOutlines(poly)
        poly.Inflate(-nm(margin), 16)
        ol = poly.Outline(0)
        pts = [(mm(ol.CPoint(k).x), mm(ol.CPoint(k).y)) for k in range(ol.PointCount())]
        return self._poly_mask(pts)

    def _keepout_mask(self):
        mask = np.zeros(self.X.shape, bool)
        zones = list(self.board.Zones())
        for fp in self.board.GetFootprints():
            zones += list(fp.Zones())
        for z in zones:
            if not z.GetIsRuleArea() or not z.GetDoNotAllowTracks():
                continue
            ol = z.Outline().Outline(0)
            pts = [(mm(ol.CPoint(k).x), mm(ol.CPoint(k).y)) for k in range(ol.PointCount())]
            mask |= self._poly_mask(pts)
        return mask

    # ---------------------------------------------------------------- rasterizing copper
    def _stamp_segment(self, m, xa, ya, xb, yb, r):
        x1, x2 = min(xa, xb) - r, max(xa, xb) + r
        y1, y2 = min(ya, yb) - r, max(ya, yb) + r
        j1, i1 = self.ij(x1, y1)
        j2, i2 = self.ij(x2, y2)
        j1, i1 = max(j1, 0), max(i1, 0)
        j2, i2 = min(j2 + 1, self.ny), min(i2 + 1, self.nx)
        if j1 >= j2 or i1 >= i2:
            return
        X, Y = self.X[j1:j2, i1:i2], self.Y[j1:j2, i1:i2]
        dx, dy = xb - xa, yb - ya
        L2 = dx * dx + dy * dy
        if L2 == 0:
            d2 = (X - xa) ** 2 + (Y - ya) ** 2
        else:
            t = np.clip(((X - xa) * dx + (Y - ya) * dy) / L2, 0, 1)
            d2 = (X - xa - t * dx) ** 2 + (Y - ya - t * dy) ** 2
        m[j1:j2, i1:i2] |= d2 <= r * r

    def _stamp_rect(self, m, bb, r):
        j1, i1 = self.ij(mm(bb.GetLeft()) - r, mm(bb.GetTop()) - r)
        j2, i2 = self.ij(mm(bb.GetRight()) + r, mm(bb.GetBottom()) + r)
        m[max(j1, 0):max(j2 + 1, 0), max(i1, 0):max(i2 + 1, 0)] = True

    def obstacles(self, netcode, half):
        """Per-layer masks of cells where a track/via of radius `half` (mm) would
        violate clearance with copper of other nets, the edge or a keep-out."""
        r = half + CLR
        masks = {L: np.zeros(self.X.shape, bool) for L in LAYERS}
        for t in self.board.GetTracks():
            if t.GetNetCode() == netcode:
                continue
            if t.GetClass() == "PCB_VIA":
                p = t.GetPosition()
                for L in LAYERS:
                    self._stamp_segment(masks[L], mm(p.x), mm(p.y), mm(p.x), mm(p.y), mm(t.GetWidth()) / 2 + r)
            else:
                L = t.GetLayer()
                if L in masks:
                    self._stamp_segment(masks[L], mm(t.GetStart().x), mm(t.GetStart().y),
                                        mm(t.GetEnd().x), mm(t.GetEnd().y), mm(t.GetWidth()) / 2 + r)
        for fp in self.board.GetFootprints():
            for p in fp.Pads():
                same = p.GetNetCode() == netcode and netcode > 0
                npth = p.GetAttribute() == pcbnew.PAD_ATTRIB_NPTH
                if same and not npth:
                    continue
                for L in LAYERS:
                    if p.IsOnLayer(L) or npth or p.GetAttribute() == pcbnew.PAD_ATTRIB_PTH:
                        self._stamp_rect(masks[L], p.GetBoundingBox(), r)
        blocked = ~self.inside | self.keepout
        for L in LAYERS:
            masks[L] |= blocked
        return masks


# -------------------------------------------------------------------- A* router
def astar(grid, starts, goals, track_obs, via_obs, max_nodes=2_500_000, via_cost=30):
    """starts/goals: sets of (layer_index, j, i). Returns list of nodes or None."""
    gj = np.array([g[1] for g in goals])
    gi = np.array([g[2] for g in goals])
    goalset = set(goals)

    def h(j, i):
        return float(np.min(np.hypot(gj - j, gi - i))) if len(gj) < 64 else 0.0

    tgt = (int(gj.mean()), int(gi.mean()))
    hfast = lambda j, i: math.hypot(tgt[0] - j, tgt[1] - i) * 0.5
    openh = []
    came = {}
    cost = {}
    for s in starts:
        cost[s] = 0.0
        heapq.heappush(openh, (hfast(s[1], s[2]), 0.0, s))
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    n = 0
    while openh and n < max_nodes:
        _, c, node = heapq.heappop(openh)
        if c > cost.get(node, 1e18):
            continue
        n += 1
        if node in goalset:
            path = [node]
            while path[-1] in came:
                path.append(came[path[-1]])
            return path[::-1]
        l, j, i = node
        obs = track_obs[l]
        for dj, di, w in moves:
            jj, ii = j + dj, i + di
            if not (0 <= jj < grid.ny and 0 <= ii < grid.nx):
                continue
            nxt = (l, jj, ii)
            if obs[jj, ii] and nxt not in goalset:
                continue
            nc = c + w
            if nc < cost.get(nxt, 1e18):
                cost[nxt] = nc
                came[nxt] = node
                heapq.heappush(openh, (nc + hfast(jj, ii), nc, nxt))
        if not via_obs[0][j, i] and not via_obs[1][j, i]:
            nxt = (1 - l, j, i)
            nc = c + via_cost
            if nc < cost.get(nxt, 1e18):
                cost[nxt] = nc
                came[nxt] = node
                heapq.heappush(openh, (nc + hfast(j, i), nc, nxt))
    return None


def commit_path(board, grid, path, net, width=TRACK_W):
    """Turn a grid path into tracks (merging straight runs) and vias."""
    pts = []
    for l, j, i in path:
        pts.append((l, *grid.xy(j, i)))
    segs, vias = [], []
    run = [pts[0]]
    for p in pts[1:]:
        if p[0] != run[-1][0]:
            vias.append((p[1], p[2]))
            if len(run) > 1:
                segs.append(run)
            run = [p]
            continue
        run.append(p)
    if len(run) > 1:
        segs.append(run)
    for run in segs:
        # simplify: keep points where direction changes
        keep = [run[0]]
        for a, b, c in zip(run, run[1:], run[2:]):
            d1 = (round((b[1] - a[1]) / G), round((b[2] - a[2]) / G))
            d2 = (round((c[1] - b[1]) / G), round((c[2] - b[2]) / G))
            if d1 != d2:
                keep.append(b)
        keep.append(run[-1])
        for a, b in zip(keep, keep[1:]):
            t = pcbnew.PCB_TRACK(board)
            t.SetStart(pcbnew.VECTOR2I(nm(a[1]), nm(a[2])))
            t.SetEnd(pcbnew.VECTOR2I(nm(b[1]), nm(b[2])))
            t.SetWidth(nm(width))
            t.SetLayer(LAYERS[a[0]])
            t.SetNet(net)
            board.Add(t)
    for x, y in vias:
        add_via(board, net, x, y)
    return len(vias)


def add_via(board, net, x, y):
    v = pcbnew.PCB_VIA(board)
    v.SetPosition(pcbnew.VECTOR2I(nm(x), nm(y)))
    v.SetWidth(nm(VIA_D))
    v.SetDrill(nm(VIA_DRILL))
    v.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
    v.SetNet(net)
    board.Add(v)


# -------------------------------------------------------------------- DRC parsing
ITEM = re.compile(r"@\(([-\d.]+) mm, ([-\d.]+) mm\): (.*)")


def unconnected(board):
    board.BuildConnectivity()
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    board.BuildConnectivity()
    rpt = tempfile.mktemp(suffix=".txt")
    pcbnew.WriteDRCReport(board, rpt, pcbnew.EDA_UNITS_MILLIMETRES, True)
    out = []
    for blk in re.split(r"\n(?=\[)", open(rpt).read()):
        if not blk.startswith("[unconnected_items]"):
            continue
        items = ITEM.findall(blk)
        if len(items) < 2:
            continue
        net = re.search(r"\[([^\]]+)\]", items[0][2]).group(1)
        out.append((net, [(float(x), float(y), d) for x, y, d in items[:2]]))
    return out


def item_layers(desc):
    if "PTH pad" in desc or desc.startswith("Via"):
        return [0, 1]
    if "on F.Cu" in desc:
        return [0]
    if "on B.Cu" in desc:
        return [1]
    return [0, 1]


def endpoints(board, net_name, x, y, desc):
    """For a track item, return both of its end points (DRC reports only one)."""
    if not desc.startswith("Track"):
        return [(x, y)]
    best, bd = None, 1e9
    for t in board.GetTracks():
        if t.GetClass() != "PCB_TRACK" or t.GetNetname() != net_name:
            continue
        for p in (t.GetStart(), t.GetEnd()):
            d = math.hypot(mm(p.x) - x, mm(p.y) - y)
            if d < bd:
                bd, best = d, t
    if best is None:
        return [(x, y)]
    return [(mm(best.GetStart().x), mm(best.GetStart().y)), (mm(best.GetEnd().x), mm(best.GetEnd().y))]


def disc(grid, x, y, layers, r=0.15):
    j, i = grid.ij(x, y)
    k = max(1, int(r / G))
    return {(l, jj, ii) for l in layers for jj in range(j - k, j + k + 1) for ii in range(i - k, i + k + 1)}


# -------------------------------------------------------------------- main passes
def route_signals(board, grid, pairs):
    fixed = 0
    for net_name, ((xa, ya, da), (xb, yb, db)) in pairs:
        if net_name == "GND":
            continue
        net = board.FindNet(net_name)
        vo = grid.obstacles(net.GetNetCode(), VIA_D / 2)
        vobs = [vo[pcbnew.F_Cu], vo[pcbnew.B_Cu]]
        starts, goals = set(), set()
        for x, y in endpoints(board, net_name, xa, ya, da):
            starts |= disc(grid, x, y, item_layers(da))
        for x, y in endpoints(board, net_name, xb, yb, db):
            goals |= disc(grid, x, y, item_layers(db))
        path = None
        for width in (TRACK_W, 0.15):   # neck down if the normal width can't get out
            to = grid.obstacles(net.GetNetCode(), width / 2)
            path = astar(grid, starts, goals, [to[pcbnew.F_Cu], to[pcbnew.B_Cu]], vobs)
            if path:
                break
        if path:
            nv = commit_path(board, grid, path, net, width=width)
            print(f"  routed {net_name}: {len(path)} cells, {nv} vias")
            fixed += 1
        else:
            print(f"  could not route {net_name} ({da[:40]} -> {db[:40]})")
    return fixed


def stitch_gnd(board, grid, pairs, radius=4.0):
    net = board.FindNet("GND")
    tobs = grid.obstacles(net.GetNetCode(), 0.2 / 2)
    tobs = [tobs[pcbnew.F_Cu], tobs[pcbnew.B_Cu]]
    vo = grid.obstacles(net.GetNetCode(), VIA_D / 2)
    vfree = ~vo[pcbnew.F_Cu] & ~vo[pcbnew.B_Cu]
    anchor = np.zeros(vfree.shape, bool)
    for t in board.GetTracks():
        if t.GetClass() == "PCB_VIA" and t.GetNetCode() == net.GetNetCode():
            p = t.GetPosition()
            grid._stamp_segment(anchor, mm(p.x), mm(p.y), mm(p.x), mm(p.y), VIA_D / 2 - 0.05)
    for fp in board.GetFootprints():
        for p in fp.Pads():
            if p.GetNetCode() == net.GetNetCode() and p.GetAttribute() == pcbnew.PAD_ATTRIB_PTH:
                grid._stamp_rect(anchor, p.GetBoundingBox(), -0.1)
    done = set()
    added = 0
    for net_name, items in pairs:
        if net_name != "GND":
            continue
        x, y, d = items[0]
        if d.startswith("Zone"):
            x, y, d = items[1]        # zone fragments: stitch the item instead
        if d.startswith("Zone"):
            continue
        if (round(x, 2), round(y, 2)) in done:
            continue
        done.add((round(x, 2), round(y, 2)))
        layers = item_layers(d)
        if len(layers) == 2:
            x, y, d = items[1]       # a through-hole item: stitch the other end instead
            layers = item_layers(d)
        l = layers[0]
        # breadth-first search outward on the item's layer to the nearest via-able cell
        j0, i0 = grid.ij(x, y)
        if not (0 <= j0 < grid.ny and 0 <= i0 < grid.nx):
            continue
        k = int(radius / G)
        seen = {(j0, i0): None}
        q = [(j0, i0)]
        found = None
        for (j, i) in q:
            if anchor[j, i] and math.hypot(j - j0, i - i0) * G > 0.3:
                found = ("anchor", j, i)
                break
            if vfree[j, i] and math.hypot(j - j0, i - i0) * G > 0.6:
                found = (j, i)
                break
            for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                jj, ii = j + dj, i + di
                if (jj, ii) in seen or abs(jj - j0) > k or abs(ii - i0) > k:
                    continue
                if not (0 <= jj < grid.ny and 0 <= ii < grid.nx):
                    continue
                if tobs[l][jj, ii] and math.hypot(jj - j0, ii - i0) * G > 0.5:
                    continue
                seen[(jj, ii)] = (j, i)
                q.append((jj, ii))
        if not found:
            print(f"  no stitching spot for GND at ({x:.2f}, {y:.2f}) {d[:40]}")
            continue
        to_anchor = found[0] == "anchor"
        if to_anchor:
            found = found[1:]
        path = [found]
        while seen[path[-1]] is not None:
            path.append(seen[path[-1]])
        cells = [(l, j, i) for j, i in path[::-1]]
        commit_path(board, grid, cells, net, width=0.2)
        added += 1
        if to_anchor:
            continue
        fx, fy = grid.xy(*found)
        add_via(board, net, fx, fy)
        # mark the new via so the next stitch keeps clear of it
        vj, vi = found
        rr = int((VIA_D + CLR + VIA_D / 2) / G) + 1
        vfree[max(vj - rr, 0):vj + rr + 1, max(vi - rr, 0):vi + rr + 1] = False
    return added


def stitch_islands(board, grid, min_area_mm2=4.0):
    """Drop pour slivers, then put a via inside every remaining pour island that has
    no GND via or through-hole pad in it, so front and back pours tie together."""
    net = board.FindNet("GND")
    for z in board.Zones():
        if not z.GetIsRuleArea():
            z.SetIslandRemovalMode(pcbnew.ISLAND_REMOVAL_MODE_AREA)
            z.SetMinIslandArea(int(min_area_mm2 * 1e12))
    board.BuildConnectivity()
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    anchors = [t.GetPosition() for t in board.GetTracks()
               if t.GetClass() == "PCB_VIA" and t.GetNetCode() == net.GetNetCode()]
    for fp in board.GetFootprints():
        anchors += [p.GetPosition() for p in fp.Pads()
                    if p.GetNetCode() == net.GetNetCode() and p.GetAttribute() == pcbnew.PAD_ATTRIB_PTH]
    vo = grid.obstacles(net.GetNetCode(), VIA_D / 2)
    vfree = ~vo[pcbnew.F_Cu] & ~vo[pcbnew.B_Cu]
    added = 0
    for z in board.Zones():
        if z.GetIsRuleArea() or z.GetNetCode() != net.GetNetCode():
            continue
        polys = z.GetFilledPolysList(z.GetLayer())
        for k in range(polys.OutlineCount()):
            one = pcbnew.SHAPE_POLY_SET()
            one.AddOutline(polys.Outline(k))
            if any(one.Contains(a) for a in anchors):
                continue
            inner = pcbnew.SHAPE_POLY_SET(one)
            inner.Inflate(-nm(VIA_D / 2 + 0.1), 8)
            if inner.OutlineCount() == 0:
                continue
            bb = one.BBox()
            j1, i1 = grid.ij(mm(bb.GetLeft()), mm(bb.GetTop()))
            j2, i2 = grid.ij(mm(bb.GetRight()), mm(bb.GetBottom()))
            cj, ci = (j1 + j2) // 2, (i1 + i2) // 2
            cands = [(abs(j - cj) + abs(i - ci), j, i)
                     for j in range(max(j1, 0), min(j2 + 1, grid.ny), 4)
                     for i in range(max(i1, 0), min(i2 + 1, grid.nx), 4) if vfree[j, i]]
            for _, j, i in sorted(cands):
                x, y = grid.xy(j, i)
                p = pcbnew.VECTOR2I(nm(x), nm(y))
                if inner.Contains(p):
                    add_via(board, net, x, y)
                    anchors.append(p)
                    rr = int((VIA_D + CLR) / G) + 1
                    vfree[max(j - rr, 0):j + rr + 1, max(i - rr, 0):i + rr + 1] = False
                    added += 1
                    break
    return added


def place_fiducials(board, grid):
    for fp in board.GetFootprints():
        if not fp.GetReference().startswith("FID"):
            continue
        L = pcbnew.B_Cu if fp.IsFlipped() else pcbnew.F_Cu
        obs = grid.obstacles(-1, 1.0)[L]      # 1 mm pad + 1 mm mask ring keeps it clean
        p = fp.GetPosition()
        j0, i0 = grid.ij(mm(p.x), mm(p.y))
        best = None
        for r in range(0, int(15 / G)):
            for dj in range(-r, r + 1):
                for di in (-r, r) if abs(dj) != r else range(-r, r + 1):
                    j, i = j0 + dj, i0 + di
                    if 0 <= j < grid.ny and 0 <= i < grid.nx and not obs[j, i]:
                        best = (j, i)
                        break
                if best:
                    break
            if best:
                break
        if best:
            x, y = grid.xy(*best)
            fp.SetPosition(pcbnew.VECTOR2I(nm(x), nm(y)))
            print(f"  {fp.GetReference()} -> ({x - 100:.1f}, {200 - y:.1f}) case mm")


def main():
    board = pcbnew.LoadBoard(PCB)
    if "--signals-only" in sys.argv:
        pairs = unconnected(board)
        route_signals(board, Grid(board), pairs)
        board.Save(PCB)
        return
    for rnd in range(4):
        pairs = unconnected(board)
        print(f"round {rnd + 1}: {len(pairs)} unconnected "
              f"({sum(1 for p in pairs if p[0] != 'GND')} signal, {sum(1 for p in pairs if p[0] == 'GND')} GND)")
        if not pairs:
            break
        grid = Grid(board)
        progress = route_signals(board, grid, pairs)
        grid = Grid(board)
        progress += stitch_gnd(board, grid, pairs)
        grid = Grid(board)
        n_isl = stitch_islands(board, grid)
        print(f"  pour-island stitching vias: {n_isl}")
        progress += n_isl
        if not progress:
            break
    place_fiducials(board, Grid(board))
    pairs = unconnected(board)
    board.Save(PCB)
    print(f"done: {len(pairs)} unconnected left; saved {PCB}")


if __name__ == "__main__":
    main()
