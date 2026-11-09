"""
Silkscreen tidy-up: move every reference designator to a free spot next to its
part (above, below, left, right; then a smaller size), so no label sits on a pad,
another label or the board edge. Labels that can't fit are hidden on the silk
layer. They're still on the Fab layer and in the assembly drawings.
"""
import pcbnew

MM = pcbnew.FromMM


def _rect(bb, grow=0.0):
    g = MM(grow)
    return (bb.GetLeft() - g, bb.GetTop() - g, bb.GetRight() + g, bb.GetBottom() + g)


def _hit(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _inside(r, edge):
    return r[0] >= edge[0] and r[1] >= edge[1] and r[2] <= edge[2] and r[3] <= edge[3]


def _poly_contains(outline, r):
    for x, y in ((r[0], r[1]), (r[2], r[1]), (r[0], r[3]), (r[2], r[3])):
        if not outline.Contains(pcbnew.VECTOR2I(int(x), int(y))):
            return False
    return True


def tidy_silk(board, edge_margin=0.4):
    # board outline as a polygon, shrunk a little so labels stay off the edge
    outline = pcbnew.SHAPE_POLY_SET()
    board.GetBoardPolygonOutlines(outline)
    outline.Inflate(-MM(edge_margin), 8)

    pads = {pcbnew.F_SilkS: [], pcbnew.B_SilkS: []}
    for fp in board.GetFootprints():
        for p in fp.Pads():
            r = _rect(p.GetBoundingBox(), 0.15)
            if p.IsOnLayer(pcbnew.F_Cu) or p.GetAttribute() in (pcbnew.PAD_ATTRIB_PTH, pcbnew.PAD_ATTRIB_NPTH):
                pads[pcbnew.F_SilkS].append(r)
            if p.IsOnLayer(pcbnew.B_Cu) or p.GetAttribute() in (pcbnew.PAD_ATTRIB_PTH, pcbnew.PAD_ATTRIB_NPTH):
                pads[pcbnew.B_SilkS].append(r)
    placed = {pcbnew.F_SilkS: [], pcbnew.B_SilkS: []}

    # Footprint artwork that can't be seen or clashes: LED outlines sit under the
    # keycaps (the pin-1 mark stays on the Fab layer for assembly); silk that runs
    # off the board edge (slide-switch levers) is dropped.
    for fp in board.GetFootprints():
        for g in list(fp.GraphicalItems()):
            if g.GetLayer() not in (pcbnew.F_SilkS, pcbnew.B_SilkS):
                continue
            if fp.GetReference().startswith("LED"):
                fp.Remove(g)
            elif not _poly_contains(outline, _rect(g.GetBoundingBox())):
                fp.Remove(g)

    # big parts first, so small ones fit around them
    fps = sorted(board.GetFootprints(), key=lambda f: -f.GetBoundingBox(False, False).GetArea())
    moved = hidden = 0
    for fp in fps:
        ref = fp.Reference()
        layer = pcbnew.B_SilkS if fp.IsFlipped() else pcbnew.F_SilkS
        if fp.GetReference().startswith(("H", "FID")):
            ref.SetVisible(False)
            continue
        ref.SetLayer(layer)
        ref.SetKeepUpright(True)
        body = fp.GetBoundingBox(False, False)
        cx, cy = body.GetCenter().x, body.GetCenter().y
        ok = False
        for size in (0.8, 0.7):
            ref.SetTextSize(pcbnew.VECTOR2I(MM(size), MM(size)))
            ref.SetTextThickness(MM(size * 0.15))
            for ang in (0, 90):
                ref.SetTextAngle(pcbnew.EDA_ANGLE(ang, pcbnew.DEGREES_T))
                ref.SetPosition(pcbnew.VECTOR2I(cx, cy))
                tb = ref.GetBoundingBox()
                hw, hh = tb.GetWidth() // 2, tb.GetHeight() // 2
                gap = MM(0.25)
                cands = [
                    (cx, body.GetTop() - hh - gap), (cx, body.GetBottom() + hh + gap),
                    (body.GetLeft() - hw - gap, cy), (body.GetRight() + hw + gap, cy),
                    (body.GetLeft() - hw - gap, body.GetTop() - hh - gap),
                    (body.GetRight() + hw + gap, body.GetTop() - hh - gap),
                    (body.GetLeft() - hw - gap, body.GetBottom() + hh + gap),
                    (body.GetRight() + hw + gap, body.GetBottom() + hh + gap),
                ]
                for x, y in cands:
                    ref.SetPosition(pcbnew.VECTOR2I(int(x), int(y)))
                    r = _rect(ref.GetBoundingBox())
                    if not _poly_contains(outline, r):
                        continue
                    if any(_hit(r, q) for q in pads[layer]) or any(_hit(r, q) for q in placed[layer]):
                        continue
                    placed[layer].append(_rect(ref.GetBoundingBox(), 0.1))
                    ok = True
                    break
                if ok:
                    break
            if ok:
                break
        if ok:
            moved += 1
        else:
            ref.SetVisible(False)
            hidden += 1
    return moved, hidden
