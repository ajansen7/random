"""
KiCad helper: draw the board outline and move footprints to where the case
expects them, using hardware/layout.py as the source of truth.

Use (KiCad 8 or 9, PCB Editor -> Tools -> Scripting Console):

    import sys; sys.path.insert(0, r"/path/to/tattle-phone/hardware")
    import place_footprints; place_footprints.run()

It is safe to run again after changing layout.py: it removes the outline it
drew last time and re-places the parts. It only moves the footprints listed in
layout.placements(). Place U1 (ESP32), J1 (USB-C) and the rest by hand using
the notes in hardware/SCHEMATIC.md.

KiCad's Y axis points down, so case (x, y) -> board (ORIGIN_X + x, ORIGIN_Y - y).
"""
import pcbnew

import layout

ORIGIN_X, ORIGIN_Y = 100.0, 200.0   # where case (0, 0) lands on the KiCad sheet, mm
OUTLINE_TAG = "tattle-outline"


def _pt(x, y):
    return pcbnew.VECTOR2I(pcbnew.FromMM(ORIGIN_X + x), pcbnew.FromMM(ORIGIN_Y - y))


def _draw_outline(board):
    for d in list(board.GetDrawings()):  # remove a previous run's outline
        if d.GetLayer() == pcbnew.Edge_Cuts and d.GetParentGroup() and \
                d.GetParentGroup().GetName() == OUTLINE_TAG:
            board.Remove(d)
    for g in list(board.Groups()):
        if g.GetName() == OUTLINE_TAG:
            board.Remove(g)

    group = pcbnew.PCB_GROUP(board)
    group.SetName(OUTLINE_TAG)
    board.Add(group)
    for seg in layout.pcb_outline():
        s = pcbnew.PCB_SHAPE(board)
        s.SetLayer(pcbnew.Edge_Cuts)
        s.SetWidth(pcbnew.FromMM(0.1))
        if seg[0] == "line":
            s.SetShape(pcbnew.SHAPE_T_SEGMENT)
            s.SetStart(_pt(*seg[1]))
            s.SetEnd(_pt(*seg[2]))
        else:
            s.SetShape(pcbnew.SHAPE_T_ARC)
            s.SetArcGeometry(_pt(*seg[1]), _pt(*seg[2]), _pt(*seg[3]))
        board.Add(s)
        group.AddItem(s)


def _set_side(fp, side):
    want_back = side == "back"
    if fp.IsFlipped() == want_back:
        return
    try:  # KiCad 9
        fp.Flip(fp.GetPosition(), pcbnew.FLIP_DIRECTION_LEFT_RIGHT)
    except (AttributeError, TypeError):  # KiCad 8
        fp.Flip(fp.GetPosition(), False)


def _pad_centroid(fp):
    pads = list(fp.Pads())
    if not pads:
        return fp.GetPosition()
    xs = [p.GetPosition().x for p in pads]
    ys = [p.GetPosition().y for p in pads]
    return pcbnew.VECTOR2I(int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))


def run():
    board = pcbnew.GetBoard()
    _draw_outline(board)
    missing = []
    for ref, (x, y, rot, side) in layout.placements().items():
        fp = board.FindFootprintByReference(ref)
        if fp is None:
            missing.append(ref)
            continue
        _set_side(fp, side)
        fp.SetOrientationDegrees(rot)
        # Footprint origins vary (pin 1 vs centre), so line up the pad centroid
        # with the target. Mounting holes have a single pad, so this is exact.
        target = _pt(x, y)
        c = _pad_centroid(fp)
        fp.Move(pcbnew.VECTOR2I(target.x - c.x, target.y - c.y))
        fp.SetLocked(True)
    pcbnew.Refresh()
    print("placed", len(layout.placements()) - len(missing), "footprints")
    if missing:
        print("not in board yet (add to schematic, then Update PCB):", ", ".join(missing))
