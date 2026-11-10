"""
Generate the Tattle Phone Lite KiCad project from design.py.

    python3 build.py [out_dir]      (default: ../kicad)

Writes tattle_lite.kicad_pro / .kicad_sch / .kicad_pcb, tattle_lite.kicad_sym and
tattle_lite.pretty/ (the mic footprint, drawn from its datasheet).
Needs KiCad's Python module (pcbnew) and the stock symbol + footprint libraries.
Built and checked with KiCad 7.0; KiCad 8-10 open and upgrade the files.

Schematic: one sheet, symbols grouped by block, a net label on every pin
(so it reads like a netlist; redraw wires later if you prefer).
Board: outline + every footprint on its net, placed per design.py, GND pours on
both layers, antenna keep-out, net classes. Tracks are NOT routed (route.py does that).
"""
import json
import math
import os
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import pcbnew  # noqa: E402

import design  # noqa: E402
import layout as LY  # noqa: E402
import place_footprints  # noqa: E402
from sexp import Sym, dump, find, find1  # noqa: E402
from symlib import load_symbol, pins as sym_pins  # noqa: E402

FP_DIR = os.environ.get("KICAD_FOOTPRINT_DIR", "/usr/share/kicad/footprints")
NAME = "tattle_lite"
LIB = "tattle_lite"
S = Sym


def uid():
    return str(uuid.uuid4())


# ------------------------------------------------------------------ custom symbols
def box_symbol(name, left, right, bottom=(), top=(), ref="U", fp=""):
    """A plain rectangle symbol. Each side is a list of (number, name, etype)."""
    n = max(len(left), len(right), 1)
    h = (n + 1) * 2.54
    w = 15.24
    body = [S("symbol"), f"{name}_0_1",
            [S("rectangle"), [S("start"), -w / 2, h / 2], [S("end"), w / 2, -h / 2],
             [S("stroke"), [S("width"), 0.254], [S("type"), S("default")]], [S("fill"), [S("type"), S("background")]]]]
    pins = [S("symbol"), f"{name}_1_1"]

    def pin(num, nm, et, x, y, ang):
        return [S("pin"), S(et), S("line"), [S("at"), x, y, ang], [S("length"), 2.54],
                [S("name"), nm, [S("effects"), [S("font"), [S("size"), 1.27, 1.27]]]],
                [S("number"), num, [S("effects"), [S("font"), [S("size"), 1.27, 1.27]]]]]

    for i, (num, nm, et) in enumerate(left):
        pins.append(pin(num, nm, et, -w / 2 - 2.54, h / 2 - 2.54 * (i + 1), 0))
    for i, (num, nm, et) in enumerate(right):
        pins.append(pin(num, nm, et, w / 2 + 2.54, h / 2 - 2.54 * (i + 1), 180))
    for i, (num, nm, et) in enumerate(bottom):
        pins.append(pin(num, nm, et, -2.54 * (len(bottom) - 1) / 2 + 2.54 * i, -h / 2 - 2.54, 90))
    for i, (num, nm, et) in enumerate(top):
        pins.append(pin(num, nm, et, -2.54 * (len(top) - 1) / 2 + 2.54 * i, h / 2 + 2.54, 270))

    def prop(k, v, y, hide=False):
        eff = [S("effects"), [S("font"), [S("size"), 1.27, 1.27]]]
        if hide:
            eff.append(S("hide"))
        return [S("property"), k, v, [S("at"), 0, y, 0], eff]

    return [S("symbol"), name, [S("in_bom"), S("yes")], [S("on_board"), S("yes")],
            prop("Reference", ref, h / 2 + 1.5), prop("Value", name, -h / 2 - 4.5),
            prop("Footprint", fp, 0, True), prop("Datasheet", "", 0, True), body, pins]


MIC_FP = "LinkMems_LMD3526B261_3.5x2.65mm"
MIC = box_symbol(
    "LMD3526B261",
    left=[("4", "VDD", "power_in"), ("3", "CLK", "input"), ("2", "L/R", "input")],
    right=[("1", "DATA", "output")],
    bottom=[("5", "GND", "power_in")],
    ref="MK", fp=f"{LIB}:{MIC_FP}")


def write_symbol_lib(path):
    lib = [S("kicad_symbol_lib"), [S("version"), 20220914], [S("generator"), S("tattle_build")], MIC]
    open(path, "w").write(dump(lib) + "\n")


def get_symbol(lib_id, local_lib):
    if lib_id.startswith(LIB + ":"):
        import copy
        s = copy.deepcopy(MIC)
        s[1] = lib_id
        return s
    return load_symbol(lib_id)


def write_mic_footprint(pretty_dir):
    """LinkMems LMD3526B261-OFA03 (PDM, bottom port), from datasheet section 7-8.
    Origin = acoustic port. Top view (component side): pads 1 DATA / 4 VDD in the row
    furthest from the port, 2 L/R / 3 CLK in the next row, 5 = GND ring round the port.
    Pads 0.73 x 0.52 on a 1.67 (x) x 0.82 (y) grid, rows 2.07 / 1.25 mm from the port;
    ring 1.62 OD / 1.03 ID; body 2.65 x 3.50, 2.46 mm above / 1.04 mm below the port.
    Board hole 0.5 mm (inside the 1.03 ring ID with 0.26 mm to copper; the mic's own port is 0.32 mm)."""
    os.makedirs(pretty_dir, exist_ok=True)
    def pad(n, x, y):
        return (f'  (pad "{n}" smd roundrect (at {x} {y}) (size 0.73 0.52) (layers "F.Cu" "F.Paste" "F.Mask")'
                f' (roundrect_rratio 0.15))\n')
    ring_r, ring_w = (1.62 + 1.03) / 4, (1.62 - 1.03) / 2
    body = [(-1.325, -2.46), (1.325, -2.46), (1.325, 1.04), (-1.325, 1.04)]
    def rect(layer, pts, w):
        out = ""
        for i in range(4):
            (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % 4]
            out += f'  (fp_line (start {x1} {y1}) (end {x2} {y2}) (stroke (width {w}) (type solid)) (layer "{layer}"))\n'
        return out
    crt = [(-1.6, -2.75), (1.6, -2.75), (1.6, 1.35), (-1.6, 1.35)]
    txt = (f'(footprint "{MIC_FP}" (version 20221018) (generator tattle_build) (layer "F.Cu")\n'
           '  (descr "LinkMems LMD3526B261-OFA03 PDM MEMS microphone, bottom port, 3.5x2.65 mm (datasheet M196 v1.0)")\n'
           '  (attr smd)\n'
           '  (fp_text reference "REF**" (at 0 -3.6) (layer "F.SilkS") (effects (font (size 1 1) (thickness 0.15))))\n'
           f'  (fp_text value "{MIC_FP}" (at 0 2.4) (layer "F.Fab") (effects (font (size 0.6 0.6) (thickness 0.1))))\n'
           + rect("F.Fab", body, 0.1) + rect("F.CrtYd", crt, 0.05)
           + '  (fp_circle (center -1.75 -2.9) (end -1.6 -2.9) (stroke (width 0.12) (type solid)) (fill solid) (layer "F.SilkS"))\n'
           + pad(1, -0.835, -2.07) + pad(4, 0.835, -2.07) + pad(2, -0.835, -1.25) + pad(3, 0.835, -1.25)
           + f'  (pad "5" smd custom (at 0 {ring_r}) (size 0.29 0.29) (layers "F.Cu" "F.Paste" "F.Mask")\n'
             '    (options (clearance outline) (anchor circle))\n'
             f'    (primitives (gr_circle (center 0 {-ring_r}) (end {ring_r} {-ring_r}) (width {ring_w}) (fill no))))\n'
           + '  (pad "" np_thru_hole circle (at 0 0) (size 0.5 0.5) (drill 0.5) (layers "*.Cu" "*.Mask"))\n'
           ')\n')
    open(os.path.join(pretty_dir, MIC_FP + ".kicad_mod"), "w").write(txt)


def write_led_footprint(pretty_dir):
    """KiCad's reverse-mount 1206 LED with the lens hole narrowed along the pad axis
    (1.8 -> 1.65 mm) so the pads clear the hole by 0.25 mm. The LTST-C230 lens is
    1.49 x 1.6 mm, so it still drops through."""
    src = open(os.path.join(FP_DIR, "LED_SMD.pretty", "LED_1206_3216Metric_ReverseMount_Hole1.8x2.4mm.kicad_mod")).read()
    name = "LED_1206_3216Metric_ReverseMount_Hole1.65x2.4mm"
    out = src.replace("LED_1206_3216Metric_ReverseMount_Hole1.8x2.4mm", name)
    out = out.replace("(size 1.8 2.4)", "(size 1.65 2.4)").replace("(drill oval 1.8 2.4)", "(drill oval 1.65 2.4)")
    assert "1.65 2.4" in out, "LED footprint hole not found"
    open(os.path.join(pretty_dir, name + ".kicad_mod"), "w").write(out)


# ------------------------------------------------------------------ pin -> net resolution
def resolve(part, pins):
    """{pin_number: net or None} for every pin of the symbol."""
    by_name = {}
    for p in pins:
        by_name.setdefault(p["name"], []).append(p["num"])
    out = {p["num"]: None for p in pins}
    for key, net in part["pins"].items():
        nums = by_name.get(key) or ([key] if key in out else None)
        if not nums:
            raise SystemExit(f"{part['ref']}: symbol {part['lib']} has no pin '{key}'")
        for n in nums:
            out[n] = net
    return out


# ------------------------------------------------------------------ schematic
GRID = 2.54


def snap(v):
    return round(v / GRID) * GRID


def sym_bbox(pins):
    if not pins:
        return (-2.54, -2.54, 2.54, 2.54)
    xs = [p["x"] for p in pins]
    ys = [p["y"] for p in pins]
    return (min(xs), min(ys), max(xs), max(ys))


def text_node(s, x, y, size=1.27, bold=False):
    font = [S("font"), [S("size"), size, size]]
    if bold:
        font.append(S("bold"))
    return [S("text"), s, [S("at"), x, y, 0], [S("effects"), font, [S("justify"), S("left"), S("bottom")]],
            [S("uuid"), uid()]]


def build_schematic(path, parts, root_uuid):
    lib_symbols = {}
    items = []
    netmap = {}  # (ref, pin_num) -> net, for cross-checking with the board
    sym_uuids = {}

    # group by block, keep order of first appearance
    blocks = []
    for p in parts:
        if p["block"] not in blocks:
            blocks.append(p["block"])

    LABEL_ROOM = 22.0     # space for net labels on each side
    col_x, col_w_max = 25.4, 0.0
    max_y = 0.0
    x0, y0 = 25.4, 30.0
    col_height_limit = 380.0
    bx, by = x0, y0
    col_width = 0.0

    # extra PWR_FLAGs so ERC knows these nets are driven
    flags = design.FLAG_NETS
    flag_parts = [dict(ref=f"#FLG{i + 1:02d}", lib="power:PWR_FLAG", value="PWR_FLAG", fp="",
                       pins={"1": n}, block="Power flags", dnp=False, note="") for i, n in enumerate(flags)]

    all_parts = parts + flag_parts
    blocks.append("Power flags")
    for block in blocks:
        members = [p for p in all_parts if p["block"] == block]
        # measure block
        laid = []
        bw = 0.0
        cy = 8.0
        row_x, row_h = 0.0, 0.0
        row_limit = 150.0
        for p in members:
            sym = get_symbol(p["lib"], None)
            ps = sym_pins(sym)
            x1, y1, x2, y2 = sym_bbox(ps)
            w = (x2 - x1) + 2 * LABEL_ROOM
            h = (y2 - y1) + 12.0
            if row_x + w > row_limit and row_x > 0:
                cy += row_h
                row_x, row_h = 0.0, 0.0
            laid.append((p, sym, ps, row_x - x1 + LABEL_ROOM, cy + y2 + 4.0))
            row_x += w
            row_h = max(row_h, h)
            bw = max(bw, row_x)
        bh = cy + row_h
        if by + bh > y0 + col_height_limit and by > y0:
            bx += col_width + 15.0
            by = y0
            col_width = 0.0
        items.append(text_node(block, bx, by + 2.0, size=2.54, bold=True))
        for p, sym, ps, ox, oy in laid:
            sx, sy = snap(bx + ox), snap(by + oy)
            lib_symbols[p["lib"]] = sym
            nets = resolve(p, ps)
            su = uid()
            sym_uuids[p["ref"]] = su
            node = [S("symbol"), [S("lib_id"), p["lib"]], [S("at"), sx, sy, 0], [S("unit"), 1],
                    [S("in_bom"), S("no") if p["ref"].startswith("#") else S("yes")],
                    [S("on_board"), S("no") if p["ref"].startswith("#") else S("yes")],
                    [S("dnp"), S("yes") if p["dnp"] else S("no")], [S("uuid"), su]]
            x1, y1, x2, y2 = sym_bbox(ps)

            def prop(k, v, x, y, hide=False):
                eff = [S("effects"), [S("font"), [S("size"), 1.27, 1.27]]]
                if hide:
                    eff.append(S("hide"))
                return [S("property"), k, v, [S("at"), x, y, 0], eff]

            node.append(prop("Reference", p["ref"], sx, sy - y2 - 3.0, p["ref"].startswith("#")))
            node.append(prop("Value", p["value"], sx, sy - y1 + 3.0, p["ref"].startswith("#")))
            node.append(prop("Footprint", p["fp"], sx, sy, True))
            if p.get("note"):
                node.append(prop("Note", p["note"], sx, sy - y1 + 5.5, True))
            for pp in ps:
                node.append([S("pin"), pp["num"], [S("uuid"), uid()]])
            node.append([S("instances"), [S("project"), NAME,
                         [S("path"), f"/{root_uuid}", [S("reference"), p["ref"]], [S("unit"), 1]]]])
            items.append(node)

            # labels / no-connects on every pin end
            for pp in ps:
                px, py = sx + pp["x"], sy - pp["y"]
                net = nets[pp["num"]]
                if not p["ref"].startswith("#"):
                    netmap[(p["ref"], pp["num"])] = net
                if net is None:
                    if pp["etype"] not in ("no_connect",):
                        items.append([S("no_connect"), [S("at"), px, py], [S("uuid"), uid()]])
                    continue
                ang = (pp["angle"] + 180) % 360
                just = "left" if ang in (0, 90) else "right"
                items.append([S("label"), net, [S("at"), px, py, ang], [S("fields_autoplaced")],
                              [S("effects"), [S("font"), [S("size"), 1.27, 1.27]], [S("justify"), S(just)]],
                              [S("uuid"), uid()]])
        by += bh + 12.0
        col_width = max(col_width, bw)
        max_y = max(max_y, by)

    width = bx + col_width + 25.0
    height = max(max_y, y0 + col_height_limit) + 25.0
    notes = ["Generated by hardware/kicad_gen/build.py from design.py. Edit design.py and rebuild",
             "rather than editing this sheet, or the board and schematic can drift apart.",
             "Every pin carries a net label; matching labels are connected."]
    for i, n in enumerate(notes):
        items.append(text_node(n, 25.4, 12.0 + i * 4.0, size=1.8))

    sch = [S("kicad_sch"), [S("version"), 20230121], [S("generator"), S("eeschema")],
           [S("uuid"), root_uuid], [S("paper"), "User", round(width), round(height)],
           [S("title_block"), [S("title"), "Tattle Phone Lite"], [S("rev"), "0.1"],
            [S("company"), ""], [S("comment"), 1, "Generated from hardware/kicad_gen/design.py"]],
           [S("lib_symbols")] + list(lib_symbols.values())]
    sch += items
    sch.append([S("sheet_instances"), [S("path"), "/", [S("page"), "1"]]])
    open(path, "w").write(dump(sch) + "\n")
    return netmap, sym_uuids


# ------------------------------------------------------------------ board
def kpt(x, y):
    return pcbnew.VECTOR2I(pcbnew.FromMM(place_footprints.ORIGIN_X + x),
                           pcbnew.FromMM(place_footprints.ORIGIN_Y - y))


def pad_centroid(fp, numbered_only=True):
    pads = [p for p in fp.Pads() if (p.GetNumber() or not numbered_only)]
    if not pads:
        return fp.GetPosition()
    return pcbnew.VECTOR2I(int(sum(p.GetPosition().x for p in pads) / len(pads)),
                           int(sum(p.GetPosition().y for p in pads) / len(pads)))


def place(fp, part):
    x, y = part["at"]
    if part["side"] == "B" and not fp.IsFlipped():
        fp.Flip(fp.GetPosition(), True)
    # KiCad flips to the back by mirroring top/bottom then turning 180 deg, so a
    # back-side part that should keep "up" pointing up has orientation rot + 180.
    fp.SetOrientationDegrees((part["rot"] + (180 if part["side"] == "B" else 0)) % 360)
    target = kpt(x, y)
    if part["fp"].startswith("Button_Switch_THT"):
        c = pad_centroid(fp)  # THT switch origins sit on pin 1
        fp.Move(pcbnew.VECTOR2I(target.x - c.x, target.y - c.y))
    elif part["fp"].startswith("Connector_JST"):
        fp.SetPosition(target)  # origin is pin 1: centre the body instead
        bb = fp.GetBoundingBox(False, False)
        fp.Move(pcbnew.VECTOR2I(target.x - bb.GetCenter().x, target.y - bb.GetCenter().y))
    else:
        fp.SetPosition(target)


def orient_slide_switch(fp, part):
    """Turn the PCM12 so its actuator (pins -> pegs direction) points out of the nearer side edge."""
    want = 1 if part["at"][0] > 0 else -1
    for rot in (0, 90, 180, 270):
        p2 = dict(part, rot=rot)
        place(fp, p2)
        pin = [p for p in fp.Pads() if p.GetNumber() == "2"][0].GetPosition()
        pegs = [p for p in fp.Pads() if p.GetAttribute() == pcbnew.PAD_ATTRIB_NPTH]
        peg = pcbnew.VECTOR2I(int(sum(p.GetPosition().x for p in pegs) / 2), int(sum(p.GetPosition().y for p in pegs) / 2))
        dx, dy = peg.x - pin.x, peg.y - pin.y
        if dx * want > 0 and abs(dx) > abs(dy):
            # put the pegs' line just inside the board edge so the actuator overhangs
            return rot
    raise RuntimeError("could not orient slide switch")


def add_zone(board, net, layer, pts, priority=0, rule_area=False):
    z = pcbnew.ZONE(board)
    z.SetLayer(layer)
    if rule_area:
        z.SetIsRuleArea(True)
        getattr(z, "SetDoNotAllowCopperPour", None) and z.SetDoNotAllowCopperPour(True)
        getattr(z, "SetDoNotAllowZoneFills", None) and z.SetDoNotAllowZoneFills(True)   # KiCad 9/10 name
        z.SetDoNotAllowTracks(True)
        z.SetDoNotAllowVias(True)
        z.SetDoNotAllowPads(False)
        z.SetDoNotAllowFootprints(False)
        ls = pcbnew.LSET()
        ls.AddLayer(pcbnew.F_Cu)
        ls.AddLayer(pcbnew.B_Cu)
        z.SetLayerSet(ls)
    else:
        z.SetNetCode(net.GetNetCode())
        z.SetAssignedPriority(priority)
        z.SetLocalClearance(pcbnew.FromMM(0.2))
        z.SetMinThickness(pcbnew.FromMM(0.15))
        z.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
        z.SetIslandRemovalMode(pcbnew.ISLAND_REMOVAL_MODE_ALWAYS)
        z.SetThermalReliefGap(pcbnew.FromMM(0.3))
        z.SetThermalReliefSpokeWidth(pcbnew.FromMM(0.35))
    ol = z.Outline()
    ol.NewOutline()
    for x, y in pts:
        p = kpt(x, y)
        ol.Append(p.x, p.y)
    board.Add(z)
    return z


def build_board(path, parts, netmap, sym_uuids):
    board = pcbnew.NewBoard(path)
    ds = board.GetDesignSettings()
    ds.SetCopperLayerCount(2)
    ds.m_TrackMinWidth = pcbnew.FromMM(0.15)
    ds.m_ViasMinSize = pcbnew.FromMM(0.5)
    ds.m_MinThroughDrill = pcbnew.FromMM(0.3)   # keep every hole >= 0.3 mm: JLC's cheap drill class
    ds.m_MinClearance = pcbnew.FromMM(0.15)

    nets = {}

    def net(name):
        if name not in nets:
            n = pcbnew.NETINFO_ITEM(board, name)
            board.Add(n)
            nets[name] = n
        return nets[name]

    for name in sorted({n for n in netmap.values() if n}):
        net(name)

    place_footprints._draw_outline(board)

    problems = []
    local_pretty = os.path.join(os.path.dirname(path), LIB + ".pretty")
    for part in parts:
        lib, name = part["fp"].split(":")
        fp = pcbnew.FootprintLoad(local_pretty if lib == LIB else os.path.join(FP_DIR, lib + ".pretty"), name)
        if fp is None:
            raise SystemExit(f"{part['ref']}: footprint {part['fp']} not found")
        fp.SetFPID(pcbnew.LIB_ID(lib, name))
        fp.SetReference(part["ref"])
        fp.SetValue(part["value"])
        fp.SetPath(pcbnew.KIID_PATH(f"/{sym_uuids[part['ref']]}"))
        if part["dnp"]:
            fp.SetAttributes(fp.GetAttributes() | pcbnew.FP_EXCLUDE_FROM_BOM | pcbnew.FP_EXCLUDE_FROM_POS_FILES)
        board.Add(fp)
        if part["fp"].endswith(("PCM12", "MSK12C02")):
            part = dict(part, rot=orient_slide_switch(fp, part))
        place(fp, part)
        fp.SetLocked(part["ref"] in place_footprints.layout.placements() or part["ref"] == "J1")
        for pad in fp.Pads():
            num = pad.GetNumber()
            if not num:
                continue
            key = (part["ref"], num)
            if key not in netmap:
                if num in ("SH", "MP"):   # mechanical tabs (slide switch), no net
                    continue
                problems.append(f"{part['ref']} pad {num} has no matching symbol pin")
                continue
            if netmap[key]:
                pad.SetNet(net(netmap[key]))
        # symbol pins with no pad
        pad_nums = {p.GetNumber() for p in fp.Pads()}
        for (ref, num), n in netmap.items():
            if ref == part["ref"] and num not in pad_nums and n:
                problems.append(f"{ref} pin {num} ({n}) has no pad in {part['fp']}")

    # GND via in the amp's exposed pad (its pin traces ring the pad, so the pour can't
    # reach it; the datasheet wants the EP on ground with vias anyway). JLC "plugged" vias.
    for ref in design.EP_VIA_PARTS:
        fp = board.FindFootprintByReference(ref)
        ep = max(fp.Pads(), key=lambda p: p.GetSize().x * p.GetSize().y)
        v = pcbnew.PCB_VIA(board)
        v.SetPosition(ep.GetPosition())
        v.SetWidth(pcbnew.FromMM(0.6))
        v.SetDrill(pcbnew.FromMM(0.3))
        v.SetNet(ep.GetNet())
        v.SetLocked(True)          # route.py keeps locked items and routes around them
        board.Add(v)

    # keep tracks and vias 0.3 mm off the LED lens holes (they are NPTH cut-outs)
    for fp in board.GetFootprints():
        if not fp.GetReference().startswith("LED"):
            continue
        for p in fp.Pads():
            if p.GetAttribute() != pcbnew.PAD_ATTRIB_NPTH:
                continue
            bb = p.GetBoundingBox()
            g = pcbnew.FromMM(0.3)
            z = pcbnew.ZONE(board)
            z.SetIsRuleArea(True)
            z.SetDoNotAllowTracks(True)
            z.SetDoNotAllowVias(True)
            z.SetDoNotAllowPads(False)
            z.SetDoNotAllowFootprints(False)
            getattr(z, "SetDoNotAllowCopperPour", None) and z.SetDoNotAllowCopperPour(False)
            getattr(z, "SetDoNotAllowZoneFills", None) and z.SetDoNotAllowZoneFills(False)
            ls = pcbnew.LSET(); ls.AddLayer(pcbnew.F_Cu); ls.AddLayer(pcbnew.B_Cu); z.SetLayerSet(ls)
            ol = z.Outline(); ol.NewOutline()
            for x, y in ((bb.GetLeft() - g, bb.GetTop() - g), (bb.GetRight() + g, bb.GetTop() - g),
                         (bb.GetRight() + g, bb.GetBottom() + g), (bb.GetLeft() - g, bb.GetBottom() + g)):
                ol.Append(int(x), int(y))
            board.Add(z)

    # copper pours and antenna keep-out
    outline = [(c[0] * 1.2, c[1]) for c in []]  # placeholder (zone uses a generous box; it is clipped to Edge.Cuts)
    box = [(-35, -5), (35, -5), (35, LY.PCB_TOP_Y + 5), (-35, LY.PCB_TOP_Y + 5)]
    add_zone(board, net("GND"), pcbnew.B_Cu, box)
    add_zone(board, net("GND"), pcbnew.F_Cu, box)
    ant_y0 = LY.PCB_TOP_Y - 6.5
    add_zone(board, None, pcbnew.F_Cu, [(-11, ant_y0), (11, ant_y0), (11, LY.PCB_TOP_Y + 5), (-11, LY.PCB_TOP_Y + 5)],
             rule_area=True)

    # silkscreen notes
    def silk(txt, x, y, layer=pcbnew.B_SilkS, size=1.0):
        t = pcbnew.PCB_TEXT(board)
        t.SetText(txt)
        t.SetPosition(kpt(x, y))
        t.SetLayer(layer)
        t.SetTextSize(pcbnew.VECTOR2I(pcbnew.FromMM(size), pcbnew.FromMM(size)))
        t.SetTextThickness(pcbnew.FromMM(size * 0.15))
        if layer == pcbnew.B_SilkS:
            t.SetMirrored(True)
        board.Add(t)

    import tidy
    moved, hidden = tidy.tidy_silk(board)
    print(f"silkscreen: {moved} labels placed, {hidden} hidden (too tight; still on Fab layer)")
    silk("TATTLE PHONE LITE v0.1", 0, 76.0, size=1.2)
    silk("POWER", 21.5, LY.POWER_SWITCH_Y - 6.0, size=0.8)
    silk("BAT+", LY.BATT_PADS[0][0] + 0.2, LY.BATT_PADS[0][1] - 2.2, size=0.8)
    silk("BAT-", LY.BATT_PADS[1][0] + 0.2, LY.BATT_PADS[1][1] + 2.2, size=0.8)
    silk("SPK+", LY.SPK_PADS[0][0] + 3.6, LY.SPK_PADS[0][1], size=0.8)
    silk("SPK-", LY.SPK_PADS[1][0] + 3.6, LY.SPK_PADS[1][1], size=0.8)
    silk("CLIP SWITCH LEADS FLUSH", 0, 45.0, size=0.8)
    silk("ANTENNA KEEP-OUT", 0, LY.PCB_TOP_Y - 3.0, layer=pcbnew.F_SilkS, size=0.8)

    board.BuildConnectivity()
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    board.Save(path)
    return board, problems


def write_project(path):
    """Add net classes to the .kicad_pro that NewBoard created."""
    try:
        pro = json.load(open(path))
    except (OSError, ValueError):
        pro = {"meta": {"filename": os.path.basename(path), "version": 1}}
    ns = pro.setdefault("net_settings", {})
    ns["meta"] = {"version": 3}
    base = {"bus_width": 12, "clearance": 0.2, "diff_pair_gap": 0.25, "diff_pair_via_gap": 0.25,
            "diff_pair_width": 0.2, "line_style": 0, "microvia_diameter": 0.3, "microvia_drill": 0.1,
            "pcb_color": "rgba(0, 0, 0, 0.000)", "schematic_color": "rgba(0, 0, 0, 0.000)",
            "track_width": 0.2, "via_diameter": 0.6, "via_drill": 0.3, "wire_width": 6}
    ns["classes"] = [dict(base, name="Default"),
                     dict(base, name="Power", track_width=0.4, via_diameter=0.7, via_drill=0.35),
                     dict(base, name="USB", track_width=0.2, diff_pair_width=0.2, diff_pair_gap=0.15)]
    ns["netclass_patterns"] = [{"netclass": "Power", "pattern": n} for n in design.POWER_NETS] + \
                              [{"netclass": "USB", "pattern": "USB_D*"}]
    pro.setdefault("libraries", {})
    json.dump(pro, open(path, "w"), indent=2)


def write_rules(out):
    """Board-specific DRC rules. KiCad treats an NPTH oval hole like the board edge (0.5 mm);
    the LED lens holes sit 0.25 mm from their own pads, which JLC accepts (NPTH-to-copper 0.2 mm)."""
    open(os.path.join(out, NAME + ".kicad_dru"), "w").write(
        '(version 1)\n'
        '(rule "reverse-mount LED lens holes"\n'
        '  (constraint edge_clearance (min 0.2mm))\n'
        "  (condition \"A.memberOfFootprint('LED*') || B.memberOfFootprint('LED*')\"))\n")


ESP32_FP = "ESP32-S3-WROOM-1_ThermalVias0.3mm"


def write_esp32_footprint(pretty_dir):
    """KiCad's ESP32-S3-WROOM-1 with its 12 ground-pad thermal vias drilled 0.3 mm instead of
    0.2 mm. 0.2 mm holes put a 2-layer board in JLC's expensive drill class (~$7 -> ~$58);
    the vias are ~1 mm apart, so 0.3 mm still leaves 0.7 mm between holes."""
    src = open(os.path.join(FP_DIR, "RF_Module.pretty", "ESP32-S3-WROOM-1.kicad_mod")).read()
    n = src.count("(drill 0.2)")
    assert n == 12, f"expected 12 thermal vias in the ESP32 footprint, found {n}"
    out = src.replace("(drill 0.2)", "(drill 0.3)")
    out = out.replace('(footprint "ESP32-S3-WROOM-1"', f'(footprint "{ESP32_FP}"', 1)
    open(os.path.join(pretty_dir, ESP32_FP + ".kicad_mod"), "w").write(out)


def write_lib_tables(out):
    open(os.path.join(out, "sym-lib-table"), "w").write(
        f'(sym_lib_table\n  (lib (name "{LIB}")(type "KiCad")(uri "${{KIPRJMOD}}/{LIB}.kicad_sym")(options "")(descr "Tattle Phone Lite parts"))\n)\n')
    open(os.path.join(out, "fp-lib-table"), "w").write(
        f'(fp_lib_table\n  (lib (name "{LIB}")(type "KiCad")(uri "${{KIPRJMOD}}/{LIB}.pretty")(options "")(descr "Tattle Phone Lite footprints"))\n)\n')


def main():
    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "kicad"))
    os.makedirs(out, exist_ok=True)
    parts = design.PARTS
    root = uid()
    write_symbol_lib(os.path.join(out, LIB + ".kicad_sym"))
    write_mic_footprint(os.path.join(out, LIB + ".pretty"))
    write_led_footprint(os.path.join(out, LIB + ".pretty"))
    write_esp32_footprint(os.path.join(out, LIB + ".pretty"))
    write_lib_tables(out)
    write_rules(out)
    netmap, sym_uuids = build_schematic(os.path.join(out, NAME + ".kicad_sch"), parts, root)
    board, problems = build_board(os.path.join(out, NAME + ".kicad_pcb"), parts, netmap, sym_uuids)
    write_project(os.path.join(out, NAME + ".kicad_pro"))
    n_nets = len({n for n in netmap.values() if n})
    print(f"wrote {out}: {len(parts)} parts, {n_nets} nets")
    for p in problems:
        print("  !", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
