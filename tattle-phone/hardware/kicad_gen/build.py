"""
Generate the Tattle Phone KiCad project from design.py.

    python3 build.py [out_dir]      (default: ../kicad)

Writes tattle_phone.kicad_pro / .kicad_sch / .kicad_pcb and tattle.kicad_sym.
Needs KiCad's Python module (pcbnew) and the stock symbol + footprint libraries.
Built and checked with KiCad 7.0; KiCad 8-10 open and upgrade the files.

Schematic: one sheet, symbols grouped by block, a net label on every pin
(so it reads like a netlist; redraw wires later if you prefer).
Board: outline + every footprint on its net, placed per design.py, GND pours on
both layers, antenna keep-out, net classes. Tracks are NOT routed.
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
NAME = "tattle_phone"
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


MAX17048 = box_symbol(
    "MAX17048",
    left=[("3", "VDD", "power_in"), ("2", "CELL", "input"), ("1", "CTG", "input"), ("6", "QSTRT", "input")],
    right=[("8", "SDA", "bidirectional"), ("7", "SCL", "input"), ("5", "ALRT", "open_collector")],
    bottom=[("4", "GND", "power_in"), ("9", "EP", "passive")],
    fp="Package_DFN_QFN:TDFN-8-1EP_2x2mm_P0.5mm_EP0.8x1.2mm")


def write_symbol_lib(path):
    lib = [S("kicad_symbol_lib"), [S("version"), 20220914], [S("generator"), S("tattle_build")], MAX17048]
    open(path, "w").write(dump(lib) + "\n")


def get_symbol(lib_id, local_lib):
    if lib_id.startswith("tattle:"):
        import copy
        s = copy.deepcopy(MAX17048)
        s[1] = lib_id
        return s
    return load_symbol(lib_id)


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
    flags = ["GND", "VBUS_C", "VIN", "VSYS", "VBAT", "+3V3", "VLED"]
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
           [S("title_block"), [S("title"), "Tattle Phone"], [S("rev"), "0.1"],
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
        z.SetDoNotAllowCopperPour(True)
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
    ds.m_MinThroughDrill = pcbnew.FromMM(0.2)   # thermal vias under U1/U3 are 0.2 mm
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
    for part in parts:
        lib, name = part["fp"].split(":")
        fp = pcbnew.FootprintLoad(os.path.join(FP_DIR, lib + ".pretty"), name)
        if fp is None:
            raise SystemExit(f"{part['ref']}: footprint {part['fp']} not found")
        fp.SetFPID(pcbnew.LIB_ID(lib, name))
        fp.SetReference(part["ref"])
        fp.SetValue(part["value"])
        fp.SetPath(pcbnew.KIID_PATH(f"/{sym_uuids[part['ref']]}"))
        if part["dnp"]:
            fp.SetAttributes(fp.GetAttributes() | pcbnew.FP_EXCLUDE_FROM_BOM | pcbnew.FP_EXCLUDE_FROM_POS_FILES)
        board.Add(fp)
        if part["ref"] in ("SW14", "SW17"):
            part = dict(part, rot=orient_slide_switch(fp, part))
        place(fp, part)
        fp.SetLocked(part["ref"] in place_footprints.layout.placements() or part["ref"] == "U1" or part["ref"] == "J1")
        for pad in fp.Pads():
            num = pad.GetNumber()
            if not num:
                continue
            key = (part["ref"], num)
            if key not in netmap:
                problems.append(f"{part['ref']} pad {num} has no matching symbol pin")
                continue
            if netmap[key]:
                pad.SetNet(net(netmap[key]))
        # symbol pins with no pad
        pad_nums = {p.GetNumber() for p in fp.Pads()}
        for (ref, num), n in netmap.items():
            if ref == part["ref"] and num not in pad_nums and n:
                problems.append(f"{ref} pin {num} ({n}) has no pad in {part['fp']}")

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
    silk("TATTLE PHONE v0.1", 0, 26.5, size=1.2)
    silk("POWER", 15.0, LY.SLIDE_SWITCH_Y + 6.5, size=0.8)
    silk("TATTLE/PLAY", -14.0, LY.SLIDE_SWITCH_Y + 6.5, size=0.8)
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


def write_lib_tables(out):
    open(os.path.join(out, "sym-lib-table"), "w").write(
        '(sym_lib_table\n  (lib (name "tattle")(type "KiCad")(uri "${KIPRJMOD}/tattle.kicad_sym")(options "")(descr "Tattle Phone parts"))\n)\n')


def main():
    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "kicad"))
    os.makedirs(out, exist_ok=True)
    parts = design.PARTS
    root = uid()
    write_symbol_lib(os.path.join(out, "tattle.kicad_sym"))
    write_lib_tables(out)
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
