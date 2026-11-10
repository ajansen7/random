"""
Tattle Phone Lite: the one place where the board and the case share numbers.

Three big buttons, 3xAA, a 2-layer board with every factory-placed part on the BACK.
Every position the PCB and the 3D-printed shell have to agree on lives here.

    python3 layout.py      -> writes ../enclosure/layout.scad (used by the case)
                              and prints every placement

Coordinates are millimetres in the *case frame*, looking at the front of the phone:
    x = 0 on the centre line, + to the right
    y = 0 at the bottom end (USB-C / mouthpiece end), + toward the earpiece
    z = 0 at the back surface, + toward the front (button) surface

Stack-up, back to front:
    back wall (battery door) -> 3xAA holder -> gap -> parts on the back of the PCB
    -> PCB -> 12x12 tact switches (front) -> button caps through the front wall
"""
from __future__ import annotations

import math
import os

# ---------------------------------------------------------------- case
CASE_LEN = 170.0          # bottom to top
CASE_W_BOTTOM = 60.0      # width at the bottom end
CASE_W_TOP = 62.0         # width at the earpiece end (gentle 90s taper)
CORNER_R = 5.0            # radius of every outside edge/corner. Boxier than the original
                          # so the flat back is wide enough for the battery door
WALL = 2.6

# ---------------------------------------------------------------- batteries
# A plain open 3xAA holder (cells side by side, flying leads) screwed to the INSIDE of
# the battery door. Changing batteries: undo the one door screw, lift door + holder out
# on their leads, swap cells, drop back in. Sized for the Keystone 2465 (open, 3xAA, 6" leads:
# 58 x 48.3 x 17 mm; Eagle 12BH331 and Pololu #142 are within 0.3 mm). Other holders: MEASURE.
BATT_HOLDER = dict(w=48.3, l=58.0, h=17.0, center_y=62.0)
DOOR_CLEAR = 0.4          # door-to-opening gap, per side
DOOR_W = BATT_HOLDER["w"] + 0.8          # opening in the flat back (holder passes through)
DOOR_Y0 = BATT_HOLDER["center_y"] - BATT_HOLDER["l"] / 2 - 7.0   # bottom end: one screw into a shelf
DOOR_Y1 = BATT_HOLDER["center_y"] + BATT_HOLDER["l"] / 2 + 5.0   # top end: hook tab under the back wall
DOOR_L = DOOR_Y1 - DOOR_Y0
BATT_FLOOR_Z = WALL       # holder sits on the door's inside face
BATT_TOP_Z = BATT_FLOOR_Z + BATT_HOLDER["h"]

# ---------------------------------------------------------------- PCB
PCB_T = 1.6
PCB_CLEARANCE = 1.0       # gap between board edge and inner wall
PCB_TOP_Y = 124.0         # top edge of the board (ESP32 antenna sits here)
BACK_PART_H = 1.2         # tallest back-side part allowed over the battery (LEDs, 0402s, QFN)
PCB_BACK_Z = BATT_TOP_Z + 0.3 + BACK_PART_H
PCB_FRONT_Z = PCB_BACK_Z + PCB_T

# 12x12 mm THT tact switch, 7.3 mm tall overall (body + round stem). Buy "12x12x7.3".
SWITCH_W = 12.0
SWITCH_H = 7.3
CAP_FLANGE_T = 1.6        # button-cap flange that sits under the front wall
INNER_FRONT_Z = PCB_FRONT_Z + SWITCH_H + CAP_FLANGE_T + 0.3
CASE_T = INNER_FRONT_Z + WALL
SPLIT_Z = PCB_BACK_Z - 0.6   # front shell is z > SPLIT_Z, back shell below

# ---------------------------------------------------------------- buttons
# name, shape, size (circle/star: outer diameter, square: side), x, y, cap colour,
# LED spots relative to the button centre (reverse-mount LEDs on the back, shining
# through holes in the board into the translucent cap). The star's LEDs sit up in its
# side arms because the switch body fills the middle.
BUTTONS = [
    dict(ref="SW1", name="TATTLE", shape="star", size=30.0, x=0.0, y=87.0, color="yellow",
         leds=[(-10.5, 3.2), (10.5, 3.2)]),
    dict(ref="SW2", name="CALL", shape="circle", size=24.0, x=0.0, y=58.0, color="green",
         leds=[(0.0, 9.0), (0.0, -9.0)]),
    dict(ref="SW3", name="HANG UP", shape="square", size=22.0, x=0.0, y=30.0, color="red",
         leds=[(0.0, 9.0), (0.0, -9.0)]),
]
STAR_INNER = 0.45         # star inner/outer radius ratio

# ---------------------------------------------------------------- misc parts
MIC_XY = (0.0, 12.0)              # bottom-port PDM mic on the back, port hole through the PCB
USBC_X = -11.0                    # HRO TYPE-C-31-M-12 on the BACK, mouth flush with the bottom edge
USBC_H = 3.26
SPEAKER_XY = (0.0, 146.0)
SPEAKER_D = 28.0                  # 28 mm 8 ohm 0.5-1 W
POWER_SWITCH_Y = 96.0             # MSK12C02 slide switch, right edge, back side (above the battery)
ESP32_MODULE = dict(w=18.0, l=25.5)   # antenna end flush with PCB_TOP_Y, on the back
MOUNT_HOLES = [(-19.0, 20.0), (19.0, 20.0), (-19.0, 110.0), (19.0, 110.0)]
MOUNT_HOLE_D = 2.7                # M2.5 clearance
BATT_PADS = [(-23.5, 93.5), (-23.5, 97.5)]   # + then -, wire pads for the holder leads
SPK_PADS = [(12.5, 112.0), (12.5, 116.0)]    # speaker leads (clear of the H4 screw boss)


# ---------------------------------------------------------------- derived
def case_width(y: float) -> float:
    return CASE_W_BOTTOM + (CASE_W_TOP - CASE_W_BOTTOM) * y / CASE_LEN


def _corner_x(y: float) -> float:
    """x of the corner-sphere centre line at height y (the taper)."""
    xb = CASE_W_BOTTOM / 2 - CORNER_R
    xt = CASE_W_TOP / 2 - CORNER_R
    return xb + (xt - xb) * (y - CORNER_R) / (CASE_LEN - 2 * CORNER_R)


PCB_CORNER_R = CORNER_R - WALL - PCB_CLEARANCE      # 3.4 mm
_pcb_top_cy = PCB_TOP_Y - PCB_CORNER_R
PCB_CORNERS = [  # circle centres; board outline = hull of these circles
    (-_corner_x(CORNER_R), CORNER_R),
    (_corner_x(CORNER_R), CORNER_R),
    (_corner_x(_pcb_top_cy), _pcb_top_cy),
    (-_corner_x(_pcb_top_cy), _pcb_top_cy),
]
PCB_BOTTOM_Y = CORNER_R - PCB_CORNER_R
USBC_Z = PCB_BACK_Z - USBC_H / 2


def pcb_half_width(y: float) -> float:
    return _corner_x(min(max(y, CORNER_R), _pcb_top_cy)) + PCB_CORNER_R


def leds():
    """[(ref, button, x, y)] for LED1..LED6, two per button in BUTTONS order."""
    out, n = [], 1
    for b in BUTTONS:
        for dx, dy in b["leds"]:
            out.append((f"LED{n}", b, b["x"] + dx, b["y"] + dy))
            n += 1
    return out


def placements():
    """Reference designator -> (x, y, rotation_deg, side) in case coords for parts whose
    position the case cares about."""
    p = {}
    for b in BUTTONS:
        p[b["ref"]] = (b["x"], b["y"], 0.0, "front")
    for ref, _b, x, y in leds():
        p[ref] = (x, y, 90.0, "back")
    p["MK1"] = (MIC_XY[0], MIC_XY[1], 0.0, "back")
    p["U1"] = (0.0, PCB_TOP_Y - ESP32_MODULE["l"] / 2, 0.0, "back")
    # mounting tabs >= 0.5 mm inside the edge; the actuator then sits ~2.5 mm inside the
    # case side (recessed on purpose: a pen tip moves it, a toddler mostly can't)
    p["SW4"] = (pcb_half_width(POWER_SWITCH_Y) - 2.0, POWER_SWITCH_Y, 90.0, "back")
    for n, (x, y) in enumerate(MOUNT_HOLES):
        p[f"H{n + 1}"] = (x, y, 0.0, "front")
    return p


def star_points(r_out: float, ratio: float = STAR_INNER):
    """10 vertices of a 5-point star, point up, centred on (0, 0)."""
    pts = []
    for i in range(10):
        r = r_out if i % 2 == 0 else r_out * ratio
        a = math.radians(90 + 36 * i)
        pts.append((r * math.cos(a), r * math.sin(a)))
    return pts


def pcb_outline():
    """Board outline as a list of ('line', a, b) and ('arc', start, mid, end)."""
    c = PCB_CORNERS
    r = PCB_CORNER_R
    n = len(c)
    normals = []
    for i in range(n):  # outward normal of edge c[i] -> c[i+1] (corners are CCW)
        ax, ay = c[i]
        bx, by = c[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy)
        normals.append((dy / L, -dx / L))
    segs = []
    for i in range(n):
        nx, ny = normals[i]
        a = (c[i][0] + r * nx, c[i][1] + r * ny)
        b = (c[(i + 1) % n][0] + r * nx, c[(i + 1) % n][1] + r * ny)
        segs.append(("line", a, b))
        n2x, n2y = normals[(i + 1) % n]
        mx, my = nx + n2x, ny + n2y
        ml = math.hypot(mx, my)
        cx, cy = c[(i + 1) % n]
        mid = (cx + r * mx / ml, cy + r * my / ml)
        end = (cx + r * n2x, cy + r * n2y)
        segs.append(("arc", b, mid, end))
    return segs


# ---------------------------------------------------------------- checks
def check():
    """Cheap geometric sanity checks; returns a list of problems (empty = OK)."""
    bad = []
    flat_half = min(case_width(b["y"]) for b in BUTTONS) / 2 - CORNER_R
    for b in BUTTONS:
        half = b["size"] / 2
        if half + 1.5 > flat_half:
            bad.append(f"{b['name']} cap ({b['size']} mm) runs onto the rounded edge")
        for dx, dy in b["leds"]:
            if b["shape"] == "circle" and math.hypot(dx, dy) + 0.8 > half:
                bad.append(f"{b['name']} LED at ({dx},{dy}) is outside the cap")
            if b["shape"] == "square" and max(abs(dx), abs(dy)) + 1.2 > half:
                bad.append(f"{b['name']} LED at ({dx},{dy}) is outside the cap")
    for a, b in zip(BUTTONS, BUTTONS[1:]):
        gap = abs(a["y"] - b["y"]) - a["size"] / 2 - b["size"] / 2
        if a["shape"] == "star" or b["shape"] == "star":
            gap += 30.0 / 2 * (1 - math.sin(math.radians(54)))   # star's lower points sit wide
        if gap < 2.0:
            bad.append(f"{a['name']} and {b['name']} caps are only {gap:.1f} mm apart")
    batt = BATT_HOLDER
    for x, y in MOUNT_HOLES:
        if abs(y - batt["center_y"]) < batt["l"] / 2 + 3.5 and abs(x) < batt["w"] / 2 + 3.5:
            bad.append(f"mount post at ({x},{y}) hits the battery holder")
    mod_bottom = PCB_TOP_Y - ESP32_MODULE["l"]
    if mod_bottom < batt["center_y"] + batt["l"] / 2:
        bad.append("ESP32 module overlaps the battery area (it is too tall to sit over it)")
    top_sw = max(BUTTONS, key=lambda b: b["y"])
    if top_sw["y"] + SWITCH_W / 2 > mod_bottom + 4:
        bad.append("top switch pins run into the ESP32 module")
    if not (batt["center_y"] + batt["l"] / 2 < POWER_SWITCH_Y - 2.5):
        bad.append("power switch hangs over the battery holder")
    flat_back = min(case_width(DOOR_Y0), case_width(DOOR_Y1)) - 2 * CORNER_R
    if DOOR_W + 2 * DOOR_CLEAR > flat_back:
        bad.append(f"battery door ({DOOR_W:.1f} mm) is wider than the flat back ({flat_back:.1f} mm)")
    for x, y in MOUNT_HOLES:
        if DOOR_Y0 - 5 < y < DOOR_Y1 + 5 and abs(x) < DOOR_W / 2 + 4:
            bad.append(f"mount post at ({x},{y}) is in the battery door area")
    if SPEAKER_XY[1] - SPEAKER_D / 2 - 2 < PCB_TOP_Y:
        bad.append("speaker ring runs into the board")
    return bad


# ---------------------------------------------------------------- SCAD export
def write_scad(path: str) -> None:
    def v(name, val):
        if isinstance(val, (list, tuple)):
            return f"{name} = {_scad_list(val)};"
        if isinstance(val, str):
            return f'{name} = "{val}";'
        return f"{name} = {val:.4f};"

    lines = ["// GENERATED by hardware/layout.py — edit that file, then run it.", ""]
    for name in [
        "CASE_LEN", "CASE_W_BOTTOM", "CASE_W_TOP", "CASE_T", "CORNER_R", "WALL", "SPLIT_Z",
        "PCB_T", "PCB_FRONT_Z", "PCB_BACK_Z", "PCB_CORNER_R", "PCB_BOTTOM_Y", "PCB_TOP_Y",
        "SWITCH_W", "SWITCH_H", "CAP_FLANGE_T", "INNER_FRONT_Z", "STAR_INNER",
        "SPEAKER_D", "POWER_SWITCH_Y", "MOUNT_HOLE_D", "USBC_X", "USBC_Z",
        "BATT_FLOOR_Z", "BATT_TOP_Z", "DOOR_W", "DOOR_L", "DOOR_Y0", "DOOR_Y1", "DOOR_CLEAR",
    ]:
        lines.append(v(name, globals()[name]))
    lines.append(v("MIC_XY", MIC_XY))
    lines.append(v("SPEAKER_XY", SPEAKER_XY))
    lines.append(v("MOUNT_HOLES", MOUNT_HOLES))
    lines.append(v("PCB_CORNERS", PCB_CORNERS))
    lines.append(v("BATT_HOLDER", [BATT_HOLDER["w"], BATT_HOLDER["l"], BATT_HOLDER["h"], BATT_HOLDER["center_y"]]))
    lines.append(v("ESP32_MODULE", [ESP32_MODULE["w"], ESP32_MODULE["l"]]))
    lines.append("BUTTONS = [" + ", ".join(
        f'["{b["name"]}", "{b["shape"]}", {b["size"]:.3f}, {b["x"]:.3f}, {b["y"]:.3f}, "{b["color"]}"]'
        for b in BUTTONS) + "];")
    lines.append("LEDS = [" + ", ".join(f"[{x:.3f}, {y:.3f}]" for _, _, x, y in leds()) + "];")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _scad_list(val):
    if isinstance(val, (list, tuple)):
        return "[" + ", ".join(_scad_list(x) for x in val) + "]"
    return f"{val:.4f}"


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "enclosure", "layout.scad")
    write_scad(out)
    print("wrote", os.path.normpath(out))
    print(f"case {CASE_LEN:.0f} x {CASE_W_BOTTOM:.0f}-{CASE_W_TOP:.0f} x {CASE_T:.1f} mm; "
          f"PCB y {PCB_BOTTOM_Y:.1f}..{PCB_TOP_Y:.1f}, back z {PCB_BACK_Z:.2f}")
    for ref, (x, y, rot, side) in sorted(placements().items()):
        print(f"  {ref:5s} x={x:7.2f} y={y:7.2f} rot={rot:5.1f} {side}")
    problems = check()
    print("checks:", "OK" if not problems else "")
    for p in problems:
        print("  !", p)
