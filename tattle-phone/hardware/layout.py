"""
Tattle Phone: the one place where the board and the case share numbers.

Every position the PCB and the 3D-printed shell have to agree on lives here:
case size, board outline, key/LED/button/mic/screw positions.

    python3 layout.py           -> writes ../enclosure/layout.scad (used by the case)
    place_footprints.run()      -> moves KiCad footprints to these spots (see that file)

Coordinates are millimetres in the *case frame*, looking at the front of the phone:
    x = 0 on the centre line, + to the right
    y = 0 at the bottom end (USB-C end), + toward the earpiece
    z = 0 at the back surface, + toward the front (keypad) surface
"""
from __future__ import annotations

import math
import os

# ---------------------------------------------------------------- case
CASE_LEN = 175.0          # bottom to top
CASE_W_BOTTOM = 52.0      # width at the bottom end
CASE_W_TOP = 58.0         # width at the earpiece end (gentle 90s taper)
CASE_T = 30.0             # front-to-back thickness
CORNER_R = 7.0            # radius of every outside edge/corner
WALL = 2.6
SPLIT_Z = 19.0            # front shell is z > SPLIT_Z, back shell below

# ---------------------------------------------------------------- PCB
PCB_T = 1.6
PCB_CLEARANCE = 1.0       # gap between board edge and inner wall
PCB_TOP_Y = 124.0         # top edge of the board (ESP32 antenna sits here)
PCB_FRONT_Z = 20.7        # z of the board's FRONT copper surface
PCB_BACK_Z = PCB_FRONT_Z - PCB_T

# Tact switches: 6x6 mm THT keys and 12x12 mm THT CALL, both 4.3 mm tall
SWITCH_H = 4.3

# ---------------------------------------------------------------- keypad
KEY_W, KEY_H = 10.5, 9.0          # keycap footprint
KEY_PITCH_X, KEY_PITCH_Y = 13.5, 13.0
KEY_COLS_X = [-KEY_PITCH_X, 0.0, KEY_PITCH_X]
KEY_ROWS_Y = [93.0, 80.0, 67.0, 54.0]   # rows "1 2 3", "4 5 6", "7 8 9", "* 0 #"
KEY_LABELS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "0", "#"]
LED_OFFSET_X = 5.0                # SK6812MINI (front side) beside each switch, under the cap edge

CALL_XY = (0.0, 34.0)
CALL_CAP_D = 22.0
CALL_LED_OFFSET_Y = 8.5

# ---------------------------------------------------------------- misc parts
MIC_XY = (0.0, 15.0)              # bottom-port mic on the back, port hole through PCB
USBC_X = 0.0                      # HRO TYPE-C-31-M-12, top-mount on the BACK of the board
USBC_H = 3.26                     # receptacle body height
USBC_Z = PCB_BACK_Z - USBC_H / 2  # centre of the opening (derived below PCB_BACK_Z)
SPEAKER_XY = (0.0, 147.0)
SPEAKER_D = 36.0                  # 36 mm 8 ohm 1 W
SLIDE_SWITCH_Y = 61.0             # power slide switch (right edge) and TATTLE/PLAY mode switch (left edge)
STATUS_LED_XY = (0.0, 104.0)      # "in use" light above the keypad, front side
BATTERY = dict(w=34.0, l=50.0, t=10.0, center_y=55.0)   # 103450 LiPo
MOUNT_HOLES = [(-19.0, 22.0), (19.0, 22.0), (-19.0, 108.0), (19.0, 108.0)]
MOUNT_HOLE_D = 2.7                # M2.5 clearance
ESP32_MODULE = dict(w=18.0, l=25.5)  # antenna end flush with PCB_TOP_Y, on the back


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


def keys():
    """[(ref_index, label, x, y)] in reading order 1..9,*,0,#"""
    out = []
    for r, y in enumerate(KEY_ROWS_Y):
        for c, x in enumerate(KEY_COLS_X):
            i = r * 3 + c
            out.append((i, KEY_LABELS[i], x, y))
    return out


def placements():
    """Reference designator -> (x, y, rotation_deg, side) in case coords.

    Reference designators to use in the KiCad schematic:
      SW1..SW12  keypad switches, in order 1 2 3 4 5 6 7 8 9 * 0 #
      SW13       CALL switch (12x12)
      LED1       CALL button LED   (first in the data chain = firmware index 0)
      LED2..13   key LEDs, same order as SW1..SW12 (firmware index 1..12), front side
      LED14      status ("in use") light above the keypad (firmware index 13)
      MK1        microphone (back side, port hole through the board)
      H1..H4     M2.5 mounting holes (case screws pass through these)
    """
    p = {}
    for i, _label, x, y in keys():
        p[f"SW{i + 1}"] = (x, y, 90.0, "front")
        p[f"LED{i + 2}"] = (x + LED_OFFSET_X, y, 0.0, "front")
    p["SW13"] = (CALL_XY[0], CALL_XY[1], 0.0, "front")
    p["LED1"] = (CALL_XY[0], CALL_XY[1] + CALL_LED_OFFSET_Y, 0.0, "front")
    p["LED14"] = (STATUS_LED_XY[0], STATUS_LED_XY[1], 0.0, "front")
    p["MK1"] = (MIC_XY[0], MIC_XY[1], 0.0, "back")
    for n, (x, y) in enumerate(MOUNT_HOLES):
        p[f"H{n + 1}"] = (x, y, 0.0, "front")
    return p


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
        # arc around corner i+1 from this edge's normal to the next edge's normal
        n2x, n2y = normals[(i + 1) % n]
        mx, my = nx + n2x, ny + n2y
        ml = math.hypot(mx, my)
        cx, cy = c[(i + 1) % n]
        mid = (cx + r * mx / ml, cy + r * my / ml)
        end = (cx + r * n2x, cy + r * n2y)
        segs.append(("arc", b, mid, end))
    return segs


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
        "CASE_LEN", "CASE_W_BOTTOM", "CASE_W_TOP", "CASE_T", "CORNER_R", "WALL",
        "SPLIT_Z", "PCB_T", "PCB_FRONT_Z", "PCB_BACK_Z", "PCB_CORNER_R", "PCB_BOTTOM_Y",
        "PCB_TOP_Y", "SWITCH_H", "KEY_W", "KEY_H", "LED_OFFSET_X", "CALL_CAP_D",
        "SPEAKER_D", "SLIDE_SWITCH_Y", "MOUNT_HOLE_D", "USBC_Z",
    ]:
        lines.append(v(name, globals()[name]))
    lines.append(v("CALL_XY", CALL_XY))
    lines.append(v("MIC_XY", MIC_XY))
    lines.append(v("SPEAKER_XY", SPEAKER_XY))
    lines.append(v("STATUS_LED_XY", STATUS_LED_XY))
    lines.append(v("USBC_X", USBC_X))
    lines.append(v("MOUNT_HOLES", MOUNT_HOLES))
    lines.append(v("PCB_CORNERS", PCB_CORNERS))
    lines.append(v("BATTERY", [BATTERY["w"], BATTERY["l"], BATTERY["t"], BATTERY["center_y"]]))
    lines.append(v("ESP32_MODULE", [ESP32_MODULE["w"], ESP32_MODULE["l"]]))
    lines.append("KEYS = [" + ", ".join(f'["{lbl}", {x:.3f}, {y:.3f}]' for _, lbl, x, y in keys()) + "];")
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
    print(f"PCB: bottom y={PCB_BOTTOM_Y:.2f}, top y={PCB_TOP_Y:.2f}, corner r={PCB_CORNER_R:.2f}")
    for ref, (x, y, rot, side) in sorted(placements().items()):
        print(f"  {ref:6s} x={x:7.2f} y={y:7.2f} rot={rot:5.1f} {side}")
