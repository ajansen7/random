// Tattle Phone enclosure: a parametric OpenSCAD model
//
// All shared dimensions come from layout.scad, which hardware/layout.py
// generates. Change the numbers there, run `python3 hardware/layout.py`, and
// the case, keypad and KiCad board outline stay in sync.
//
// Pick a part with -D part="..." (or edit the default below):
//   closed     the finished phone (default)
//   assembly   everything in place, front shell see-through
//   exploded   parts spread out along z
//   front      front shell        (print face-down)
//   back       back shell         (print back-down; supports under the antenna nub)
//   keypad     12 separate translucent keycaps, each captive under the front wall (print as-is)
//   status_lens  little translucent plug for the "in use" light (print as-is)
//   call_cap   big CALL button cap (print as-is)
//   speaker_cup sealed back box for the speaker (print as-is)
//   pcb        2D board outline + holes, for DXF export to KiCad
//
// Printed parts are exported in print orientation by the Makefile.

include <layout.scad>

// What to show when you open this file. Try: "closed" (the finished phone),
// "assembly" (x-ray: see-through front shell with the board inside), "exploded",
// or a single printable part: "front", "back", "keypad", "call_cap", "status_lens",
// "speaker_cup". Press F5 for a quick colored preview. F6 renders in full quality but
// shows everything in OpenSCAD's default yellow/teal colors.
part = "closed";
print_orient = false;      // the Makefile sets true for STL export

$fn = $preview ? 40 : 72;

// ------------------------------------------------------------ tunables
KEY_CLEAR   = 0.4;         // keycap-to-hole gap, per side
KEY_PROUD   = 1.6;         // how far keys stand out of the front face
KEY_FLANGE  = 1.2;         // keycap flange overhang under the front wall, per side
WEB_T       = 1.0;
WEB_BOT     = PCB_FRONT_Z + SWITCH_H + 0.05;     // web rests on the switch tops
CALL_FLANGE = 5.0;         // flange overhang (diameter) under the front wall
INSERT_D    = 3.6;         // M2.5 heat-set insert hole (check your inserts)
INSERT_L    = 4.2;
SCREW_D     = 2.9;
SCREW_HEAD_D= 5.4;
SCREW_CB    = 6.0;         // counterbore depth from the back -> M2.5 x 18 screws
LIP_T       = 0.6;
LIP_H       = 1.5;
USBC_SLOT   = [9.4, 3.6];
SLIDE_SLOT  = [9.0, 3.2];
SLIDE_Z     = PCB_BACK_Z - 1.6;
FONT        = "Liberation Sans:style=Bold";

INNER_FRONT = CASE_T - WALL;

// ------------------------------------------------------------ basic shapes
function cx(y) = (CASE_W_BOTTOM/2 - CORNER_R) +
                 ((CASE_W_TOP - CASE_W_BOTTOM)/2) * (y - CORNER_R) / (CASE_LEN - 2*CORNER_R);

// The handset: a hull of 8 spheres -> flat faces, every edge rounded by CORNER_R.
// inset > 0 gives the same shape shrunk evenly (used for the cavity).
module body(inset = 0) {
    r = CORNER_R - inset;
    hull() for (y = [CORNER_R, CASE_LEN - CORNER_R], s = [-1, 1], z = [CORNER_R, CASE_T - CORNER_R])
        translate([s * cx(y), y, z]) sphere(r = r);
}

module rrect(w, h, r) { offset(r = r) square([w - 2*r, h - 2*r], center = true); }

module pcb2d() { hull() for (c = PCB_CORNERS) translate(c) circle(r = PCB_CORNER_R); }

module above(z) { translate([-100, -100, z]) cube([200, 500, 100]); }
module below(z) { translate([-100, -100, z - 100]) cube([200, 500, 100]); }

// ------------------------------------------------------------ cut-outs through both shells
module shared_cutouts() {
    // USB-C in the bottom end, with a counterbore so the plug's overmold seats
    translate([USBC_X, -1, USBC_Z]) rotate([-90, 0, 0]) {
        linear_extrude(WALL + 4) rrect(USBC_SLOT[0], USBC_SLOT[1], USBC_SLOT[1]/2 - 0.01);
        linear_extrude(1 + 1.2) rrect(12.5, 7.0, 3.0);
    }
    // slide switches: POWER on the right, TATTLE/PLAY mode on the left
    for (s = [-1, 1]) translate([s * (cx(SLIDE_SWITCH_Y) + CORNER_R - WALL - 1), SLIDE_SWITCH_Y, SLIDE_Z])
        cube([WALL + 4, SLIDE_SLOT[0], SLIDE_SLOT[1]], center = true);
}

// ------------------------------------------------------------ front shell
module grille_holes() {
    // Five smooth slots, flush with the face: easy to wipe, nothing for crumbs to
    // pack into. Glue a scrap of speaker cloth behind them inside the case.
    r = 12;
    for (dy = [-8, -4, 0, 4, 8]) {
        half = sqrt(r*r - dy*dy) - 1.5;
        translate([SPEAKER_XY[0], SPEAKER_XY[1] + dy, INNER_FRONT - 1])
            hull() for (sx = [-1, 1]) translate([sx * half, 0, 0]) cylinder(d = 2.0, h = WALL + 2, $fn = 20);
    }
}

module front_features() {
    // screw bosses with heat-set inserts (installed from below with an iron)
    for (h = MOUNT_HOLES) translate([h[0], h[1], PCB_FRONT_Z]) difference() {
        cylinder(d = 6.5, h = CASE_T - PCB_FRONT_Z - 0.5);
        translate([0, 0, -0.01]) cylinder(d = INSERT_D, h = INSERT_L);
    }
    // speaker locating ring
    translate([SPEAKER_XY[0], SPEAKER_XY[1], INNER_FRONT - 4]) difference() {
        cylinder(d = SPEAKER_D + 0.4 + 3.2, h = 5);
        translate([0, 0, -1]) cylinder(d = SPEAKER_D + 0.4, h = 7);
    }
    // light pipe for the status LED: keeps its glow in one spot and holds the lens
    translate([STATUS_LED_XY[0], STATUS_LED_XY[1], PCB_FRONT_Z + 2.0]) difference() {
        cylinder(d = 7.4, h = CASE_T - PCB_FRONT_Z - 2.0);
        translate([0, 0, -1]) cylinder(d = 5.4, h = CASE_T);
    }
    // mic duct: front holes -> gasket -> port hole in the PCB
    translate([MIC_XY[0], MIC_XY[1], PCB_FRONT_Z + 0.6]) difference() {
        cylinder(d = 5, h = CASE_T - PCB_FRONT_Z);
        translate([0, 0, -1]) cylinder(d = 2.2, h = CASE_T);
    }
}

module front_cutouts() {
    for (k = KEYS) translate([k[1], k[2], INNER_FRONT - 1])
        linear_extrude(WALL + 2) rrect(KEY_W + 2*KEY_CLEAR, KEY_H + 2*KEY_CLEAR, 1.6 + KEY_CLEAR);
    translate([CALL_XY[0], CALL_XY[1], INNER_FRONT - 1]) cylinder(d = CALL_CAP_D + 2*KEY_CLEAR, h = WALL + 2);
    grille_holes();
    // status light window (the printed lens sits in it)
    translate([STATUS_LED_XY[0], STATUS_LED_XY[1], INNER_FRONT - 1]) cylinder(d = 3.3, h = WALL + 2);
    // mic: three pin-holes in a little dish
    translate([MIC_XY[0], MIC_XY[1], CASE_T - 0.5]) cylinder(d = 7, h = 2);
    for (dx = [-1.6, 0, 1.6]) translate([MIC_XY[0] + dx, MIC_XY[1], INNER_FRONT - 1]) cylinder(d = 1.1, h = WALL + 2, $fn = 12);
    // label between keypad and earpiece
    translate([0, (KEYS[0][2] + KEY_H/2 + SPEAKER_XY[1] - 15) / 2, CASE_T - 0.6])
        linear_extrude(2) text("TATTLE LINE", size = 4.2, font = FONT, halign = "center", valign = "center");
}

module front_shell() {
    difference() {
        union() {
            intersection() { difference() { body(); body(WALL); } above(SPLIT_Z); }
            intersection() { body(1.0); front_features(); }
        }
        front_cutouts();
        shared_cutouts();
    }
}

// ------------------------------------------------------------ back shell
module lip() {
    difference() {
        intersection() { difference() { body(WALL); body(WALL + LIP_T); } translate([0, 0, SPLIT_Z - 0.01]) above(0) ; translate([0,0,SPLIT_Z + LIP_H]) below(0); }
        translate([USBC_X - 9, -1, 0]) cube([18, 15, 50]);   // clear the USB-C receptacle
    }
}

module back_features() {
    for (h = MOUNT_HOLES) translate([h[0], h[1], 0]) cylinder(d = 7, h = PCB_BACK_Z);
    // battery pocket ribs (line with thin foam)
    translate([0, BATTERY[3], WALL - 0.01]) linear_extrude(6) difference() {
        rrect(BATTERY[0] + 1.5 + 2.4, BATTERY[1] + 1.5 + 2.4, 2);
        rrect(BATTERY[0] + 1.5, BATTERY[1] + 1.5, 1);
    }
}

module antenna_nub() {
    // Stubby 90s antenna. Part of the shell on purpose: nothing a toddler can snap off.
    x = cx(CASE_LEN - CORNER_R) - 4;
    translate([x, CASE_LEN - 9, 10]) rotate([-90, 0, 0]) {
        cylinder(d1 = 11, d2 = 9.5, h = 24);
        translate([0, 0, 24]) sphere(d = 9.5);
    }
}

module back_shell() {
    difference() {
        union() {
            intersection() { difference() { body(); body(WALL); } below(SPLIT_Z); }
            lip();
            intersection() { body(1.0); back_features(); }
            antenna_nub();
        }
        for (h = MOUNT_HOLES) translate([h[0], h[1], -1]) {
            cylinder(d = SCREW_D, h = PCB_BACK_Z + 2);
            cylinder(d = SCREW_HEAD_D, h = SCREW_CB + 1);
        }
        shared_cutouts();
    }
}

// ------------------------------------------------------------ keycaps
// Twelve separate caps. Each has a flange wider than its hole in the front
// wall, so once the case is screwed shut a key can't be pulled out, and there
// is no thin hinge to fatigue. The cap rests on its tact switch.
module key_rect() { rrect(KEY_W, KEY_H, 1.6); }

module keycap(label) {
    top = CASE_T + KEY_PROUD;
    difference() {
        union() {
            translate([0, 0, WEB_BOT]) linear_extrude(INNER_FRONT - WEB_BOT - 0.35)
                rrect(KEY_W + 2*KEY_FLANGE, KEY_H + 2*KEY_FLANGE, 1.6 + KEY_FLANGE);
            hull() {
                translate([0, 0, WEB_BOT]) linear_extrude(top - WEB_BOT - 0.8) key_rect();
                translate([0, 0, top - 0.01]) linear_extrude(0.01) rrect(KEY_W - 1.4, KEY_H - 1.4, 1.0);
            }
        }
        translate([0, 0, top - 0.6]) linear_extrude(1)
            text(label, size = label == "*" ? 7 : 5, font = FONT, halign = "center",
                 valign = label == "*" ? "top" : "center");
    }
}

module keypad() {
    for (k = KEYS) translate([k[1], k[2], 0]) keycap(k[0]);
}

// ------------------------------------------------------------ status light lens
// A translucent plug: rod through the window, flange inside the light pipe so it
// can't come out the front. Rests just above the LED.
module status_lens() {
    translate([STATUS_LED_XY[0], STATUS_LED_XY[1], 0]) {
        translate([0, 0, PCB_FRONT_Z + 1.8]) cylinder(d = 2.9, h = CASE_T - PCB_FRONT_Z - 1.8);
        translate([0, 0, INNER_FRONT - 1.4]) cylinder(d = 5.0, h = 1.2);
    }
}

// ------------------------------------------------------------ CALL cap
module call_cap() {
    d = CALL_CAP_D;
    top = CASE_T + KEY_PROUD + 0.8;
    translate([CALL_XY[0], CALL_XY[1], 0]) difference() {
        union() {
            translate([0, 0, WEB_BOT]) cylinder(d = d + CALL_FLANGE, h = INNER_FRONT - WEB_BOT - 0.3);
            hull() {
                translate([0, 0, WEB_BOT]) cylinder(d = d, h = top - WEB_BOT - 1);
                translate([0, 0, top - 0.01]) cylinder(d = d - 2, h = 0.01);
            }
        }
        // handset icon: two ear/mouth pads joined by a bar
        translate([0, 0, top - 0.6]) linear_extrude(1) rotate(-30) {
            hull() { translate([-5, 0]) circle(d = 3.4); translate([5, 0]) circle(d = 3.4); }
            for (s = [-1, 1]) translate([s * 5.6, -1.6]) rrect(4.2, 3.4, 1.2);
        }
    }
}

// ------------------------------------------------------------ speaker back box
module speaker_cup() {
    ring_od = SPEAKER_D + 0.4 + 3.2;
    h = 12;
    translate([SPEAKER_XY[0], SPEAKER_XY[1], INNER_FRONT - 4 + 3 - h]) difference() {
        cylinder(d = ring_od + 0.3 + 3.2, h = h);
        translate([0, 0, 1.6]) cylinder(d = ring_od + 0.3, h = h);
        translate([0, ring_od/2 - 2, 4]) rotate([-90, 0, 0]) cylinder(d = 3, h = 10);  // wire hole (seal with glue)
    }
}

// ------------------------------------------------------------ dummies for the assembly view
module pcb_dummy() {
    color("darkgreen") translate([0, 0, PCB_BACK_Z]) linear_extrude(PCB_T) difference() {
        pcb2d();
        for (h = MOUNT_HOLES) translate(h) circle(d = MOUNT_HOLE_D);
    }
    color("silver") translate([0, PCB_TOP_Y - ESP32_MODULE[1]/2, PCB_BACK_Z - 3.1])
        linear_extrude(3.1) square([ESP32_MODULE[0], ESP32_MODULE[1]], center = true);
    color("dimgray") for (k = KEYS) translate([k[1], k[2], PCB_FRONT_Z]) linear_extrude(SWITCH_H) square(6, center = true);
    color("dimgray") translate([CALL_XY[0], CALL_XY[1], PCB_FRONT_Z]) linear_extrude(SWITCH_H) square(12, center = true);
    color("silver") translate([USBC_X, PCB_BOTTOM_Y + 3.6, USBC_Z]) cube([8.9, 7.3, 3.2], center = true);
}
module battery_dummy() {
    color("royalblue") translate([0, BATTERY[3], WALL + BATTERY[2]/2]) cube([BATTERY[0], BATTERY[1], BATTERY[2]], center = true);
}
module speaker_dummy() {
    color("black") translate([SPEAKER_XY[0], SPEAKER_XY[1], INNER_FRONT - 5]) cylinder(d = SPEAKER_D, h = 5);
}

// ------------------------------------------------------------ 2D board outline for KiCad
module pcb_outline_2d() {
    difference() {
        pcb2d();
        for (h = MOUNT_HOLES) translate(h) circle(d = MOUNT_HOLE_D);
    }
}

// ------------------------------------------------------------ part selector
SHELL_COLOR = [0.98, 0.93, 0.80];   // 90s "almond" plastic
KEY_COLOR   = [0.85, 0.95, 1.0];

module place_print(flip = false) {
    if (print_orient && flip) translate([0, 0, CASE_T]) rotate([180, 0, 0]) children();
    else if (print_orient) children();
    else children();
}

if (part == "assembly") {
    color(SHELL_COLOR) back_shell();
    pcb_dummy(); battery_dummy(); speaker_dummy();
    color("gray") speaker_cup();
    color(KEY_COLOR) keypad();
    color("limegreen") call_cap();
    color("white") status_lens();
    color([0.98, 0.93, 0.80, 0.25]) front_shell();
} else if (part == "closed") {
    color(SHELL_COLOR) back_shell();
    color(SHELL_COLOR) front_shell();
    color(KEY_COLOR) keypad();
    color("limegreen") call_cap();
    color("white") status_lens();
} else if (part == "exploded") {
    color(SHELL_COLOR) back_shell();
    translate([0, 0, 18]) { pcb_dummy(); speaker_dummy(); }
    translate([0, 0, -2]) battery_dummy();
    translate([0, 0, 18]) color("gray") speaker_cup();
    translate([0, 0, 34]) { color(KEY_COLOR) keypad(); color("limegreen") call_cap(); }
    translate([0, 0, 50]) color(SHELL_COLOR) front_shell();
} else if (part == "front") {
    place_print(true) front_shell();
} else if (part == "back") {
    back_shell();
} else if (part == "keypad") {
    if (print_orient) translate([0, 0, -WEB_BOT]) keypad(); else color(KEY_COLOR) keypad();
} else if (part == "status_lens") {
    if (print_orient) translate([0, 0, -(PCB_FRONT_Z + 1.8)]) status_lens(); else status_lens();
} else if (part == "call_cap") {
    if (print_orient) translate([0, 0, -WEB_BOT]) call_cap(); else call_cap();
} else if (part == "speaker_cup") {
    if (print_orient) translate([0, 0, -(INNER_FRONT - 4 + 3 - 12)]) speaker_cup(); else speaker_cup();
} else if (part == "pcb") {
    pcb_outline_2d();
}
