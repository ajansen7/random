// Tattle Phone Lite enclosure: a parametric OpenSCAD model
//
// All shared dimensions come from layout.scad, which hardware/layout.py generates.
// Change the numbers there, run `python3 hardware/layout.py`, and the case, the
// buttons and the KiCad board outline stay in sync.
//
// Pick a part with -D part="..." (or edit the default below):
//   closed       the finished phone (default)
//   back_view    the finished phone seen from the back (battery door)
//   assembly     everything in place, front shell see-through
//   exploded     parts spread out along z
//   front        front shell                    (print face-down)
//   back         back shell                     (print back-down)
//   door         battery door + holder rim      (print outside-down)
//   cap_tattle / cap_call / cap_hangup   button caps (print as-is, translucent filament)
//   speaker_cup  sealed back box for the speaker (print as-is)
//   pcb          2D board outline + holes, for DXF export to KiCad

include <layout.scad>

part = "closed";
print_orient = false;      // the Makefile sets true for STL export

$fn = $preview ? 40 : 72;

// ------------------------------------------------------------ tunables
CAP_CLEAR   = 0.4;         // cap-to-hole gap, per side
CAP_PROUD   = 2.5;         // how far caps stand out of the front face
CAP_FLANGE  = 1.8;         // cap flange overhang under the front wall, per side
CAP_BOT     = PCB_FRONT_Z + SWITCH_H + 0.05;   // cap rests on the switch stem
INSERT_D    = 3.7;         // M2.5 heat-set insert hole: CNC Kitchen "M2.5 Standard" (M2.5 x 4.0).
INSERT_L    = 4.6;         // Other brands: use their recommended hole, depth >= insert + 0.5.
                           // Screws: 4x M2.5x20 (case, from the back), 1x M2.5x6 countersunk (door)
SCREW_D     = 2.9;
SCREW_HEAD_D= 5.4;
SCREW_CB    = 6.0;         // counterbore depth from the back
LIP_T       = 0.6;
LIP_H       = 1.5;
USBC_SLOT   = [9.4, 3.6];
SLIDE_SLOT  = [7.5, 3.4];  // recessed: push with a fingernail or pen tip (toddler-resistant)
SLIDE_Z     = PCB_BACK_Z - 1.4;
DOOR_SCREW_Y = DOOR_Y0 + 2.4;          // M2.5 countersunk screw, into an insert in the shelf
DOOR_TAB    = 4.0;         // hook tab length past the top end of the opening
DOOR_TAB_T  = 1.2;
FONT        = "Liberation Sans:style=Bold";

INNER_FRONT = INNER_FRONT_Z;
BTN_COLORS  = ["yellow", "limegreen", "red"];

// ------------------------------------------------------------ basic shapes
function cx(y) = (CASE_W_BOTTOM/2 - CORNER_R) +
                 ((CASE_W_TOP - CASE_W_BOTTOM)/2) * (y - CORNER_R) / (CASE_LEN - 2*CORNER_R);

module body(inset = 0) {
    r = CORNER_R - inset;
    hull() for (y = [CORNER_R, CASE_LEN - CORNER_R], s = [-1, 1], z = [CORNER_R, CASE_T - CORNER_R])
        translate([s * cx(y), y, z]) sphere(r = r);
}

module rrect(w, h, r) { offset(r = r) square([w - 2*r, h - 2*r], center = true); }
module pcb2d() { hull() for (c = PCB_CORNERS) translate(c) circle(r = PCB_CORNER_R); }
module above(z) { translate([-100, -100, z]) cube([200, 500, 100]); }
module below(z) { translate([-100, -100, z - 100]) cube([200, 500, 100]); }

function star_pts(r) = [for (i = [0:9]) let(rr = i % 2 == 0 ? r : r * STAR_INNER, a = 90 + 36 * i)
                        [rr * cos(a), rr * sin(a)]];

// 2D outline of a button's visible top, grown by `grow` (clearance or flange)
module btn2d(shape, size, grow = 0) {
    if (shape == "circle") circle(d = size + 2*grow);
    else if (shape == "square") rrect(size + 2*grow, size + 2*grow, 3 + grow);
    else if (shape == "star") offset(r = 1.2 + grow) offset(delta = -1.2) polygon(star_pts(size/2));
}

// ------------------------------------------------------------ cut-outs through both shells
module shared_cutouts() {
    // USB-C in the bottom end (programming only), with a counterbore for the plug
    translate([USBC_X, -1, USBC_Z]) rotate([-90, 0, 0]) {
        linear_extrude(WALL + 4) rrect(USBC_SLOT[0], USBC_SLOT[1], USBC_SLOT[1]/2 - 0.01);
        linear_extrude(1 + 1.2) rrect(12.5, 7.0, 3.0);
    }
    // recessed POWER slide switch, right side
    translate([cx(POWER_SWITCH_Y) + CORNER_R - WALL/2, POWER_SWITCH_Y, SLIDE_Z])
        cube([WALL + 4, SLIDE_SLOT[0], SLIDE_SLOT[1]], center = true);
}

// ------------------------------------------------------------ front shell
module grille_holes() {
    r = 10;
    for (dy = [-6, -3, 0, 3, 6]) {
        half = sqrt(r*r - dy*dy) - 1.5;
        translate([SPEAKER_XY[0], SPEAKER_XY[1] + dy, INNER_FRONT - 1])
            hull() for (sx = [-1, 1]) translate([sx * half, 0, 0]) cylinder(d = 1.8, h = WALL + 2, $fn = 20);
    }
}

module front_features() {
    for (h = MOUNT_HOLES) translate([h[0], h[1], PCB_FRONT_Z]) difference() {
        cylinder(d = 6.5, h = CASE_T - PCB_FRONT_Z - 0.5);
        translate([0, 0, -0.01]) cylinder(d = INSERT_D, h = INSERT_L);
    }
    // speaker locating ring
    translate([SPEAKER_XY[0], SPEAKER_XY[1], INNER_FRONT - 4]) difference() {
        cylinder(d = SPEAKER_D + 0.4 + 3.2, h = 5);
        translate([0, 0, -1]) cylinder(d = SPEAKER_D + 0.4, h = 7);
    }
    // mic duct: front holes -> (foam gasket) -> port hole in the PCB
    translate([MIC_XY[0], MIC_XY[1], PCB_FRONT_Z + 0.6]) difference() {
        cylinder(d = 5, h = CASE_T - PCB_FRONT_Z);
        translate([0, 0, -1]) cylinder(d = 2.2, h = CASE_T);
    }
}

module front_cutouts() {
    for (b = BUTTONS) translate([b[3], b[4], INNER_FRONT - 1])
        linear_extrude(WALL + 2) btn2d(b[1], b[2], CAP_CLEAR);
    grille_holes();
    translate([MIC_XY[0], MIC_XY[1], CASE_T - 0.5]) cylinder(d = 7, h = 2);
    for (dx = [-1.6, 0, 1.6]) translate([MIC_XY[0] + dx, MIC_XY[1], INNER_FRONT - 1]) cylinder(d = 1.1, h = WALL + 2, $fn = 12);
    top_btn = max([for (b = BUTTONS) b[4] + b[2]/2]);
    translate([0, (top_btn + SPEAKER_XY[1] - SPEAKER_D/2) / 2, CASE_T - 0.6])
        linear_extrude(2) text("TATTLE LINE", size = 4.4, font = FONT, halign = "center", valign = "center");
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
module door2d(grow = 0) {
    translate([0, (DOOR_Y0 + DOOR_Y1) / 2]) rrect(DOOR_W + 2*grow, DOOR_L + 2*grow, 3 + grow);
}

module lip() {
    difference() {
        intersection() { difference() { body(WALL); body(WALL + LIP_T); } translate([0, 0, SPLIT_Z - 0.01]) above(0); translate([0, 0, SPLIT_Z + LIP_H]) below(0); }
        translate([USBC_X - 9, -1, 0]) cube([18, 15, 50]);   // clear the USB-C receptacle
    }
}

module back_features() {
    for (h = MOUNT_HOLES) translate([h[0], h[1], 0]) cylinder(d = 7, h = PCB_BACK_Z);
    // shelf over the bottom end of the door: stops it going in; the door screw bites here
    translate([0, DOOR_Y0 + 0.5, WALL - 0.01]) linear_extrude(5) rrect(14, 9, 1.5);
}

module antenna_nub() {
    x = cx(CASE_LEN - CORNER_R) - 4;
    translate([x, CASE_LEN - 9, 10]) rotate([-90, 0, 0]) {
        cylinder(d1 = 11, d2 = 9.5, h = 22);
        translate([0, 0, 22]) sphere(d = 9.5);
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
        // battery door opening
        translate([0, 0, -1]) linear_extrude(WALL + 1.01) door2d(DOOR_CLEAR);
        // insert hole in the shelf for the door screw (insert pressed in from below)
        translate([0, DOOR_SCREW_Y, WALL - 0.02]) cylinder(d = INSERT_D, h = INSERT_L);
        // finger notch so the door can be lifted
        translate([0, DOOR_Y1 + 0.5, -1]) scale([1, 0.6, 1]) cylinder(d = 10, h = 1.6);
    }
}

// ------------------------------------------------------------ battery door
// Flush plate in the back. Top end: a hook tab that slides under the back wall.
// Bottom end: one countersunk M2.5 screw into the shelf. The 3xAA holder is stuck to
// the inside (double-sided foam tape, or screws through its own holes) inside a rim.
module door() {
    difference() {
        union() {
            intersection() {   // outer face follows the case back exactly
                translate([0, 0, 0]) linear_extrude(WALL) door2d();
                body();
            }
            // hook tab
            translate([0, DOOR_Y1 - 1, WALL]) linear_extrude(DOOR_TAB_T)
                rrect(DOOR_W - 16, 2 * (DOOR_TAB + 1), 1.5);
            translate([-(DOOR_W - 16)/2, DOOR_Y1 - 2, 0]) cube([DOOR_W - 16, 2, WALL + DOOR_TAB_T]);
            // holder locating rim
            translate([0, BATT_HOLDER[3], WALL - 0.01]) linear_extrude(1.5) difference() {
                rrect(BATT_HOLDER[0] + 0.8 + 2.4, BATT_HOLDER[1] + 0.8 + 2.4, 2);
                rrect(BATT_HOLDER[0] + 0.8, BATT_HOLDER[1] + 0.8, 1);
            }
        }
        // countersunk screw hole
        translate([0, DOOR_SCREW_Y, -0.01]) {
            cylinder(d = SCREW_D, h = WALL + 2);
            cylinder(d1 = 5.6, d2 = SCREW_D, h = 1.4);
        }
        // wire notch in the rim (holder leads exit at the top end)
        translate([-BATT_HOLDER[0]/2 + 4, BATT_HOLDER[3] + BATT_HOLDER[1]/2 - 1, WALL]) cube([8, 4, 3]);
    }
}

// ------------------------------------------------------------ button caps
// Each cap rests on its 12x12 tact switch; a flange under the front wall makes it
// captive once the case is screwed shut. Print in translucent filament: the LEDs
// shine up through holes in the board into it.
module cap(b) {
    shape = b[1]; size = b[2];
    top = CASE_T + CAP_PROUD;
    translate([b[3], b[4], 0]) difference() {
        union() {
            translate([0, 0, CAP_BOT]) linear_extrude(INNER_FRONT - CAP_BOT - 0.3)
                btn2d(shape, size, CAP_FLANGE);
            // body + a 1 mm bevel. A scaled extrude (not hull) so the star keeps its points.
            translate([0, 0, CAP_BOT]) linear_extrude(top - CAP_BOT - 1.0) btn2d(shape, size);
            translate([0, 0, top - 1.0]) linear_extrude(1.0, scale = (size - 2) / size) btn2d(shape, size);
        }
        // shallow icons
        translate([0, 0, top - 0.6]) linear_extrude(1) {
            if (shape == "circle") handset2d(-30);
            if (shape == "square") handset2d(150);
        }
        // hollow underside: lighter, lets the LED light in, stem sits in the middle
        translate([0, 0, CAP_BOT - 0.01]) linear_extrude(top - CAP_BOT - 3.0)
            btn2d(shape, size, -2.0);
    }
    // centre post down onto the switch stem
    translate([b[3], b[4], CAP_BOT]) cylinder(d = 5, h = top - CAP_BOT - 2.0);
}

module handset2d(rot) {
    rotate(rot) {
        hull() { translate([-5, 0]) circle(d = 3.4); translate([5, 0]) circle(d = 3.4); }
        for (s = [-1, 1]) translate([s * 5.6, -1.6]) rrect(4.2, 3.4, 1.2);
    }
}

// ------------------------------------------------------------ speaker back box
module speaker_cup() {
    ring_od = SPEAKER_D + 0.4 + 3.2;
    h = 10;
    translate([SPEAKER_XY[0], SPEAKER_XY[1], INNER_FRONT - 4 + 3 - h]) difference() {
        cylinder(d = ring_od + 0.3 + 3.2, h = h);
        translate([0, 0, 1.6]) cylinder(d = ring_od + 0.3, h = h);
        translate([0, -ring_od/2 + 2, 4]) rotate([90, 0, 0]) cylinder(d = 3, h = 10);  // wire hole
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
    color("dimgray") for (b = BUTTONS) translate([b[3], b[4], PCB_FRONT_Z]) {
        linear_extrude(3.6) square(SWITCH_W, center = true);
        cylinder(d = 6.6, h = SWITCH_H);
    }
    color("red") for (l = LEDS) translate([l[0], l[1], PCB_BACK_Z - 0.5]) cube([1.6, 3.2, 1], center = true);
    color("silver") translate([USBC_X, PCB_BOTTOM_Y + 3.6, USBC_Z]) cube([8.9, 7.3, 3.2], center = true);
}
module battery_dummy() {
    color("royalblue") translate([0, BATT_HOLDER[3], BATT_FLOOR_Z]) {
        translate([0, 0, BATT_HOLDER[2]/2]) cube([BATT_HOLDER[0], BATT_HOLDER[1], BATT_HOLDER[2]], center = true);
    }
    color("silver") for (i = [-1, 0, 1]) translate([i * 14.6, BATT_HOLDER[3], BATT_FLOOR_Z + 1.2 + 7.25])
        rotate([90, 0, 0]) cylinder(d = 14.5, h = 50.5, center = true);
}
module speaker_dummy() {
    color("black") translate([SPEAKER_XY[0], SPEAKER_XY[1], INNER_FRONT - 5]) cylinder(d = SPEAKER_D, h = 5);
}

module pcb_outline_2d() {
    difference() {
        pcb2d();
        for (h = MOUNT_HOLES) translate(h) circle(d = MOUNT_HOLE_D);
    }
}

// ------------------------------------------------------------ part selector
SHELL_COLOR = [0.98, 0.93, 0.80];   // 90s "almond" plastic

module caps_colored() { for (i = [0 : len(BUTTONS) - 1]) color(BTN_COLORS[i]) cap(BUTTONS[i]); }
function btn(name) = [for (b = BUTTONS) if (b[0] == name) b][0];

if (part == "assembly") {
    color(SHELL_COLOR) back_shell();
    color([0.9, 0.85, 0.72]) door();
    pcb_dummy(); battery_dummy(); speaker_dummy();
    color("gray") speaker_cup();
    caps_colored();
    color([0.98, 0.93, 0.80, 0.25]) front_shell();
} else if (part == "closed") {
    color(SHELL_COLOR) back_shell();
    color([0.95, 0.89, 0.75]) door();
    color(SHELL_COLOR) front_shell();
    caps_colored();
} else if (part == "back_view") {
    translate([0, 0, CASE_T]) rotate([0, 180, 0]) {
        color(SHELL_COLOR) back_shell();
        color([0.92, 0.86, 0.72]) door();
        color(SHELL_COLOR) front_shell();
    }
} else if (part == "exploded") {
    translate([0, 0, -22]) color([0.9, 0.85, 0.72]) door();
    translate([0, 0, -14]) battery_dummy();
    color(SHELL_COLOR) back_shell();
    translate([0, 0, 18]) { pcb_dummy(); speaker_dummy(); color("gray") speaker_cup(); }
    translate([0, 0, 46]) caps_colored();
    translate([0, 0, 66]) color(SHELL_COLOR) front_shell();
} else if (part == "front") {
    if (print_orient) translate([0, 0, CASE_T]) rotate([180, 0, 0]) front_shell(); else front_shell();
} else if (part == "back") {
    back_shell();
} else if (part == "door") {
    door();
} else if (part == "cap_tattle") {
    translate(print_orient ? [-btn("TATTLE")[3], -btn("TATTLE")[4], -CAP_BOT] : [0, 0, 0]) cap(btn("TATTLE"));
} else if (part == "cap_call") {
    translate(print_orient ? [-btn("CALL")[3], -btn("CALL")[4], -CAP_BOT] : [0, 0, 0]) cap(btn("CALL"));
} else if (part == "cap_hangup") {
    translate(print_orient ? [-btn("HANG UP")[3], -btn("HANG UP")[4], -CAP_BOT] : [0, 0, 0]) cap(btn("HANG UP"));
} else if (part == "speaker_cup") {
    if (print_orient) translate([0, 0, -(INNER_FRONT - 4 + 3 - 10)]) speaker_cup(); else speaker_cup();
} else if (part == "pcb") {
    pcb_outline_2d();
}
