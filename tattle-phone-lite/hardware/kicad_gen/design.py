"""
Tattle Phone Lite circuit: every part, its footprint, and what each pin connects to.

build.py turns this into a KiCad schematic (a net label on every pin, so it reads as a
netlist) and a board with every footprint placed and every pad on its net.
Edit this file, re-run build.py, and both stay in agreement.

Pins are given by NAME as the KiCad library symbol spells it (e.g. "~{SD_MODE}", "IO7"),
or by number for passives ("1"/"2"). Pins left out are marked no-connect.

Placement is in the case frame from ../layout.py (mm; x right, y toward the earpiece).
side "F" = front (faces the buttons), "B" = back. Every factory-placed part is on the
BACK, so the board needs single-sided assembly. Over the battery (y 33..91) back-side
parts must stay <= 1.2 mm tall: 0402s, the QFN, LEDs. No 0805/1206 caps there.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import layout as LY  # noqa: E402

R0402 = "Resistor_SMD:R_0402_1005Metric"
C0402 = "Capacitor_SMD:C_0402_1005Metric"
C0805 = "Capacitor_SMD:C_0805_2012Metric"
C1206 = "Capacitor_SMD:C_1206_3216Metric"
WIRE = "Connector_Wire:SolderWire-0.5sqmm_1x01_D0.9mm_OD2.1mm"
TP = "TestPoint:TestPoint_Pad_D1.0mm"

PARTS = []      # filled by part()


def part(ref, lib, value, fp, pins, at=None, side="B", rot=0, block="", dnp=False, note=""):
    PARTS.append(dict(ref=ref, lib=lib, value=value, fp=fp, pins=pins, at=at,
                      side=side, rot=rot, block=block, dnp=dnp, note=note))


def R(ref, val, a, b, at=None, rot=0, block="", fp=R0402, note=""):
    part(ref, "Device:R", val, fp, {"1": a, "2": b}, at, "B", rot, block, note=note)


def C(ref, val, a, b="GND", at=None, rot=0, block="", fp=C0402):
    part(ref, "Device:C", val, fp, {"1": a, "2": b}, at, "B", rot, block)


P = LY.placements()

# ======================================================================= batteries + power
B = "Power: 3xAA -> 3.3 V"
part("BT1", "Connector_Generic:Conn_01x01", "BAT+", WIRE, {"Pin_1": "BAT_RAW"},
     at=LY.BATT_PADS[0], block=B, note="red lead of the 3xAA holder")
part("BT2", "Connector_Generic:Conn_01x01", "BAT-", WIRE, {"Pin_1": "GND"},
     at=LY.BATT_PADS[1], block=B, note="black lead of the 3xAA holder")
part("Q1", "Transistor_FET:AO3401A", "AO3401A", "Package_TO_SOT_SMD:SOT-23",
     {"G": "GND", "D": "BAT_RAW", "S": "VBAT"}, at=(-19.5, 94.5), block=B,
     note="reverse-battery protection: off if the cells go in backwards")
part("SW4", "Switch:SW_SPDT", "POWER", "Button_Switch_SMD:SW_SPDT_Shouhan_MSK12C02",
     {"A": "VBAT", "B": "PWR_EN", "C": "GND"}, at=P["SW4"][:2], block=B,
     note="switches only the LDO enable (MSK12C02 is a 50 mA switch)")
R("R1", "1M", "PWR_EN", "GND", at=(19.5, 92.0), block=B)
part("U3", "Regulator_Linear:AP2112K-3.3", "RT9080-33", "Package_TO_SOT_SMD:SOT-23-5",
     {"VIN": "VBAT", "GND": "GND", "EN": "PWR_EN", "VOUT": "+3V3"}, at=(-14.5, 94.5), block=B,
     note="RT9080-33GJ5: same SOT-23-5 pinout as AP2112K; 2 uA idle, 5.5 V max in")
C("C1", "22u", "VBAT", at=(-14.5, 98.5), fp=C0805, block=B)
C("C2", "22u", "+3V3", at=(-11.6, 113.5), fp=C0805, rot=90, block=B)
C("C3", "100u", "+3V3", at=(-14.5, 114.5), fp=C1206, rot=90, block=B)
R("R2", "1M", "VBAT", "VBAT_SENSE", at=(-20.5, 99.5), rot=90, block=B)
R("R3", "1M", "VBAT_SENSE", "GND", at=(-19.0, 99.5), rot=90, block=B)
C("C4", "0.1u", "VBAT_SENSE", at=(-17.5, 99.5), rot=90, block=B)

# ======================================================================= ESP32-S3
B = "ESP32-S3"
part("U1", "RF_Module:ESP32-S3-WROOM-1", "ESP32-S3-WROOM-1-N8R2", "tattle_lite:ESP32-S3-WROOM-1_ThermalVias0.3mm",
     {"GND": "GND", "3V3": "+3V3", "EN": "EN",
      "IO0": "BTN_CALL", "IO4": "BTN_TATTLE", "IO5": "BTN_HANGUP",
      "IO6": "LED_Y", "IO7": "LED_G", "IO15": "LED_R",
      "IO16": "MIC_CLK", "IO17": "MIC_DATA",
      "IO10": "AMP_BCLK", "IO11": "AMP_LRC", "IO12": "AMP_DIN", "IO13": "AMP_SD",
      "IO1": "VBAT_SENSE", "USB_D-": "USB_DM", "USB_D+": "USB_DP",
      "TXD0": "TX0", "RXD0": "RX0"},
     at=P["U1"][:2], block=B, note="antenna end flush with the top edge; no copper under the antenna")
C("C5", "0.1u", "+3V3", at=(-11.6, 110.5), rot=90, block=B)
R("R4", "10k", "EN", "+3V3", at=(-11.6, 107.5), rot=90, block=B)
C("C6", "1u", "EN", at=(-11.6, 105.0), rot=90, block=B)
R("R5", "10k", "BTN_CALL", "+3V3", at=(11.6, 107.5), rot=90, block=B,
  note="IO0 strap: hold CALL while switching on = programming mode")

# ======================================================================= buttons
B = "Buttons"
for b in LY.BUTTONS:
    part(b["ref"], "Switch:SW_Push", b["name"], "Button_Switch_THT:SW_PUSH-12mm",
         {"1": "BTN_" + b["name"].replace(" ", ""), "2": "GND"}, at=(b["x"], b["y"]), side="F", block=B,
         note="12x12x7.3 mm THT, hand-soldered; clip the leads flush (battery below)")

# ======================================================================= LEDs
B = "LEDs"
led_net = {"TATTLE": "LED_Y", "CALL": "LED_G", "HANG UP": "LED_R"}
led_val = {"TATTLE": "LTST-C230KSKT yellow", "CALL": "LTST-C230KGKT green", "HANG UP": "LTST-C230KRKT red"}
for i, (ref, b, x, y) in enumerate(LY.leds()):
    n = i + 1
    part(ref, "Device:LED", led_val[b["name"]], "tattle_lite:LED_1206_3216Metric_ReverseMount_Hole1.65x2.4mm",
         {"K": "GND", "A": f"LED{n}_A"}, at=(x, y), rot=90, block=B,
         note="reverse mount: shines through the board into the button cap")
    rx = x + 2.6 if x >= 0 else x - 2.6
    R(f"R{10 + n}", "150", led_net[b["name"]], f"LED{n}_A", at=(rx, y), rot=90, block=B)

# ======================================================================= microphone
B = "Microphone"
part("MK1", "tattle_lite:LMD3526B261", "LMD3526B261-OFA03", "tattle_lite:LinkMems_LMD3526B261_3.5x2.65mm",
     {"DATA": "MIC_DATA", "L/R": "GND", "CLK": "MIC_CLK", "VDD": "+3V3", "GND": "GND"},
     at=LY.MIC_XY, block=B, note="PDM, bottom port over the board hole; L/R low = left channel")
C("C7", "0.1u", "+3V3", at=(3.6, 13.5), rot=90, block=B)

# ======================================================================= amp + speaker
B = "Speaker amp"
part("U2", "Audio:MAX98357A", "MAX98357A", "Package_DFN_QFN:TQFN-16-1EP_3x3mm_P0.5mm_EP1.23x1.23mm",
     {"DIN": "AMP_DIN", "GAIN_SLOT": "VBAT", "GND": "GND", "~{SD_MODE}": "AMP_SD",
      "VDD": "VBAT", "OUTP": "SPK_P", "OUTN": "SPK_N", "LRCLK": "AMP_LRC", "BCLK": "AMP_BCLK",
      "PAD": "GND"},
     at=(16.0, 102.0), block=B, note="runs straight from the batteries; GAIN to VDD = 6 dB")
C("C8", "22u", "VBAT", at=(20.6, 102.0), fp=C0805, rot=90, block=B)
C("C9", "0.1u", "VBAT", at=(21.0, 105.6), rot=90, block=B)
R("R6", "100k", "AMP_SD", "GND", at=(14.5, 95.5), block=B)   # in open pour below the module
part("LS1", "Connector_Generic:Conn_01x01", "SPK+", WIRE, {"Pin_1": "SPK_P"}, at=LY.SPK_PADS[0], block=B)
part("LS2", "Connector_Generic:Conn_01x01", "SPK-", WIRE, {"Pin_1": "SPK_N"}, at=LY.SPK_PADS[1], block=B)

# ======================================================================= USB-C (programming only)
B = "USB-C (programming only)"
part("J1", "Connector:USB_C_Receptacle_USB2.0_16P", "HRO TYPE-C-31-M-12",
     "Connector_USB:USB_C_Receptacle_HRO_TYPE-C-31-M-12",
     {"GND": "GND", "SHIELD": "GND", "CC1": "CC1", "CC2": "CC2", "D+": "USB_DP", "D-": "USB_DM",
      "VBUS": "VBUS"},
     at=(LY.USBC_X, LY.PCB_BOTTOM_Y + 3.65), block=B,
     note="VBUS goes nowhere (own net, only J1's pads): USB never powers the board, so it can't charge alkalines")
R("R7", "5.1k", "CC1", "GND", at=(LY.USBC_X - 6.2, 12.0), rot=90, block=B)
R("R8", "5.1k", "CC2", "GND", at=(LY.USBC_X + 6.2, 12.0), rot=90, block=B)

# ======================================================================= mechanical / test
B = "Mechanical"
for n, (x, y) in enumerate(LY.MOUNT_HOLES):
    part(f"H{n + 1}", "Mechanical:MountingHole", "M2.5", "MountingHole:MountingHole_2.7mm_M2.5",
         {}, at=(x, y), side="F", block=B)
for n, (x, y) in enumerate([(-22.0, 30.0), (22.0, 30.0), (-25.0, 103.0)]):
    part(f"FID{n + 1}", "Mechanical:Fiducial", "Fiducial", "Fiducial:Fiducial_1mm_Mask2mm",
         {}, at=(x, y), block=B)
for n, (net, x, y) in enumerate([("TX0", 8.0, 7.0), ("RX0", 10.6, 7.0), ("GND", 13.2, 7.0),
                                 ("+3V3", -25.0, 113.5), ("VBAT", -22.0, 116.0)]):
    part(f"TP{n + 1}", "Connector:TestPoint", net, TP, {"1": net}, at=(x, y), block=B)

# Parts that get a GND via in their exposed (thermal) pad
EP_VIA_PARTS = ["U2"]

# Nets that get wide tracks
POWER_NETS = ["GND", "BAT_RAW", "VBAT", "+3V3"]   # speaker lines: default 0.2 mm is plenty for a small earpiece
# Nets that need a PWR_FLAG for ERC
FLAG_NETS = ["GND", "BAT_RAW", "VBAT", "+3V3"]
