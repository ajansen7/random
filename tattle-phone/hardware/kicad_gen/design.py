"""
Tattle Phone circuit: every part, its footprint, and what each pin connects to.

build.py turns this into a KiCad schematic (labels on every pin, so it reads as
a netlist) and a board with every footprint placed and every pad on its net.
Edit this file, re-run build.py, and both stay in agreement.

Pins are given by NAME as the KiCad library symbol spells it (e.g. "~{CE}", "IO7"),
or by number for passives ("1"/"2"). Pins left out are marked no-connect.

Placement is in the case frame from ../layout.py (mm; x right, y toward the
earpiece). side "F" = front (faces the keypad), "B" = back.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import layout as LY

R0402 = "Resistor_SMD:R_0402_1005Metric"
C0402 = "Capacitor_SMD:C_0402_1005Metric"
C0603 = "Capacitor_SMD:C_0603_1608Metric"
C0805 = "Capacitor_SMD:C_0805_2012Metric"
C1206 = "Capacitor_SMD:C_1206_3216Metric"
JST2 = "Connector_JST:JST_PH_S2B-PH-K_1x02_P2.00mm_Horizontal"
TP = "TestPoint:TestPoint_Pad_D1.0mm"

PARTS = []      # filled by part()


def part(ref, lib, value, fp, pins, at=None, side="B", rot=0, block="", dnp=False, note=""):
    PARTS.append(dict(ref=ref, lib=lib, value=value, fp=fp, pins=pins, at=at,
                      side=side, rot=rot, block=block, dnp=dnp, note=note))


def R(ref, val, a, b, at=None, rot=0, block="", dnp=False, fp=R0402):
    part(ref, "Device:R", val, fp, {"1": a, "2": b}, at, "B", rot, block, dnp)


def C(ref, val, a, b="GND", at=None, rot=0, block="", fp=C0402):
    part(ref, "Device:C", val, fp, {"1": a, "2": b}, at, "B", rot, block)


# ======================================================================= USB-C in
B = "USB-C input"
part("J1", "Connector:USB_C_Receptacle_USB2.0_16P", "HRO TYPE-C-31-M-12",
     "Connector_USB:USB_C_Receptacle_HRO_TYPE-C-31-M-12",
     {"VBUS": "VBUS_C", "GND": "GND", "SHIELD": "GND", "CC1": "CC1", "CC2": "CC2",
      "D+": "USB_DP", "D-": "USB_DM"},
     at=(0, LY.PCB_BOTTOM_Y + 3.65), block=B, note="mouth flush with board edge")
R("R1", "5.1k", "CC1", "GND", at=(-6.2, 9.5), rot=90, block=B)
R("R2", "5.1k", "CC2", "GND", at=(6.2, 9.5), rot=90, block=B)
part("F1", "Device:Polyfuse", "0.75A hold", "Fuse:Fuse_1206_3216Metric",
     {"1": "VBUS_C", "2": "VIN"}, at=(-9.5, 7.5), rot=90, block=B)
part("D1", "Power_Protection:USBLC6-2SC6", "USBLC6-2SC6", "Package_TO_SOT_SMD:SOT-23-6",
     {"I/O1": "USB_DP", "I/O2": "USB_DM", "VBUS": "VIN", "GND": "GND"},
     at=(7.8, 17.0), rot=90, block=B)
part("D2", "Diode:SMF5V0A", "SMF5.0A", "Diode_SMD:D_SMF",
     {"A1": "VIN", "A2": "GND"}, at=(-13.5, 8.5), rot=90, block=B)

# ======================================================================= charger
B = "Charger / power path"
part("U2", "Battery_Management:BQ24075RGT", "BQ24075RGT",
     "Package_DFN_QFN:VQFN-16-1EP_3x3mm_P0.5mm_EP1.6x1.6mm",
     {"TS": "NTC", "BAT": "VBAT", "~{CE}": "GND", "EN2": "EN2", "EN1": "GND",
      "~{PGOOD}": "PGOOD", "VSS": "GND", "~{CHG}": "CHG", "OUT": "VSYS", "ILIM": "ILIM",
      "IN": "VIN", "TMR": "TMR", "SYSOFF": "SYSOFF", "ISET": "ISET"},
     at=(-11.0, 18.5), block=B, note="OUT regulation level: check datasheet; all loads OK to 5.5 V")
C("C1", "10u", "VIN", at=(-15.5, 15.0), fp=C0805, rot=90, block=B)
C("C2", "10u", "VSYS", at=(-15.5, 19.5), fp=C0805, rot=90, block=B)
C("C3", "4.7u", "VBAT", at=(-11.0, 23.0), fp=C0603, block=B)
R("R3", "10k", "EN2", "VIN", at=(-5.8, 14.6), block=B)
R("R4", "1.5k", "ILIM", "GND", at=(-5.8, 16.3), block=B)      # ~1 A input limit
R("R5", "DNP", "TMR", "GND", at=(-5.8, 18.0), block=B, dnp=True)  # open = default timers
R("R6", "1.2k", "ISET", "GND", at=(-5.8, 19.7), block=B)      # ~750 mA charge
R("R7", "10k", "CHG", "+3V3", at=(-5.8, 21.4), block=B)
R("R8", "10k", "PGOOD", "+3V3", at=(-5.8, 23.1), block=B)
part("J2", "Connector_Generic:Conn_01x02", "Battery (JST-PH)", JST2,
     {"Pin_1": "VBAT", "Pin_2": "GND"}, at=(-18.5, 33.0), block=B,
     note="check your cell's lead polarity")
part("J4", "Connector_Generic:Conn_01x02", "NTC 10k (glue to cell)", JST2,
     {"Pin_1": "NTC", "Pin_2": "GND"}, at=(-18.5, 43.0), block=B)
part("SW14", "Switch:SW_SPDT", "PCM12 power", "Button_Switch_SMD:SW_SPDT_PCM12",
     {"A": "GND", "B": "SYSOFF", "C": "VBAT"},
     at=(LY.PCB_CORNERS[1][0] + 1.4, LY.SLIDE_SWITCH_Y), block=B,
     note="actuator out the right edge: up = ON (SYSOFF low)")
part("SW17", "Switch:SW_SPDT", "PCM12 TATTLE/PLAY", "Button_Switch_SMD:SW_SPDT_PCM12",
     {"A": "GND", "B": "MODE_SW"},
     at=(-(LY.PCB_CORNERS[1][0] + 1.4), LY.SLIDE_SWITCH_Y), block="ESP32-S3",
     note="actuator out the left edge: LOW = PLAY (stay on Wi-Fi), HIGH = TATTLE-only low power")

# ======================================================================= 3.3 V
B = "3.3 V buck-boost"
part("U3", "Regulator_Switching:TPS63001", "TPS63001", "Package_SON:Texas_DRC0010J_ThermalVias",
     {"VOUT": "+3V3", "L2": "SW2", "PGND": "GND", "L1": "SW1", "VIN": "VSYS", "EN": "VSYS",
      "PS/SYNC": "GND", "VINA": "VSYS", "GND": "GND", "FB": "+3V3"},
     at=(-9.0, 44.5), block=B)
part("L1", "Device:L", "1.5u XAL4020", "Inductor_SMD:L_Coilcraft_XAL4020-XXX",
     {"1": "SW1", "2": "SW2"}, at=(-9.0, 39.5), block=B)
C("C4", "10u", "VSYS", at=(-4.5, 43.0), fp=C0805, rot=90, block=B)
C("C5", "22u", "+3V3", at=(-13.5, 43.0), fp=C0805, rot=90, block=B)
C("C6", "22u", "+3V3", at=(-13.5, 47.0), fp=C0805, rot=90, block=B)
C("C7", "0.1u", "VSYS", at=(-6.0, 48.0), block=B)

# ======================================================================= ESP32-S3
B = "ESP32-S3"
part("U1", "RF_Module:ESP32-S3-WROOM-1", "ESP32-S3-WROOM-1-N16R8", "RF_Module:ESP32-S3-WROOM-1",
     {"GND": "GND", "3V3": "+3V3", "EN": "EN",
      "IO4": "MIC_BCLK", "IO5": "MIC_WS", "IO6": "MIC_SD", "IO7": "CALL_BTN",
      "IO15": "AMP_BCLK", "IO16": "AMP_LRC", "IO17": "AMP_DIN", "IO18": "AMP_SD",
      "IO8": "LED_DATA", "IO19": "USB_DM", "IO20": "USB_DP", "IO9": "SDA", "IO10": "SCL",
      "IO11": "LED_PWR_EN", "IO12": "CHG", "IO13": "PGOOD", "IO14": "GAUGE_ALRT",
      "IO21": "KEY_INT", "IO0": "BOOT", "IO38": "MODE_SW", "IO1": "SPARE_IO1", "TXD0": "TX0", "RXD0": "RX0"},
     at=(0, LY.PCB_TOP_Y - LY.ESP32_MODULE["l"] / 2), block=B,
     note="antenna end flush with top edge; no copper under antenna")
C("C8", "22u", "+3V3", at=(-11.6, 113.5), fp=C0805, rot=90, block=B)
C("C9", "0.1u", "+3V3", at=(-11.6, 110.5), rot=90, block=B)
C("C10", "100u", "+3V3", at=(-14.5, 114.0), fp=C1206, rot=90, block=B)
R("R9", "10k", "EN", "+3V3", at=(-11.6, 107.0), rot=90, block=B)
C("C11", "1u", "EN", at=(-11.6, 104.5), rot=90, block=B)
part("SW15", "Switch:SW_Push", "RESET", "Button_Switch_SMD:SW_SPST_B3U-1000P",
     {"1": "EN", "2": "GND"}, at=(-14.5, 101.0), rot=90, block=B)
R("R10", "10k", "BOOT", "+3V3", at=(11.6, 107.0), rot=90, block=B)
part("SW16", "Switch:SW_Push", "BOOT", "Button_Switch_SMD:SW_SPST_B3U-1000P",
     {"1": "BOOT", "2": "GND"}, at=(14.5, 101.0), rot=90, block=B)
R("R21", "10k", "MODE_SW", "+3V3", at=(-19.5, 52.0), rot=90, block=B)

# ======================================================================= microphone
B = "Microphone"
part("MK1", "Sensor_Audio:ICS-43434", "ICS-43434", "Sensor_Audio:InvenSense_ICS-43434-6_3.5x2.65mm",
     {"WS": "MIC_WS", "LR": "GND", "GND": "GND", "SCK": "MIC_BCLK", "VDD": "+3V3", "SD": "MIC_SD"},
     at=LY.MIC_XY, block=B, note="port hole faces the front mic duct")
C("C12", "0.1u", "+3V3", at=(-3.5, 15.5), block=B)
R("R11", "100k", "MIC_SD", "GND", at=(-3.5, 13.0), block=B)

# ======================================================================= amp
B = "Speaker amp"
part("U4", "Audio:MAX98357A", "MAX98357A", "Package_DFN_QFN:TQFN-16-1EP_3x3mm_P0.5mm_EP1.23x1.23mm",
     {"DIN": "AMP_DIN", "GAIN_SLOT": "VSYS", "GND": "GND", "~{SD_MODE}": "AMP_SD",
      "VDD": "VSYS", "OUTP": "SPK_P", "OUTN": "SPK_N", "LRCLK": "AMP_LRC", "BCLK": "AMP_BCLK",
      "PAD": "GND"},
     at=(9.0, 44.5), block=B, note="GAIN to VDD = 6 dB")
C("C13", "10u", "VSYS", at=(13.5, 43.0), fp=C0805, rot=90, block=B)
C("C14", "0.1u", "VSYS", at=(5.5, 43.0), block=B)
R("R12", "100k", "AMP_SD", "GND", at=(5.5, 46.5), block=B)
part("J3", "Connector_Generic:Conn_01x02", "Speaker 8R (JST-PH)", JST2,
     {"Pin_1": "SPK_P", "Pin_2": "SPK_N"}, at=(18.5, 33.0), block=B,
     note="speaker leads run up the inside of the case to the earpiece")

# ======================================================================= battery gauge
B = "Battery gauge"
part("U6", "tattle:MAX17048", "MAX17048G+", "Package_DFN_QFN:TDFN-8-1EP_2x2mm_P0.5mm_EP0.8x1.2mm",
     {"CTG": "GND", "CELL": "VBAT", "VDD": "VBAT", "GND": "GND", "ALRT": "GAUGE_ALRT",
      "QSTRT": "GND", "SCL": "SCL", "SDA": "SDA", "EP": "GND"},
     at=(6.75, 59.5), block=B)
C("C15", "0.1u", "VBAT", at=(6.75, 62.0), block=B)
R("R13", "4.7k", "SDA", "+3V3", at=(9.0, 57.0), block=B)
R("R14", "4.7k", "SCL", "+3V3", at=(4.5, 57.0), block=B)
R("R15", "10k", "GAUGE_ALRT", "+3V3", at=(9.0, 62.0), block=B)

# ======================================================================= keypad
B = "Keypad"
key_nets = ["KEY_1", "KEY_2", "KEY_3", "KEY_4", "KEY_5", "KEY_6", "KEY_7", "KEY_8", "KEY_9",
            "KEY_STAR", "KEY_0", "KEY_HASH"]
io_pins = [f"P0{i}" for i in range(8)] + [f"P1{i}" for i in range(4)]
pins = {"~{INT}": "KEY_INT", "A0": "GND", "A1": "GND", "A2": "GND", "GND": "GND", "EPAD": "GND",
        "VCC": "+3V3", "SCL": "SCL", "SDA": "SDA"}
pins.update({p: n for p, n in zip(io_pins, key_nets)})
part("U7", "Interface_Expansion:TCA9555RTWR", "TCA9555RTWR",
     "Package_DFN_QFN:WQFN-24-1EP_4x4mm_P0.5mm_EP2.45x2.45mm",
     pins, at=(-6.75, 73.5), block=B, note="I2C 0x20, 100k internal pull-ups; QFN fits between the key-switch legs")
C("C16", "0.1u", "+3V3", at=(-6.75, 79.0), rot=90, block=B)
R("R16", "10k", "KEY_INT", "+3V3", at=(-6.75, 68.5), rot=90, block=B)
for i, (_, label, x, y) in enumerate(LY.keys()):
    part(f"SW{i + 1}", "Switch:SW_Push", f"KEY {label}", "Button_Switch_THT:SW_PUSH_6mm_H4.3mm",
         {"1": key_nets[i], "2": "GND"}, at=(x, y), side="F", rot=90, block=B)
part("SW13", "Switch:SW_Push", "CALL", "Button_Switch_THT:SW_PUSH-12mm",
     {"1": "CALL_BTN", "2": "GND"}, at=LY.CALL_XY, side="F", block=B)
R("R17", "10k", "CALL_BTN", "+3V3", at=(-9.5, 28.0), block=B)

# ======================================================================= LEDs
B = "LEDs"
part("Q1", "Transistor_FET:AO3401A", "AO3401A", "Package_TO_SOT_SMD:SOT-23",
     {"G": "LED_GATE", "S": "VSYS", "D": "VLED"}, at=(6.75, 85.5), block=B)
part("Q2", "Transistor_FET:2N7002", "2N7002", "Package_TO_SOT_SMD:SOT-23",
     {"G": "LED_PWR_EN", "S": "GND", "D": "LED_GATE"}, at=(6.75, 89.5), block=B)
R("R18", "100k", "LED_GATE", "VSYS", at=(9.3, 87.5), rot=90, block=B)
R("R19", "100k", "LED_PWR_EN", "GND", at=(4.2, 87.5), rot=90, block=B)
part("U8", "74xGxx:74AHCT1G125", "74AHCT1G125", "Package_TO_SOT_SMD:SOT-23-5",
     {"1": "GND", "2": "LED_DATA", "3": "GND", "4": "LED_DATA_5V", "5": "VLED"},
     at=(6.75, 74.5), block=B, note="3.3 V -> VLED level shift")
C("C17", "0.1u", "VLED", at=(9.3, 74.5), rot=90, block=B)
R("R20", "330", "LED_DATA_5V", "LED_D0", at=(4.2, 74.5), rot=90, block=B)
C("C18", "10u", "VLED", at=(8.0, 77.6), fp=C0805, block=B)
led_xy = [(LY.CALL_XY[0], LY.CALL_XY[1] + LY.CALL_LED_OFFSET_Y)] + \
         [(x + LY.LED_OFFSET_X, y) for _, _, x, y in LY.keys()] + [LY.STATUS_LED_XY]
for i, (x, y) in enumerate(led_xy):
    part(f"LED{i + 1}", "LED:SK6812MINI", "SK6812MINI", "LED_SMD:LED_SK6812MINI_PLCC4_3.5x3.5mm_P1.75mm",
         {"DIN": f"LED_D{i}", "DOUT": f"LED_D{i + 1}" if i < len(led_xy) - 1 else None,
          "VDD": "VLED", "VSS": "GND"},
         at=(x, y), side="F", block=B)
    cap_at = {0: (x, y + 2.6), len(led_xy) - 1: (6.5, 96.8)}.get(i, (x, y))
    C(f"C{20 + i}", "0.1u", "VLED", at=cap_at, block=B)

# ======================================================================= mechanical / test
B = "Mechanical"
for i, (x, y) in enumerate(LY.MOUNT_HOLES):
    part(f"H{i + 1}", "Mechanical:MountingHole", "M2.5", "MountingHole:MountingHole_2.7mm_M2.5",
         {}, at=(x, y), side="F", block=B)
for i, (x, y) in enumerate([(-17.0, 12.0), (17.0, 12.0), (-19.0, 97.0)]):
    part(f"FID{i + 1}", "Mechanical:Fiducial", "Fiducial", "Fiducial:Fiducial_1mm_Mask2mm",
         {}, at=(x, y), block=B)
for i, (net, x, y) in enumerate([("TX0", 21.0, 41.5), ("RX0", 21.0, 44.0), ("GND", 21.0, 46.5),
                                 ("VBAT", -12.0, 27.0), ("VSYS", -14.5, 27.0), ("+3V3", -17.0, 27.0),
                                 ("SPARE_IO1", 21.0, 49.0)]):
    part(f"TP{i + 1}", "Connector:TestPoint", net, TP, {"1": net}, at=(x, y), block=B)

# Nets that get wide tracks
POWER_NETS = ["GND", "VBUS_C", "VIN", "VSYS", "VBAT", "+3V3", "VLED", "SW1", "SW2", "SPK_P", "SPK_N"]
