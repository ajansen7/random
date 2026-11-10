"""
LCSC / JLCPCB part numbers for the assembly BOM, keyed by (value, footprint name) as
make_fab.py groups them. Checked against JLC's parts library (stock and datasheets) on
2026-11-10. Re-check stock before each order; JLC stock moves.

Substitutions (the design value stays the same, the part differs):
  - L1: Sunlord SWPA4020S1R5NT instead of Coilcraft XAL4020-152 ($0.06 vs $13.42).
    1.5 uH, Isat 4.45 A, Irms 1.98 A, 35 mOhm. Its 0.95 mm end terminals overlap the
    XAL4020 pads (0.70-1.68 mm from centre) by about two thirds; pad length matches.
  - LED1-14: OPSCO SK6812mini-012 (3.7 x 3.5 x 1.1 mm). The plain "SK6812MINI" is out of
    stock at JLC. Its datasheet pad map (top view: DOUT top-left, GND bottom-left,
    VDD top-right, DIN bottom-right) matches our footprint's signals; only the printed pin
    numbers differ, so check LED rotation in JLC's placement preview.
    Not usable: SK6812MINI-HS (pins rotated one position), WS2812B-Mini (6.2 mm gull-wing
    leads), SK6812MINI-B-012 and XL-3535RGBC-2812B-T (need >= 4.5 V; VLED is the battery).
  - F1: 16 V-rated nSMD075 instead of the 6 V one, for margin on VBUS.
"""

LCSC = {
    ("HRO TYPE-C-31-M-12", "USB_C_Receptacle_HRO_TYPE-C-31-M-12"): "C165948",
    ("5.1k", "R_0402_1005Metric"): "C25905",
    ("0.75A hold", "Fuse_1206_3216Metric"): "C52195112",      # NSMD075/16, 750 mA hold, 16 V
    ("USBLC6-2SC6", "SOT-23-6"): "C7519",                     # ST original
    ("SMF5.0A", "D_SMF"): "C193402",                          # SOD-123FL body fits D_SMF
    ("BQ24075RGT", "VQFN-16-1EP_3x3mm_P0.5mm_EP1.6x1.6mm"): "C15464",
    ("10u", "C_0805_2012Metric"): "C15850",                   # 25 V X5R (basic)
    ("4.7u", "C_0603_1608Metric"): "C19666",                  # 16 V X5R (basic)
    ("10k", "R_0402_1005Metric"): "C25744",
    ("1.5k", "R_0402_1005Metric"): "C25867",
    ("1.2k", "R_0402_1005Metric"): "C25862",
    ("PCM12 power", "SW_SPDT_PCM12"): "C221841",              # C&K PCM12SMTR
    ("PCM12 TATTLE/PLAY", "SW_SPDT_PCM12"): "C221841",
    ("TPS63001", "Texas_DRC0010J_ThermalVias"): "C28060",
    ("1.5u XAL4020", "L_Coilcraft_XAL4020-XXX"): "C42099",    # Sunlord SWPA4020S1R5NT (see above)
    ("22u", "C_0805_2012Metric"): "C45783",                   # 25 V X5R (basic)
    ("0.1u", "C_0402_1005Metric"): "C1525",                   # 16 V X7R (basic)
    ("ESP32-S3-WROOM-1-N16R8", "ESP32-S3-WROOM-1"): "C2913202",
    ("100u", "C_1206_3216Metric"): "C15008",                  # 6.3 V X5R on +3V3 (basic)
    ("1u", "C_0402_1005Metric"): "C52923",                    # 25 V X5R (basic)
    ("RESET", "SW_SPST_B3U-1000P"): "C231329",                # Omron B3U-1000P
    ("BOOT", "SW_SPST_B3U-1000P"): "C231329",
    ("ICS-43434", "InvenSense_ICS-43434-6_3.5x2.65mm"): "C5656610",
    ("100k", "R_0402_1005Metric"): "C25741",
    ("MAX98357A", "TQFN-16-1EP_3x3mm_P0.5mm_EP1.23x1.23mm"): "C910544",
    ("MAX17048G+", "TDFN-8-1EP_2x2mm_P0.5mm_EP0.8x1.2mm"): "C2682616",
    ("4.7k", "R_0402_1005Metric"): "C25900",
    ("TCA9555RTWR", "WQFN-24-1EP_4x4mm_P0.5mm_EP2.45x2.45mm"): "C140277",
    ("AO3401A", "SOT-23"): "C15127",                          # basic
    ("2N7002", "SOT-23"): "C8545",                            # basic
    ("74AHCT1G125", "SOT-23-5"): "C842287",                   # Diodes 74AHCT1G125W5-7, standard SOT-25 pinout
    ("330", "R_0402_1005Metric"): "C25104",
    ("SK6812MINI", "LED_SK6812MINI_PLCC4_3.5x3.5mm_P1.75mm"): "C2886570",  # SK6812mini-012 (see above)
}
