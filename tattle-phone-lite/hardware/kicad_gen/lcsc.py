"""
LCSC / JLCPCB part numbers for the Lite assembly BOM, keyed by (value, footprint name) as
make_fab.py groups them (one BOM line per LCSC number). Checked against JLC's parts
library (stock + datasheets) on 2026-11-10; re-check stock before ordering.

Hand-soldered, so not here: SW1-3 (12x12 THT tact), the battery and speaker wire pads.
"""

LCSC = {
    ("ESP32-S3-WROOM-1-N8R2", "ESP32-S3-WROOM-1_ThermalVias0.3mm"): "C2913204",
    ("LMD3526B261-OFA03", "LinkMems_LMD3526B261_3.5x2.65mm"): "C41347997",   # PDM mic, bottom port
    ("MAX98357A", "TQFN-16-1EP_3x3mm_P0.5mm_EP1.23x1.23mm"): "C910544",
    ("RT9080-33", "SOT-23-5"): "C841192",                     # RT9080-33GJ5
    ("AO3401A", "SOT-23"): "C15127",                          # basic
    ("POWER", "SW_SPDT_Shouhan_MSK12C02"): "C431540",
    ("HRO TYPE-C-31-M-12", "USB_C_Receptacle_HRO_TYPE-C-31-M-12"): "C165948",
    ("LTST-C230KSKT yellow", "LED_1206_3216Metric_ReverseMount_Hole1.65x2.4mm"): "C125108",
    ("LTST-C230KGKT green", "LED_1206_3216Metric_ReverseMount_Hole1.65x2.4mm"): "C125106",
    ("LTST-C230KRKT red", "LED_1206_3216Metric_ReverseMount_Hole1.65x2.4mm"): "C125107",
    ("22u", "C_0805_2012Metric"): "C45783",                   # 25 V X5R (basic)
    ("100u", "C_1206_3216Metric"): "C15008",                  # 6.3 V X5R on +3V3 (basic)
    ("0.1u", "C_0402_1005Metric"): "C1525",                   # basic
    ("1u", "C_0402_1005Metric"): "C52923",                    # basic
    ("1M", "R_0402_1005Metric"): "C26083",                    # basic
    ("10k", "R_0402_1005Metric"): "C25744",                   # basic
    ("100k", "R_0402_1005Metric"): "C25741",                  # basic
    ("150", "R_0402_1005Metric"): "C25082",                   # preferred
    ("5.1k", "R_0402_1005Metric"): "C25905",                  # basic
}
