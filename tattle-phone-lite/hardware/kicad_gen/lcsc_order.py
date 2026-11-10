"""
Parts order for building the Lite boards yourself, as a CSV for LCSC's BOM Tool.

    python3 lcsc_order.py [boards]        (default 3) -> ../fab/lcsc_order_<n>_boards.csv

Columns: Quantity, LCSC Part Number, Comment. In the BOM Tool map exactly those two
(Quantity / LCSC Part Number) and leave Comment unmapped or as "Description".
Quantities = per-board count x boards + spares. LCSC rounds small parts up to its own
minimum order, so you'll often get more passives than this asks for.
JLC's parts stock (what the assembly BOM uses) and LCSC's store are separate warehouses,
so this checks LCSC's own stock for every line and warns about anything short. Parts that
JLC has but LCSC doesn't sell are swapped for their store equivalent (STORE_EQUIVALENT).
Not included (buy elsewhere): 3xAA holder with leads, 28 mm 8 ohm speaker, M2.5 heat-set
inserts + screws, solder paste/flux.
"""
import csv
import json
import math
import time
import urllib.request
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FAB = os.path.join(HERE, "..", "fab")

# JLC-assembly-only part -> the identical part LCSC's store sells (checked datasheets)
STORE_EQUIVALENT = {
    # LinkMems mic: "-DM2" is the bag-packed version; same datasheet (M196 v1.0), pinout, size
    "C41347997": ("C53446635", "LMD3526B261-OFA03-DM2"),
}

# Hand-soldered parts that the factory BOM leaves out
EXTRA = [("C84931", "Omron B3F-4055 12x12x7.3 mm THT tact switch (SW1-3, the big buttons)", 3)]

# Spares per order (not per board). Cheap fiddly parts get plenty; the ESP32 none.
SPARES = {
    "C2913204": 0,     # ESP32-S3 module ($5): solder it carefully
    "C41347997": 2,    # mic: no leads, easy to cook or lose
    "C53446635": 2,    # the same mic, LCSC store version
    "C910544": 1,      # MAX98357A amp
    "C841192": 1,      # LDO
    "C15127": 2,       # P-FET
    "C431540": 1,      # slide switch (melts under hot air)
    "C165948": 1,      # USB-C
    "C84931": 2,       # big buttons
}


def spare(code, comment, per_order):
    if code in SPARES:
        return SPARES[code]
    if "LTST" in comment:
        return 3                                   # LEDs: 3 extra of each colour
    return max(5, math.ceil(per_order * 0.5))      # resistors / capacitors


def lcsc_stock(code):
    """Units in LCSC's store, or None if LCSC doesn't sell it / can't be reached."""
    url = f"https://wmsc.lcsc.com/ftps/wm/product/detail?productCode={code}"
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30))
        res = r.get("result") or {}
        return res.get("stockNumber") if res else None
    except Exception:
        return None


def main():
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    boards = int(nums[0]) if nums else 3
    rows = []
    for r in csv.DictReader(open(os.path.join(FAB, "bom_jlc.csv"))):
        n = len([d for d in r["Designator"].split(",") if d.strip()])
        code, comment = r["LCSC Part #"], f'{r["Comment"]} ({r["Designator"]})'
        if code in STORE_EQUIVALENT:
            code, model = STORE_EQUIVALENT[code]
            comment += f" -> store part {model}"
        rows.append((code, comment, n))
    rows += EXTRA
    out = os.path.join(FAB, f"lcsc_order_{boards}_boards.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Quantity", "LCSC Part Number", "Comment"])
        short = []
        for code, comment, n in rows:
            need = n * boards
            qty = need + spare(code, comment, need)
            w.writerow([qty, code, comment])
            if "--no-check" not in sys.argv:
                st = lcsc_stock(code)
                if st is None or st < qty:
                    short.append(f"{code} {comment[:40]}: need {qty}, LCSC has {st}")
                time.sleep(0.3)
    print(f"wrote {os.path.normpath(out)}: {len(rows)} lines for {boards} boards")
    for s_ in short:
        print("  ! short at LCSC:", s_)
    if not short and "--no-check" not in sys.argv:
        print("  every line is in stock at LCSC")


if __name__ == "__main__":
    main()
