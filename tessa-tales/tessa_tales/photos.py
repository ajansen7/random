"""Reading pictures: when they were taken, small copies, and cheap pixel math (no model)."""

import hashlib
import re
from datetime import datetime
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
EXIF_IFD = 0x8769
DATETIME_ORIGINAL = 36867
DATETIME = 306


def is_photo(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")


def natural_key(name: str):
    """IMG_9 before IMG_10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def exif_time(img: Image.Image) -> datetime | None:
    try:
        ex = img.getexif()
        raw = ex.get_ifd(EXIF_IFD).get(DATETIME_ORIGINAL) or ex.get(DATETIME)
    except Exception:
        return None
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw).strip("\x00 ").strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def open_small(path: Path, max_side: int) -> tuple[Image.Image, datetime | None]:
    """Open a photo, upright and shrunk to max_side, plus its EXIF time if it has one."""
    img = Image.open(path)
    taken = exif_time(img)
    try:
        img.draft("RGB", (max_side, max_side))  # fast JPEG downscale while decoding
    except Exception:
        pass
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img, taken


def pixel_features(img: Image.Image) -> dict:
    """Brightness, spread and sharpness from a small grey copy. Sharpness only picks stack covers."""
    w, h = img.size
    scale = 256 / max(w, h)
    g = np.asarray(img.convert("L").resize((max(8, int(w * scale)), max(8, int(h * scale)))), dtype=np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return {
        "brightness": round(float(g.mean()), 1),
        "spread": round(float(g.std()), 1),
        "sharpness": round(float(lap.var()), 1),
        "phash": str(imagehash.phash(img)),
    }


def phash_distance(a: str, b: str) -> int:
    return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)
