"""The ~/TessaTales folder: originals copied off the card, small copies, cached features, imports."""

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from .photos import is_photo, natural_key, open_small, pixel_features, sha1_file

THUMB_SIDE = 384


class Library:
    def __init__(self, home: Path):
        self.home = Path(home)
        self.originals = self.home / "originals"
        self.thumbs = self.home / "cache" / "thumb"
        self.feats = self.home / "cache" / "feat"
        self.imports = self.home / "imports"
        self.sheets = self.home / "sheets"
        for d in (self.originals, self.thumbs, self.feats, self.imports, self.sheets):
            d.mkdir(parents=True, exist_ok=True)
        self.seen_path = self.home / "seen.json"

    # ---- bookkeeping -------------------------------------------------
    def _load_seen(self) -> dict:
        if self.seen_path.exists():
            return json.loads(self.seen_path.read_text())
        return {}

    def manifest_path(self, import_id: str) -> Path:
        return self.imports / f"{import_id}.json"

    def load_manifest(self, import_id: str) -> dict:
        return json.loads(self.manifest_path(import_id).read_text())

    def save_manifest(self, man: dict) -> None:
        self.manifest_path(man["id"]).write_text(json.dumps(man, indent=1))

    def latest_import(self) -> str | None:
        ids = sorted((p.stem for p in self.imports.glob("*.json")), key=natural_key)
        return ids[-1] if ids else None

    def _new_import_id(self) -> str:
        base = datetime.now().strftime("%Y-%m-%d")
        n, import_id = 1, base
        while self.manifest_path(import_id).exists() or (self.originals / import_id).exists():
            n += 1
            import_id = f"{base}-{n}"
        return import_id

    # ---- copying off the card ----------------------------------------
    def import_card(self, source: Path, log=print) -> str | None:
        """Copy every new photo from source into originals/<import-id>/. Never touches the card."""
        files = sorted((p for p in source.rglob("*") if is_photo(p)), key=lambda p: natural_key(str(p.relative_to(source))))
        seen = self._load_seen()
        import_id = self._new_import_id()
        dest = self.originals / import_id
        photos, skipped = [], 0
        for i, p in enumerate(files, 1):
            digest = sha1_file(p)
            if digest in seen:
                skipped += 1
                continue
            name = "_".join(p.relative_to(source).parts)
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest / name)
            rel = f"{import_id}/{name}"
            seen[digest] = rel
            photos.append({"sha1": digest, "file": rel, "name": p.name, "mtime": p.stat().st_mtime})
            if i % 50 == 0:
                log(f"  copied {i}/{len(files)}")
        if not photos:
            log(f"Nothing new on {source} ({skipped} pictures were already imported).")
            return None
        self.seen_path.write_text(json.dumps(seen, indent=0))
        self.save_manifest({
            "id": import_id,
            "source": str(source),
            "imported_at": datetime.now().isoformat(timespec="seconds"),
            "already_imported": skipped,
            "photos": photos,
        })
        log(f"Copied {len(photos)} new pictures into {dest} ({skipped} already imported).")
        return import_id

    # ---- per-photo cache ---------------------------------------------
    def thumb_path(self, sha1: str) -> Path:
        return self.thumbs / f"{sha1}.jpg"

    def features(self, photo: dict) -> dict:
        """Pixel features + EXIF time, cached by content hash. Also writes the small copy."""
        cache = self.feats / f"{photo['sha1']}.json"
        if cache.exists() and self.thumb_path(photo["sha1"]).exists():
            return json.loads(cache.read_text())
        img, taken = open_small(self.originals / photo["file"], THUMB_SIDE)
        img.save(self.thumb_path(photo["sha1"]), "JPEG", quality=82)
        feat = pixel_features(img)
        feat["exif_time"] = taken.isoformat() if taken else None
        cache.write_text(json.dumps(feat))
        return feat

    def embedding_path(self, sha1: str, model_tag: str) -> Path:
        return self.feats / f"{sha1}.{model_tag}.npy"

    def load_embedding(self, sha1: str, model_tag: str):
        p = self.embedding_path(sha1, model_tag)
        return np.load(p) if p.exists() else None

    def save_embedding(self, sha1: str, model_tag: str, vec) -> None:
        np.save(self.embedding_path(sha1, model_tag), vec.astype(np.float32))
