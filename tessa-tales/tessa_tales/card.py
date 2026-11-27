"""The camera card: notice it, copy and sort new pictures in the background, eject it."""

import subprocess
import threading
from pathlib import Path

from . import sieve, stories
from .library import Library


def find_cards() -> list[Path]:
    vols = Path("/Volumes")
    if not vols.exists():
        return []
    return [v for v in vols.iterdir() if v.is_dir() and (v / "DCIM").is_dir()]


class Importer:
    """One import at a time, run in the background so the page can show Olaf working."""

    def __init__(self, lib: Library, cfg: dict):
        self.lib, self.cfg = lib, cfg
        self.lock = threading.Lock()
        self.state = {"phase": "idle"}
        self._seen: dict = {"card": None}
        self._watching = False

    def card_status(self) -> dict:
        """Never touches the card: returns the watcher's last look. A stuck SD reader can freeze
        any file call on its mount, so only the watcher thread ever waits on it."""
        if not self._watching:
            self._watching = True
            threading.Thread(target=self._watch, daemon=True).start()
        return dict(self._seen)

    def _watch(self):
        import time
        while True:
            try:
                cards = find_cards()
                if not cards:
                    self._seen = {"card": None}
                else:
                    card = cards[0]
                    if self._seen.get("card") != str(card) or self.state.get("phase") == "done":
                        self._seen = {"card": str(card), "name": card.name, "new": None}
                        new = self.lib.count_new(card)
                        self._seen = {"card": str(card), "name": card.name, "new": new}
                        if self.state.get("phase") == "done":
                            self.state = {**self.state, "phase": "settled"}
            except OSError:
                self._seen = {"card": None}
            time.sleep(3)

    def start(self, card: str) -> dict:
        with self.lock:
            if self.state.get("phase") in ("copying", "sorting", "stories"):
                return self.state
            if card != self._seen.get("card"):
                raise ValueError("that card isn't plugged in")
            self.state = {"phase": "copying", "done": 0, "total": 0}
        threading.Thread(target=self._run, args=(Path(card),), daemon=True).start()
        return self.state

    def _run(self, card: Path):
        def progress(i, total):
            self.state.update(done=i, total=total)
        try:
            import_id = self.lib.import_card(card, log=lambda *a: None, progress=progress)
            if not import_id:
                self.state = {"phase": "done", "new_pictures": 0, "new_stories": []}
                return
            self.state.update(phase="sorting", import_id=import_id)
            man = sieve.run(self.lib, self.cfg, import_id, log=lambda *a: None)
            self.state.update(phase="stories")
            made = stories.ensure_day_stories(self.lib, import_id)
            self.state = {"phase": "done", "import_id": import_id, "new_pictures": len(man["photos"]),
                          "buried": man["sieve"]["photo_counts"]["snowdrift"], "new_stories": made}
        except Exception as e:
            self.state = {"phase": "error", "error": str(e)}

    def eject(self, card: str) -> None:
        if card != self._seen.get("card"):
            raise ValueError("that card isn't plugged in")
        if self.state.get("phase") in ("copying",):
            raise ValueError("still copying")
        r = subprocess.run(["diskutil", "eject", card], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise ValueError(r.stderr.strip() or "couldn't eject")
