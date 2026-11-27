"""Tessa's voice: save each recording, then let Whisper (on the Mac's GPU) write down her words.

Kept in stories/<book>/voice.json, separate from story.json, so the page saving the book
can never overwrite words that Whisper just wrote.
  { "<page id>|cover|end": {"audio": "audio/<file>", "recorded_at": ..., "words": str|None,
                            "status": "listening"|"done"|"quiet", "edited": bool} }
"""

import json
import queue
import re
import threading
import warnings
from datetime import datetime

from .library import Library

KEY = re.compile(r"^(cover|end|p-[A-Za-z0-9-]{1,60})$")
EXTS = {"audio/webm": "webm", "audio/mp4": "m4a", "audio/ogg": "ogg", "audio/wav": "wav", "audio/x-wav": "wav", "audio/mpeg": "mp3"}
# Whisper sometimes "hears" these in silence
HALLUCINATIONS = {"", "you", "thank you", "thanks for watching", "bye", "the end", "."}


class Voice:
    def __init__(self, lib: Library, cfg: dict):
        self.lib = lib
        self.model = cfg.get("model", "mlx-community/whisper-large-v3-turbo")
        self.prompt = cfg.get("prompt", "")
        self.lock = threading.Lock()
        self.jobs: queue.Queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    # ---- the voice.json file ------------------------------------------------
    def _path(self, book_id: str):
        return self.lib.home / "stories" / book_id / "voice.json"

    def load(self, book_id: str) -> dict:
        p = self._path(book_id)
        return json.loads(p.read_text()) if p.exists() else {}

    def _update(self, book_id: str, key: str, fn) -> dict:
        with self.lock:
            data = self.load(book_id)
            entry = fn(data.get(key))
            if entry is None:
                data.pop(key, None)
            else:
                data[key] = entry
            p = self._path(book_id)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1))
            tmp.replace(p)
            return entry

    # ---- recordings -----------------------------------------------------------
    def save_clip(self, book_id: str, key: str, body: bytes, content_type: str) -> dict:
        if not KEY.match(key):
            raise ValueError("bad page")
        ext = EXTS.get(content_type.split(";")[0].strip().lower())
        if not ext:
            raise ValueError("unknown audio type")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        rel = f"audio/{key}-{stamp}.{ext}"
        path = self.lib.home / "stories" / book_id / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)  # older takes stay on disk; voice.json points at the newest
        entry = self._update(book_id, key, lambda old: {
            "audio": rel, "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "words": None, "status": "listening", "edited": False,
        })
        self.jobs.put((book_id, key, rel))
        return entry

    def set_words(self, book_id: str, key: str, words: str) -> dict:
        if not KEY.match(key):
            raise ValueError("bad page")
        def fn(old):
            if not old:
                raise ValueError("nothing recorded yet")
            return {**old, "words": words.strip()[:400] or None, "edited": True, "status": "done"}
        return self._update(book_id, key, fn)

    def audio_path(self, book_id: str, name: str):
        if not re.match(r"^[A-Za-z0-9-]+\.(webm|m4a|ogg|wav|mp3)$", name):
            return None
        return self.lib.home / "stories" / book_id / "audio" / name

    # ---- Whisper, one clip at a time ---------------------------------------
    def _worker(self):
        whisper = None
        while True:
            book_id, key, rel = self.jobs.get()
            words = None
            try:
                if whisper is None:
                    warnings.filterwarnings("ignore")
                    import mlx_whisper as whisper
                result = whisper.transcribe(
                    str(self.lib.home / "stories" / book_id / rel), path_or_hf_repo=self.model,
                    language="en", initial_prompt=self.prompt or None, condition_on_previous_text=False,
                )
                text = " ".join(result.get("text", "").split())
                if text.lower().strip(" .!?,") not in HALLUCINATIONS:
                    words = text
            except Exception as e:  # never let one bad clip stop the listener
                print(f"  (couldn't hear {key}: {e})")

            def fn(old):
                if not old or old.get("audio") != rel or old.get("edited"):
                    return old  # re-recorded or hand-edited meanwhile: leave it
                return {**old, "words": words, "status": "done" if words else "quiet"}
            self._update(book_id, key, fn)
