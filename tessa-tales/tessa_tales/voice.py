"""Tessa's voice: save each recording, then let Whisper (on the Mac's GPU) write down her words.

Two boxes of recordings:
  talk   her talk about a picture, keyed by the picture (sha1). Global, so "look like a dino
         neck!" follows that picture into any story.     ~/TessaTales/talk.json + talk/
  story  a story's own clips: "cover" (she says the title) and "end".
                                                 ~/TessaTales/stories/<id>/voice.json + audio/
Entries: {"audio": "<file>", "recorded_at", "words": str|None,
          "status": "listening"|"done"|"quiet", "edited": bool}
"""

import json
import queue
import re
import threading
import warnings
from datetime import datetime
from pathlib import Path

from .library import Library

SHA = re.compile(r"^[0-9a-f]{40}$")
STORY_KEY = re.compile(r"^(cover|end)$")
STORY_ID = re.compile(r"^[a-z0-9-]{1,64}$")
EXTS = {"audio/webm": "webm", "audio/mp4": "m4a", "audio/ogg": "ogg", "audio/wav": "wav", "audio/x-wav": "wav", "audio/mpeg": "mp3"}
AUDIO_NAME = re.compile(r"^[A-Za-z0-9-]+\.(webm|m4a|ogg|wav|mp3)$")
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

    # ---- where things live --------------------------------------------------
    def _box(self, box: str, owner: str | None) -> tuple[Path, Path]:
        """(json file, audio folder) for a box."""
        if box == "talk":
            return self.lib.home / "talk.json", self.lib.home / "talk"
        if box == "story" and owner and STORY_ID.match(owner):
            d = self.lib.home / "stories" / owner
            return d / "voice.json", d / "audio"
        raise ValueError("bad box")

    def _check_key(self, box: str, key: str):
        if not (SHA if box == "talk" else STORY_KEY).match(key):
            raise ValueError("bad key")

    def load(self, box: str, owner: str | None = None) -> dict:
        path, _ = self._box(box, owner)
        return json.loads(path.read_text()) if path.exists() else {}

    def _update(self, box: str, owner: str | None, key: str, fn) -> dict | None:
        path, _ = self._box(box, owner)
        with self.lock:
            data = json.loads(path.read_text()) if path.exists() else {}
            entry = fn(data.get(key))
            if entry is None:
                data.pop(key, None)
            else:
                data[key] = entry
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1))
            tmp.replace(path)
            return entry

    # ---- recordings -----------------------------------------------------------
    def save_clip(self, box: str, owner: str | None, key: str, body: bytes, content_type: str) -> dict:
        self._check_key(box, key)
        ext = EXTS.get(content_type.split(";")[0].strip().lower())
        if not ext:
            raise ValueError("unknown audio type")
        _, folder = self._box(box, owner)
        name = f"{key}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.{ext}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(body)  # older takes stay on disk; the json points at the newest
        entry = self._update(box, owner, key, lambda old: {
            "audio": name, "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "words": None, "status": "listening", "edited": False,
        })
        self.jobs.put((box, owner, key, name))
        return entry

    def set_words(self, box: str, owner: str | None, key: str, words: str) -> dict:
        self._check_key(box, key)

        def fn(old):
            if not old:
                raise ValueError("nothing recorded yet")
            return {**old, "words": words.strip()[:400] or None, "edited": True, "status": "done"}
        return self._update(box, owner, key, fn)

    def audio_path(self, box: str, owner: str | None, name: str) -> Path | None:
        if not AUDIO_NAME.match(name):
            return None
        _, folder = self._box(box, owner)
        return folder / name

    # ---- Whisper, one clip at a time ---------------------------------------
    def _worker(self):
        whisper = None
        while True:
            box, owner, key, name = self.jobs.get()
            words = None
            try:
                if whisper is None:
                    warnings.filterwarnings("ignore")
                    import mlx_whisper as whisper
                _, folder = self._box(box, owner)
                result = whisper.transcribe(
                    str(folder / name), path_or_hf_repo=self.model,
                    language="en", initial_prompt=self.prompt or None, condition_on_previous_text=False,
                )
                text = " ".join(result.get("text", "").split())
                if text.lower().strip(" .!?,") not in HALLUCINATIONS:
                    words = text
            except Exception as e:  # never let one bad clip stop the listener
                print(f"  (couldn't hear {key}: {e})")

            def fn(old):
                if not old or old.get("audio") != name or old.get("edited"):
                    return old  # re-recorded or hand-edited meanwhile: leave it
                return {**old, "words": words, "status": "done" if words else "quiet"}
            self._update(box, owner, key, fn)
