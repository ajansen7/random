"""A tiny local web server: the shelf, each story's page, Tessa's turn, her pictures and voice.

Listens on 127.0.0.1 only. No accounts. Nothing leaves the Mac.
"""

import json
import re
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import stories
from .card import Importer
from .library import Library
from .photos import open_small
from .voice import STORY_ID, Voice
from .weave import WeaveError, weave

WEB = Path(__file__).resolve().parent.parent / "web"
LARGE_SIDE = 1600
SHA = re.compile(r"^[0-9a-f]{40}$")
AUDIO_TYPES = {"webm": "audio/webm", "m4a": "audio/mp4", "ogg": "audio/ogg", "wav": "audio/wav", "mp3": "audio/mpeg"}


def shelf(lib: Library, voice: Voice) -> list[dict]:
    talk = voice.load("talk")
    out = []
    for s in stories.list_all(lib):
        pages = s.get("pages", [])
        told = sum(1 for p in pages if any(h in talk for h in [p["photo"], *p.get("frames", [])]))
        cover = pages[0]["photo"] if pages else (s["pool"][0] if s["pool"] else None)
        out.append({"id": s["id"], "title": s["title"], "cover": cover, "pages": len(pages), "told": told,
                    "new": s.get("new", False), "woven": bool(s.get("woven_at")), "import": s["import"]})
    return out


def make_handler(lib: Library, voice: Voice, importer: Importer):
    photo_by_sha = {}

    def original_for(sha: str) -> Path | None:
        if sha not in photo_by_sha:
            seen = json.loads(lib.seen_path.read_text()) if lib.seen_path.exists() else {}
            photo_by_sha.update(seen)
        rel = photo_by_sha.get(sha)
        return lib.originals / rel if rel else None

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the terminal quiet
            pass

        # -- helpers ---------------------------------------------------------------
        def _json(self, data, status=HTTPStatus.OK):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _oops(self, msg, status=HTTPStatus.BAD_REQUEST):
            return self._json({"error": str(msg)}, status)

        def _file(self, path: Path | None, ctype: str, cache: bool = False):
            if not path or not path.is_file():
                return self.send_error(HTTPStatus.NOT_FOUND)
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _redirect(self, where):
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", where)
            self.end_headers()

        def _body(self, limit):
            length = int(self.headers.get("Content-Length", "0"))
            if length > limit:
                raise ValueError("too big")
            return self.rfile.read(length)

        def _json_body(self, limit=2_000_000) -> dict:
            data = json.loads(self._body(limit) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("expected an object")
            return data

        def _story_ok(self, sid):
            return STORY_ID.match(sid) and stories.exists(lib, sid)

        # -- GET -------------------------------------------------------------------
        def do_GET(self):
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            html = "text/html; charset=utf-8"
            if not parts:
                return self._file(WEB / "shelf.html", html)
            if parts[0] == "book":
                return self._redirect("/")  # old links from the one-book-per-card days
            if parts[0] == "story" and len(parts) == 2 and self._story_ok(parts[1]):
                return self._file(WEB / "story.html", html)
            if parts[0] == "tell" and len(parts) == 2 and self._story_ok(parts[1]):
                return self._redirect(f"/story/{parts[1]}")  # Tessa's turn and the story page are one screen now
            if parts[0] == "web" and len(parts) == 2 and re.match(r"^[a-z0-9_-]+\.(html|js|css|svg)$", parts[1]):
                ctype = {"html": "text/html", "js": "text/javascript", "css": "text/css", "svg": "image/svg+xml"}[parts[1].rsplit(".", 1)[1]]
                return self._file(WEB / parts[1], ctype + "; charset=utf-8")
            if parts[0] == "audio":
                if len(parts) == 3 and parts[1] == "talk":
                    path = voice.audio_path("talk", None, parts[2])
                elif len(parts) == 4 and parts[1] == "story" and self._story_ok(parts[2]):
                    path = voice.audio_path("story", parts[2], parts[3])
                else:
                    path = None
                if not path:
                    return self.send_error(HTTPStatus.NOT_FOUND)
                return self._file(path, AUDIO_TYPES[path.suffix[1:]], cache=True)
            if parts[0] in ("thumb", "large") and len(parts) == 2 and parts[1].endswith(".jpg"):
                sha = parts[1][:-4]
                if not SHA.match(sha):
                    return self.send_error(HTTPStatus.NOT_FOUND)
                if parts[0] == "thumb":
                    return self._file(lib.thumb_path(sha), "image/jpeg", cache=True)
                large = lib.home / "cache" / "large" / f"{sha}.jpg"
                if not large.exists():
                    src = original_for(sha)
                    if not src or not src.exists():
                        return self.send_error(HTTPStatus.NOT_FOUND)
                    large.parent.mkdir(parents=True, exist_ok=True)
                    open_small(src, LARGE_SIDE)[0].save(large, "JPEG", quality=86)
                return self._file(large, "image/jpeg", cache=True)
            if parts[0] == "api":
                api = parts[1:]
                if api == ["shelf"]:
                    return self._json(shelf(lib, voice))
                if api == ["talk"]:
                    return self._json(voice.load("talk"))
                if api == ["card"]:
                    return self._json({**importer.card_status(), "import": importer.state})
                if len(api) == 2 and api[0] == "story" and self._story_ok(api[1]):
                    return self._json(stories.view(lib, stories.load(lib, api[1])))
                if len(api) == 2 and api[0] == "voice" and self._story_ok(api[1]):
                    return self._json(voice.load("story", api[1]))
            return self.send_error(HTTPStatus.NOT_FOUND)

        # -- POST ------------------------------------------------------------------
        def do_POST(self):
            api = [p for p in self.path.split("/") if p][1:]
            try:
                ctype = self.headers.get("Content-Type", "")
                if len(api) == 2 and api[0] == "talk":
                    return self._json(voice.save_clip("talk", None, api[1], self._body(50_000_000), ctype))
                if len(api) == 3 and api[0] == "voice" and self._story_ok(api[1]):
                    return self._json(voice.save_clip("story", api[1], api[2], self._body(50_000_000), ctype))
                if len(api) == 3 and api[0] == "story" and api[2] == "weave" and self._story_ok(api[1]):
                    hint = str(self._json_body(10_000).get("hint", ""))[:300]
                    st = stories.load(lib, api[1])
                    st["hint"] = hint
                    stories.save(lib, st)
                    try:
                        weave(lib, api[1], voice.load("talk"), voice.load("story", api[1]), importer.cfg.get("weave", {}))
                    except WeaveError as e:
                        return self._oops(e, HTTPStatus.UNPROCESSABLE_ENTITY)
                    return self._json(stories.view(lib, stories.load(lib, api[1])))
                if api == ["import"]:
                    return self._json(importer.start(str(self._json_body(10_000).get("card", ""))))
                if api == ["eject"]:
                    importer.eject(str(self._json_body(10_000).get("card", "")))
                    return self._json({"ejected": True})
            except (ValueError, json.JSONDecodeError) as e:
                return self._oops(e)
            return self.send_error(HTTPStatus.NOT_FOUND)

        # -- PUT -------------------------------------------------------------------
        def do_PUT(self):
            api = [p for p in self.path.split("/") if p][1:]
            try:
                if len(api) == 3 and api[0] == "talk" and api[2] == "words":
                    return self._json(voice.set_words("talk", None, api[1], str(self._json_body(10_000).get("words", ""))))
                if len(api) == 4 and api[0] == "voice" and api[3] == "words" and self._story_ok(api[1]):
                    return self._json(voice.set_words("story", api[1], api[2], str(self._json_body(10_000).get("words", ""))))
                if len(api) == 2 and api[0] == "story" and self._story_ok(api[1]):
                    data = self._json_body()
                    if not isinstance(data.get("pages"), list):
                        raise ValueError("no pages")
                    if not all(isinstance(h, str) and SHA.match(h) for h in data.get("nope", [])):
                        raise ValueError("bad picture id")
                    for page in data["pages"]:
                        if not all(isinstance(h, str) and SHA.match(h) for h in [page.get("photo", "")] + list(page.get("frames", []))):
                            raise ValueError("bad picture id")
                        if page.get("bridge") is not None and not isinstance(page["bridge"], str):
                            raise ValueError("bad bridge")
                    story = stories.save_from_page(lib, api[1], data)
                    return self._json({"saved": True, "updated": story["updated"]})
            except (ValueError, json.JSONDecodeError) as e:
                return self._oops(e)
            return self.send_error(HTTPStatus.NOT_FOUND)

    return Handler


def serve(lib: Library, cfg: dict, port: int, open_path: str | None = None) -> None:
    stories.migrate_old_books(lib)
    for p in sorted(lib.imports.glob("*.json")):
        if "sieve" in json.loads(p.read_text()):
            stories.ensure_day_stories(lib, p.stem)
    handler = make_handler(lib, Voice(lib, cfg.get("voice", {})), Importer(lib, cfg))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://localhost:{port}{open_path or '/'}"
    print(f"Tessa Tales is open at {url}  (Ctrl-C to stop)")
    if open_path is not None:
        import subprocess
        subprocess.run(["open", url], check=False)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBye! ❄")
