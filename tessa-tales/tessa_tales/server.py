"""A tiny local web server: the book-making page, Tessa's turn, her pictures and voice, saving.

Listens on 127.0.0.1 only. No accounts, nothing leaves the Mac.
"""

import json
import re
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import book as books
from .library import Library
from .photos import open_small
from .voice import Voice

WEB = Path(__file__).resolve().parent.parent / "web"
LARGE_SIDE = 1600
SHA = re.compile(r"^[0-9a-f]{40}$")
IMPORT_ID = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}(-[0-9]+)?$")


def make_handler(lib: Library, voice: Voice):
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

        def _file(self, path: Path, ctype: str, cache: bool = False):
            if not path.is_file():
                return self.send_error(HTTPStatus.NOT_FOUND)
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(data)

        # -- routes ----------------------------------------------------------------
        def do_GET(self):
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            if not parts:
                latest = lib.latest_import()
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Location", f"/book/{latest}" if latest else "/web/empty.html")
                return self.end_headers()
            if parts[0] in ("book", "tell") and len(parts) == 2 and IMPORT_ID.match(parts[1]):
                return self._file(WEB / f"{parts[0]}.html", "text/html; charset=utf-8")
            if parts[0] == "audio" and len(parts) == 3 and IMPORT_ID.match(parts[1]):
                path = voice.audio_path(parts[1], parts[2])
                if not path:
                    return self.send_error(HTTPStatus.NOT_FOUND)
                ctype = {"webm": "audio/webm", "m4a": "audio/mp4", "ogg": "audio/ogg", "wav": "audio/wav", "mp3": "audio/mpeg"}
                return self._file(path, ctype[parts[2].rsplit(".", 1)[1]], cache=True)
            if parts[0] == "web" and len(parts) == 2 and re.match(r"^[a-z0-9_-]+\.(html|js|css)$", parts[1]):
                ctype = {"html": "text/html", "js": "text/javascript", "css": "text/css"}[parts[1].rsplit(".", 1)[1]]
                return self._file(WEB / parts[1], ctype + "; charset=utf-8")
            if parts[0] == "api" and len(parts) == 3 and IMPORT_ID.match(parts[2]):
                if not lib.manifest_path(parts[2]).exists():
                    return self.send_error(HTTPStatus.NOT_FOUND)
                if parts[1] == "import":
                    man = lib.load_manifest(parts[2])
                    return self._json({"id": man["id"], "total": len(man["photos"]), "sieve": man["sieve"]})
                if parts[1] == "book":
                    return self._json(books.load_or_draft(lib, parts[2]))
                if parts[1] == "voice":
                    return self._json(voice.load(parts[2]))
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
            return self.send_error(HTTPStatus.NOT_FOUND)

        def _body(self, limit):
            length = int(self.headers.get("Content-Length", "0"))
            if length > limit:
                raise ValueError("too big")
            return self.rfile.read(length)

        def do_POST(self):
            parts = [p for p in self.path.split("/") if p]
            if len(parts) == 4 and parts[:2] == ["api", "voice"] and IMPORT_ID.match(parts[2]):
                try:
                    entry = voice.save_clip(parts[2], parts[3], self._body(50_000_000), self.headers.get("Content-Type", ""))
                except ValueError as e:
                    return self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
                return self._json(entry)
            return self.send_error(HTTPStatus.NOT_FOUND)

        def do_PUT(self):
            parts = [p for p in self.path.split("/") if p]
            if len(parts) == 5 and parts[:2] == ["api", "voice"] and parts[4] == "words" and IMPORT_ID.match(parts[2]):
                try:
                    entry = voice.set_words(parts[2], parts[3], str(json.loads(self._body(10_000)).get("words", "")))
                except (ValueError, json.JSONDecodeError) as e:
                    return self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
                return self._json(entry)
            if len(parts) != 3 or parts[:2] != ["api", "book"] or not IMPORT_ID.match(parts[2]):
                return self.send_error(HTTPStatus.NOT_FOUND)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(min(length, 2_000_000)))
                if data.get("id") != parts[2] or not isinstance(data.get("pages"), list):
                    raise ValueError("book doesn't match")
                if not all(isinstance(h, str) and SHA.match(h) for h in data.get("nope", [])):
                    raise ValueError("bad picture id")
                for page in data["pages"]:
                    if not all(SHA.match(h) for h in [page.get("photo", "")] + list(page.get("frames", []))):
                        raise ValueError("bad picture id")
                books.save(lib, data)
            except (ValueError, json.JSONDecodeError) as e:
                return self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
            return self._json({"saved": True, "updated": data["updated"]})

    return Handler


def serve(lib: Library, port: int, open_path: str | None = None, voice_cfg: dict | None = None) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(lib, Voice(lib, voice_cfg or {})))
    url = f"http://localhost:{port}{open_path or '/'}"
    print(f"Tessa Tales is open at {url}  (Ctrl-C to stop)")
    if open_path is not None:
        import subprocess
        subprocess.run(["open", url], check=False)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBye! ❄")
