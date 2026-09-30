"""Static server for the paper-summary app, plus a comment endpoint.

``python -m http.server`` answers GET only, so the app could never save a note.
This adds one small API on top of the standard static handler:

    GET  /api/comments?paper=<file>   -> that paper's comments
    POST /api/comments                -> add, edit the status of, or delete one

Comments are written to ``papers/comments/<paper-stem>.json`` as plain JSON, so
they are readable without the app, diff cleanly, and can be committed. They are
the reader's own notes on a paper, which is exactly the kind of thing that
should survive a laptop.

Bound to the loopback interface only. Writes are confined to papers/comments/
and the paper name is checked against papers/topics.json before anything is
written, so a crafted request cannot place a file elsewhere.
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

MAX_BODY = 64 * 1024          # a note, not a file upload
MAX_TEXT = 8_000
MAX_QUOTE = 2_000
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+\.md$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def comments_dir() -> Path:
    d = repo_root() / "papers" / "comments"
    d.mkdir(parents=True, exist_ok=True)
    return d


def known_papers() -> dict[str, str]:
    """Map summary filename -> topic id, from topics.json. The allow-list."""
    try:
        data = json.loads((repo_root() / "papers" / "topics.json").read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out = {}
    for topic in data.get("topics", []):
        for paper in topic.get("papers", []):
            if paper.get("file"):
                out[paper["file"]] = topic.get("id", "")
    return out


def store_path(paper: str) -> Path:
    return comments_dir() / (paper[:-3] + ".json")


def load_store(paper: str) -> dict:
    p = store_path(paper)
    if p.exists():
        try:
            d = json.loads(p.read_text("utf-8"))
            d.setdefault("comments", [])
            return d
        except json.JSONDecodeError:
            # Never lose a note to a parse error: keep the bad file aside.
            p.rename(p.with_suffix(".json.corrupt"))
    return {"paper": paper, "topic": known_papers().get(paper, ""), "comments": []}


def save_store(paper: str, store: dict) -> None:
    p = store_path(paper)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(store, indent=2, ensure_ascii=False) + "\n", "utf-8")
    tmp.replace(p)                      # atomic: a crash cannot truncate the file


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------------ util

    def _send_json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _fail(self, code: int, why: str) -> None:
        self._send_json({"error": why}, code)

    def _paper_param(self, name) -> str | None:
        if not isinstance(name, str) or not SAFE_NAME.match(name):
            return None
        return name if name in known_papers() else None

    # ------------------------------------------------------------------- GET

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/comments":
            q = parse_qs(parsed.query)
            paper = self._paper_param((q.get("paper") or [None])[0])
            if not paper:
                return self._fail(400, "unknown or malformed paper")
            return self._send_json(load_store(paper))
        if parsed.path == "/api/health":
            return self._send_json({"ok": True, "papers": len(known_papers())})
        return super().do_GET()

    # ------------------------------------------------------------------ POST

    def do_POST(self):  # noqa: N802
        if urlparse(self.path).path != "/api/comments":
            return self._fail(404, "no such endpoint")

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._fail(400, "bad Content-Length")
        if length <= 0 or length > MAX_BODY:
            return self._fail(413, "body missing or too large")

        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._fail(400, "body is not JSON")
        if not isinstance(payload, dict):
            return self._fail(400, "body is not an object")

        paper = self._paper_param(payload.get("paper"))
        if not paper:
            return self._fail(400, "unknown or malformed paper")

        action = payload.get("action", "add")
        store = load_store(paper)

        if action == "add":
            text = (payload.get("text") or "").strip()
            if not text:
                return self._fail(400, "a comment needs text")
            comment = {
                "id": uuid.uuid4().hex[:10],
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "section": str(payload.get("section") or "")[:200],
                "quote": str(payload.get("quote") or "")[:MAX_QUOTE],
                "text": text[:MAX_TEXT],
                "status": "open",
            }
            store["comments"].append(comment)
            save_store(paper, store)
            return self._send_json(comment, 201)

        if action in ("status", "delete"):
            cid = payload.get("id")
            hit = next((c for c in store["comments"] if c.get("id") == cid), None)
            if hit is None:
                return self._fail(404, "no such comment")
            if action == "delete":
                store["comments"].remove(hit)
            else:
                want = payload.get("status")
                if want not in ("open", "resolved"):
                    return self._fail(400, "status must be open or resolved")
                hit["status"] = want
            save_store(paper, store)
            return self._send_json(load_store(paper))

        return self._fail(400, f"unknown action {action!r}")

    # Quieter log: one line per request, no HTML error pages for the API.
    def log_message(self, fmt, *args):
        sys.stderr.write("[http] %s\n" % (fmt % args))


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
    root = str(repo_root())
    handler = partial(Handler, directory=root)
    with ThreadingHTTPServer((host, port), handler) as srv:
        sys.stderr.write(f"[http] serving {root} on {host}:{port}\n")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
