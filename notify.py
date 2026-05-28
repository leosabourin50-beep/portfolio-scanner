"""Shared Telegram delivery + alert logging.

Single source of truth for sending to Telegram, replacing the near-duplicate
`_send_telegram` implementations that used to live in both `scanner_cron.py`
and `intraday_watcher.py`. Adds photo support (so alerts can carry the daily
chart) and a tiny JSONL append helper used to log delivered alerts for the
weekly signal scorecard.

Stdlib only — no third-party HTTP dependency — so it imports cleanly in the
slim Fly image and in GitHub Actions.

The scanner-cron bot and the intraday-watcher bot run on SEPARATE tokens, so
every send accepts an explicit (token, chat_id) — or use a `Notifier` bound to
a process's own credentials. The module-level functions fall back to the
TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID env vars for the default (cron) bot.

Env vars (defaults; each process may bind its own via Notifier):
  TELEGRAM_BOT_TOKEN   default bot token (used by the daily cron)
  TELEGRAM_CHAT_ID     default destination chat id
  DRY_RUN              "1" prints instead of sending (returns success)
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

API_BASE = "https://api.telegram.org"

# Telegram hard limits.
_MSG_LIMIT = 4096
_CAPTION_LIMIT = 1024


def _creds() -> tuple[str | None, str | None]:
    return os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def send_message(text: str, chat_id: str | None = None,
                 parse_mode: str = "HTML", token: str | None = None) -> bool:
    """Send a text message. Returns True iff delivered (or simulated under
    DRY_RUN). A False return means the caller must NOT mark the alert as
    deduped — otherwise a transient failure silently swallows the ping.

    `token`/`chat_id` override the default env credentials, so a process on a
    dedicated bot (e.g. the intraday watcher) can send via its own token."""
    if os.environ.get("DRY_RUN") == "1":
        print("[DRY_RUN] would send message:\n" + text + "\n---")
        return True
    env_token, default_chat = _creds()
    token = token or env_token
    chat_id = chat_id or default_chat
    if not token or not chat_id:
        print("[skip] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set", file=sys.stderr)
        return False
    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": _truncate(text, _MSG_LIMIT),
        "parse_mode": parse_mode,
        "disable_web_page_preview": "true",
    }).encode()
    url = f"{API_BASE}/bot{token}/sendMessage"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=payload), timeout=15) as resp:
            body = resp.read()
        if b'"ok":true' in body:
            return True
        print(f"[error] Telegram sendMessage non-ok: {body[:200]!r}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[error] Telegram sendMessage failed: {e}", file=sys.stderr)
        return False


def _multipart(fields: dict[str, str], file_field: str,
               filename: str, file_bytes: bytes,
               content_type: str = "image/png") -> tuple[bytes, str]:
    """Build a multipart/form-data body with stdlib only. Returns
    (body, content_type_header)."""
    boundary = "----portfolioscanner7c1f9b2e4a"
    crlf = b"\r\n"
    out: list[bytes] = []
    for name, value in fields.items():
        out.append(f"--{boundary}".encode())
        out.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        out.append(b"")
        out.append(str(value).encode())
    out.append(f"--{boundary}".encode())
    out.append(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"'.encode()
    )
    out.append(f"Content-Type: {content_type}".encode())
    out.append(b"")
    body = crlf.join(out) + crlf + file_bytes + crlf + f"--{boundary}--".encode() + crlf
    return body, f"multipart/form-data; boundary={boundary}"


def send_photo(png_bytes: bytes, caption: str = "",
               chat_id: str | None = None, parse_mode: str = "HTML",
               token: str | None = None) -> bool:
    """Send a PNG with an HTML caption via sendPhoto. Returns True iff
    delivered. Captions over Telegram's 1024-char limit are truncated."""
    if os.environ.get("DRY_RUN") == "1":
        print(f"[DRY_RUN] would send photo ({len(png_bytes)} bytes) caption:\n{caption}\n---")
        return True
    env_token, default_chat = _creds()
    token = token or env_token
    chat_id = chat_id or default_chat
    if not token or not chat_id:
        print("[skip] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set", file=sys.stderr)
        return False
    fields = {
        "chat_id": str(chat_id),
        "caption": _truncate(caption, _CAPTION_LIMIT),
        "parse_mode": parse_mode,
    }
    body, content_type = _multipart(fields, "photo", "chart.png", png_bytes)
    url = f"{API_BASE}/bot{token}/sendPhoto"
    req = urllib.request.Request(url, data=body)
    req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp_body = resp.read()
        if b'"ok":true' in resp_body:
            return True
        print(f"[error] Telegram sendPhoto non-ok: {resp_body[:200]!r}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[error] Telegram sendPhoto failed: {e}", file=sys.stderr)
        return False


def send_alert(text: str, png_bytes: bytes | None = None,
               chat_id: str | None = None, token: str | None = None) -> bool:
    """Preferred entry point for alerts: if a chart PNG is available, send it
    as a photo with the alert text as caption; otherwise (or if the photo
    send fails) fall back to a plain text message. Charts are a best-effort
    enhancement and never block delivery of the underlying alert."""
    if png_bytes:
        if send_photo(png_bytes, caption=text, chat_id=chat_id, token=token):
            return True
        # Photo path failed (e.g. caption too long, transient) — make sure the
        # user still gets the signal as text.
        print("[warn] photo send failed — falling back to text", file=sys.stderr)
    return send_message(text, chat_id=chat_id, token=token)


class Notifier:
    """A Telegram sender bound to one bot's (token, chat_id). Lets the cron
    bot and the watcher bot run on separate tokens while sharing all the
    delivery/formatting logic. `from_env` resolves a primary env var with an
    optional fallback so single-token deployments keep working."""

    def __init__(self, token: str | None = None, chat_id: str | None = None):
        env_token, env_chat = _creds()
        self.token = token or env_token
        self.chat_id = chat_id or env_chat

    @classmethod
    def from_env(cls, token_var: str, chat_var: str,
                 fallback_token_var: str = "TELEGRAM_BOT_TOKEN",
                 fallback_chat_var: str = "TELEGRAM_CHAT_ID") -> "Notifier":
        token = os.environ.get(token_var) or os.environ.get(fallback_token_var)
        chat = os.environ.get(chat_var) or os.environ.get(fallback_chat_var)
        return cls(token=token, chat_id=chat)

    def message(self, text: str, parse_mode: str = "HTML") -> bool:
        return send_message(text, chat_id=self.chat_id, parse_mode=parse_mode, token=self.token)

    def photo(self, png_bytes: bytes, caption: str = "", parse_mode: str = "HTML") -> bool:
        return send_photo(png_bytes, caption=caption, chat_id=self.chat_id,
                          parse_mode=parse_mode, token=self.token)

    def alert(self, text: str, png_bytes: bytes | None = None) -> bool:
        return send_alert(text, png_bytes=png_bytes, chat_id=self.chat_id, token=self.token)


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append one JSON record as a line. Best-effort: logging must never take
    down an alert, so failures are swallowed with a stderr note."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"[warn] alert-log append failed: {e}", file=sys.stderr)
