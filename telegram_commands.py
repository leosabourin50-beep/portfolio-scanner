"""Interactive Telegram command handler (the two-way bot).

Long-polls getUpdates on the watcher's bot token and dispatches commands. It
runs as a daemon thread inside `intraday_watcher.py` because that is the only
always-on process — GitHub Actions cron is ephemeral and can't hold a
long-poll, and Telegram allows only ONE getUpdates consumer per token.

Authorized senders only: the alert chat (TELEGRAM_WATCHER_CHAT_ID /
TELEGRAM_CHAT_ID) plus any ids in TELEGRAM_OWNER_ID (comma/space list). Set
TELEGRAM_OWNER_ID to your Telegram user id to DM commands while alerts post to
a separate channel. Everyone else is ignored.

Commands:
  /help                       this list
  /status                     watcher liveness + portfolio summary
  /list                       show the portfolio (with positions / mutes)
  /positions                  owned positions with live P/L
  /scan TICKER                on-demand read + chart for one ticker
  /add TICKER [N @ PRICE]     add ticker (optionally as an owned position)
  /remove TICKER              remove ticker
  /mute TICKER                silence alerts for a ticker
  /unmute TICKER              clear a mute/snooze
  /snooze TICKER [30m|2h|1d]  temporarily silence (default 1h)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import portfolio as pf
import portfolio_scanner as ps
import single_name_analyzer as sna
import chart_render
from polygon_adapter import get_snapshot_all

API_BASE = "https://api.telegram.org"
OFFSET_PATH = pf.DATA_DIR / "command_offset.txt"
# Resolve these the same way the watcher does (env override wins) so /status
# reads the same files the watcher writes.
HEARTBEAT_PATH = Path(os.environ.get("HEARTBEAT_PATH") or pf.DATA_DIR / "heartbeat.txt")
WATCHLIST_PATH = Path(os.environ.get("WATCHLIST_PATH") or pf.DATA_DIR / "intraday_watchlist.json")

_DUR_RE = re.compile(r"^(\d+)\s*([mhd])$", re.IGNORECASE)
_CFG: dict | None = None


def _cfg() -> dict:
    global _CFG
    if _CFG is None:
        _CFG = sna.load_config()
    return _CFG


# ─────────────────────────────────────────────────────────────
# Update offset persistence (so a restart doesn't replay commands)
# ─────────────────────────────────────────────────────────────

def _load_offset() -> int:
    try:
        return int(OFFSET_PATH.read_text().strip())
    except (OSError, ValueError):
        return 0


def _save_offset(offset: int) -> None:
    try:
        OFFSET_PATH.parent.mkdir(parents=True, exist_ok=True)
        OFFSET_PATH.write_text(str(offset))
    except OSError:
        pass


# ─────────────────────────────────────────────────────────────
# Telegram getUpdates
# ─────────────────────────────────────────────────────────────

def _get_updates(token: str, offset: int, timeout: int) -> list[dict]:
    params = urllib.parse.urlencode({
        "offset": offset,
        "timeout": timeout,
        "allowed_updates": json.dumps(["message"]),
    })
    url = f"{API_BASE}/bot{token}/getUpdates?{params}"
    with urllib.request.urlopen(url, timeout=timeout + 15) as resp:
        data = json.loads(resp.read())
    if not data.get("ok"):
        raise RuntimeError(f"getUpdates not ok: {str(data)[:200]}")
    return data.get("result", [])


# ─────────────────────────────────────────────────────────────
# Command parsing helpers
# ─────────────────────────────────────────────────────────────

def _parse_duration(s: str) -> timedelta | None:
    m = _DUR_RE.match(s.strip())
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2).lower()
    return {"m": timedelta(minutes=n), "h": timedelta(hours=n),
            "d": timedelta(days=n)}[unit]


def _heartbeat_age() -> str:
    try:
        ts = datetime.fromisoformat(HEARTBEAT_PATH.read_text().strip())
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        secs = (datetime.now(timezone.utc) - ts).total_seconds()
        if secs < 120:
            return f"{int(secs)}s ago"
        if secs < 7200:
            return f"{int(secs // 60)}m ago"
        return f"{int(secs // 3600)}h ago"
    except (OSError, ValueError):
        return "unknown"


def _watchlist_summary() -> tuple[int, int]:
    """(armed tickers, total levels) from the persisted watchlist, or (0,0)."""
    try:
        data = json.loads(WATCHLIST_PATH.read_text())
        tickers = data.get("tickers") or {}
        return len(tickers), sum(len(v) for v in tickers.values())
    except (OSError, json.JSONDecodeError):
        return 0, 0


# ─────────────────────────────────────────────────────────────
# Command handlers
# ─────────────────────────────────────────────────────────────

HELP = (
    "<b>Portfolio scanner — commands</b>\n"
    "/status — watcher + portfolio summary\n"
    "/list — show portfolio\n"
    "/positions — owned positions + P/L\n"
    "/scan TICKER — read + chart\n"
    "/add TICKER [N @ PRICE] — add (optionally as a holding)\n"
    "/remove TICKER — remove\n"
    "/mute TICKER · /unmute TICKER\n"
    "/snooze TICKER 30m|2h|1d (default 1h)"
)


def _cmd_status(notifier, args) -> None:
    positions = pf.load_positions()
    owned = [p for p in positions if p.is_owned]
    mutes = pf.load_mutes()
    armed, levels = _watchlist_summary()
    lines = [
        "<b>📡 STATUS</b>",
        f"Last poll: <code>{_heartbeat_age()}</code>",
        f"Portfolio: {len(positions)} tickers ({len(owned)} held)",
        f"Watchlist: {armed} armed · {levels} levels",
    ]
    if mutes:
        lines.append("Muted: " + ", ".join(sorted(mutes.keys())))
    notifier.message("\n".join(lines))


def _cmd_list(notifier, args) -> None:
    positions = pf.load_positions()
    if not positions:
        notifier.message("Portfolio is empty. Add one with /add TICKER")
        return
    mutes = pf.load_mutes()
    lines = ["<b>Portfolio</b>"]
    for p in positions:
        tag = " 🔇" if p.ticker in mutes else ""
        if p.is_owned:
            lines.append(f"• <b>{p.ticker}</b> — {p.shares:g} @ ${p.entry:.2f}{tag}")
        else:
            lines.append(f"• <b>{p.ticker}</b>{tag}")
    notifier.message("\n".join(lines))


def _cmd_positions(notifier, args) -> None:
    owned = [p for p in pf.load_positions() if p.is_owned]
    if not owned:
        notifier.message("No owned positions. Add one with /add TICKER N @ PRICE")
        return
    snap = get_snapshot_all([p.ticker for p in owned])
    lines = ["<b>📦 Positions</b>"]
    for p in owned:
        row = snap.get(p.ticker) or {}
        last = row.get("last")
        if last and p.entry:
            pl = (last / p.entry - 1) * 100.0
            sign = "+" if pl >= 0 else ""
            val = p.shares * last
            lines.append(
                f"• <b>{p.ticker}</b> {p.shares:g} @ ${p.entry:.2f} → "
                f"<code>${last:.2f}</code> ({sign}{pl:.1f}%, ${val:,.0f})"
            )
        else:
            lines.append(f"• <b>{p.ticker}</b> {p.shares:g} @ ${p.entry:.2f} (price n/a)")
    notifier.message("\n".join(lines))


def _cmd_scan(notifier, args) -> None:
    if not args:
        notifier.message("Usage: /scan TICKER")
        return
    ticker = args[0].upper()
    notifier.message(f"Scanning <b>{ticker}</b>…")
    try:
        result = sna.analyze_ticker(ticker, cfg=_cfg())
    except Exception as e:  # noqa: BLE001
        notifier.message(f"Scan failed for {ticker}: {e}")
        return
    if result.get("error"):
        notifier.message(f"{ticker}: {result['error']}")
        return

    headline = result.get("headline") or {}
    price = result.get("price") or {}
    last = float(price.get("last") or 0.0)
    chg = float(price.get("chg_pct") or 0.0)
    _score, tier, anchor = ps.score_actionability(result)

    lines = [
        f"<b>{headline.get('icon', '◆')} {ticker}</b> — {headline.get('title', '')}",
        f"<code>${last:.2f}  ({'+' if chg >= 0 else ''}{chg:.2f}%)</code>",
    ]
    if headline.get("detail"):
        lines.append(headline["detail"])
    lines.append(f"<i>{ps.TIER_LABELS.get(tier, tier)}</i>")
    caption = "\n".join(lines)

    highlight = None
    if anchor is not None:
        highlight = {"price": float(anchor.trigger_price),
                     "label": anchor.label, "direction": anchor.direction}
    png = chart_render.render_daily_png(result, _cfg(), highlight=highlight)
    notifier.alert(caption, png_bytes=png)


def _cmd_add(notifier, args) -> None:
    if not args:
        notifier.message("Usage: /add TICKER  or  /add TICKER 100 @ 240.50")
        return
    pos = pf._parse_line(" ".join(args))
    if pos is None:
        notifier.message("Couldn't parse that. Try /add TICKER or /add TICKER 100 @ 240.50")
        return
    added = pf.add_entry(pos.ticker, pos.shares, pos.entry)
    verb = "Added" if added else "Updated"
    if pos.is_owned:
        notifier.message(f"{verb} <b>{pos.ticker}</b> — {pos.shares:g} @ ${pos.entry:.2f}")
    else:
        notifier.message(f"{verb} <b>{pos.ticker}</b> (watch-only)")


def _cmd_remove(notifier, args) -> None:
    if not args:
        notifier.message("Usage: /remove TICKER")
        return
    ticker = args[0].upper()
    if pf.remove_ticker(ticker):
        notifier.message(f"Removed <b>{ticker}</b>")
    else:
        notifier.message(f"{ticker} wasn't in the portfolio")


def _cmd_mute(notifier, args) -> None:
    if not args:
        notifier.message("Usage: /mute TICKER")
        return
    ticker = args[0].upper()
    pf.set_mute(ticker)
    notifier.message(f"🔇 Muted <b>{ticker}</b> — /unmute {ticker} to restore")


def _cmd_unmute(notifier, args) -> None:
    if not args:
        notifier.message("Usage: /unmute TICKER")
        return
    ticker = args[0].upper()
    if pf.clear_mute(ticker):
        notifier.message(f"🔔 Unmuted <b>{ticker}</b>")
    else:
        notifier.message(f"{ticker} wasn't muted")


def _cmd_snooze(notifier, args) -> None:
    if not args:
        notifier.message("Usage: /snooze TICKER 30m|2h|1d (default 1h)")
        return
    ticker = args[0].upper()
    dur = _parse_duration(args[1]) if len(args) > 1 else timedelta(hours=1)
    if dur is None:
        notifier.message("Bad duration. Use e.g. 30m, 2h, 1d")
        return
    until = datetime.now(timezone.utc) + dur
    pf.set_snooze(ticker, until)
    label = args[1] if len(args) > 1 else "1h"
    notifier.message(f"😴 Snoozed <b>{ticker}</b> for {label}")


_COMMANDS = {
    "help": lambda n, a: n.message(HELP),
    "start": lambda n, a: n.message(HELP),
    "status": _cmd_status,
    "list": _cmd_list,
    "positions": _cmd_positions,
    "scan": _cmd_scan,
    "add": _cmd_add,
    "remove": _cmd_remove,
    "mute": _cmd_mute,
    "unmute": _cmd_unmute,
    "snooze": _cmd_snooze,
}


def _allowed_ids(notifier) -> set[str]:
    """IDs allowed to command the bot: TELEGRAM_OWNER_ID (comma/space list) plus
    the notifier's chat. Setting TELEGRAM_OWNER_ID to your user id lets you DM
    commands even when alerts post to a different chat (e.g. a channel)."""
    allowed: set[str] = set()
    for tok in os.environ.get("TELEGRAM_OWNER_ID", "").replace(",", " ").split():
        if tok.strip():
            allowed.add(tok.strip())
    if notifier.chat_id:
        allowed.add(str(notifier.chat_id))
    return allowed


def _handle_update(upd: dict, notifier) -> None:
    msg = upd.get("message") or {}
    text = (msg.get("text") or "").strip()
    chat = msg.get("chat") or {}
    chat_id = str(chat.get("id", ""))
    sender_id = str((msg.get("from") or {}).get("id", ""))

    # Authorize on EITHER the chat (alert channel) or the sender (owner DM).
    allowed = _allowed_ids(notifier)
    if allowed and chat_id not in allowed and sender_id not in allowed:
        print(f"[commands] ignoring unauthorized chat {chat_id} (sender {sender_id})",
              file=sys.stderr)
        return
    if not text.startswith("/"):
        return

    parts = text.split()
    cmd = parts[0].lstrip("/").split("@", 1)[0].lower()
    args = parts[1:]
    handler = _COMMANDS.get(cmd)
    if handler is None:
        notifier.message(f"Unknown command /{cmd}. Try /help")
        return
    handler(notifier, args)


# ─────────────────────────────────────────────────────────────
# Long-poll loop (runs in a daemon thread)
# ─────────────────────────────────────────────────────────────

# Transient long-poll hiccups — the held-open HTTPS connection gets cut by the
# network/server. Benign: retry quietly. (urllib lacks requests' connection
# resilience, so these surface as exceptions.)
_BENIGN_POLL_ERRORS = ("UNEXPECTED_EOF", "EOF occurred", "timed out",
                       "Connection reset", "Remote end closed", "Temporary failure")


def run_command_loop(notifier, poll_timeout: int = 30) -> None:
    token = getattr(notifier, "token", None)
    if not token:
        print("[commands] no token — command loop not started", file=sys.stderr)
        return
    offset = _load_offset()
    print("[commands] polling for commands…")
    while True:
        try:
            updates = _get_updates(token, offset, poll_timeout)
        except Exception as e:  # noqa: BLE001
            # A dropped long-poll connection just means "no updates" — retry
            # quietly. Only surface genuinely unexpected errors.
            if any(s in str(e) for s in _BENIGN_POLL_ERRORS):
                time.sleep(2)
            else:
                print(f"[commands] getUpdates error: {e}", file=sys.stderr)
                time.sleep(5)
            continue
        for upd in updates:
            offset = max(offset, int(upd.get("update_id", 0)) + 1)
            _save_offset(offset)
            try:
                _handle_update(upd, notifier)
            except Exception as e:  # noqa: BLE001
                print(f"[commands] handler error: {e}", file=sys.stderr)


if __name__ == "__main__":
    import notify
    run_command_loop(notify.Notifier.from_env(
        "TELEGRAM_WATCHER_BOT_TOKEN", "TELEGRAM_WATCHER_CHAT_ID"))
