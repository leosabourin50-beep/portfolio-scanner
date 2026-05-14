"""Cron-driven portfolio scanner that sends Telegram alerts on
breakout/breakdown triggers.

Designed to be invoked by GitHub Actions on a cron schedule. Each run:
  1. Loads the portfolio from `portfolio.txt`.
  2. Runs `portfolio_scanner.scan_portfolio` (same engine as the app).
  3. Filters reads down to actionable tiers (FRESH_TRIGGER /
     RECENT_TRIGGER / IMMEDIATE_SETUP — configurable via env var).
  4. De-dupes against `.alerts_state.json` so a given (ticker, setup_kind,
     trading_date) only alerts once.
  5. Sends one Telegram message per new alert.
  6. Writes the updated state file (the GH Actions workflow commits it
     back to the repo so dedup persists across runs).

Required env vars:
  POLYGON_API_KEY      — same key the app uses
  TELEGRAM_BOT_TOKEN   — from @BotFather
  TELEGRAM_CHAT_ID     — your personal chat id (see README setup)

Optional env vars:
  ALERT_TIERS          — comma list, default "FRESH_TRIGGER,RECENT_TRIGGER,IMMEDIATE_SETUP"
  SCAN_PORTFOLIO       — comma list override; if set, ignores portfolio.txt
  ALERTS_STATE_PATH    — default ".alerts_state.json"
  DRY_RUN              — "1" to skip Telegram send (prints to stdout instead)
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import portfolio_scanner as ps
import single_name_analyzer as sna


DEFAULT_TIERS = ("FRESH_TRIGGER", "RECENT_TRIGGER", "IMMEDIATE_SETUP")
PORTFOLIO_FILE = Path("portfolio.txt")
STATE_FILE = Path(os.environ.get("ALERTS_STATE_PATH", ".alerts_state.json"))


# ─────────────────────────────────────────────────────────────
# Portfolio loading
# ─────────────────────────────────────────────────────────────

def load_portfolio() -> list[str]:
    """Env var override > portfolio.txt > empty."""
    env = os.environ.get("SCAN_PORTFOLIO", "").strip()
    if env:
        return [t.strip().upper() for t in env.replace(",", " ").split() if t.strip()]
    if PORTFOLIO_FILE.exists():
        raw = PORTFOLIO_FILE.read_text()
        return [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]
    return []


# ─────────────────────────────────────────────────────────────
# Dedup state (JSON file committed by the workflow)
# ─────────────────────────────────────────────────────────────

def load_state() -> dict:
    if not STATE_FILE.exists():
        return {"alerted": []}
    try:
        data = json.loads(STATE_FILE.read_text())
    except json.JSONDecodeError:
        return {"alerted": []}
    if "alerted" not in data or not isinstance(data["alerted"], list):
        data["alerted"] = []
    return data


def save_state(state: dict) -> None:
    # Prune anything older than ~30 days to keep the file small. Keys are
    # "TICKER|KIND|YYYY-MM-DD" so we can sort/cull on the date suffix.
    today = datetime.now(timezone.utc).date()
    pruned: list[str] = []
    for key in state.get("alerted", []):
        try:
            d = key.rsplit("|", 1)[-1]
            day = datetime.strptime(d, "%Y-%m-%d").date()
        except (ValueError, IndexError):
            pruned.append(key)
            continue
        if (today - day).days <= 30:
            pruned.append(key)
    state["alerted"] = sorted(set(pruned))
    STATE_FILE.write_text(json.dumps(state, indent=2))


def alert_key(read: ps.ActionableRead) -> str:
    """Stable dedup key. Date comes from the last daily bar so a Sunday
    run won't generate a new key vs Friday's run."""
    last_bar = None
    df = read.result.get("df_daily")
    if df is not None and not df.empty:
        last_bar = df.index[-1].strftime("%Y-%m-%d")
    if last_bar is None:
        last_bar = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"{read.ticker}|{read.setup_kind or 'NEUTRAL'}|{last_bar}"


# ─────────────────────────────────────────────────────────────
# Telegram delivery
# ─────────────────────────────────────────────────────────────

def _send_telegram(text: str) -> None:
    if os.environ.get("DRY_RUN") == "1":
        print("[DRY_RUN] would send:\n" + text + "\n---")
        return
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[skip] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set", file=sys.stderr)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": "true",
    }).encode()
    req = urllib.request.Request(url, data=payload)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
    except Exception as e:
        print(f"[error] Telegram send failed: {e}", file=sys.stderr)


def format_message(read: ps.ActionableRead) -> str:
    """Format a single alert as HTML for Telegram."""
    chg_sign = "+" if read.chg_pct >= 0 else ""
    dir_arrow = "▲" if read.direction == "BREAKOUT" else "▼"

    lines = [
        f"<b>{read.icon} {read.ticker}</b> — {read.headline_title}",
        f"<code>${read.last_price:.2f}  ({chg_sign}{read.chg_pct:.2f}%)</code>",
        f"{dir_arrow} <b>{read.setup_label}</b>",
    ]
    if read.headline_detail:
        lines.append(read.headline_detail)
    if read.headline_level:
        lines.append(f"Level: <code>{read.headline_level}</code>")
    lines.append(f"<i>{read.tier_label} · score {int(read.score)}</i>")
    return "\n".join(lines)


def format_batch_header(n: int, total: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"<b>PORTFOLIO SCAN · {ts}</b>\n"
        f"<i>{n} new alert{'s' if n != 1 else ''} across {total} tickers</i>"
    )


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> int:
    tickers = load_portfolio()
    if not tickers:
        print("[skip] no portfolio configured (set SCAN_PORTFOLIO or portfolio.txt)")
        return 0

    tier_env = os.environ.get("ALERT_TIERS", "").strip()
    alert_tiers = (
        tuple(t.strip() for t in tier_env.split(",") if t.strip())
        if tier_env else DEFAULT_TIERS
    )

    print(f"[scan] {len(tickers)} tickers, alert tiers: {alert_tiers}")
    cfg = sna.load_config()
    reads = ps.scan_portfolio(tickers, cfg=cfg, max_workers=8)

    actionable = [r for r in reads if r.tier in alert_tiers]
    if not actionable:
        print("[done] no actionable reads this scan")
        return 0

    state = load_state()
    already = set(state.get("alerted", []))

    new_alerts: list[ps.ActionableRead] = []
    for r in actionable:
        key = alert_key(r)
        if key in already:
            continue
        new_alerts.append(r)
        already.add(key)

    if not new_alerts:
        print(f"[done] {len(actionable)} actionable, all already alerted today")
        return 0

    print(f"[alert] sending {len(new_alerts)} new alerts")
    _send_telegram(format_batch_header(len(new_alerts), len(tickers)))
    for r in new_alerts:
        _send_telegram(format_message(r))

    state["alerted"] = sorted(already)
    save_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
