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
from datetime import datetime, time as dtime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import portfolio_scanner as ps
import single_name_analyzer as sna
import portfolio as pf
import notify
import chart_render


DEFAULT_TIERS = ("FRESH_TRIGGER", "RECENT_TRIGGER", "IMMEDIATE_SETUP")
STATE_FILE = Path(os.environ.get("ALERTS_STATE_PATH", ".alerts_state.json"))
ALERT_LOG_FILE = Path(os.environ.get("ALERT_LOG_PATH", ".alert_log.jsonl"))

# The GitHub Actions cron fires on a fixed UTC window (it can't do
# timezones), so the script itself gates delivery to US market hours in
# Eastern time. ZoneInfo handles EST/EDT automatically — no DST juggling.
# Window is 09:30–16:05 ET: covers the full session plus the 4:00 PM ET
# closing-bar read, and excludes the post-close (5 PM+) runs.
ET = ZoneInfo("America/New_York")
MARKET_OPEN_ET  = dtime(9, 30)
MARKET_CLOSE_ET = dtime(16, 5)


def within_market_hours(now_utc: datetime | None = None) -> bool:
    now_et = (now_utc or datetime.now(timezone.utc)).astimezone(ET)
    if now_et.weekday() >= 5:          # Sat/Sun
        return False
    return MARKET_OPEN_ET <= now_et.time() <= MARKET_CLOSE_ET


# ─────────────────────────────────────────────────────────────
# Dedup state (persisted between runs via GitHub Actions cache)
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


def _trading_date(read: ps.ActionableRead) -> str:
    """Date of the last daily bar (so a weekend run reuses Friday's date and
    doesn't re-alert). Falls back to today UTC."""
    df = read.result.get("df_daily")
    if df is not None and not df.empty:
        return df.index[-1].strftime("%Y-%m-%d")
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def alert_key(read: ps.ActionableRead) -> str:
    """Stable dedup key. Date comes from the last daily bar so a Sunday
    run won't generate a new key vs Friday's run."""
    return f"{read.ticker}|{read.setup_kind or 'NEUTRAL'}|{_trading_date(read)}"


def alert_record(read: ps.ActionableRead) -> dict:
    """One row for the alert log the weekly scorecard grades."""
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": "cron",
        "ticker": read.ticker,
        "kind": read.setup_kind,
        "direction": read.direction,
        "tier": read.tier,
        "trigger": round(read.trigger_price, 4) if read.trigger_price else None,
        "price_at_alert": round(read.last_price, 4) if read.last_price else None,
        "trading_date": _trading_date(read),
    }


# ─────────────────────────────────────────────────────────────
# Message formatting
# ─────────────────────────────────────────────────────────────

def format_message(read: ps.ActionableRead, position: pf.Position | None = None) -> str:
    """Format a single alert as HTML for Telegram. If the ticker is an owned
    position, prepend a holdings/P&L line so a breakdown on a name you HOLD
    reads as a risk alert, not just an opportunity."""
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

    if position is not None and position.is_owned and position.entry and read.last_price > 0:
        pl = (read.last_price / position.entry - 1) * 100.0
        pl_sign = "+" if pl >= 0 else ""
        risk = " ⚠️" if read.direction == "BREAKDOWN" else ""
        lines.append(
            f"📦 <b>HELD</b>{risk} {position.shares:g} sh @ "
            f"<code>${position.entry:.2f}</code> · P/L {pl_sign}{pl:.1f}%"
        )

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
    if os.environ.get("IGNORE_MARKET_HOURS") != "1" and not within_market_hours():
        now_et = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
        print(f"[skip] outside market hours ({now_et}) — no alerts sent")
        return 0

    tickers = pf.load_portfolio()
    if not tickers:
        print("[skip] no portfolio configured (set SCAN_PORTFOLIO or portfolio.txt)")
        return 0
    positions = pf.positions_map()

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

    if not new_alerts:
        print(f"[done] {len(actionable)} actionable, all already alerted today")
        return 0

    print(f"[alert] attempting {len(new_alerts)} new alerts")
    # Best-effort send the batch header — even if it fails, the per-ticker
    # messages below carry their own context so the user still gets the data.
    notify.send_message(format_batch_header(len(new_alerts), len(tickers)))

    delivered = 0
    for r in new_alerts:
        pos = positions.get(r.ticker)
        # Render the daily chart with the firing level emphasized. Best-effort:
        # a None PNG (no kaleido / render error) falls back to a text alert.
        png = chart_render.render_daily_png(
            r.result, cfg,
            highlight={"price": r.trigger_price, "label": r.setup_label,
                       "direction": r.direction} if r.trigger_price else None,
        )
        if notify.send_alert(format_message(r, pos), png_bytes=png):
            already.add(alert_key(r))
            delivered += 1
            notify.append_jsonl(ALERT_LOG_FILE, alert_record(r))
        else:
            print(f"[retry] {r.ticker} delivery failed — will retry next run", file=sys.stderr)

    if delivered != len(new_alerts):
        print(f"[done] delivered {delivered}/{len(new_alerts)} — undelivered will retry", file=sys.stderr)
    else:
        print(f"[done] delivered {delivered}/{len(new_alerts)} alerts")

    state["alerted"] = sorted(already)
    save_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
