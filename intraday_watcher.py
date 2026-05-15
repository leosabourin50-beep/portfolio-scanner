"""Intraday breakout watcher.

Watches a curated list of high-signal levels (generated once per session by
running the full daily scan) and pings Telegram the first time price closes
through a level intraday.

Designed for two runtime modes:
  • Long-running (default) — `python intraday_watcher.py`
      Infinite loop, sleeps 60s between iterations. Intended for a Fly.io VM.
  • Single-pass — `python intraday_watcher.py --once`
      One scan + check, then exits. Useful for testing.

Signal-quality filters (keep the bot quiet, not noisy):
  • Only STRONG-quality breakout/breakdown setups are watched
  • Bar close must clear the trigger by ≥ 0.1 ATR — no wick fakeouts
  • Skip first 10 min of the session (opening-range noise)
  • One alert per (ticker, kind, trigger, date) — first cross only

Required env vars:
  POLYGON_API_KEY     — Polygon Stocks Starter or higher
  TELEGRAM_BOT_TOKEN  — same bot as the daily cron
  TELEGRAM_CHAT_ID    — your chat id

Optional env vars:
  WATCHER_POLL_SEC    — seconds between iterations (default 60)
  WATCHLIST_PATH      — where the per-day watchlist JSON lives (default /data/intraday_watchlist.json on Fly, .intraday_watchlist.json locally)
  ALERTS_STATE_PATH   — dedup state file (default /data/intraday_alerts_state.json on Fly, .intraday_alerts_state.json locally)
  DRY_RUN             — "1" to print messages instead of sending
  PER_TICKER_LIMIT    — max watched setups per ticker (default 3)
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict, field
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import portfolio_scanner as ps
import single_name_analyzer as sna
from polygon_adapter import get_client


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

ET = ZoneInfo("America/New_York")

POLL_SEC = int(os.environ.get("WATCHER_POLL_SEC", "60"))
PER_TICKER_LIMIT = int(os.environ.get("PER_TICKER_LIMIT", "3"))

# Default state paths: prefer /data (Fly volume) if it exists, else cwd
_DATA_DIR = Path("/data") if Path("/data").is_dir() else Path(".")
WATCHLIST_PATH = Path(os.environ.get("WATCHLIST_PATH", _DATA_DIR / "intraday_watchlist.json"))
ALERTS_STATE_PATH = Path(os.environ.get("ALERTS_STATE_PATH", _DATA_DIR / "intraday_alerts_state.json"))

PORTFOLIO_FILE = Path("portfolio.txt")

# Bump when the watchlist schema changes so cached files get regenerated
# on deploy instead of sticking around until the next trading day.
WATCHLIST_SCHEMA_VERSION = 3

# Session timing (US market, ET)
MARKET_OPEN  = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)
OPENING_SKIP_MIN = 10        # ignore first 10 min after open (fakeout window)
ATR_CONFIRMATION = 0.10      # bar close must clear level by >= this many ATRs
ATR_FOLLOW_THROUGH = 0.50    # for already-triggered setups, require a stronger continuation

# Setups that get the special "🏆 NEW 52-WEEK HIGH" / "📉 NEW 52-WEEK LOW"
# treatment — wick-based detection (any bar.high above the level fires) with
# distinct alert formatting. These bypass the close-based confirmation buffer
# because the moment a 52w high prints is the moment that matters.
_52W_HIGH_KINDS = {"DONCHIAN_252_HIGH", "WEEKLY_52W_HIGH"}
_52W_LOW_KINDS  = {"DONCHIAN_252_LOW", "WEEKLY_52W_LOW"}


def _is_channel_kind(kind: str) -> bool:
    """Channel rail setups get wick-based 'rail touch' alerts — a tag in
    formatted alerts and bypassed close-buffer logic, since touching the
    rail (even on a wick that rejects) is itself the signal."""
    return kind.startswith("CHANNEL_")


def _is_touch_eligible_kind(kind: str) -> bool:
    """Horizontal S/R and moving-average levels get an ADDITIONAL wick-based
    touch twin (alongside the normal close-based break alert). Touching the
    level is the decision point — bounce ('breakdown avoided') or break —
    so the user wants the ping the moment price tags it."""
    return (
        kind in ("HORIZONTAL_SUPPORT", "HORIZONTAL_RESISTANCE")
        or kind.startswith("MA_")
    )


# Rebuild the watchlist mid-day to catch setups that only formed after the
# morning scan — e.g., a fresh pattern that triggered into IMMEDIATE proximity
# during the session. Hours expressed in ET.
MIDDAY_REBUILD_HOURS = (12, 14)  # 12:00 ET and 14:00 ET


# ─────────────────────────────────────────────────────────────
# Watchlist data model
# ─────────────────────────────────────────────────────────────

@dataclass
class WatchSetup:
    kind: str
    direction: str         # BREAKOUT or BREAKDOWN
    trigger: float
    label: str
    quality_label: str
    atr: float             # daily ATR — used as the wick-fakeout buffer
    # When True, this is a follow-through alert on a level that already
    # triggered today. Watcher uses a 0.5-ATR confirmation buffer for these
    # so we only ping on real continuation, not noise around the breakout.
    is_follow_through: bool = False
    # When True, this is a wick-based TOUCH twin of a horizontal S/R or MA
    # level — fires the moment price tags the level (decision point: bounce
    # or break) rather than waiting for a confirming close.
    is_touch: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WatchSetup":
        # Tolerate older watchlists missing newer fields
        return cls(
            kind=d["kind"],
            direction=d["direction"],
            trigger=d["trigger"],
            label=d["label"],
            quality_label=d["quality_label"],
            atr=d["atr"],
            is_follow_through=bool(d.get("is_follow_through", False)),
            is_touch=bool(d.get("is_touch", False)),
        )


# ─────────────────────────────────────────────────────────────
# Portfolio + watchlist generation
# ─────────────────────────────────────────────────────────────

def load_portfolio() -> list[str]:
    env = os.environ.get("SCAN_PORTFOLIO", "").strip()
    if env:
        return [t.strip().upper() for t in env.replace(",", " ").split() if t.strip()]
    if PORTFOLIO_FILE.exists():
        raw = PORTFOLIO_FILE.read_text()
        return [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]
    return []


def _atr_from_result(result: dict) -> float:
    """Pull the daily ATR(14) out of the analyzer result, or compute it from
    the daily DataFrame as a fallback."""
    signals = result.get("signals") or {}
    deep = result.get("deep") or {}
    for key in ("atr_14", "atr", "atr14"):
        for d in (signals, deep):
            v = d.get(key)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
    df = result.get("df_daily")
    if df is None or df.empty or len(df) < 15:
        return 0.0
    high = df["High"]; low = df["Low"]; close = df["Close"].shift(1)
    tr = (high - low).combine((high - close).abs(), max).combine((low - close).abs(), max)
    return float(tr.rolling(14).mean().iloc[-1] or 0.0)


def build_watchlist(tickers: list[str]) -> dict[str, list[WatchSetup]]:
    """Run the daily scan and extract STRONG-quality breakout / breakdown
    triggers per ticker. Limited to top PER_TICKER_LIMIT setups by quality."""
    cfg = sna.load_config()
    print(f"[watchlist] scanning {len(tickers)} tickers to build watchlist...")
    reads = ps.scan_portfolio(tickers, cfg=cfg, max_workers=8)

    watchlist: dict[str, list[WatchSetup]] = {}
    for r in reads:
        if r.result.get("error"):
            continue
        atr = _atr_from_result(r.result)
        candidates: list[WatchSetup] = []

        def _qualifies(s) -> bool:
            """STRONG within 5% OR MODERATE within 2% — looser net so things
            like a MODERATE inverse H&S sitting right at the neckline get
            watched alongside the pure A+ setups."""
            d = abs(s.distance_pct)
            if s.quality_label == "STRONG" and d <= 5.0:
                return True
            if s.quality_label == "MODERATE" and d <= 2.0:
                return True
            return False

        # Pending setups
        for s in list(r.result.get("breakouts") or []) + list(r.result.get("breakdowns") or []):
            if not _qualifies(s):
                continue
            candidates.append(WatchSetup(
                kind=s.kind,
                direction=s.direction,
                trigger=float(s.trigger_price),
                label=s.label,
                quality_label=s.quality_label,
                atr=atr,
                is_follow_through=False,
            ))

        # Recently-triggered setups → follow-through (close >= trigger + 0.5
        # ATR for breakouts, symmetric for breakdowns) so we capture
        # continuation moves the strict-pending watcher would miss.
        for s in list(r.result.get("triggered") or []):
            if (s.bars_since_trigger or 0) > 5:
                continue
            if s.quality_label not in ("STRONG", "MODERATE"):
                continue
            candidates.append(WatchSetup(
                kind=s.kind,
                direction=s.direction,
                trigger=float(s.trigger_price),
                label=s.label,
                quality_label=s.quality_label,
                atr=atr,
                is_follow_through=True,
            ))

        if not candidates:
            continue
        # Sort by absolute distance from current price, then cap. The limit
        # governs the number of distinct LEVELS — touch twins added below
        # are bonus and don't count against it.
        last = (r.result.get("price") or {}).get("last") or 0.0
        if last:
            candidates.sort(key=lambda w: abs(w.trigger - float(last)))
        candidates = candidates[:PER_TICKER_LIMIT]

        # For horizontal S/R and MA levels, add a wick-based TOUCH twin so
        # the user gets pinged the moment price tags the level (decision
        # point) in addition to the close-based break confirmation.
        twins: list[WatchSetup] = []
        for w in candidates:
            if w.is_follow_through:
                continue
            if _is_touch_eligible_kind(w.kind):
                twins.append(WatchSetup(
                    kind=w.kind,
                    direction=w.direction,
                    trigger=w.trigger,
                    label=w.label,
                    quality_label=w.quality_label,
                    atr=w.atr,
                    is_follow_through=False,
                    is_touch=True,
                ))
        watchlist[r.ticker] = candidates + twins

    return watchlist


def save_watchlist(watchlist: dict[str, list[WatchSetup]]) -> None:
    payload = {
        "schema_version": WATCHLIST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tickers": {tk: [w.to_dict() for w in ws] for tk, ws in watchlist.items()},
    }
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    WATCHLIST_PATH.write_text(json.dumps(payload, indent=2))


def load_watchlist() -> tuple[dict[str, list[WatchSetup]], str | None]:
    if not WATCHLIST_PATH.exists():
        return {}, None
    try:
        data = json.loads(WATCHLIST_PATH.read_text())
    except json.JSONDecodeError:
        return {}, None
    # Force regenerate if the schema version doesn't match (the in-disk
    # format may be missing new fields the watcher now depends on)
    if data.get("schema_version") != WATCHLIST_SCHEMA_VERSION:
        return {}, None
    out: dict[str, list[WatchSetup]] = {}
    for tk, ws in (data.get("tickers") or {}).items():
        out[tk] = [WatchSetup.from_dict(w) for w in ws]
    return out, data.get("generated_at")


# ─────────────────────────────────────────────────────────────
# Dedup state
# ─────────────────────────────────────────────────────────────

def load_state() -> dict:
    if not ALERTS_STATE_PATH.exists():
        return {"alerted": []}
    try:
        return json.loads(ALERTS_STATE_PATH.read_text())
    except json.JSONDecodeError:
        return {"alerted": []}


def save_state(state: dict) -> None:
    today = datetime.now(timezone.utc).date()
    pruned: list[str] = []
    for key in state.get("alerted", []):
        try:
            d = key.rsplit("|", 1)[-1]
            day = datetime.strptime(d, "%Y-%m-%d").date()
        except (ValueError, IndexError):
            pruned.append(key)
            continue
        if (today - day).days <= 7:
            pruned.append(key)
    state["alerted"] = sorted(set(pruned))
    ALERTS_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALERTS_STATE_PATH.write_text(json.dumps(state, indent=2))


def alert_key(ticker: str, setup: WatchSetup, date_str: str) -> str:
    # Round trigger to 2dp so float jitter doesn't break dedup. The marker
    # keeps follow-through / touch / initial-cross alerts dedup-separate so
    # a touch ping and its later break ping don't suppress each other.
    if setup.is_touch:
        marker = "TCH"
    elif setup.is_follow_through:
        marker = "FT"
    else:
        marker = "X"
    return f"{ticker}|{setup.kind}|{marker}|{setup.trigger:.2f}|{date_str}"


# ─────────────────────────────────────────────────────────────
# Polygon intraday fetch
# ─────────────────────────────────────────────────────────────

def fetch_latest_5m_bar(ticker: str) -> dict | None:
    """Returns the most recent completed 5-min bar as a dict, or None."""
    client = get_client()
    today_et = datetime.now(ET).date()
    today_str = today_et.strftime("%Y-%m-%d")
    try:
        # Last-3-bar fetch — gives us the latest bar plus a buffer for any
        # delayed bar prints.
        bars = list(client.get_aggs(
            ticker=ticker,
            multiplier=5, timespan="minute",
            from_=today_str, to=today_str,
            adjusted=True, sort="desc", limit=3,
        ))
    except Exception as e:
        print(f"[error] polygon fetch failed for {ticker}: {e}", file=sys.stderr)
        return None
    if not bars:
        return None
    # bars[0] is the most recent. Polygon timestamp is ms unix.
    b = bars[0]
    return {
        "ts": datetime.fromtimestamp(b.timestamp / 1000, tz=timezone.utc),
        "open": float(b.open),
        "high": float(b.high),
        "low": float(b.low),
        "close": float(b.close),
        "volume": float(b.volume or 0),
    }


# ─────────────────────────────────────────────────────────────
# Telegram delivery
# ─────────────────────────────────────────────────────────────

def _send_telegram(text: str) -> bool:
    """Return True iff the message was successfully delivered (or simulated
    in DRY_RUN). False means the caller should NOT mark the alert as
    delivered — so the next poll will retry it instead of silently deduping."""
    if os.environ.get("DRY_RUN") == "1":
        print("[DRY_RUN] would send:\n" + text + "\n---")
        return True
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[skip] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set", file=sys.stderr)
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": "true",
    }).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=payload), timeout=15) as resp:
            body = resp.read()
        # Telegram returns 200 with {"ok": true} on success
        if b'"ok":true' in body:
            return True
        print(f"[error] Telegram send returned non-ok: {body[:200]!r}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[error] Telegram send failed: {e}", file=sys.stderr)
        return False


def format_alert(ticker: str, setup: WatchSetup, bar: dict) -> str:
    bar_time_et = bar["ts"].astimezone(ET).strftime("%H:%M ET")

    # Special-case 52-week high / low — distinct trophy/down-trend formatting
    if setup.kind in _52W_HIGH_KINDS:
        chg = (bar["high"] - setup.trigger) / setup.trigger * 100
        return "\n".join([
            f"<b>🏆 NEW 52-WEEK HIGH — {ticker}</b>",
            f"<code>${bar['high']:.2f}</code>  ({chg:+.2f}% past prior 52wh)",
            f"Prior high <code>${setup.trigger:.2f}</code>",
            f"<i>{bar_time_et}</i>",
        ])
    if setup.kind in _52W_LOW_KINDS:
        chg = (bar["low"] - setup.trigger) / setup.trigger * 100
        return "\n".join([
            f"<b>📉 NEW 52-WEEK LOW — {ticker}</b>",
            f"<code>${bar['low']:.2f}</code>  ({chg:+.2f}% past prior 52wl)",
            f"Prior low <code>${setup.trigger:.2f}</code>",
            f"<i>{bar_time_et}</i>",
        ])

    # Horizontal S/R or MA level touch — decision-point ping
    if setup.is_touch:
        touched_price = bar["high"] if setup.direction == "BREAKOUT" else bar["low"]
        chg = (touched_price - setup.trigger) / setup.trigger * 100
        held = "resistance" if setup.direction == "BREAKOUT" else "support"
        return "\n".join([
            f"<b>📍 LEVEL TOUCH — {ticker}</b>",
            f"<code>${touched_price:.2f}</code>  ({chg:+.2f}% vs level)",
            f"Tagged {held} <b>{setup.label.upper()}</b> @ <code>${setup.trigger:.2f}</code>",
            f"<i>watch for bounce or break · {bar_time_et}</i>",
        ])

    # Channel rail touch — distinct from a breakout ping
    if _is_channel_kind(setup.kind):
        rail_side = "UPPER" if "UPPER" in setup.kind else ("LOWER" if "LOWER" in setup.kind else "RAIL")
        tf = "WEEKLY" if "WEEKLY" in setup.kind else "DAILY"
        slope = "ASCENDING" if "ASCENDING" in setup.kind else (
                "DESCENDING" if "DESCENDING" in setup.kind else "CHANNEL")
        touched_price = bar["high"] if setup.direction == "BREAKOUT" else bar["low"]
        chg = (touched_price - setup.trigger) / setup.trigger * 100
        return "\n".join([
            f"<b>📍 CHANNEL TOUCH — {ticker}</b>",
            f"<code>${touched_price:.2f}</code>  ({chg:+.2f}% vs rail)",
            f"{tf} {slope} channel — {rail_side.lower()} rail at <code>${setup.trigger:.2f}</code>",
            f"<i>{bar_time_et}</i>",
        ])

    arrow = "▲" if setup.direction == "BREAKOUT" else "▼"
    if setup.is_follow_through:
        side = "FOLLOW-THROUGH" + ("" if setup.direction == "BREAKOUT" else " ↓")
        body_verb = "Extending past"
    else:
        side = "BREAKOUT" if setup.direction == "BREAKOUT" else "BREAKDOWN"
        body_verb = "Cleared"
    chg_from_trigger = (bar["close"] - setup.trigger) / setup.trigger * 100
    lines = [
        f"<b>{arrow} INTRADAY {side} — {ticker}</b>",
        f"<code>${bar['close']:.2f}</code>  ({chg_from_trigger:+.2f}% vs trigger)",
        f"{body_verb} <b>{setup.label.upper()}</b> @ <code>${setup.trigger:.2f}</code>",
        f"<i>5-min bar close · {bar_time_et}</i>",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Cross detection
# ─────────────────────────────────────────────────────────────

def is_cross(bar: dict, setup: WatchSetup) -> bool:
    """Return True if the bar represents a confirmed cross of the level.
    52-week high/low setups fire wick-based on any bar high/low printing past
    the level — the moment a new 52wh prints is the moment that matters.
    Pending setups: bar close must clear by >= 0.1 ATR and the bar must have
    touched the trigger (rules out gap-throughs that retraced).
    Follow-through setups (level already triggered today): bar close must
    extend >= 0.5 ATR past the level — captures real continuation rather
    than chop around the breakout level."""
    # 52-week high/low: pure wick check, no buffer
    if setup.kind in _52W_HIGH_KINDS:
        return bar["high"] > setup.trigger
    if setup.kind in _52W_LOW_KINDS:
        return bar["low"] < setup.trigger

    # Channel rail TOUCH — wick-based, fires when price reaches the rail
    # in the direction the analyzer flagged. Captures both reversal and
    # breakout scenarios; user decides what to do from the chart.
    if _is_channel_kind(setup.kind):
        if setup.direction == "BREAKOUT":
            return bar["high"] >= setup.trigger
        return bar["low"] <= setup.trigger

    # Horizontal S/R or MA TOUCH twin — wick-based, same decision-point
    # logic as channel touches. The close-based break alert for this same
    # level is a separate WatchSetup and fires independently.
    if setup.is_touch:
        if setup.direction == "BREAKOUT":
            return bar["high"] >= setup.trigger
        return bar["low"] <= setup.trigger

    atr_mult = ATR_FOLLOW_THROUGH if setup.is_follow_through else ATR_CONFIRMATION
    buffer = max(setup.atr * atr_mult, setup.trigger * 0.0005)
    if setup.direction == "BREAKOUT":
        if setup.is_follow_through:
            return bar["close"] >= setup.trigger + buffer
        return bar["close"] >= setup.trigger + buffer and bar["high"] >= setup.trigger
    # BREAKDOWN
    if setup.is_follow_through:
        return bar["close"] <= setup.trigger - buffer
    return bar["close"] <= setup.trigger - buffer and bar["low"] <= setup.trigger


# ─────────────────────────────────────────────────────────────
# Session timing
# ─────────────────────────────────────────────────────────────

def market_state(now_utc: datetime) -> str:
    """Return one of: 'pre', 'open', 'opening_skip', 'closed_weekday', 'weekend'."""
    now_et = now_utc.astimezone(ET)
    if now_et.weekday() >= 5:
        return "weekend"
    t = now_et.time()
    if t < MARKET_OPEN:
        return "pre"
    if t < (datetime.combine(now_et.date(), MARKET_OPEN) + timedelta(minutes=OPENING_SKIP_MIN)).time():
        return "opening_skip"
    if t < MARKET_CLOSE:
        return "open"
    return "closed_weekday"


def seconds_until_open(now_utc: datetime) -> float:
    """Seconds until the next market open + OPENING_SKIP_MIN."""
    now_et = now_utc.astimezone(ET)
    target_date = now_et.date()
    open_t = (datetime.combine(target_date, MARKET_OPEN) + timedelta(minutes=OPENING_SKIP_MIN)).time()
    target = datetime.combine(target_date, open_t, tzinfo=ET)
    if now_et >= target:
        target += timedelta(days=1)
    # Skip weekends
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return (target - now_et).total_seconds()


# ─────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────

def check_once(watchlist: dict[str, list[WatchSetup]]) -> int:
    """One pass through every watched setup. Returns number of alerts sent."""
    state = load_state()
    already = set(state.get("alerted", []))
    sent = 0
    today_et = datetime.now(ET).strftime("%Y-%m-%d")

    for ticker, setups in watchlist.items():
        if not setups:
            continue
        # Skip the fetch if every setup for this ticker is already alerted today
        if all(alert_key(ticker, s, today_et) in already for s in setups):
            continue
        bar = fetch_latest_5m_bar(ticker)
        if bar is None:
            continue
        for s in setups:
            key = alert_key(ticker, s, today_et)
            if key in already:
                continue
            if is_cross(bar, s):
                if _send_telegram(format_alert(ticker, s, bar)):
                    already.add(key)
                    sent += 1
                    print(f"[alert] {ticker} {s.direction} {s.kind} @ ${s.trigger:.2f}")
                else:
                    print(
                        f"[retry] {ticker} {s.kind} delivery failed — will retry next poll",
                        file=sys.stderr,
                    )

    if sent:
        state["alerted"] = sorted(already)
        save_state(state)
    return sent


def ensure_today_watchlist(portfolio: list[str]) -> dict[str, list[WatchSetup]]:
    """Load the persisted watchlist; regenerate it if missing or stale."""
    watchlist, generated_at = load_watchlist()
    today_et = datetime.now(ET).date().isoformat()
    fresh = False
    if generated_at:
        try:
            gen_date = datetime.fromisoformat(generated_at).astimezone(ET).date().isoformat()
            fresh = (gen_date == today_et)
        except ValueError:
            fresh = False
    if not fresh or not watchlist:
        watchlist = build_watchlist(portfolio)
        save_watchlist(watchlist)
        print(f"[watchlist] generated {sum(len(v) for v in watchlist.values())} levels across {len(watchlist)} tickers")
    else:
        n_levels = sum(len(v) for v in watchlist.values())
        print(f"[watchlist] using cached: {n_levels} levels across {len(watchlist)} tickers")
    return watchlist


def _force_rebuild_watchlist(portfolio: list[str]) -> dict[str, list[WatchSetup]]:
    print("[watchlist] mid-day rebuild — re-scanning to catch newly-formed setups")
    watchlist = build_watchlist(portfolio)
    save_watchlist(watchlist)
    print(f"[watchlist] rebuilt: {sum(len(v) for v in watchlist.values())} levels across {len(watchlist)} tickers")
    return watchlist


def main_loop() -> int:
    portfolio = load_portfolio()
    if not portfolio:
        print("[fatal] no portfolio configured", file=sys.stderr)
        return 1
    print(f"[start] intraday watcher | poll {POLL_SEC}s | portfolio: {portfolio}")

    watchlist: dict[str, list[WatchSetup]] = {}
    rebuilt_hours_today: set[int] = set()

    while True:
        try:
            now = datetime.now(timezone.utc)
            state = market_state(now)

            if state in ("weekend", "closed_weekday", "pre"):
                wait = min(seconds_until_open(now), 3600)  # cap at 1 hour
                print(f"[sleep] market state={state}, sleeping {int(wait)}s")
                # New trading day starts after a sleep through the close — clear
                # the rebuild bookkeeping so tomorrow's mid-day rebuilds fire.
                rebuilt_hours_today.clear()
                time.sleep(max(wait, 60))
                continue

            if state == "opening_skip":
                # Pre-build today's watchlist while we wait through the noisy open
                if not watchlist:
                    watchlist = ensure_today_watchlist(portfolio)
                time.sleep(POLL_SEC)
                continue

            # state == "open"
            watchlist = ensure_today_watchlist(portfolio)

            # Mid-day rebuild: at the configured hours (12:00 ET, 14:00 ET) do
            # a fresh build so setups that only formed during the session get
            # picked up. Once per hour, dedup'd via rebuilt_hours_today.
            now_et = now.astimezone(ET)
            for hr in MIDDAY_REBUILD_HOURS:
                if now_et.hour == hr and hr not in rebuilt_hours_today:
                    watchlist = _force_rebuild_watchlist(portfolio)
                    rebuilt_hours_today.add(hr)
                    break

            sent = check_once(watchlist)
            if sent == 0:
                print(f"[tick] {now_et.strftime('%H:%M:%S ET')} — no crosses")
            time.sleep(POLL_SEC)
        except KeyboardInterrupt:
            print("[stop] interrupted")
            return 0
        except Exception:
            print("[error] iteration failed:\n" + traceback.format_exc(), file=sys.stderr)
            time.sleep(POLL_SEC)


def main() -> int:
    if "--once" in sys.argv:
        portfolio = load_portfolio()
        if not portfolio:
            print("[fatal] no portfolio configured", file=sys.stderr)
            return 1
        watchlist = ensure_today_watchlist(portfolio)
        sent = check_once(watchlist)
        print(f"[done] {sent} alert(s) sent")
        return 0
    return main_loop()


if __name__ == "__main__":
    sys.exit(main())
