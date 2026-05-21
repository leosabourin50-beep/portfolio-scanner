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
from dataclasses import dataclass, asdict, field, replace
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

# Violent-move detection. Fires when a single 5-min bar's close differs from
# the previous bar's close by at least VIOLENT_MOVE_PCT, AND the bar's
# volume is at least VIOLENT_MOVE_VOL_MULT times the avg 5-min bar. Catches
# vertical surges/drops that don't happen to cross any pre-curated level.
VIOLENT_MOVE_PCT      = float(os.environ.get("VIOLENT_MOVE_PCT", "3.0"))
VIOLENT_MOVE_VOL_MULT = float(os.environ.get("VIOLENT_MOVE_VOL_MULT", "2.5"))

# Default state paths: prefer /data (Fly volume) if it exists, else cwd
_DATA_DIR = Path("/data") if Path("/data").is_dir() else Path(".")
WATCHLIST_PATH = Path(os.environ.get("WATCHLIST_PATH", _DATA_DIR / "intraday_watchlist.json"))
ALERTS_STATE_PATH = Path(os.environ.get("ALERTS_STATE_PATH", _DATA_DIR / "intraday_alerts_state.json"))

PORTFOLIO_FILE = Path("portfolio.txt")

# Bump when the watchlist schema changes so cached files get regenerated
# on deploy instead of sticking around until the next trading day.
WATCHLIST_SCHEMA_VERSION = 8

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


def _setup_has_kind(setup, kinds: set) -> bool:
    """True if the setup's own kind OR any kind absorbed in its confluence
    merge matches. Critical so a PATTERN_* anchor that absorbed e.g.
    WEEKLY_52W_HIGH still triggers the 52w-high special handling (wick-
    based detection + trophy formatting) — without this, the merge silently
    downgraded ATH alerts to standard close-buffer breakouts."""
    if setup.kind in kinds:
        return True
    for k in (setup.confluence_kinds or []):
        if k in kinds:
            return True
    return False


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
    # 20-day average daily volume (from daily bars) — kept for reference /
    # potential digest use. NOT used for RVOL (different volume universe
    # than intraday aggs — see avg_5m_vol).
    avg_daily_vol: float = 0.0
    # Mean volume of a 5-min bar over recent intraday history. RVOL is
    # computed against THIS so numerator and denominator come from the
    # same (intraday-aggregate) volume universe.
    avg_5m_vol: float = 0.0
    # Last price at watchlist-build time — lets the refresh digest rank
    # levels by proximity-to-trigger without extra API calls.
    last_price: float = 0.0
    # Kinds of other setups absorbed into this one during confluence merge.
    # Critical for not losing special semantics: a PATTERN_* that absorbs a
    # WEEKLY_52W_HIGH should still fire wick-based with the trophy format,
    # not the standard close-buffer breakout path.
    confluence_kinds: list[str] = field(default_factory=list)

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
            avg_daily_vol=float(d.get("avg_daily_vol", 0.0)),
            avg_5m_vol=float(d.get("avg_5m_vol", 0.0)),
            last_price=float(d.get("last_price", 0.0)),
            confluence_kinds=list(d.get("confluence_kinds") or []),
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


def _rolling_52w_extreme(df, want_high: bool):
    """True trailing-252-trading-day extreme EXCLUDING today's live/partial
    bar. This is the *correct* reference for a 'new 52-week high' — not the
    analyzer's setup trigger_price, which is the stale original breakout
    level (price may sit far above it, making a wick check always-true and
    firing false 'new high' pings even on a -4% day). Returns None if there
    isn't enough history to be meaningful."""
    if df is None or df.empty or len(df) < 30:
        return None
    window = df.iloc[-253:-1]            # 252 bars ending yesterday
    if window.empty:
        return None
    return float(window["High"].max()) if want_high else float(window["Low"].min())


def build_watchlist(
    tickers: list[str],
) -> tuple[dict[str, list[WatchSetup]], dict[str, dict]]:
    """Run the daily scan and extract STRONG-quality breakout / breakdown
    triggers per ticker. Returns:
      • watchlist: ticker -> list[WatchSetup] (only tickers with qualifying
        level setups appear here)
      • portfolio_meta: ticker -> {avg_5m_vol, last_price} for EVERY
        analyzed portfolio ticker, including "quiet" ones with no setups.
        Powers level-independent alerts (violent-move) on all names."""
    cfg = sna.load_config()
    print(f"[watchlist] scanning {len(tickers)} tickers to build watchlist...")
    reads = ps.scan_portfolio(tickers, cfg=cfg, max_workers=8)

    watchlist: dict[str, list[WatchSetup]] = {}
    portfolio_meta: dict[str, dict] = {}
    for r in reads:
        if r.result.get("error"):
            continue
        atr = _atr_from_result(r.result)
        # Stamp per-ticker metadata for EVERY analyzed ticker (incl. quiet
        # ones with no setups) so the violent-move scan can run on them.
        avg_5m = _avg_5m_volume(r.ticker)
        last_px = float((r.result.get("price") or {}).get("last") or 0.0)
        portfolio_meta[r.ticker] = {"avg_5m_vol": avg_5m, "last_price": last_px}
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
                confluence_kinds=list(s.confluence_with or []),
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
                confluence_kinds=list(s.confluence_with or []),
            ))

        # ── 52-week high/low correction ─────────────────────────────────
        # Replace the analyzer's stale breakout-level trigger with the TRUE
        # trailing-252-day extreme (excluding today), force off the
        # follow-through path (these always use the fresh wick check), and
        # collapse duplicate 52w kinds so a genuine new high pings once, not
        # once per detector (DONCHIAN_252_HIGH + WEEKLY_52W_HIGH).
        df_daily = r.result.get("df_daily")
        hi_252 = _rolling_52w_extreme(df_daily, want_high=True)
        lo_252 = _rolling_52w_extreme(df_daily, want_high=False)
        fixed: list[WatchSetup] = []
        seen_hi = seen_lo = False
        for w in candidates:
            if w.kind in _52W_HIGH_KINDS:
                if seen_hi or hi_252 is None:
                    continue
                seen_hi = True
                w = replace(w, trigger=hi_252, is_follow_through=False)
            elif w.kind in _52W_LOW_KINDS:
                if seen_lo or lo_252 is None:
                    continue
                seen_lo = True
                w = replace(w, trigger=lo_252, is_follow_through=False)
            fixed.append(w)
        candidates = fixed

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
                    confluence_kinds=list(w.confluence_kinds or []),
                ))
        final = candidates + twins

        # Stamp volume references + last price on every entry. avg_5m and
        # last_px were already computed above for portfolio_meta — reuse.
        avg_dv = 0.0
        ddf = r.result.get("df_daily")
        if ddf is not None and not ddf.empty and len(ddf) > 21:
            avg_dv = float(ddf["Volume"].iloc[-21:-1].mean() or 0.0)
        for w in final:
            w.avg_daily_vol = avg_dv
            w.avg_5m_vol = avg_5m
            w.last_price = last_px

        watchlist[r.ticker] = final

    return watchlist, portfolio_meta


def save_watchlist(
    watchlist: dict[str, list[WatchSetup]],
    portfolio_meta: dict[str, dict],
) -> None:
    payload = {
        "schema_version": WATCHLIST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tickers": {tk: [w.to_dict() for w in ws] for tk, ws in watchlist.items()},
        "portfolio_meta": portfolio_meta,
    }
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    WATCHLIST_PATH.write_text(json.dumps(payload, indent=2))


def load_watchlist() -> tuple[dict[str, list[WatchSetup]], dict[str, dict], str | None]:
    if not WATCHLIST_PATH.exists():
        return {}, {}, None
    try:
        data = json.loads(WATCHLIST_PATH.read_text())
    except json.JSONDecodeError:
        return {}, {}, None
    if data.get("schema_version") != WATCHLIST_SCHEMA_VERSION:
        return {}, {}, None
    out: dict[str, list[WatchSetup]] = {}
    for tk, ws in (data.get("tickers") or {}).items():
        out[tk] = [WatchSetup.from_dict(w) for w in ws]
    meta = data.get("portfolio_meta") or {}
    return out, meta, data.get("generated_at")


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

def fetch_recent_5m_bars(ticker: str, n: int = 3) -> list[dict] | None:
    """Returns the most-recent n 5-min bars (newest-first), or None on
    error/empty. Used by both the level-cross check (needs bars[0]) and the
    violent-move detector (needs bars[0].close vs bars[1].close)."""
    client = get_client()
    today_str = datetime.now(ET).date().strftime("%Y-%m-%d")
    try:
        bars = list(client.get_aggs(
            ticker=ticker, multiplier=5, timespan="minute",
            from_=today_str, to=today_str,
            adjusted=True, sort="desc", limit=n,
        ))
    except Exception as e:
        print(f"[error] polygon fetch failed for {ticker}: {e}", file=sys.stderr)
        return None
    if not bars:
        return None
    return [
        {
            "ts": datetime.fromtimestamp(b.timestamp / 1000, tz=timezone.utc),
            "open": float(b.open), "high": float(b.high),
            "low": float(b.low), "close": float(b.close),
            "volume": float(b.volume or 0),
        }
        for b in bars
    ]


def fetch_latest_5m_bar(ticker: str) -> dict | None:
    """Returns the most recent completed 5-min bar as a dict, or None."""
    bars = fetch_recent_5m_bars(ticker, n=3)
    return bars[0] if bars else None


def _is_rth_bar(b) -> bool:
    """Polygon's intraday aggs include pre-market (4:00–9:30 ET) and after-
    hours (16:00–20:00 ET), where 5-min bars trade hundreds to single-digit
    thousand shares. Mixing those into the avg destroys the baseline (a
    typical regular-hours bar then looks like 10× normal). Restrict to
    regular trading hours: 9:30 ≤ bar start < 16:00 ET."""
    bar_et = datetime.fromtimestamp(b.timestamp / 1000, tz=timezone.utc).astimezone(ET)
    if bar_et.weekday() >= 5:
        return False
    t = bar_et.time()
    return MARKET_OPEN <= t < MARKET_CLOSE


def _avg_5m_volume(ticker: str, lookback_days: int = 10) -> float:
    """Mean volume of a regular-hours 5-min bar over the last
    `lookback_days` calendar days. RTH-only so the baseline reflects normal
    session participation, not pre/post-market thin bars."""
    client = get_client()
    end = datetime.now(ET).date()
    start = end - timedelta(days=lookback_days)
    try:
        bars = list(client.get_aggs(
            ticker=ticker, multiplier=5, timespan="minute",
            from_=start.strftime("%Y-%m-%d"), to=end.strftime("%Y-%m-%d"),
            adjusted=True, sort="asc", limit=5000,
        ))
    except Exception:
        return 0.0
    vols = [float(b.volume or 0) for b in bars
            if (b.volume or 0) > 0 and _is_rth_bar(b)]
    if not vols:
        return 0.0
    return sum(vols) / len(vols)


def _volume_context(ticker: str, avg_5m_vol: float, bar: dict) -> str | None:
    """Relative-volume annotation for an alert, e.g.
    'Vol 2.4× normal pace · this bar 3.1× avg 5m', or None if unavailable.

    Both metrics are measured purely in the intraday 5-min-aggregate volume
    universe so the ratios are meaningful:
      • this bar Nx  = firing bar volume / mean 5-min-bar volume
      • Nx normal pace = today's cumulative vol so far / (mean 5-min-bar
        volume × number of 5-min bars elapsed this session) — >1 means
        today is running heavier than a typical day at this point."""
    if not avg_5m_vol or avg_5m_vol <= 0:
        return None
    client = get_client()
    today = datetime.now(ET).date().strftime("%Y-%m-%d")
    try:
        bars = list(client.get_aggs(
            ticker=ticker, multiplier=5, timespan="minute",
            from_=today, to=today, adjusted=True, sort="asc", limit=120,
        ))
    except Exception:
        bars = []
    # RTH-only cumulative: pre/post-market volume isn't comparable to the
    # RTH-baseline avg_5m_vol, so it has to be excluded from both sides.
    cum_vol = sum(float(b.volume or 0) for b in bars if _is_rth_bar(b))

    bar_et = bar["ts"].astimezone(ET)
    # Bars elapsed since 9:30 ET (clamped to session bounds). If the firing
    # bar is pre-market, the session pace ratio is undefined — fall back to
    # the per-bar multiple alone.
    open_today = bar_et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_today = bar_et.replace(hour=16, minute=0, second=0, microsecond=0)
    in_rth = open_today <= bar_et < close_today
    elapsed_min = (bar_et - open_today).total_seconds() / 60.0
    bars_elapsed = max(min(elapsed_min / 5.0, 78.0), 1.0) if in_rth else 0.0

    expected_cum = avg_5m_vol * bars_elapsed
    bar_mult = bar["volume"] / avg_5m_vol if avg_5m_vol > 0 else 0.0

    if in_rth and expected_cum > 0 and cum_vol > 0:
        rvol = cum_vol / expected_cum
        return f"📊 Vol {rvol:.1f}× normal pace · this bar {bar_mult:.1f}× avg 5m"
    # Pre-market or no RTH cumulative yet — just the per-bar multiple.
    return f"📊 This bar {bar_mult:.1f}× avg 5m volume"


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


def format_alert(ticker: str, setup: WatchSetup, bar: dict,
                  vol_note: str | None = None) -> str:
    bar_time_et = bar["ts"].astimezone(ET).strftime("%H:%M ET")

    # Special-case 52-week high / low — distinct trophy/down-trend formatting.
    # Also fires when a pattern setup absorbed a 52w-kind via confluence
    # merge (e.g. PATTERN_BULL_BULL_FLAG sitting on a WEEKLY_52W_HIGH).
    if _setup_has_kind(setup, _52W_HIGH_KINDS):
        chg = (bar["high"] - setup.trigger) / setup.trigger * 100
        lines = [
            f"<b>🏆 NEW 52-WEEK HIGH — {ticker}</b>",
            f"<code>${bar['high']:.2f}</code>  ({chg:+.2f}% past prior 52wh)",
            f"Prior high <code>${setup.trigger:.2f}</code>",
        ]
        # If a chart pattern sits on this level too, surface that for context
        if setup.kind.startswith("PATTERN_"):
            lines.append(f"<i>Confluence: {setup.label}</i>")
        footer = bar_time_et
    elif _setup_has_kind(setup, _52W_LOW_KINDS):
        chg = (bar["low"] - setup.trigger) / setup.trigger * 100
        lines = [
            f"<b>📉 NEW 52-WEEK LOW — {ticker}</b>",
            f"<code>${bar['low']:.2f}</code>  ({chg:+.2f}% past prior 52wl)",
            f"Prior low <code>${setup.trigger:.2f}</code>",
        ]
        if setup.kind.startswith("PATTERN_"):
            lines.append(f"<i>Confluence: {setup.label}</i>")
        footer = bar_time_et
    elif setup.is_touch:
        # Horizontal S/R or MA level touch — decision-point ping
        held = "resistance" if setup.direction == "BREAKOUT" else "support"
        lines = [
            f"<b>📍 LEVEL TOUCH — {ticker}</b>",
            f"Tagged {held} <b>{setup.label.upper()}</b> @ <code>${setup.trigger:.2f}</code>",
            f"Now <code>${bar['close']:.2f}</code>",
        ]
        footer = f"watch for bounce or break · {bar_time_et}"
    elif _is_channel_kind(setup.kind):
        # Channel rail touch — distinct from a breakout ping
        rail_side = "UPPER" if "UPPER" in setup.kind else ("LOWER" if "LOWER" in setup.kind else "RAIL")
        tf = "WEEKLY" if "WEEKLY" in setup.kind else "DAILY"
        slope = "ASCENDING" if "ASCENDING" in setup.kind else (
                "DESCENDING" if "DESCENDING" in setup.kind else "CHANNEL")
        lines = [
            f"<b>📍 CHANNEL TOUCH — {ticker}</b>",
            f"{tf} {slope} channel — {rail_side.lower()} rail @ <code>${setup.trigger:.2f}</code>",
            f"Now <code>${bar['close']:.2f}</code>",
        ]
        footer = f"watch for bounce or break · {bar_time_et}"
    else:
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
        ]
        footer = f"5-min bar close · {bar_time_et}"

    if vol_note:
        lines.append(vol_note)
    lines.append(f"<i>{footer}</i>")
    return "\n".join(lines)


def format_violent_move(
    ticker: str, bar: dict, chg_pct: float, vol_mult: float,
) -> str:
    """Big single-bar move alert — independent of any pre-curated level.
    Fires on vertical surges/drops the level watcher can't see."""
    bar_time_et = bar["ts"].astimezone(ET).strftime("%H:%M ET")
    if chg_pct >= 0:
        return "\n".join([
            f"<b>🚀 VIOLENT MOVE UP — {ticker}</b>",
            f"<code>${bar['close']:.2f}</code>  ({chg_pct:+.2f}% on the 5-min bar)",
            f"📊 {vol_mult:.1f}× avg 5m volume",
            f"<i>5-min bar close · {bar_time_et}</i>",
        ])
    return "\n".join([
        f"<b>🔻 VIOLENT DROP — {ticker}</b>",
        f"<code>${bar['close']:.2f}</code>  ({chg_pct:+.2f}% on the 5-min bar)",
        f"📊 {vol_mult:.1f}× avg 5m volume",
        f"<i>5-min bar close · {bar_time_et}</i>",
    ])


def format_refresh_digest(
    watchlist: dict[str, list[WatchSetup]],
    portfolio_meta: dict[str, dict],
    label: str, top_n: int = 6,
) -> str | None:
    """A 'state of the portfolio' digest sent at each (re)build: how many
    tickers are armed (have level setups) vs monitored (full portfolio),
    and the handful closest to firing, ranked by distance from build-time
    price to the trigger. The 'monitored' count includes quiet tickers
    that get violent-move coverage but no level alerts."""
    n_armed = len(watchlist)
    n_monitored = max(len(portfolio_meta), n_armed)
    n_levels = sum(len(v) for v in watchlist.values())
    n_quiet = max(n_monitored - n_armed, 0)

    # Collapse touch twins (same ticker+kind+trigger) so a level isn't
    # listed twice; keep the closest representative per level.
    seen: dict[tuple, float] = {}
    rows: list[tuple[float, str, str, str]] = []  # (dist_pct, glyph, ticker, label)
    for tk, setups in watchlist.items():
        for s in setups:
            if not s.last_price or s.last_price <= 0 or not s.trigger:
                continue
            key = (tk, s.kind, round(s.trigger, 2))
            if key in seen:
                continue
            seen[key] = 1.0
            dist = (s.trigger - s.last_price) / s.last_price * 100.0
            glyph = "▲" if s.direction == "BREAKOUT" else "▼"
            rows.append((abs(dist), glyph, tk, f"{s.label} {dist:+.1f}%"))

    if not rows:
        return None
    rows.sort(key=lambda x: x[0])
    quiet_note = f" ({n_quiet} quiet, violent-move only)" if n_quiet else ""
    head = (
        f"<b>🔄 WATCHLIST · {label} REFRESH</b>\n"
        f"<i>{n_armed} of {n_monitored} tickers armed{quiet_note} · {n_levels} levels</i>\n"
        f"Closest to firing:"
    )
    body = "\n".join(
        f"{g} <b>{tk}</b> — {lbl}" for _, g, tk, lbl in rows[:top_n]
    )
    return head + "\n" + body


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
    # 52-week high/low: pure wick check, no buffer. Also covers patterns
    # that absorbed a WEEKLY_52W_HIGH via confluence merge — without this
    # check, a Bull Flag sitting on top of a 52w high silently downgrades
    # to the close-buffer breakout path (caught STM today).
    if _setup_has_kind(setup, _52W_HIGH_KINDS):
        return bar["high"] > setup.trigger
    if _setup_has_kind(setup, _52W_LOW_KINDS):
        return bar["low"] < setup.trigger

    # Channel rail / level TOUCH — a genuine touch means price actually
    # TRADED at the level during this bar, i.e. the trigger falls inside the
    # bar's [low, high] range. A one-sided wick check (bar.high >= trigger)
    # is trivially always-true whenever price sits far from the level (e.g.
    # KLAC -4% but still well above a low rail), producing false touches.
    if _is_channel_kind(setup.kind) or setup.is_touch:
        return bar["low"] <= setup.trigger <= bar["high"]

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

def check_once(
    watchlist: dict[str, list[WatchSetup]],
    portfolio_meta: dict[str, dict] | None = None,
) -> int:
    """One pass per portfolio ticker. Returns alerts sent.

    Two independent alert paths run per ticker:
      • Level-cross: only for tickers with armed setups (watchlist entries)
      • Violent-move: runs on EVERY portfolio ticker (uses portfolio_meta
        for avg_5m_vol), so a vertical bar on a 'quiet' name still pings."""
    state = load_state()
    already = set(state.get("alerted", []))
    sent = 0
    today_et = datetime.now(ET).strftime("%Y-%m-%d")
    portfolio_meta = portfolio_meta or {}

    # Iterate every monitored ticker (union of watchlist + portfolio_meta).
    # Quiet tickers (no setups) only do the violent-move check.
    all_tickers = sorted(set(watchlist) | set(portfolio_meta))

    for ticker in all_tickers:
        setups = watchlist.get(ticker) or []
        meta = portfolio_meta.get(ticker) or {}

        vm_up_key   = f"{ticker}|VIOLENT_MOVE|UP|{today_et}"
        vm_down_key = f"{ticker}|VIOLENT_MOVE|DOWN|{today_et}"
        levels_done = (not setups) or all(alert_key(ticker, s, today_et) in already for s in setups)
        vm_done = vm_up_key in already and vm_down_key in already
        if levels_done and vm_done:
            continue

        # One fetch per ticker — bars[0] is the latest, bars[1] is the
        # prior bar (used as the baseline for violent-move % change).
        bars = fetch_recent_5m_bars(ticker, n=3)
        if not bars:
            continue
        bar = bars[0]
        prev_close = float(bars[1]["close"]) if len(bars) >= 2 else None
        # Prefer the per-setup stamp (more recent if setups exist), fall
        # back to portfolio_meta for quiet tickers.
        avg_5m_vol = setups[0].avg_5m_vol if setups else float(meta.get("avg_5m_vol") or 0.0)

        # ── Level-cross alerts ─────────────────────────────────────────
        for s in setups:
            key = alert_key(ticker, s, today_et)
            if key in already:
                continue
            if is_cross(bar, s):
                vol_note = _volume_context(ticker, s.avg_5m_vol, bar)
                if _send_telegram(format_alert(ticker, s, bar, vol_note)):
                    already.add(key)
                    sent += 1
                    print(f"[alert] {ticker} {s.direction} {s.kind} @ ${s.trigger:.2f}")
                else:
                    print(
                        f"[retry] {ticker} {s.kind} delivery failed — will retry next poll",
                        file=sys.stderr,
                    )

        # ── Violent-move alert (independent of pre-curated levels) ─────
        if prev_close and prev_close > 0 and avg_5m_vol > 0:
            chg_pct = (bar["close"] - prev_close) / prev_close * 100.0
            vol_mult = bar["volume"] / avg_5m_vol
            if abs(chg_pct) >= VIOLENT_MOVE_PCT and vol_mult >= VIOLENT_MOVE_VOL_MULT:
                vm_key = vm_up_key if chg_pct > 0 else vm_down_key
                if vm_key not in already:
                    if _send_telegram(format_violent_move(ticker, bar, chg_pct, vol_mult)):
                        already.add(vm_key)
                        sent += 1
                        print(f"[alert] {ticker} VIOLENT_MOVE {chg_pct:+.2f}% on {vol_mult:.1f}× vol")
                    else:
                        print(
                            f"[retry] {ticker} violent-move delivery failed — will retry next poll",
                            file=sys.stderr,
                        )

    if sent:
        state["alerted"] = sorted(already)
        save_state(state)
    return sent


def ensure_today_watchlist(
    portfolio: list[str],
) -> tuple[dict[str, list[WatchSetup]], dict[str, dict]]:
    """Load the persisted watchlist; regenerate it if missing or stale."""
    watchlist, meta, generated_at = load_watchlist()
    today_et = datetime.now(ET).date().isoformat()
    fresh = False
    if generated_at:
        try:
            gen_date = datetime.fromisoformat(generated_at).astimezone(ET).date().isoformat()
            fresh = (gen_date == today_et)
        except ValueError:
            fresh = False
    if not fresh or not watchlist:
        watchlist, meta = build_watchlist(portfolio)
        save_watchlist(watchlist, meta)
        print(f"[watchlist] generated {sum(len(v) for v in watchlist.values())} levels across {len(watchlist)} armed of {len(meta)} monitored")
    else:
        n_levels = sum(len(v) for v in watchlist.values())
        print(f"[watchlist] using cached: {n_levels} levels across {len(watchlist)} armed of {len(meta)} monitored")
    return watchlist, meta


def _force_rebuild_watchlist(
    portfolio: list[str],
) -> tuple[dict[str, list[WatchSetup]], dict[str, dict]]:
    print("[watchlist] mid-day rebuild — re-scanning to catch newly-formed setups")
    watchlist, meta = build_watchlist(portfolio)
    save_watchlist(watchlist, meta)
    print(f"[watchlist] rebuilt: {sum(len(v) for v in watchlist.values())} levels across {len(watchlist)} armed of {len(meta)} monitored")
    return watchlist, meta


def main_loop() -> int:
    portfolio = load_portfolio()
    if not portfolio:
        print("[fatal] no portfolio configured", file=sys.stderr)
        return 1
    print(f"[start] intraday watcher | poll {POLL_SEC}s | portfolio: {portfolio}")

    watchlist: dict[str, list[WatchSetup]] = {}
    portfolio_meta: dict[str, dict] = {}
    rebuilt_hours_today: set[int] = set()
    digests_sent_today: set[str] = set()

    def _send_digest(label: str) -> None:
        msg = format_refresh_digest(watchlist, portfolio_meta, label)
        if msg and _send_telegram(msg):
            print(f"[digest] sent {label} refresh digest")

    while True:
        try:
            now = datetime.now(timezone.utc)
            state = market_state(now)

            if state in ("weekend", "closed_weekday", "pre"):
                wait = min(seconds_until_open(now), 3600)  # cap at 1 hour
                print(f"[sleep] market state={state}, sleeping {int(wait)}s")
                # New trading day starts after a sleep through the close — clear
                # the rebuild/digest bookkeeping so tomorrow's fire.
                rebuilt_hours_today.clear()
                digests_sent_today.clear()
                time.sleep(max(wait, 60))
                continue

            today_key = datetime.now(ET).strftime("%Y-%m-%d")

            if state == "opening_skip":
                # Pre-build today's watchlist while we wait through the noisy open
                if not watchlist:
                    watchlist, portfolio_meta = ensure_today_watchlist(portfolio)
                # Morning "here's the day's board" digest — once per day
                dk = f"{today_key}|OPEN"
                if watchlist and dk not in digests_sent_today:
                    _send_digest("MARKET OPEN")
                    digests_sent_today.add(dk)
                time.sleep(POLL_SEC)
                continue

            # state == "open"
            watchlist, portfolio_meta = ensure_today_watchlist(portfolio)

            # Cover the case where the watcher first comes up after the
            # opening-skip window (deploy mid-session): still send the
            # day's first digest once.
            dk_open = f"{today_key}|OPEN"
            if watchlist and dk_open not in digests_sent_today:
                _send_digest("MARKET OPEN")
                digests_sent_today.add(dk_open)

            # Mid-day rebuild: at the configured hours (12:00 ET, 14:00 ET) do
            # a fresh build so setups that only formed during the session get
            # picked up. Once per hour, dedup'd via rebuilt_hours_today.
            now_et = now.astimezone(ET)
            for hr in MIDDAY_REBUILD_HOURS:
                if now_et.hour == hr and hr not in rebuilt_hours_today:
                    watchlist, portfolio_meta = _force_rebuild_watchlist(portfolio)
                    rebuilt_hours_today.add(hr)
                    dk_mid = f"{today_key}|{hr}"
                    if dk_mid not in digests_sent_today:
                        _send_digest(f"{hr}:00 ET")
                        digests_sent_today.add(dk_mid)
                    break

            sent = check_once(watchlist, portfolio_meta)
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
        watchlist, portfolio_meta = ensure_today_watchlist(portfolio)
        sent = check_once(watchlist, portfolio_meta)
        print(f"[done] {sent} alert(s) sent")
        return 0
    return main_loop()


if __name__ == "__main__":
    sys.exit(main())
