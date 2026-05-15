"""Setup discovery — enumerate every plausible FUTURE breakout/breakdown trigger.

Replaces the single-verdict grading model with a multi-setup map. For one ticker
we scan the daily + weekly chart and emit a list of `Setup` records, each
describing one potential future trigger with: direction, kind, trigger price,
invalidation, target, distance, maturity, tests, and a quality score.

Setup sources:
  • Donchian highs/lows at multiple lookbacks (20/50/100/252)
  • Moving-average pivots (20/50/200) — reclaim if below, loss if above
  • Detected chart patterns (daily + weekly) from patterns.py
  • Horizontal swing pivots tested 2+ times within ATR tolerance
  • Weekly Donchian on resampled weekly bars
  • Open / unfilled gaps above + below current price
  • Bollinger-squeeze resolution levels

After collection, setups whose trigger levels overlap within ATR tolerance are
merged into confluence groups — confluence boosts quality. Each setup is then
scored 0-100 from maturity, tests, proximity, trend alignment, and RS context.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────

Direction = Literal["BREAKOUT", "BREAKDOWN"]
ProximityClass = Literal["IMMEDIATE", "NEAR", "WATCH", "FAR"]
QualityLabel = Literal["STRONG", "MODERATE", "EARLY", "WEAK"]


@dataclass
class Setup:
    direction: Direction
    kind: str
    label: str
    trigger_price: float
    trigger_condition: str
    invalidation_price: Optional[float]
    target_price: Optional[float]
    distance_pct: float            # signed: positive = trigger above current, negative = below
    distance_atr: Optional[float]  # ATRs from current price (always >= 0)
    proximity_class: ProximityClass
    maturity_bars: Optional[int]
    tests: Optional[int]
    quality: float
    quality_label: QualityLabel
    rationale: list[str] = field(default_factory=list)
    timeframe: str = "daily"
    confluence_with: list[str] = field(default_factory=list)
    # None = pending future setup. 0 = triggered today. 1 = triggered yesterday, etc.
    # Setups with a non-None bars_since_trigger are routed to result["triggered"]
    # instead of the pending breakouts/breakdowns lists.
    bars_since_trigger: Optional[int] = None
    # For diagonal trendline setups: {"x0", "y0", "x1", "y1"} (timestamps + prices)
    # tells the UI to render the level as a sloped line rather than a horizontal
    # hline. None = render as horizontal (the default).
    line_meta: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_triggered(self) -> bool:
        return self.bars_since_trigger is not None


# ─────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────

MAX_DISTANCE_PCT = 20.0       # drop setups farther than this from current price
ATR_TOLERANCE_MERGE = 0.6     # setups within this many ATRs are merged

PROX_IMMEDIATE = 1.0
PROX_NEAR = 3.0
PROX_WATCH = 7.0


# ─────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1:
        return float("nan")
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def _proximity_class(distance_pct: float) -> ProximityClass:
    ad = abs(distance_pct)
    if ad < PROX_IMMEDIATE:
        return "IMMEDIATE"
    if ad < PROX_NEAR:
        return "NEAR"
    if ad < PROX_WATCH:
        return "WATCH"
    return "FAR"


def _quality_label(score: float) -> QualityLabel:
    if score >= 70:
        return "STRONG"
    if score >= 50:
        return "MODERATE"
    if score >= 30:
        return "EARLY"
    return "WEAK"


def _distances(trigger: float, last: float, atr: float) -> tuple[float, Optional[float]]:
    """Returns (signed distance_pct, distance_atr)."""
    if last <= 0:
        return 0.0, None
    dist_pct = (trigger - last) / last * 100
    dist_atr = abs(trigger - last) / atr if atr and atr == atr and atr > 0 else None
    return round(dist_pct, 2), None if dist_atr is None else round(dist_atr, 2)


def _swing_highs(highs: np.ndarray, distance: int = 3) -> list[int]:
    idx = []
    n = len(highs)
    for i in range(distance, n - distance):
        window = highs[i - distance:i + distance + 1]
        if highs[i] == window.max() and (window == highs[i]).sum() == 1:
            idx.append(i)
    return idx


def _swing_lows(lows: np.ndarray, distance: int = 3) -> list[int]:
    idx = []
    n = len(lows)
    for i in range(distance, n - distance):
        window = lows[i - distance:i + distance + 1]
        if lows[i] == window.min() and (window == lows[i]).sum() == 1:
            idx.append(i)
    return idx


# ─────────────────────────────────────────────────
# Setup source: Donchian channels at multiple lookbacks
# ─────────────────────────────────────────────────

def _count_touches_high(df: pd.DataFrame, level: float, atr: float, lookback: int) -> tuple[int, int]:
    """Returns (touches, bars_since_set) for an upper level."""
    if atr <= 0 or len(df) < 2:
        return 0, 0
    window = df.iloc[-lookback:] if len(df) > lookback else df
    tol = max(atr * 0.4, level * 0.005)
    highs = window["High"].to_numpy(dtype=float)
    touches = int(((highs >= level - tol) & (highs <= level + tol)).sum())
    # bars_since_set: distance back to the most recent bar whose high equalled the level
    bars_since = 0
    for i in range(len(window) - 1, -1, -1):
        if window["High"].iloc[i] >= level - 1e-9:
            bars_since = len(window) - 1 - i
            break
    return touches, bars_since


def _count_touches_low(df: pd.DataFrame, level: float, atr: float, lookback: int) -> tuple[int, int]:
    if atr <= 0 or len(df) < 2:
        return 0, 0
    window = df.iloc[-lookback:] if len(df) > lookback else df
    tol = max(atr * 0.4, level * 0.005)
    lows = window["Low"].to_numpy(dtype=float)
    touches = int(((lows >= level - tol) & (lows <= level + tol)).sum())
    bars_since = 0
    for i in range(len(window) - 1, -1, -1):
        if window["Low"].iloc[i] <= level + 1e-9:
            bars_since = len(window) - 1 - i
            break
    return touches, bars_since


def _find_recent_cross_above(
    df: pd.DataFrame, period: int, max_lookback: int = 5,
) -> tuple[Optional[int], Optional[float]]:
    """Find the most recent bar where close FIRST crossed above its prior-N-day high.

    Walks back up to max_lookback bars from the latest bar. For each candidate
    bar, requires:
        close[bar] > max(High[bar-period : bar])      # this bar is above
        AND close[bar-1] <= max(High[bar-period-1 : bar-1])   # prior bar wasn't

    Returns (bars_back_from_latest, level_that_was_crossed) or (None, None).
    """
    if len(df) < period + 2:
        return None, None
    for back in range(max_lookback + 1):
        idx = len(df) - 1 - back
        if idx < period + 1:
            return None, None
        prior_hi = float(df["High"].iloc[idx - period:idx].max())
        cur_close = float(df["Close"].iloc[idx])
        if cur_close <= prior_hi:
            continue
        prev_close = float(df["Close"].iloc[idx - 1])
        prev_prior_hi = float(df["High"].iloc[idx - period - 1:idx - 1].max())
        if prev_close <= prev_prior_hi:
            return back, prior_hi
    return None, None


def _find_recent_cross_below(
    df: pd.DataFrame, period: int, max_lookback: int = 5,
) -> tuple[Optional[int], Optional[float]]:
    """Mirror of _find_recent_cross_above for breakdowns."""
    if len(df) < period + 2:
        return None, None
    for back in range(max_lookback + 1):
        idx = len(df) - 1 - back
        if idx < period + 1:
            return None, None
        prior_lo = float(df["Low"].iloc[idx - period:idx].min())
        cur_close = float(df["Close"].iloc[idx])
        if cur_close >= prior_lo:
            continue
        prev_close = float(df["Close"].iloc[idx - 1])
        prev_prior_lo = float(df["Low"].iloc[idx - period - 1:idx - 1].min())
        if prev_close >= prev_prior_lo:
            return back, prior_lo
    return None, None


def _bars_since_level_cross(
    df: pd.DataFrame, level: float, direction: str, max_lookback: int = 60,
) -> Optional[int]:
    """Bars since Close FIRST crossed `level` in `direction` (the transition
    bar where prev was on one side and current is on the other). Returns the
    bars-back count, or None if no clean transition within max_lookback —
    meaning price has been beyond the level longer than the window, i.e. the
    breakout is sustained/old, not a fresh same-day event.

    This replaces hardcoding bars_since_trigger=0 for "Breaking out" patterns,
    which made a multi-day consolidation around a trigger re-stamp as a brand
    new breakout on every daily scan.
    """
    closes = df["Close"]
    n = len(closes)
    if n < 2:
        return None
    for back in range(max_lookback + 1):
        idx = n - 1 - back
        if idx < 1:
            return None
        cur = float(closes.iloc[idx])
        prev = float(closes.iloc[idx - 1])
        if direction == "BREAKOUT":
            if cur > level and prev <= level:
                return back
        else:  # BREAKDOWN
            if cur < level and prev >= level:
                return back
    return None


def find_donchian_setups(df: pd.DataFrame, atr: float, periods: list[int]) -> list[Setup]:
    """Emit Donchian setups using the PRIOR N-day window (excluding today).

    If today's close has already cleared the prior window's high (or low), the
    setup is emitted with bars_since_trigger set so the UI can route it to the
    'recently triggered' bucket instead of re-flagging it as a future setup.
    """
    if df.empty:
        return []
    last = float(df["Close"].iloc[-1])
    setups: list[Setup] = []

    for period in periods:
        if len(df) < period + 2:
            continue
        # Prior window excludes today's bar — this is the level that
        # was meaningful before today's session printed.
        prior_window = df.iloc[-period - 1:-1]
        prior_hi = float(prior_window["High"].max())
        prior_lo = float(prior_window["Low"].min())
        rng = prior_hi - prior_lo if prior_hi > prior_lo else prior_hi * 0.1

        # ── Breakout side ─────────────────────────────────────────
        if last <= prior_hi:
            # Pending future breakout at prior_hi.
            d_pct, d_atr = _distances(prior_hi, last, atr)
            if abs(d_pct) <= MAX_DISTANCE_PCT:
                touches, _ = _count_touches_high(df.iloc[:-1], prior_hi, atr, period)
                setups.append(Setup(
                    direction="BREAKOUT",
                    kind=f"DONCHIAN_{period}_HIGH",
                    label=f"{period}-day high",
                    trigger_price=round(prior_hi, 2),
                    trigger_condition=f"Close above ${prior_hi:.2f} on >1.5x volume, upper-third close",
                    invalidation_price=round(prior_lo, 2),
                    target_price=round(prior_hi + rng, 2),
                    distance_pct=d_pct,
                    distance_atr=d_atr,
                    proximity_class=_proximity_class(d_pct),
                    maturity_bars=period,
                    tests=touches,
                    quality=0.0,
                    quality_label="WEAK",
                    rationale=[
                        f"{period}-day Donchian high",
                        f"Level tested {touches}× in window" if touches > 0 else "Fresh high",
                    ],
                ))
        else:
            # Today is above prior_hi — find when the first close crossed and
            # surface it as a triggered event (not a future setup).
            bars_back, level = _find_recent_cross_above(df, period, max_lookback=5)
            if bars_back is not None and level is not None:
                d_pct, d_atr = _distances(level, last, atr)
                if abs(d_pct) <= MAX_DISTANCE_PCT:
                    when = "today" if bars_back == 0 else (
                        "yesterday" if bars_back == 1 else f"{bars_back} bars ago"
                    )
                    setups.append(Setup(
                        direction="BREAKOUT",
                        kind=f"DONCHIAN_{period}_HIGH",
                        label=f"{period}-day high cleared",
                        trigger_price=round(level, 2),
                        trigger_condition=f"Cleared ${level:.2f} {when} — watch for hold",
                        invalidation_price=round(level, 2),
                        target_price=round(level + rng, 2),
                        distance_pct=d_pct,
                        distance_atr=d_atr,
                        proximity_class=_proximity_class(d_pct),
                        maturity_bars=period,
                        tests=None,
                        quality=0.0,
                        quality_label="WEAK",
                        rationale=[f"{period}-day Donchian high cleared {when}"],
                        bars_since_trigger=bars_back,
                    ))

        # ── Breakdown side ────────────────────────────────────────
        if last >= prior_lo:
            d_pct, d_atr = _distances(prior_lo, last, atr)
            if abs(d_pct) <= MAX_DISTANCE_PCT:
                touches, _ = _count_touches_low(df.iloc[:-1], prior_lo, atr, period)
                setups.append(Setup(
                    direction="BREAKDOWN",
                    kind=f"DONCHIAN_{period}_LOW",
                    label=f"{period}-day low",
                    trigger_price=round(prior_lo, 2),
                    trigger_condition=f"Close below ${prior_lo:.2f} on >1.5x volume",
                    invalidation_price=round(prior_hi, 2),
                    target_price=round(prior_lo - rng, 2),
                    distance_pct=d_pct,
                    distance_atr=d_atr,
                    proximity_class=_proximity_class(d_pct),
                    maturity_bars=period,
                    tests=touches,
                    quality=0.0,
                    quality_label="WEAK",
                    rationale=[
                        f"{period}-day Donchian low",
                        f"Level tested {touches}× in window" if touches > 0 else "Fresh low",
                    ],
                ))
        else:
            bars_back, level = _find_recent_cross_below(df, period, max_lookback=5)
            if bars_back is not None and level is not None:
                d_pct, d_atr = _distances(level, last, atr)
                if abs(d_pct) <= MAX_DISTANCE_PCT:
                    when = "today" if bars_back == 0 else (
                        "yesterday" if bars_back == 1 else f"{bars_back} bars ago"
                    )
                    setups.append(Setup(
                        direction="BREAKDOWN",
                        kind=f"DONCHIAN_{period}_LOW",
                        label=f"{period}-day low broken",
                        trigger_price=round(level, 2),
                        trigger_condition=f"Lost ${level:.2f} {when} — watch for follow-through",
                        invalidation_price=round(level, 2),
                        target_price=round(level - rng, 2),
                        distance_pct=d_pct,
                        distance_atr=d_atr,
                        proximity_class=_proximity_class(d_pct),
                        maturity_bars=period,
                        tests=None,
                        quality=0.0,
                        quality_label="WEAK",
                        rationale=[f"{period}-day Donchian low broken {when}"],
                        bars_since_trigger=bars_back,
                    ))
    return setups


# ─────────────────────────────────────────────────
# Setup source: moving-average pivots
# ─────────────────────────────────────────────────

def _ma_slope_pct(ma: pd.Series, period: int = 10) -> float:
    if len(ma.dropna()) < period + 1:
        return 0.0
    s = ma.dropna()
    return float((s.iloc[-1] / s.iloc[-period] - 1) * 100)


def _count_ma_tests(df: pd.DataFrame, ma: pd.Series, atr: float, lookback: int = 60) -> int:
    if atr <= 0:
        return 0
    n = min(lookback, len(df))
    tail_df = df.iloc[-n:]
    tail_ma = ma.iloc[-n:]
    tol = atr * 0.4
    touches = 0
    for i in range(len(tail_df)):
        m = float(tail_ma.iloc[i]) if not pd.isna(tail_ma.iloc[i]) else None
        if m is None:
            continue
        h = float(tail_df["High"].iloc[i])
        l = float(tail_df["Low"].iloc[i])
        if l <= m + tol and h >= m - tol:
            touches += 1
    return touches


def find_ma_setups(df: pd.DataFrame, atr: float, periods: list[int]) -> list[Setup]:
    if df.empty:
        return []
    last = float(df["Close"].iloc[-1])
    setups: list[Setup] = []
    close = df["Close"]
    for period in periods:
        if len(close) < period + 11:
            continue
        ma = close.rolling(period).mean()
        ma_now = float(ma.iloc[-1])
        if not (ma_now == ma_now):  # NaN check
            continue
        slope = _ma_slope_pct(ma, period=10)
        touches = _count_ma_tests(df, ma, atr)
        d_pct, d_atr = _distances(ma_now, last, atr)
        if abs(d_pct) > MAX_DISTANCE_PCT:
            continue
        if last < ma_now:
            # Reclaim setup (breakout)
            stretch = atr if atr > 0 else ma_now * 0.02
            setups.append(Setup(
                direction="BREAKOUT",
                kind=f"MA_{period}_RECLAIM",
                label=f"{period}-day MA reclaim",
                trigger_price=round(ma_now, 2),
                trigger_condition=f"Daily close above {period}-MA (${ma_now:.2f})",
                invalidation_price=round(ma_now - stretch * 1.5, 2),
                target_price=None,
                distance_pct=d_pct,
                distance_atr=d_atr,
                proximity_class=_proximity_class(d_pct),
                maturity_bars=period,
                tests=touches,
                quality=0.0,
                quality_label="WEAK",
                rationale=[
                    f"{period}-MA at ${ma_now:.2f}, slope {slope:+.1f}%",
                    f"{touches} touches in last 60 bars" if touches else "untested recently",
                ],
            ))
        else:
            # Loss setup (breakdown)
            stretch = atr if atr > 0 else ma_now * 0.02
            setups.append(Setup(
                direction="BREAKDOWN",
                kind=f"MA_{period}_LOSS",
                label=f"{period}-day MA loss",
                trigger_price=round(ma_now, 2),
                trigger_condition=f"Daily close below {period}-MA (${ma_now:.2f})",
                invalidation_price=round(ma_now + stretch * 1.5, 2),
                target_price=None,
                distance_pct=d_pct,
                distance_atr=d_atr,
                proximity_class=_proximity_class(d_pct),
                maturity_bars=period,
                tests=touches,
                quality=0.0,
                quality_label="WEAK",
                rationale=[
                    f"{period}-MA at ${ma_now:.2f}, slope {slope:+.1f}%",
                    f"{touches} touches in last 60 bars" if touches else "untested recently",
                ],
            ))
    return setups


# ─────────────────────────────────────────────────
# Setup source: detected patterns
# ─────────────────────────────────────────────────

_BEARISH_PATTERNS = {
    "Bear Flag", "Descending Triangle", "Inv Cup & Handle", "Double Top",
    "H&S Top", "Distribution VCP", "HT Bear Flag", "Rising Wedge", "Bear Pennant",
}


# Pattern staleness thresholds (calendar days). A pattern whose end_date is
# older than this is considered stale — the chart has moved on and the
# pattern's geometric breakout_level is no longer current resistance.
PATTERN_STALE_DAILY_DAYS = 14
PATTERN_STALE_WEEKLY_DAYS = 28


def find_pattern_setups(pattern_results, df: pd.DataFrame, atr: float) -> list[Setup]:
    """Pattern-derived setups with staleness + status filtering.

    Filters applied:
      • Drop status == "Confirmed" — already past the trigger, history.
      • Drop patterns whose end_date is older than the staleness threshold:
            daily patterns: 14 calendar days
            weekly patterns: 28 calendar days
        Stale patterns reference geometry that's no longer the current pivot.
      • Route status == "Breaking out" patterns to triggered (bars_since_trigger=0)
        — the trigger has just been crossed, surface as "just happened" not pending.
    """
    if df is None or df.empty:
        return []
    last = float(df["Close"].iloc[-1])
    last_date = df.index[-1]
    setups: list[Setup] = []
    for p in pattern_results or []:
        if p.status == "Confirmed":
            continue
        weekly = "[weekly]" in (p.notes or "")
        timeframe = "weekly" if weekly else "daily"
        threshold = PATTERN_STALE_WEEKLY_DAYS if weekly else PATTERN_STALE_DAILY_DAYS
        try:
            age_days = (last_date - p.end_date).days
        except Exception:
            age_days = 0
        if age_days > threshold:
            continue

        is_bear = p.pattern in _BEARISH_PATTERNS
        direction: Direction = "BREAKDOWN" if is_bear else "BREAKOUT"
        trigger = float(p.breakout_level)
        d_pct, d_atr = _distances(trigger, last, atr)
        if abs(d_pct) > MAX_DISTANCE_PCT:
            continue
        # Skip patterns whose trigger has already been cleared meaningfully
        # in our direction. The "Breaking out" route below catches the
        # narrow band right around the trigger.
        if direction == "BREAKOUT" and trigger < last * 0.99:
            continue
        if direction == "BREAKDOWN" and trigger > last * 1.01:
            continue

        eff_conf = float(p.confidence)
        if weekly:
            eff_conf = min(1.0, eff_conf * 1.2)
        rationale = [
            f"{p.pattern} ({p.status})",
            f"{eff_conf:.0%} effective confidence",
            f"end_date {age_days}d ago",
        ]
        if weekly:
            rationale.append("weekly timeframe — higher-conviction")
        if p.notes and "rescored" in p.notes:
            rationale.append("distant pattern — early-forming")

        if p.status == "Breaking out":
            # Pin to the ACTUAL cross bar instead of hardcoding 0. A pattern
            # can stay "Breaking out" (price within 1% of the trigger) for
            # many sessions; without this it re-stamps as a fresh same-day
            # breakout every scan and monopolises the headline.
            bst = _bars_since_level_cross(df, trigger, direction, max_lookback=60)
            if bst is None:
                # No clean close transition in 60 bars — price has held
                # beyond the trigger for a long time. Treat as a stale
                # trigger (use pattern age, floored above the fresh/recent
                # window) so it never shows as FRESH BREAKOUT today.
                bst = max(age_days, 6)
            bars_since_trigger = bst
        else:
            bars_since_trigger = None

        setups.append(Setup(
            direction=direction,
            kind=f"PATTERN_{'BEAR' if is_bear else 'BULL'}_{p.pattern.replace(' ', '_').upper()}",
            label=f"{'Weekly ' if weekly else ''}{p.pattern}",
            trigger_price=round(trigger, 2),
            trigger_condition=(
                f"{'Close above' if direction == 'BREAKOUT' else 'Close below'} "
                f"${trigger:.2f} ({p.pattern} trigger)"
                if bars_since_trigger is None else
                f"Cleared ${trigger:.2f} on the {p.pattern} trigger — watch for hold"
            ),
            invalidation_price=None,
            target_price=round(float(p.target), 2) if p.target else None,
            distance_pct=d_pct,
            distance_atr=d_atr,
            proximity_class=_proximity_class(d_pct),
            maturity_bars=None,
            tests=None,
            quality=0.0,
            quality_label="WEAK",
            rationale=rationale,
            timeframe=timeframe,
            bars_since_trigger=bars_since_trigger,
        ))
    return setups


# ─────────────────────────────────────────────────
# Setup source: horizontal pivot clusters
# ─────────────────────────────────────────────────

def find_horizontal_setups(df: pd.DataFrame, atr: float, lookback: int = 500) -> list[Setup]:
    """Multi-touch horizontal pivot levels.

    Lookback extended to 500 bars (~2yr) so swing highs/lows from a year+ ago
    surface as long-held levels. Each cluster's `maturity_bars` is the age of
    the OLDEST swing point in the cluster, so the quality scorer can reward
    levels that have stood for a long time.
    """
    if df.empty or atr <= 0:
        return []
    window = df.iloc[-lookback:] if len(df) > lookback else df
    if len(window) < 30:
        return []
    last = float(df["Close"].iloc[-1])
    n = len(window)
    today_x = n - 1

    highs = window["High"].to_numpy(dtype=float)
    lows = window["Low"].to_numpy(dtype=float)
    sh = _swing_highs(highs)
    sl = _swing_lows(lows)
    tol = atr * 0.5

    # Carry indices so we can track per-cluster age (oldest swing).
    sh_pts = sorted([(float(highs[i]), int(i)) for i in sh], key=lambda p: p[0])
    sl_pts = sorted([(float(lows[i]), int(i)) for i in sl], key=lambda p: p[0])

    def cluster(pts: list[tuple[float, int]], side: str):
        out: list[tuple[float, int, str, int]] = []   # (level, touches, side, oldest_x)
        i = 0
        while i < len(pts):
            j = i
            group_prices = [pts[i][0]]
            group_idxs = [pts[i][1]]
            while j + 1 < len(pts) and pts[j + 1][0] - pts[i][0] <= tol:
                group_prices.append(pts[j + 1][0])
                group_idxs.append(pts[j + 1][1])
                j += 1
            if len(group_prices) >= 2:
                out.append((
                    sum(group_prices) / len(group_prices),
                    len(group_prices),
                    side,
                    min(group_idxs),
                ))
            i = j + 1
        return out

    pivot_levels: list[tuple[float, int, str, int]] = []
    pivot_levels += cluster(sh_pts, "high")
    pivot_levels += cluster(sl_pts, "low")

    setups: list[Setup] = []
    for level, touches, side, oldest_x in pivot_levels:
        d_pct, d_atr = _distances(level, last, atr)
        if abs(d_pct) > MAX_DISTANCE_PCT:
            continue
        age = today_x - oldest_x
        if level > last * 1.001 and side == "high":
            setups.append(Setup(
                direction="BREAKOUT",
                kind="HORIZONTAL_RESISTANCE",
                label=f"Resistance shelf ({touches}× tested)",
                trigger_price=round(level, 2),
                trigger_condition=f"Close above ${level:.2f} with confirming volume",
                invalidation_price=None,
                target_price=None,
                distance_pct=d_pct,
                distance_atr=d_atr,
                proximity_class=_proximity_class(d_pct),
                maturity_bars=age,
                tests=touches,
                quality=0.0,
                quality_label="WEAK",
                rationale=[
                    f"{touches} swing-high rejections at this level",
                    f"Oldest test {age} bars ago",
                ],
            ))
        elif level < last * 0.999 and side == "low":
            setups.append(Setup(
                direction="BREAKDOWN",
                kind="HORIZONTAL_SUPPORT",
                label=f"Support shelf ({touches}× tested)",
                trigger_price=round(level, 2),
                trigger_condition=f"Close below ${level:.2f} with confirming volume",
                invalidation_price=None,
                target_price=None,
                distance_pct=d_pct,
                distance_atr=d_atr,
                proximity_class=_proximity_class(d_pct),
                maturity_bars=age,
                tests=touches,
                quality=0.0,
                quality_label="WEAK",
                rationale=[
                    f"{touches} swing-low bounces at this level",
                    f"Oldest test {age} bars ago",
                ],
            ))
    return setups


# ─────────────────────────────────────────────────
# Setup source: weekly Donchian
# ─────────────────────────────────────────────────

def find_weekly_setups(df_weekly: pd.DataFrame, atr_daily: float) -> list[Setup]:
    """Same prior-window logic as daily Donchian, applied to weekly bars.

    If this week's close already cleared the prior-N-week high (or low), the
    setup is emitted with bars_since_trigger (in weeks) so the UI can mark it
    as a recently-triggered event rather than re-flag it as a future setup.
    """
    if df_weekly is None or df_weekly.empty or len(df_weekly) < 21:
        return []
    last = float(df_weekly["Close"].iloc[-1])
    setups: list[Setup] = []
    for period_w, label in [(20, "20-week"), (52, "52-week")]:
        if len(df_weekly) < period_w + 2:
            continue
        prior_window = df_weekly.iloc[-period_w - 1:-1]
        prior_hi = float(prior_window["High"].max())
        prior_lo = float(prior_window["Low"].min())
        rng = prior_hi - prior_lo if prior_hi > prior_lo else prior_hi * 0.1

        # Breakout
        if last <= prior_hi:
            d_pct, d_atr = _distances(prior_hi, last, atr_daily)
            if abs(d_pct) <= MAX_DISTANCE_PCT:
                setups.append(Setup(
                    direction="BREAKOUT",
                    kind=f"WEEKLY_{period_w}W_HIGH",
                    label=f"{label} high",
                    trigger_price=round(prior_hi, 2),
                    trigger_condition=f"Weekly close above ${prior_hi:.2f}",
                    invalidation_price=round(prior_lo, 2),
                    target_price=round(prior_hi + rng, 2),
                    distance_pct=d_pct,
                    distance_atr=d_atr,
                    proximity_class=_proximity_class(d_pct),
                    maturity_bars=period_w,
                    tests=None,
                    quality=0.0,
                    quality_label="WEAK",
                    rationale=[f"{label} Donchian high — higher-timeframe pivot"],
                    timeframe="weekly",
                ))
        else:
            bars_back, level = _find_recent_cross_above(df_weekly, period_w, max_lookback=3)
            if bars_back is not None and level is not None:
                d_pct, d_atr = _distances(level, last, atr_daily)
                if abs(d_pct) <= MAX_DISTANCE_PCT:
                    when = "this week" if bars_back == 0 else (
                        "last week" if bars_back == 1 else f"{bars_back} weeks ago"
                    )
                    setups.append(Setup(
                        direction="BREAKOUT",
                        kind=f"WEEKLY_{period_w}W_HIGH",
                        label=f"{label} high cleared",
                        trigger_price=round(level, 2),
                        trigger_condition=f"Cleared ${level:.2f} {when} — watch for weekly hold",
                        invalidation_price=round(level, 2),
                        target_price=round(level + rng, 2),
                        distance_pct=d_pct,
                        distance_atr=d_atr,
                        proximity_class=_proximity_class(d_pct),
                        maturity_bars=period_w,
                        tests=None,
                        quality=0.0,
                        quality_label="WEAK",
                        rationale=[f"{label} Donchian high cleared {when}"],
                        timeframe="weekly",
                        bars_since_trigger=bars_back,
                    ))

        # Breakdown
        if last >= prior_lo:
            d_pct, d_atr = _distances(prior_lo, last, atr_daily)
            if abs(d_pct) <= MAX_DISTANCE_PCT:
                setups.append(Setup(
                    direction="BREAKDOWN",
                    kind=f"WEEKLY_{period_w}W_LOW",
                    label=f"{label} low",
                    trigger_price=round(prior_lo, 2),
                    trigger_condition=f"Weekly close below ${prior_lo:.2f}",
                    invalidation_price=round(prior_hi, 2),
                    target_price=round(prior_lo - rng, 2),
                    distance_pct=d_pct,
                    distance_atr=d_atr,
                    proximity_class=_proximity_class(d_pct),
                    maturity_bars=period_w,
                    tests=None,
                    quality=0.0,
                    quality_label="WEAK",
                    rationale=[f"{label} Donchian low — higher-timeframe pivot"],
                    timeframe="weekly",
                ))
        else:
            bars_back, level = _find_recent_cross_below(df_weekly, period_w, max_lookback=3)
            if bars_back is not None and level is not None:
                d_pct, d_atr = _distances(level, last, atr_daily)
                if abs(d_pct) <= MAX_DISTANCE_PCT:
                    when = "this week" if bars_back == 0 else (
                        "last week" if bars_back == 1 else f"{bars_back} weeks ago"
                    )
                    setups.append(Setup(
                        direction="BREAKDOWN",
                        kind=f"WEEKLY_{period_w}W_LOW",
                        label=f"{label} low broken",
                        trigger_price=round(level, 2),
                        trigger_condition=f"Lost ${level:.2f} {when} — watch for weekly follow-through",
                        invalidation_price=round(level, 2),
                        target_price=round(level - rng, 2),
                        distance_pct=d_pct,
                        distance_atr=d_atr,
                        proximity_class=_proximity_class(d_pct),
                        maturity_bars=period_w,
                        tests=None,
                        quality=0.0,
                        quality_label="WEAK",
                        rationale=[f"{label} Donchian low broken {when}"],
                        timeframe="weekly",
                        bars_since_trigger=bars_back,
                    ))
    return setups


# ─────────────────────────────────────────────────
# Setup source: open / unfilled gaps
# ─────────────────────────────────────────────────

def find_gap_setups(df: pd.DataFrame, atr: float, lookback: int = 120) -> list[Setup]:
    if df.empty or atr <= 0:
        return []
    last = float(df["Close"].iloc[-1])
    n = min(lookback, len(df) - 1)
    window = df.iloc[-n - 1:]
    setups: list[Setup] = []
    # Walk from oldest to newest; only emit gaps that remain unfilled through today.
    for i in range(1, len(window)):
        prev_high = float(window["High"].iloc[i - 1])
        prev_low = float(window["Low"].iloc[i - 1])
        today_low = float(window["Low"].iloc[i])
        today_high = float(window["High"].iloc[i])
        # Gap up: today's low > prior high. Gap is "open" if no subsequent low ≤ prev_high.
        if today_low > prev_high * 1.002 and (today_low - prev_high) > atr * 0.5:
            future = window.iloc[i + 1:]
            if future.empty or float(future["Low"].min()) > prev_high:
                gap_low = prev_high
                gap_high = today_low
                # If we're currently above the gap, the gap acts as support (breakdown setup at gap_high → fill).
                # If we're currently below the gap, the gap is overhead resistance (breakout setup at gap_low → fill).
                if last > gap_high:
                    d_pct, d_atr = _distances(gap_high, last, atr)
                    if abs(d_pct) <= MAX_DISTANCE_PCT:
                        setups.append(Setup(
                            direction="BREAKDOWN",
                            kind="GAP_FILL_DOWN",
                            label=f"Unfilled gap ${gap_low:.2f}-${gap_high:.2f}",
                            trigger_price=round(gap_high, 2),
                            trigger_condition=f"Loss of ${gap_high:.2f} (top of unfilled up-gap)",
                            invalidation_price=None,
                            target_price=round(gap_low, 2),
                            distance_pct=d_pct,
                            distance_atr=d_atr,
                            proximity_class=_proximity_class(d_pct),
                            maturity_bars=int(len(window) - 1 - i),
                            tests=None,
                            quality=0.0,
                            quality_label="WEAK",
                            rationale=["Unfilled gap acts as magnet target"],
                        ))
                elif last < gap_low:
                    d_pct, d_atr = _distances(gap_low, last, atr)
                    if abs(d_pct) <= MAX_DISTANCE_PCT:
                        setups.append(Setup(
                            direction="BREAKOUT",
                            kind="GAP_FILL_UP",
                            label=f"Gap fill ${gap_low:.2f}-${gap_high:.2f}",
                            trigger_price=round(gap_low, 2),
                            trigger_condition=f"Reclaim of ${gap_low:.2f} (bottom of overhead gap)",
                            invalidation_price=None,
                            target_price=round(gap_high, 2),
                            distance_pct=d_pct,
                            distance_atr=d_atr,
                            proximity_class=_proximity_class(d_pct),
                            maturity_bars=int(len(window) - 1 - i),
                            tests=None,
                            quality=0.0,
                            quality_label="WEAK",
                            rationale=["Overhead gap acts as magnet target"],
                        ))
        # Gap down: today's high < prior low.
        elif today_high < prev_low * 0.998 and (prev_low - today_high) > atr * 0.5:
            future = window.iloc[i + 1:]
            if future.empty or float(future["High"].max()) < prev_low:
                gap_low = today_high
                gap_high = prev_low
                if last < gap_low:
                    d_pct, d_atr = _distances(gap_low, last, atr)
                    if abs(d_pct) <= MAX_DISTANCE_PCT:
                        setups.append(Setup(
                            direction="BREAKOUT",
                            kind="GAP_FILL_UP",
                            label=f"Unfilled down-gap ${gap_low:.2f}-${gap_high:.2f}",
                            trigger_price=round(gap_low, 2),
                            trigger_condition=f"Reclaim of ${gap_low:.2f} (bottom of unfilled down-gap)",
                            invalidation_price=None,
                            target_price=round(gap_high, 2),
                            distance_pct=d_pct,
                            distance_atr=d_atr,
                            proximity_class=_proximity_class(d_pct),
                            maturity_bars=int(len(window) - 1 - i),
                            tests=None,
                            quality=0.0,
                            quality_label="WEAK",
                            rationale=["Unfilled down-gap acts as overhead magnet"],
                        ))
                elif last > gap_high:
                    d_pct, d_atr = _distances(gap_high, last, atr)
                    if abs(d_pct) <= MAX_DISTANCE_PCT:
                        setups.append(Setup(
                            direction="BREAKDOWN",
                            kind="GAP_FILL_DOWN",
                            label=f"Gap target ${gap_low:.2f}-${gap_high:.2f}",
                            trigger_price=round(gap_high, 2),
                            trigger_condition=f"Loss of ${gap_high:.2f} (top of below-price down-gap)",
                            invalidation_price=None,
                            target_price=round(gap_low, 2),
                            distance_pct=d_pct,
                            distance_atr=d_atr,
                            proximity_class=_proximity_class(d_pct),
                            maturity_bars=int(len(window) - 1 - i),
                            tests=None,
                            quality=0.0,
                            quality_label="WEAK",
                            rationale=["Below-price down-gap acts as magnet"],
                        ))
    return setups


# ─────────────────────────────────────────────────
# Setup source: Bollinger squeeze resolution
# ─────────────────────────────────────────────────

def find_squeeze_setups(df: pd.DataFrame, signals: dict, atr: float) -> list[Setup]:
    boll = signals.get("bollinger") or {}
    consol = signals.get("consolidation") or {}
    is_squeeze = (
        boll.get("signal") == "SQUEEZE"
        or boll.get("was_squeezed")
        or consol.get("had_squeeze")
    )
    if not is_squeeze or df.empty or atr <= 0:
        return []
    # Recompute current Bollinger bands quickly
    close = df["Close"]
    period = 20
    std_mult = 2.0
    if len(close) < period + 1:
        return []
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper = float((ma + std_mult * sd).iloc[-1])
    lower = float((ma - std_mult * sd).iloc[-1])
    last = float(close.iloc[-1])
    bw_pct = boll.get("bandwidth_percentile")
    setups: list[Setup] = []

    if upper > last:
        d_pct, d_atr = _distances(upper, last, atr)
        if abs(d_pct) <= MAX_DISTANCE_PCT:
            setups.append(Setup(
                direction="BREAKOUT",
                kind="SQUEEZE_UPPER",
                label="Squeeze upper-band break",
                trigger_price=round(upper, 2),
                trigger_condition=f"Close above ${upper:.2f} (upper BB) on volume",
                invalidation_price=round(lower, 2),
                target_price=None,
                distance_pct=d_pct,
                distance_atr=d_atr,
                proximity_class=_proximity_class(d_pct),
                maturity_bars=None,
                tests=None,
                quality=0.0,
                quality_label="WEAK",
                rationale=[
                    "Bollinger squeeze active — energy build-up",
                    f"BB-width {bw_pct:.0f}%-ile" if isinstance(bw_pct, (int, float)) else "",
                ],
            ))
    if lower < last:
        d_pct, d_atr = _distances(lower, last, atr)
        if abs(d_pct) <= MAX_DISTANCE_PCT:
            setups.append(Setup(
                direction="BREAKDOWN",
                kind="SQUEEZE_LOWER",
                label="Squeeze lower-band break",
                trigger_price=round(lower, 2),
                trigger_condition=f"Close below ${lower:.2f} (lower BB) on volume",
                invalidation_price=round(upper, 2),
                target_price=None,
                distance_pct=d_pct,
                distance_atr=d_atr,
                proximity_class=_proximity_class(d_pct),
                maturity_bars=None,
                tests=None,
                quality=0.0,
                quality_label="WEAK",
                rationale=[
                    "Bollinger squeeze active — energy build-up",
                    f"BB-width {bw_pct:.0f}%-ile" if isinstance(bw_pct, (int, float)) else "",
                ],
            ))
    # Strip blank rationale lines
    for s in setups:
        s.rationale = [r for r in s.rationale if r]
    return setups


# ─────────────────────────────────────────────────
# Setup source: anchored VWAP
# ─────────────────────────────────────────────────

def _avwap_from(df: pd.DataFrame, anchor_loc: int) -> pd.Series:
    """Volume-weighted average price computed from anchor_loc forward.
    Returns the series indexed by date over [anchor_loc, end]."""
    seg = df.iloc[anchor_loc:]
    if seg.empty:
        return pd.Series(dtype=float)
    typical = (seg["High"] + seg["Low"] + seg["Close"]) / 3.0
    pv = typical * seg["Volume"]
    cum_v = seg["Volume"].cumsum().replace(0, float("nan"))
    return pv.cumsum() / cum_v


def find_anchored_vwap_setups(df: pd.DataFrame, atr: float, lookback: int = 252) -> list[Setup]:
    """Anchored VWAP from the major swing high and swing low in the lookback window.

    AVWAPs from major swing points are institutional-grade reference levels:
    price reclaiming a high-anchored AVWAP from below = buyers in control of
    the move from the high; price losing a low-anchored AVWAP from above =
    sellers reasserting from the rally low. We compute both as potential
    future trigger levels and let confluence merge with anything nearby.
    """
    if df.empty or atr <= 0 or len(df) < 20:
        return []
    window = df.iloc[-lookback:] if len(df) > lookback else df
    last = float(df["Close"].iloc[-1])
    setups: list[Setup] = []

    # Map window bar to df bar index
    base_loc = len(df) - len(window)
    hi_local = int(window["High"].values.argmax())
    lo_local = int(window["Low"].values.argmin())

    for local_idx, anchor_kind, anchor_word in (
        (hi_local, "HIGH", "swing high"),
        (lo_local, "LOW", "swing low"),
    ):
        anchor_loc = base_loc + local_idx
        if anchor_loc >= len(df) - 4:
            continue   # too recent — AVWAP barely exists
        anchor_date = df.index[anchor_loc]
        anchor_price = float(
            df["High"].iloc[anchor_loc] if anchor_kind == "HIGH"
            else df["Low"].iloc[anchor_loc]
        )
        vwap = _avwap_from(df, anchor_loc)
        if vwap.empty:
            continue
        vwap_now = float(vwap.iloc[-1])
        if not (vwap_now == vwap_now):
            continue
        bars_since_anchor = len(df) - 1 - anchor_loc
        d_pct, d_atr = _distances(vwap_now, last, atr)
        if abs(d_pct) > MAX_DISTANCE_PCT:
            continue

        if vwap_now > last:
            # AVWAP is above price — acts as overhead resistance / reclaim setup.
            setups.append(Setup(
                direction="BREAKOUT",
                kind=f"AVWAP_FROM_{anchor_kind}",
                label=f"AVWAP from {anchor_word} ({bars_since_anchor}d back)",
                trigger_price=round(vwap_now, 2),
                trigger_condition=f"Close above AVWAP at ${vwap_now:.2f}",
                invalidation_price=None,
                target_price=None,
                distance_pct=d_pct,
                distance_atr=d_atr,
                proximity_class=_proximity_class(d_pct),
                maturity_bars=bars_since_anchor,
                tests=None,
                quality=0.0,
                quality_label="WEAK",
                rationale=[
                    f"AVWAP anchored at {anchor_word} ${anchor_price:.2f} "
                    f"({bars_since_anchor} bars ago)",
                    "Institutional reference price for the move from anchor",
                ],
            ))
        else:
            # AVWAP below price — acts as support / loss setup.
            setups.append(Setup(
                direction="BREAKDOWN",
                kind=f"AVWAP_FROM_{anchor_kind}",
                label=f"AVWAP from {anchor_word} ({bars_since_anchor}d back)",
                trigger_price=round(vwap_now, 2),
                trigger_condition=f"Close below AVWAP at ${vwap_now:.2f}",
                invalidation_price=None,
                target_price=None,
                distance_pct=d_pct,
                distance_atr=d_atr,
                proximity_class=_proximity_class(d_pct),
                maturity_bars=bars_since_anchor,
                tests=None,
                quality=0.0,
                quality_label="WEAK",
                rationale=[
                    f"AVWAP anchored at {anchor_word} ${anchor_price:.2f} "
                    f"({bars_since_anchor} bars ago)",
                    "Institutional reference price for the move from anchor",
                ],
            ))
    return setups


# ─────────────────────────────────────────────────
# Setup source: round-number psychological levels
# ─────────────────────────────────────────────────

def _round_increment(price: float) -> float:
    """Pick a sensible round-number increment for the given price magnitude."""
    if price < 5:
        return 0.50
    if price < 20:
        return 1.0
    if price < 100:
        return 5.0
    if price < 500:
        return 10.0
    if price < 1500:
        return 50.0
    if price < 5000:
        return 100.0
    return 500.0


def find_round_number_setups(df: pd.DataFrame, atr: float) -> list[Setup]:
    """Nearest round number above and below the current price.

    Psychological levels — they don't have structural backing, so they get
    a modest base quality. Useful primarily for confluence with stronger
    levels (e.g. $200 sitting right at a 50-day high reinforces it).
    """
    if df.empty or atr <= 0:
        return []
    import math
    last = float(df["Close"].iloc[-1])
    inc = _round_increment(last)
    above = math.ceil(last / inc) * inc
    below = math.floor(last / inc) * inc
    if abs(above - last) < 1e-6:
        above += inc
    if abs(below - last) < 1e-6:
        below -= inc

    setups: list[Setup] = []
    for level, direction in ((above, "BREAKOUT"), (below, "BREAKDOWN")):
        if level <= 0:
            continue
        d_pct, d_atr = _distances(level, last, atr)
        if abs(d_pct) > MAX_DISTANCE_PCT:
            continue
        # Skip if too close to last close in ATR terms — usually not actionable
        if d_atr is not None and d_atr < 0.1:
            continue
        setups.append(Setup(
            direction=direction,  # type: ignore[arg-type]
            kind=f"ROUND_NUMBER_{'ABOVE' if direction == 'BREAKOUT' else 'BELOW'}",
            label=f"${level:.0f} round number",
            trigger_price=round(level, 2),
            trigger_condition=(
                f"Close above ${level:.2f}" if direction == "BREAKOUT"
                else f"Close below ${level:.2f}"
            ),
            invalidation_price=None,
            target_price=None,
            distance_pct=d_pct,
            distance_atr=d_atr,
            proximity_class=_proximity_class(d_pct),
            maturity_bars=None,
            tests=None,
            quality=0.0,
            quality_label="WEAK",
            rationale=[f"${level:.0f} — psychological round-number level"],
        ))
    return setups


# ─────────────────────────────────────────────────
# Setup source: diagonal trendlines
# ─────────────────────────────────────────────────

def _trendline_candidates(
    pts: list[tuple[int, float]],
    side: str,                    # "high" = resistance (descending), "low" = support (ascending)
    tolerance: float,
    min_touches: int = 3,
) -> list[tuple[int, float, float, int, int]]:
    """Find trendlines through swing points.

    For each candidate line (defined by a pair of swing points), count how
    many other swing points lie within `tolerance` of the line between
    them. Reject lines that have swing points clearly violating the line
    (a swing high ABOVE a resistance trendline, or a swing low BELOW a
    support trendline) — those mean the line was broken historically.

    Returns: list of (touch_count, slope, intercept, anchor_x, end_x).
    """
    if len(pts) < min_touches:
        return []
    pts = sorted(pts, key=lambda p: p[0])
    out: list[tuple[int, float, float, int, int]] = []
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if side == "high" and slope >= 0:
                continue
            if side == "low" and slope <= 0:
                continue
            intercept = y1 - slope * x1
            touches = 0
            broken = False
            for x, y in pts:
                if x < x1 or x > x2:
                    continue
                projected = slope * x + intercept
                if abs(y - projected) <= tolerance:
                    touches += 1
                    continue
                # Disqualify if a swing point on the wrong side
                if side == "high" and y > projected + tolerance:
                    broken = True
                    break
                if side == "low" and y < projected - tolerance:
                    broken = True
                    break
            if broken:
                continue
            if touches >= min_touches:
                out.append((touches, slope, intercept, x1, x2))
    # Dedupe near-duplicates by slope/intercept
    out.sort(key=lambda t: -t[0])
    keep: list[tuple[int, float, float, int, int]] = []
    for tl in out:
        too_similar = any(
            abs(tl[1] - k[1]) < 0.1 and abs(tl[2] - k[2]) < tolerance * 2
            for k in keep
        )
        if not too_similar:
            keep.append(tl)
        if len(keep) >= 3:
            break
    return keep


def find_trendline_setups(df: pd.DataFrame, atr: float, lookback: int = 252) -> list[Setup]:
    """Diagonal trendlines through 3+ swing points.

    Two kinds:
      • Descending resistance through swing highs: a future BREAKOUT setup,
        the trigger being the line projected to today.
      • Ascending support through swing lows: a future BREAKDOWN setup if
        broken, the trigger being the line projected to today.
    Lines that have already been broken (price currently above resistance line
    or below support line) are not emitted — the geometry has changed.
    """
    if df.empty or atr <= 0 or len(df) < 30:
        return []
    window = df.iloc[-lookback:] if len(df) > lookback else df
    if len(window) < 30:
        return []
    last = float(df["Close"].iloc[-1])
    base_loc = len(df) - len(window)
    n = len(window)
    today_x = n - 1
    today_date = df.index[-1]

    highs = window["High"].to_numpy(dtype=float)
    lows = window["Low"].to_numpy(dtype=float)
    sh = _swing_highs(highs, distance=4)
    sl = _swing_lows(lows, distance=4)
    sh_pts = [(i, float(highs[i])) for i in sh]
    sl_pts = [(i, float(lows[i])) for i in sl]
    tol = atr * 0.6

    setups: list[Setup] = []

    # Descending resistance → BREAKOUT
    for touches, slope, intercept, anchor_x, end_x in _trendline_candidates(
        sh_pts, "high", tolerance=tol
    ):
        projected_today = slope * today_x + intercept
        if projected_today <= last:
            continue  # already broken
        d_pct, d_atr = _distances(projected_today, last, atr)
        if abs(d_pct) > MAX_DISTANCE_PCT:
            continue
        anchor_date = df.index[base_loc + anchor_x]
        anchor_price = slope * anchor_x + intercept
        time_span = today_x - anchor_x
        setups.append(Setup(
            direction="BREAKOUT",
            kind="TRENDLINE_DESCENDING",
            label=f"Descending trendline ({touches}× touched)",
            trigger_price=round(projected_today, 2),
            trigger_condition=(
                f"Close above descending trendline (at ${projected_today:.2f} today)"
            ),
            invalidation_price=None,
            target_price=None,
            distance_pct=d_pct,
            distance_atr=d_atr,
            proximity_class=_proximity_class(d_pct),
            maturity_bars=time_span,
            tests=touches,
            quality=0.0,
            quality_label="WEAK",
            rationale=[
                f"Descending trendline through {touches} swing highs",
                f"Spans {time_span} bars",
            ],
            line_meta={
                "x0": anchor_date, "y0": round(anchor_price, 2),
                "x1": today_date, "y1": round(projected_today, 2),
            },
        ))

    # Ascending support → BREAKDOWN
    for touches, slope, intercept, anchor_x, end_x in _trendline_candidates(
        sl_pts, "low", tolerance=tol
    ):
        projected_today = slope * today_x + intercept
        if projected_today >= last:
            continue
        d_pct, d_atr = _distances(projected_today, last, atr)
        if abs(d_pct) > MAX_DISTANCE_PCT:
            continue
        anchor_date = df.index[base_loc + anchor_x]
        anchor_price = slope * anchor_x + intercept
        time_span = today_x - anchor_x
        setups.append(Setup(
            direction="BREAKDOWN",
            kind="TRENDLINE_ASCENDING",
            label=f"Ascending trendline ({touches}× touched)",
            trigger_price=round(projected_today, 2),
            trigger_condition=(
                f"Close below ascending trendline (at ${projected_today:.2f} today)"
            ),
            invalidation_price=None,
            target_price=None,
            distance_pct=d_pct,
            distance_atr=d_atr,
            proximity_class=_proximity_class(d_pct),
            maturity_bars=time_span,
            tests=touches,
            quality=0.0,
            quality_label="WEAK",
            rationale=[
                f"Ascending trendline through {touches} swing lows",
                f"Spans {time_span} bars",
            ],
            line_meta={
                "x0": anchor_date, "y0": round(anchor_price, 2),
                "x1": today_date, "y1": round(projected_today, 2),
            },
        ))
    return setups


# ─────────────────────────────────────────────────
# Setup source: parallel-rail price channels
# ─────────────────────────────────────────────────

def _rail_candidates(
    pts: list[tuple[int, float]],
    side: str,             # "upper" rejects swing points above the line; "lower" below
    tolerance: float,
    min_touches: int = 2,
    max_keep: int = 8,
) -> list[tuple[int, float, float, int, int]]:
    """Like _trendline_candidates but accepts BOTH slope directions.

    Used for channel detection where an upper rail might be ascending (in an
    uptrend channel) or descending (downtrend), and same for the lower rail.
    Returns (touches, slope, intercept, anchor_x, end_x), deduped.
    """
    if len(pts) < min_touches:
        return []
    pts = sorted(pts, key=lambda p: p[0])
    out: list[tuple[int, float, float, int, int]] = []
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            touches = 0
            broken = False
            for x, y in pts:
                if x < x1 or x > x2:
                    continue
                projected = slope * x + intercept
                if abs(y - projected) <= tolerance:
                    touches += 1
                    continue
                if side == "upper" and y > projected + tolerance:
                    broken = True
                    break
                if side == "lower" and y < projected - tolerance:
                    broken = True
                    break
            if broken:
                continue
            if touches >= min_touches:
                out.append((touches, slope, intercept, x1, x2))
    out.sort(key=lambda t: -t[0])
    keep: list[tuple[int, float, float, int, int]] = []
    for tl in out:
        too_similar = any(
            abs(tl[1] - k[1]) < 0.05 and abs(tl[2] - k[2]) < tolerance * 2
            for k in keep
        )
        if not too_similar:
            keep.append(tl)
        if len(keep) >= max_keep:
            break
    return keep


# Channel detection tunables.
CHANNEL_SLOPE_TOL = 0.25      # max relative slope variance between upper/lower rail
CHANNEL_MIN_WIDTH_ATR = 3.0   # narrower than this is a Donchian range, not a channel
CHANNEL_MIN_TOTAL_TOUCHES = 4 # 2 per rail minimum


def find_channel_setups(
    df: pd.DataFrame,
    atr: float,
    lookback: int = 252,
    timeframe: str = "daily",
) -> list[Setup]:
    """Detect parallel-rail price channels and emit one Setup per rail.

    A channel is two parallel trendlines bounding price action, with:
      • At least 4 total touches (2+ on each rail)
      • Rails non-crossing throughout the window
      • Slopes within 25% of each other (real channels rarely perfectly parallel)
      • Same slope sign (both rising or both falling — diverging rails = wedge)
      • Width >= 3 ATRs at today's bar (horizontal channels caught by Donchian)
      • No in-window swing point violating either rail
      • Price currently INSIDE the channel (else the channel was broken — skip)

    Setups emitted per channel:
      • Upper rail → BREAKOUT. In an ascending channel this is continuation
        acceleration; in a descending channel it's a trend-reversal break.
      • Lower rail → BREAKDOWN. Mirror.

    The `timeframe` arg ("daily" or "weekly") drives:
      • The minimum bar-count check (30 daily, 25 weekly — fewer bars per year)
      • The kind label suffix (CHANNEL_* vs CHANNEL_WEEKLY_*)
      • The setup's `timeframe` field for downstream UI grouping
    """
    min_bars = 25 if timeframe == "weekly" else 30
    if df.empty or atr <= 0 or len(df) < min_bars:
        return []
    window = df.iloc[-lookback:] if len(df) > lookback else df
    if len(window) < min_bars:
        return []
    last = float(df["Close"].iloc[-1])
    base_loc = len(df) - len(window)
    n = len(window)
    today_x = n - 1
    today_date = df.index[-1]

    highs = window["High"].to_numpy(dtype=float)
    lows = window["Low"].to_numpy(dtype=float)
    sh = list(_swing_highs(highs, distance=4))
    sl = list(_swing_lows(lows, distance=4))

    # Include "current local extremum" as a virtual swing point. _swing_highs
    # requires `distance` future bars to confirm, so a fresh peak in the last
    # 4 bars never registers — but visually it's the bar that defines the
    # rail's latest touch. Without this, fast-moving names (e.g. LRCX with
    # a parabolic recent leg) have their upper rail anchored to OLD swings
    # well below current price, and the "price inside channel" filter rejects
    # every candidate. We add the max-high (min-low) bar from the last
    # 2*distance bars when it's more recent (and higher / lower) than any
    # formal swing already in the set.
    edge_window = min(8, len(highs))
    edge_start = len(highs) - edge_window
    if edge_window >= 2:
        edge_max = edge_start + int(np.argmax(highs[edge_start:]))
        edge_min = edge_start + int(np.argmin(lows[edge_start:]))
        if not sh or edge_max > sh[-1]:
            if not sh or float(highs[edge_max]) > float(highs[sh[-1]]):
                sh.append(edge_max)
        if not sl or edge_min > sl[-1]:
            if not sl or float(lows[edge_min]) < float(lows[sl[-1]]):
                sl.append(edge_min)

    sh_pts = [(int(i), float(highs[i])) for i in sh]
    sl_pts = [(int(i), float(lows[i])) for i in sl]
    tol = atr * 0.6

    upper_rails = _rail_candidates(sh_pts, "upper", tolerance=tol, min_touches=2)
    lower_rails = _rail_candidates(sl_pts, "lower", tolerance=tol, min_touches=2)
    if not upper_rails or not lower_rails:
        return []

    # First pass: collect every viable channel as a tuple, then rank by touches
    # and keep at most CHANNEL_MAX_PER_TICKER so the chart isn't spaghetti.
    candidates: list[dict] = []

    for u_touches, u_slope, u_intercept, u_x1, u_x2 in upper_rails:
        for l_touches, l_slope, l_intercept, l_x1, l_x2 in lower_rails:
            total_touches = u_touches + l_touches
            if total_touches < CHANNEL_MIN_TOTAL_TOUCHES:
                continue
            if (u_slope > 0) != (l_slope > 0):
                continue
            max_abs = max(abs(u_slope), abs(l_slope))
            if max_abs == 0:
                continue
            if abs(u_slope - l_slope) / max_abs > CHANNEL_SLOPE_TOL:
                continue
            u_today = u_slope * today_x + u_intercept
            l_today = l_slope * today_x + l_intercept
            width = u_today - l_today
            if width < CHANNEL_MIN_WIDTH_ATR * atr:
                continue
            crossed = any(
                (l_slope * x + l_intercept) >= (u_slope * x + u_intercept)
                for x in range(0, n)
            )
            if crossed:
                continue
            avg_slope = (u_slope + l_slope) / 2.0
            if abs(avg_slope * n) < 0.5 * width:
                continue
            if last > u_today or last < l_today:
                continue
            candidates.append({
                "total_touches": total_touches,
                "u_touches": u_touches, "l_touches": l_touches,
                "u_slope": u_slope, "u_intercept": u_intercept,
                "l_slope": l_slope, "l_intercept": l_intercept,
                "u_today": u_today, "l_today": l_today, "width": width,
                "avg_slope": avg_slope,
                "u_x1": u_x1, "l_x1": l_x1,
            })

    # Rank by total touches (more touches = more respected channel), break ties
    # by tightness (smaller width relative to ATR), and dedupe similar channels.
    candidates.sort(key=lambda c: (-c["total_touches"], c["width"] / atr))
    setups: list[Setup] = []
    used: list[tuple[float, float, float, float]] = []
    CHANNEL_MAX_PER_TICKER = 2

    for c in candidates:
        sig = (round(c["u_slope"], 4), round(c["l_slope"], 4),
               round(c["u_intercept"]), round(c["l_intercept"]))
        # Looser dedupe than the rail-level dedupe — channels with similar
        # endpoint prices are functionally the same chart object even if
        # geometry differs at the start.
        endpoint_tol = atr * 1.0
        is_dupe = any(
            abs(c["u_today"] - (s[0] * today_x + s[2])) < endpoint_tol
            and abs(c["l_today"] - (s[1] * today_x + s[3])) < endpoint_tol
            for s in used
        )
        if is_dupe:
            continue
        used.append(sig)
        if len(used) > CHANNEL_MAX_PER_TICKER:
            break

        u_slope, u_intercept = c["u_slope"], c["u_intercept"]
        l_slope, l_intercept = c["l_slope"], c["l_intercept"]
        u_today, l_today, width = c["u_today"], c["l_today"], c["width"]
        avg_slope = c["avg_slope"]
        u_touches, l_touches = c["u_touches"], c["l_touches"]
        total_touches = c["total_touches"]
        channel_dir = "ASCENDING" if avg_slope > 0 else "DESCENDING"

        # Endpoint dates for rendering — each rail anchors at its OWN swing
        # point (not the shared older one) so we never extrapolate one rail's
        # slope backward past its actual touch point and get absurd values.
        u_anchor_x = c["u_x1"]
        l_anchor_x = c["l_x1"]
        u_anchor_date = df.index[base_loc + u_anchor_x]
        l_anchor_date = df.index[base_loc + l_anchor_x]
        u_at_anchor = u_slope * u_anchor_x + u_intercept
        l_at_anchor = l_slope * l_anchor_x + l_intercept
        # Span = how long this channel has been in effect (older of the two anchors)
        span = today_x - min(u_anchor_x, l_anchor_x)

        prefix = "CHANNEL_WEEKLY_" if timeframe == "weekly" else "CHANNEL_"
        if channel_dir == "ASCENDING":
            upper_kind, upper_role = f"{prefix}ASCENDING_UPPER", "continuation"
            lower_kind, lower_role = f"{prefix}ASCENDING_LOWER", "trend reversal"
        else:
            upper_kind, upper_role = f"{prefix}DESCENDING_UPPER", "trend reversal"
            lower_kind, lower_role = f"{prefix}DESCENDING_LOWER", "continuation"
        label_prefix = "Weekly " if timeframe == "weekly" else ""

        u_d_pct, u_d_atr = _distances(u_today, last, atr)
        l_d_pct, l_d_atr = _distances(l_today, last, atr)

        span_unit = "weeks" if timeframe == "weekly" else "bars"
        # Upper-rail break (BREAKOUT)
        setups.append(Setup(
            direction="BREAKOUT",
            kind=upper_kind,
            label=f"{label_prefix}{channel_dir.title()} channel — upper rail",
            trigger_price=round(u_today, 2),
            trigger_condition=(
                f"Close above upper rail (${u_today:.2f} today, "
                f"slope {u_slope:+.3f}/bar)"
            ),
            invalidation_price=round(l_today, 2),
            target_price=round(u_today + width, 2),
            distance_pct=u_d_pct,
            distance_atr=u_d_atr,
            proximity_class=_proximity_class(u_d_pct),
            maturity_bars=span,
            tests=u_touches,
            quality=0.0,
            quality_label="WEAK",
            rationale=[
                f"{label_prefix}{channel_dir.title()} channel, {total_touches} total touches",
                f"Width ${width:.2f} ({width/atr:.1f} ATR), spans {span} {span_unit}",
                f"Upper-rail break = {upper_role}",
            ],
            timeframe=timeframe,
            line_meta={
                "x0": u_anchor_date, "y0": round(u_at_anchor, 2),
                "x1": today_date, "y1": round(u_today, 2),
            },
        ))
        # Lower-rail break (BREAKDOWN)
        setups.append(Setup(
            direction="BREAKDOWN",
            kind=lower_kind,
            label=f"{label_prefix}{channel_dir.title()} channel — lower rail",
            trigger_price=round(l_today, 2),
            trigger_condition=(
                f"Close below lower rail (${l_today:.2f} today, "
                f"slope {l_slope:+.3f}/bar)"
            ),
            invalidation_price=round(u_today, 2),
            target_price=round(l_today - width, 2),
            distance_pct=l_d_pct,
            distance_atr=l_d_atr,
            proximity_class=_proximity_class(l_d_pct),
            maturity_bars=span,
            tests=l_touches,
            quality=0.0,
            quality_label="WEAK",
            rationale=[
                f"{label_prefix}{channel_dir.title()} channel, {total_touches} total touches",
                f"Width ${width:.2f} ({width/atr:.1f} ATR), spans {span} {span_unit}",
                f"Lower-rail break = {lower_role}",
            ],
            timeframe=timeframe,
            line_meta={
                "x0": l_anchor_date, "y0": round(l_at_anchor, 2),
                "x1": today_date, "y1": round(l_today, 2),
            },
        ))

    # ─── Forming-channel pass: 2-touch confirmed rail + 1-touch parallel ───
    # An "up, down, up" or "down, up, down" sequence — one rail is structurally
    # tested twice, the other is constructed parallel through the single
    # opposing swing point. This catches channels EARLIER in their formation
    # at the cost of one rail being inferred rather than tested.
    confirmed_endpoints = [(c["u_today"], c["l_today"]) for c in candidates if c in [used_c for used_c in []] or True]
    # Actually just rebuild from `used` which is the list we appended to
    confirmed_today_pairs: list[tuple[float, float]] = []
    for s_tuple in used:
        u_s, l_s, u_i, l_i = s_tuple
        confirmed_today_pairs.append((u_s * today_x + u_i, l_s * today_x + l_i))

    forming: list[dict] = []

    # Pass A: confirmed UPPER (≥2 touches) + parallel LOWER through a single low
    for u_touches, u_slope, u_intercept, u_x1, u_x2 in upper_rails:
        if not sl_pts:
            continue
        # Try every swing low as the parallel anchor; keep the deepest valid one
        best: Optional[dict] = None
        for lx, ly in sl_pts:
            l_int_try = ly - u_slope * lx
            ok = True
            l_count = 1
            for x, y in sl_pts:
                proj = u_slope * x + l_int_try
                if y < proj - tol:
                    ok = False; break
                if (x, y) != (lx, ly) and abs(y - proj) <= tol:
                    l_count += 1
            if not ok or l_count >= 2:
                continue
            u_today_v = u_slope * today_x + u_intercept
            l_today_v = u_slope * today_x + l_int_try
            width_v = u_today_v - l_today_v
            if width_v < CHANNEL_MIN_WIDTH_ATR * atr:
                continue
            if abs(u_slope * n) < 0.5 * width_v:
                continue
            if last > u_today_v or last < l_today_v:
                continue
            # For forming channels both rails must be within MAX_DISTANCE_PCT
            # of current — the parallel rail to prevent ancient-swing reach,
            # the confirmed rail because a stale 2-touch line that's 50%+
            # above current isn't actionable in the trading window we care
            # about even if its geometry is "valid" in the data.
            u_d_pct_v = (u_today_v - last) / last * 100 if last > 0 else 0
            l_d_pct_v = (l_today_v - last) / last * 100 if last > 0 else 0
            if abs(l_d_pct_v) > MAX_DISTANCE_PCT or abs(u_d_pct_v) > MAX_DISTANCE_PCT:
                continue
            if best is None or l_int_try < best["l_intercept"]:
                best = {
                    "u_touches": u_touches, "u_slope": u_slope, "u_intercept": u_intercept,
                    "l_slope": u_slope, "l_intercept": l_int_try, "l_touches": l_count,
                    "u_today": u_today_v, "l_today": l_today_v, "width": width_v,
                    "u_x1": u_x1, "l_x1": lx,
                    "total_touches": u_touches + l_count,
                }
        if best is not None:
            forming.append(best)

    # Pass B: confirmed LOWER (≥2 touches) + parallel UPPER through a single high
    for l_touches, l_slope, l_intercept, l_x1, l_x2 in lower_rails:
        if not sh_pts:
            continue
        best = None
        for hx, hy in sh_pts:
            u_int_try = hy - l_slope * hx
            ok = True
            u_count = 1
            for x, y in sh_pts:
                proj = l_slope * x + u_int_try
                if y > proj + tol:
                    ok = False; break
                if (x, y) != (hx, hy) and abs(y - proj) <= tol:
                    u_count += 1
            if not ok or u_count >= 2:
                continue
            u_today_v = l_slope * today_x + u_int_try
            l_today_v = l_slope * today_x + l_intercept
            width_v = u_today_v - l_today_v
            if width_v < CHANNEL_MIN_WIDTH_ATR * atr:
                continue
            if abs(l_slope * n) < 0.5 * width_v:
                continue
            if last > u_today_v or last < l_today_v:
                continue
            # Both rails must be within MAX_DISTANCE_PCT for forming channels
            # (see Pass A comment for rationale).
            u_d_pct_v = (u_today_v - last) / last * 100 if last > 0 else 0
            l_d_pct_v = (l_today_v - last) / last * 100 if last > 0 else 0
            if abs(u_d_pct_v) > MAX_DISTANCE_PCT or abs(l_d_pct_v) > MAX_DISTANCE_PCT:
                continue
            if best is None or u_int_try > best["u_intercept"]:
                best = {
                    "u_touches": u_count, "u_slope": l_slope, "u_intercept": u_int_try,
                    "l_slope": l_slope, "l_intercept": l_intercept, "l_touches": l_touches,
                    "u_today": u_today_v, "l_today": l_today_v, "width": width_v,
                    "u_x1": hx, "l_x1": l_x1,
                    "total_touches": u_count + l_touches,
                }
        if best is not None:
            forming.append(best)

    # Rank forming by total touches, then tightness. Dedupe against confirmed and self.
    forming.sort(key=lambda c: (-c["total_touches"], c["width"] / atr))
    endpoint_tol = atr * 1.0
    FORMING_MAX = 1
    forming_kept: list[dict] = []
    for c in forming:
        # Skip if endpoints match an already-emitted confirmed channel
        is_dupe_confirmed = any(
            abs(c["u_today"] - up) < endpoint_tol and abs(c["l_today"] - lp) < endpoint_tol
            for up, lp in confirmed_today_pairs
        )
        if is_dupe_confirmed:
            continue
        # Skip if matches another forming we've kept
        is_dupe_forming = any(
            abs(c["u_today"] - k["u_today"]) < endpoint_tol
            and abs(c["l_today"] - k["l_today"]) < endpoint_tol
            for k in forming_kept
        )
        if is_dupe_forming:
            continue
        forming_kept.append(c)
        if len(forming_kept) >= FORMING_MAX:
            break

    for c in forming_kept:
        u_slope = c["u_slope"]; u_intercept = c["u_intercept"]
        l_slope = c["l_slope"]; l_intercept = c["l_intercept"]
        u_today = c["u_today"]; l_today = c["l_today"]; width = c["width"]
        avg_slope = (u_slope + l_slope) / 2.0
        u_touches = c["u_touches"]; l_touches = c["l_touches"]
        total_touches = c["total_touches"]
        u_anchor_x = c["u_x1"]; l_anchor_x = c["l_x1"]
        u_anchor_date = df.index[base_loc + u_anchor_x]
        l_anchor_date = df.index[base_loc + l_anchor_x]
        u_at_anchor = u_slope * u_anchor_x + u_intercept
        l_at_anchor = l_slope * l_anchor_x + l_intercept
        span = today_x - min(u_anchor_x, l_anchor_x)
        channel_dir = "ASCENDING" if avg_slope > 0 else "DESCENDING"

        prefix = "CHANNEL_WEEKLY_" if timeframe == "weekly" else "CHANNEL_"
        if channel_dir == "ASCENDING":
            upper_kind = f"{prefix}ASCENDING_UPPER_FORMING"
            lower_kind = f"{prefix}ASCENDING_LOWER_FORMING"
            upper_role, lower_role = "continuation", "trend reversal"
        else:
            upper_kind = f"{prefix}DESCENDING_UPPER_FORMING"
            lower_kind = f"{prefix}DESCENDING_LOWER_FORMING"
            upper_role, lower_role = "trend reversal", "continuation"
        label_prefix = "Weekly " if timeframe == "weekly" else ""
        span_unit = "weeks" if timeframe == "weekly" else "bars"

        u_d_pct, u_d_atr = _distances(u_today, last, atr)
        l_d_pct, l_d_atr = _distances(l_today, last, atr)

        # Identify which rail is the confirmed one (≥2 touches) so the
        # rationale can label them correctly.
        confirmed_side = "upper" if u_touches >= 2 else "lower"

        setups.append(Setup(
            direction="BREAKOUT",
            kind=upper_kind,
            label=f"{label_prefix}{channel_dir.title()} channel (forming) — upper rail",
            trigger_price=round(u_today, 2),
            trigger_condition=(
                f"Close above upper rail (${u_today:.2f} today, "
                f"slope {u_slope:+.3f}/bar)"
            ),
            invalidation_price=round(l_today, 2),
            target_price=round(u_today + width, 2),
            distance_pct=u_d_pct,
            distance_atr=u_d_atr,
            proximity_class=_proximity_class(u_d_pct),
            maturity_bars=span,
            tests=u_touches,
            quality=0.0,
            quality_label="WEAK",
            rationale=[
                f"{label_prefix}{channel_dir.title()} channel forming, "
                f"{total_touches} touches ({confirmed_side} rail confirmed)",
                f"Width ${width:.2f} ({width/atr:.1f} ATR), spans {span} {span_unit}",
                "Forming — opposite rail constructed parallel through single touch",
                f"Upper-rail break = {upper_role}",
            ],
            timeframe=timeframe,
            line_meta={
                "x0": u_anchor_date, "y0": round(u_at_anchor, 2),
                "x1": today_date, "y1": round(u_today, 2),
            },
        ))
        setups.append(Setup(
            direction="BREAKDOWN",
            kind=lower_kind,
            label=f"{label_prefix}{channel_dir.title()} channel (forming) — lower rail",
            trigger_price=round(l_today, 2),
            trigger_condition=(
                f"Close below lower rail (${l_today:.2f} today, "
                f"slope {l_slope:+.3f}/bar)"
            ),
            invalidation_price=round(u_today, 2),
            target_price=round(l_today - width, 2),
            distance_pct=l_d_pct,
            distance_atr=l_d_atr,
            proximity_class=_proximity_class(l_d_pct),
            maturity_bars=span,
            tests=l_touches,
            quality=0.0,
            quality_label="WEAK",
            rationale=[
                f"{label_prefix}{channel_dir.title()} channel forming, "
                f"{total_touches} touches ({confirmed_side} rail confirmed)",
                f"Width ${width:.2f} ({width/atr:.1f} ATR), spans {span} {span_unit}",
                "Forming — opposite rail constructed parallel through single touch",
                f"Lower-rail break = {lower_role}",
            ],
            timeframe=timeframe,
            line_meta={
                "x0": l_anchor_date, "y0": round(l_at_anchor, 2),
                "x1": today_date, "y1": round(l_today, 2),
            },
        ))

    return setups


# ─────────────────────────────────────────────────
# Setup source: reversal candles at S/R
# ─────────────────────────────────────────────────

def _is_hammer(o: float, h: float, l: float, c: float) -> bool:
    body = abs(c - o)
    upper = h - max(c, o)
    lower = min(c, o) - l
    total = h - l
    if total <= 0 or body <= 0:
        return False
    return (
        lower >= 2 * body
        and upper <= 0.4 * body
        and body / total < 0.45
    )


def _is_shooting_star(o: float, h: float, l: float, c: float) -> bool:
    body = abs(c - o)
    upper = h - max(c, o)
    lower = min(c, o) - l
    total = h - l
    if total <= 0 or body <= 0:
        return False
    return (
        upper >= 2 * body
        and lower <= 0.4 * body
        and body / total < 0.45
    )


def _is_bullish_engulfing(po: float, pc: float, o: float, c: float) -> bool:
    return pc < po and c > o and o <= pc and c >= po


def _is_bearish_engulfing(po: float, pc: float, o: float, c: float) -> bool:
    return pc > po and c < o and o >= pc and c <= po


def find_reversal_candle_setups(df: pd.DataFrame, atr: float, signals: dict) -> list[Setup]:
    """Reversal candle setups, gated on being at/near a structural S/R level.

    A hammer or bullish engulfing in the middle of nowhere isn't a setup —
    it's a candle. Tied to support, it becomes a tradeable signal. Trigger
    is today's high (for bullish) or low (for bearish) — confirmation that
    the reversal candle held by the next bar's print.
    """
    if df.empty or len(df) < 3 or atr <= 0:
        return []
    last_close = float(df["Close"].iloc[-1])
    today_high = float(df["High"].iloc[-1])
    today_low = float(df["Low"].iloc[-1])
    today_open = float(df["Open"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2])
    prev_open = float(df["Open"].iloc[-2])

    consol = signals.get("consolidation") or {}
    range_high = consol.get("range_high")
    range_low = consol.get("range_low")
    setups: list[Setup] = []

    near_support = (
        range_low is not None
        and abs(today_low - float(range_low)) <= atr * 1.0
    )
    near_resistance = (
        range_high is not None
        and abs(today_high - float(range_high)) <= atr * 1.0
    )

    # ── Bullish reversal at support ─────────────────────────────────
    hammer = _is_hammer(today_open, today_high, today_low, last_close)
    bull_engulf = _is_bullish_engulfing(prev_open, prev_close, today_open, last_close)
    if (hammer or bull_engulf) and near_support:
        pname = "Hammer" if hammer else "Bullish Engulfing"
        d_pct, d_atr = _distances(today_high, last_close, atr)
        setups.append(Setup(
            direction="BREAKOUT",
            kind=f"REVERSAL_BULL_{pname.replace(' ', '_').upper()}",
            label=f"{pname} at support",
            trigger_price=round(today_high, 2),
            trigger_condition=f"Close above ${today_high:.2f} (today's {pname.lower()} high)",
            invalidation_price=round(today_low, 2),
            target_price=None,
            distance_pct=d_pct,
            distance_atr=d_atr,
            proximity_class=_proximity_class(d_pct),
            maturity_bars=None,
            tests=None,
            quality=0.0,
            quality_label="WEAK",
            rationale=[
                f"{pname} at daily Donchian low (${float(range_low):.2f})",
                "Reversal signal — needs follow-through close",
            ],
        ))

    # ── Bearish reversal at resistance ──────────────────────────────
    star = _is_shooting_star(today_open, today_high, today_low, last_close)
    bear_engulf = _is_bearish_engulfing(prev_open, prev_close, today_open, last_close)
    if (star or bear_engulf) and near_resistance:
        pname = "Shooting Star" if star else "Bearish Engulfing"
        d_pct, d_atr = _distances(today_low, last_close, atr)
        setups.append(Setup(
            direction="BREAKDOWN",
            kind=f"REVERSAL_BEAR_{pname.replace(' ', '_').upper()}",
            label=f"{pname} at resistance",
            trigger_price=round(today_low, 2),
            trigger_condition=f"Close below ${today_low:.2f} (today's {pname.lower()} low)",
            invalidation_price=round(today_high, 2),
            target_price=None,
            distance_pct=d_pct,
            distance_atr=d_atr,
            proximity_class=_proximity_class(d_pct),
            maturity_bars=None,
            tests=None,
            quality=0.0,
            quality_label="WEAK",
            rationale=[
                f"{pname} at daily Donchian high (${float(range_high):.2f})",
                "Reversal signal — needs follow-through close",
            ],
        ))
    return setups


# ─────────────────────────────────────────────────
# Confluence merging
# ─────────────────────────────────────────────────

def merge_confluence(setups: list[Setup], atr: float) -> list[Setup]:
    """Setups within ATR_TOLERANCE_MERGE ATRs are merged — the highest-quality
    base is kept; the merged setup gets a confluence_with list and a quality bump.

    Triggered setups (bars_since_trigger != None) are merged separately from
    pending setups so a triggered breakout doesn't absorb a pending pattern at
    a nearby level — they answer different questions ("what just happened" vs
    "what to watch for next").
    """
    if not setups or atr <= 0:
        return setups
    # Group by direction AND triggered/pending state, then cluster by trigger.
    by_bucket: dict[tuple[str, bool], list[Setup]] = {}
    for s in setups:
        key = (s.direction, s.is_triggered)
        by_bucket.setdefault(key, []).append(s)
    merged: list[Setup] = []
    for _key, group in by_bucket.items():
        group = sorted(group, key=lambda s: s.trigger_price)
        tol = atr * ATR_TOLERANCE_MERGE
        clusters: list[list[Setup]] = []
        for s in group:
            if clusters and abs(s.trigger_price - clusters[-1][0].trigger_price) <= tol:
                clusters[-1].append(s)
            else:
                clusters.append([s])
        for cluster in clusters:
            if len(cluster) == 1:
                merged.append(cluster[0])
                continue
            anchor = max(cluster, key=lambda s: _base_quality_for_kind(s.kind))
            others = [s for s in cluster if s is not anchor]
            anchor.confluence_with = [s.kind for s in others]
            extra = ", ".join(s.label for s in others)
            anchor.rationale.append(f"Confluence: {extra}")
            merged.append(anchor)
    return merged


# ─────────────────────────────────────────────────
# Quality scoring
# ─────────────────────────────────────────────────

# Base quality per setup kind. Higher = more inherently meaningful level.
_BASE_QUALITY = {
    "DONCHIAN_252_HIGH": 60, "DONCHIAN_252_LOW": 60,
    "DONCHIAN_100_HIGH": 50, "DONCHIAN_100_LOW": 50,
    "DONCHIAN_50_HIGH": 45,  "DONCHIAN_50_LOW": 45,
    "DONCHIAN_20_HIGH": 30,  "DONCHIAN_20_LOW": 30,
    "MA_200_RECLAIM": 45,    "MA_200_LOSS": 50,
    "MA_50_RECLAIM": 35,     "MA_50_LOSS": 40,
    "MA_20_RECLAIM": 25,     "MA_20_LOSS": 25,
    "HORIZONTAL_RESISTANCE": 40, "HORIZONTAL_SUPPORT": 40,
    "WEEKLY_52W_HIGH": 65, "WEEKLY_52W_LOW": 65,
    "WEEKLY_20W_HIGH": 50, "WEEKLY_20W_LOW": 50,
    "GAP_FILL_UP": 30, "GAP_FILL_DOWN": 30,
    "SQUEEZE_UPPER": 35, "SQUEEZE_LOWER": 35,
    "AVWAP_FROM_HIGH": 45, "AVWAP_FROM_LOW": 45,
    "ROUND_NUMBER_ABOVE": 25, "ROUND_NUMBER_BELOW": 25,
    "TRENDLINE_DESCENDING": 50, "TRENDLINE_ASCENDING": 50,
    "REVERSAL_BULL_HAMMER": 30,
    "REVERSAL_BULL_BULLISH_ENGULFING": 30,
    "REVERSAL_BEAR_SHOOTING_STAR": 30,
    "REVERSAL_BEAR_BEARISH_ENGULFING": 30,
    # Continuation breakouts (upper rail of ascending / lower rail of descending).
    # Bumped above DONCHIAN_252_HIGH/LOW (60) so the channel wins as the merge
    # anchor when its rail overlaps with a Donchian high/low — otherwise the
    # channel kind and its line_meta (for diagonal rendering) get absorbed.
    "CHANNEL_ASCENDING_UPPER": 62,
    "CHANNEL_DESCENDING_LOWER": 62,
    # Trend-reversal breakouts (opposite-rail of established channel)
    "CHANNEL_ASCENDING_LOWER": 68,
    "CHANNEL_DESCENDING_UPPER": 68,
    # Weekly channels — multi-month/year geometry, harder to coincidence and
    # more respected by institutional flow. Bumped above WEEKLY_52W_HIGH (65)
    # for the same anchor-survival reason.
    "CHANNEL_WEEKLY_ASCENDING_UPPER": 72,
    "CHANNEL_WEEKLY_DESCENDING_LOWER": 72,
    "CHANNEL_WEEKLY_ASCENDING_LOWER": 80,
    "CHANNEL_WEEKLY_DESCENDING_UPPER": 80,
    # Forming channels — 2+1 touch geometry (one rail real, one constructed
    # parallel through a single swing). Lower base than confirmed so they
    # show up in the setup map but rarely beat real 4+ channels for the
    # headline. Continuation < Reversal within each tier, same as confirmed.
    "CHANNEL_ASCENDING_UPPER_FORMING": 42,
    "CHANNEL_DESCENDING_LOWER_FORMING": 42,
    "CHANNEL_ASCENDING_LOWER_FORMING": 48,
    "CHANNEL_DESCENDING_UPPER_FORMING": 48,
    "CHANNEL_WEEKLY_ASCENDING_UPPER_FORMING": 52,
    "CHANNEL_WEEKLY_DESCENDING_LOWER_FORMING": 52,
    "CHANNEL_WEEKLY_ASCENDING_LOWER_FORMING": 58,
    "CHANNEL_WEEKLY_DESCENDING_UPPER_FORMING": 58,
}


def _base_quality_for_kind(kind: str) -> int:
    if kind in _BASE_QUALITY:
        return _BASE_QUALITY[kind]
    if kind.startswith("PATTERN_"):
        return 40  # base for patterns; pattern confidence shapes the rest
    return 25


def score_quality(setup: Setup, context: dict) -> Setup:
    """Compute the 0-100 quality score for a setup.

    context provides:
        last_price, atr, ma_stack, rs_trend_20d, rs_leading, vol_ratio,
        pattern_confidence_by_kind (dict)
    """
    base = _base_quality_for_kind(setup.kind)
    score = float(base)

    # Pattern-specific confidence override
    if setup.kind.startswith("PATTERN_"):
        conf = context.get("pattern_confidence_by_kind", {}).get(setup.kind, 0.5)
        score = 20 + conf * 60   # 0.5 conf → 50, 1.0 conf → 80

    # Maturity boost: levels held for longer mean more
    if setup.maturity_bars:
        if setup.maturity_bars >= 200:
            score += 10
        elif setup.maturity_bars >= 80:
            score += 6
        elif setup.maturity_bars >= 30:
            score += 3

    # Tests boost — up to 4 tests strengthens; beyond that the level is "tired"
    if setup.tests is not None:
        if 2 <= setup.tests <= 4:
            score += 8
        elif setup.tests >= 5:
            score += 4   # still meaningful but less fresh

    # Proximity boost — closer = more actionable
    if setup.proximity_class == "IMMEDIATE":
        score += 10
    elif setup.proximity_class == "NEAR":
        score += 6
    elif setup.proximity_class == "WATCH":
        score += 2

    # Trend / RS alignment
    ma_stack = context.get("ma_stack", "MIXED")
    rs_trend = context.get("rs_trend_20d", "FLAT")
    rs_leading = context.get("rs_leading", False)
    if setup.direction == "BREAKOUT":
        if ma_stack == "BULLISH_ALIGNED":
            score += 8
        elif ma_stack == "PARTIALLY_ALIGNED":
            score += 4
        elif ma_stack == "BEARISH_ALIGNED":
            score -= 8
        if rs_trend == "UP":
            score += 6
        elif rs_trend == "DOWN":
            score -= 4
        if rs_leading:
            score += 4
    else:
        if ma_stack == "BEARISH_ALIGNED":
            score += 8
        elif ma_stack == "BULLISH_ALIGNED":
            score -= 8
        if rs_trend == "DOWN":
            score += 6
        elif rs_trend == "UP":
            score -= 4

    # Confluence boost
    if setup.confluence_with:
        score += min(12, 4 * len(setup.confluence_with))

    score = max(0.0, min(100.0, score))
    setup.quality = round(score, 1)
    setup.quality_label = _quality_label(score)
    return setup


# ─────────────────────────────────────────────────
# Top-level entrypoint
# ─────────────────────────────────────────────────

def find_all_setups(
    df: pd.DataFrame,
    df_weekly: pd.DataFrame | None,
    pattern_results: list,
    signals: dict,
    deep: dict,
    cfg: dict,
) -> list[Setup]:
    """Run every detection source, merge confluences, score quality, return ranked list."""
    if df is None or df.empty:
        return []
    atr = _atr(df, period=14)
    last = float(df["Close"].iloc[-1])

    donchian_periods = cfg.get("indicators", {}).get("donchian_periods", [20, 50, 252])
    # Add 100 if not present — useful intermediate
    if 100 not in donchian_periods:
        donchian_periods = sorted(set(donchian_periods) | {100})
    ma_periods = cfg.get("indicators", {}).get("ma_periods", [20, 50, 200])

    setups: list[Setup] = []
    setups += find_donchian_setups(df, atr, donchian_periods)
    setups += find_ma_setups(df, atr, ma_periods)
    setups += find_pattern_setups(pattern_results, df, atr)
    setups += find_horizontal_setups(df, atr)
    setups += find_weekly_setups(df_weekly, atr)
    setups += find_gap_setups(df, atr)
    setups += find_squeeze_setups(df, signals, atr)
    # New sources (added in the "five-source" pass)
    setups += find_anchored_vwap_setups(df, atr)
    # Round numbers disabled — added too much chart noise without enough signal.
    # Function kept in source for now in case we want to bring it back.
    # setups += find_round_number_setups(df, atr)
    setups += find_trendline_setups(df, atr)
    setups += find_reversal_candle_setups(df, atr, signals)
    setups += find_channel_setups(df, atr, timeframe="daily")
    # Weekly channels — higher-signal structural read on multi-month geometry.
    if df_weekly is not None and not df_weekly.empty and len(df_weekly) >= 25:
        atr_weekly = _atr(df_weekly, period=14)
        if atr_weekly and atr_weekly == atr_weekly and atr_weekly > 0:
            setups += find_channel_setups(
                df_weekly, atr_weekly, lookback=104, timeframe="weekly",
            )

    setups = merge_confluence(setups, atr)

    # Build pattern-confidence lookup so score_quality can refer to it.
    pat_conf: dict[str, float] = {}
    for p in pattern_results or []:
        is_bear = p.pattern in _BEARISH_PATTERNS
        kind = f"PATTERN_{'BEAR' if is_bear else 'BULL'}_{p.pattern.replace(' ', '_').upper()}"
        eff = float(p.confidence)
        if "[weekly]" in (p.notes or ""):
            eff = min(1.0, eff * 1.2)
        # Keep the max if duplicates
        pat_conf[kind] = max(pat_conf.get(kind, 0.0), eff)

    context = {
        "last_price": last,
        "atr": atr,
        "ma_stack": deep.get("ma_stack", "MIXED"),
        "rs_trend_20d": deep.get("rs_trend_20d", "FLAT"),
        "rs_leading": deep.get("rs_leading", False),
        "pattern_confidence_by_kind": pat_conf,
    }
    for s in setups:
        score_quality(s, context)

    # Sort: primary by proximity_class (IMMEDIATE first), secondary by quality desc
    prox_order = {"IMMEDIATE": 0, "NEAR": 1, "WATCH": 2, "FAR": 3}
    setups.sort(key=lambda s: (prox_order.get(s.proximity_class, 9), -s.quality))
    return setups


def split_by_direction(setups: list[Setup]) -> tuple[list[Setup], list[Setup]]:
    """Pending (non-triggered) setups, split by direction."""
    breakouts = [s for s in setups if s.direction == "BREAKOUT" and not s.is_triggered]
    breakdowns = [s for s in setups if s.direction == "BREAKDOWN" and not s.is_triggered]
    return breakouts, breakdowns


def triggered_recently(setups: list[Setup]) -> list[Setup]:
    """Setups whose trigger was crossed within the recent lookback window.
    Sorted by recency (most recent first), then by quality."""
    trig = [s for s in setups if s.is_triggered]
    trig.sort(key=lambda s: (s.bars_since_trigger or 0, -s.quality))
    return trig


# ─────────────────────────────────────────────────
# Headline takeaway — the single highest-signal read for a ticker
# ─────────────────────────────────────────────────

# Color palette mirrors app.py constants so we can render the same way.
_COLOR_BULL_STRONG = "#00ff8c"     # PHOSPHOR
_COLOR_BULL_SOFT = "#9aff5a"       # LIME
_COLOR_BEAR_STRONG = "#ff3b3b"     # WARN
_COLOR_BEAR_SOFT = "#ff7a7a"
_COLOR_NEUTRAL = "#ffb000"         # AMBER


def headline_takeaway(setups: list[Setup], last_price: float) -> dict:
    """Pick the single most informative read for the ticker.

    Priority:
      1. Fresh trigger today (bars_since_trigger == 0)
      2. Channel structure (highest-touch wins)
      3. IMMEDIATE pending setup with STRONG quality
      4. NEAR pending setup with STRONG quality
      5. Highest-quality setup of any proximity
      6. Fallback: range / no clear bias
    """
    triggered = [s for s in setups if s.is_triggered]
    pending = [s for s in setups if not s.is_triggered]

    # ── 1. Fresh trigger today ──────────────────────────────────────
    # DAILY timeframe only: a weekly setup with bars_since_trigger == 0
    # means "this week" (could be Monday's cross showing all week), not a
    # genuine same-day breakout. Weekly structure is represented by the
    # channel/weekly tiers below — it shouldn't claim the FRESH BREAKOUT
    # headline, which is specifically a "this just happened today" signal.
    fresh_today = [
        s for s in triggered
        if s.bars_since_trigger == 0 and s.timeframe == "daily"
    ]
    if fresh_today:
        s = max(fresh_today, key=lambda x: x.quality)
        if s.direction == "BREAKOUT":
            return {
                "title": "FRESH BREAKOUT",
                "bias": "BULLISH",
                "detail": f"{s.label}",
                "level": f"${s.trigger_price:.2f}",
                "subdetail": s.trigger_condition,
                "color": _COLOR_BULL_STRONG,
                "icon": "▲",
            }
        return {
            "title": "FRESH BREAKDOWN",
            "bias": "BEARISH",
            "detail": f"{s.label}",
            "level": f"${s.trigger_price:.2f}",
            "subdetail": s.trigger_condition,
            "color": _COLOR_BEAR_STRONG,
            "icon": "▼",
        }

    # ── 2. Channel structure (weekly-confirmed > daily-confirmed > forming) ─
    def _is_forming(s):
        return s.kind.endswith("_FORMING")
    weekly_confirmed = [s for s in setups if s.kind.startswith("CHANNEL_WEEKLY_") and not _is_forming(s)]
    daily_confirmed = [s for s in setups if s.kind.startswith("CHANNEL_") and not s.kind.startswith("CHANNEL_WEEKLY_") and not _is_forming(s)]
    weekly_forming = [s for s in setups if s.kind.startswith("CHANNEL_WEEKLY_") and _is_forming(s)]
    daily_forming = [s for s in setups if s.kind.startswith("CHANNEL_") and not s.kind.startswith("CHANNEL_WEEKLY_") and _is_forming(s)]
    channels = weekly_confirmed or daily_confirmed or weekly_forming or daily_forming
    is_weekly = bool(weekly_confirmed or weekly_forming)
    is_forming = bool(channels) and not (weekly_confirmed or daily_confirmed)
    if channels:
        upper = next((s for s in channels if "UPPER" in s.kind), None)
        lower = next((s for s in channels if "LOWER" in s.kind), None)
        if upper and lower:
            is_ascending = "ASCENDING" in upper.kind
            touches = max((upper.tests or 0) + (lower.tests or 0), 3 if is_forming else 4)
            tf_prefix = "WEEKLY " if is_weekly else ""
            span_unit = "weekly" if is_weekly else "daily"
            forming_suffix = " (FORMING)" if is_forming else ""
            forming_detail = (
                f"Forming structure, {touches}× touches — one rail constructed parallel"
                if is_forming else
                f"{span_unit.title()} trend intact, {touches}× total rail touches"
            )
            if is_ascending:
                return {
                    "title": f"{tf_prefix}ASCENDING CHANNEL{forming_suffix}",
                    "bias": "BULLISH",
                    "detail": forming_detail,
                    "level": f"${lower.trigger_price:.2f} → ${upper.trigger_price:.2f}",
                    "subdetail": (
                        f"Buy lower rail bounces; upper-rail break = acceleration; "
                        f"lower-rail break = trend reversal"
                    ),
                    "color": _COLOR_BULL_SOFT,
                    "icon": "↗",
                }
            return {
                "title": f"{tf_prefix}DESCENDING CHANNEL{forming_suffix}",
                "bias": "BEARISH",
                "detail": forming_detail,
                "level": f"${upper.trigger_price:.2f} → ${lower.trigger_price:.2f}",
                "subdetail": (
                    f"Sell upper rail rejections; lower-rail break = acceleration; "
                    f"upper-rail break = trend reversal"
                ),
                "color": _COLOR_BEAR_SOFT,
                "icon": "↘",
            }

    # ── 3. IMMEDIATE pending setup with STRONG quality ─────────────
    imm_strong = [
        s for s in pending
        if s.proximity_class == "IMMEDIATE" and s.quality_label == "STRONG"
    ]
    if imm_strong:
        s = max(imm_strong, key=lambda x: x.quality)
        if s.direction == "BREAKOUT":
            return {
                "title": "PRIMED FOR BREAKOUT",
                "bias": "BULLISH",
                "detail": f"At {s.label}",
                "level": f"${s.trigger_price:.2f}",
                "subdetail": s.trigger_condition,
                "color": _COLOR_BULL_STRONG,
                "icon": "▲",
            }
        return {
            "title": "PRIMED FOR BREAKDOWN",
            "bias": "BEARISH",
            "detail": f"At {s.label}",
            "level": f"${s.trigger_price:.2f}",
            "subdetail": s.trigger_condition,
            "color": _COLOR_BEAR_STRONG,
            "icon": "▼",
        }

    # ── 4. NEAR pending setup with STRONG quality ──────────────────
    near_strong = [
        s for s in pending
        if s.proximity_class == "NEAR" and s.quality_label == "STRONG"
    ]
    if near_strong:
        s = max(near_strong, key=lambda x: x.quality)
        if s.direction == "BREAKOUT":
            return {
                "title": "APPROACHING RESISTANCE",
                "bias": "BULLISH-WATCH",
                "detail": f"Closest: {s.label}",
                "level": f"${s.trigger_price:.2f}",
                "subdetail": f"{s.distance_pct:+.2f}% away — {s.trigger_condition}",
                "color": _COLOR_BULL_SOFT,
                "icon": "▲",
            }
        return {
            "title": "APPROACHING SUPPORT",
            "bias": "BEARISH-WATCH",
            "detail": f"Closest: {s.label}",
            "level": f"${s.trigger_price:.2f}",
            "subdetail": f"{s.distance_pct:+.2f}% away — {s.trigger_condition}",
            "color": _COLOR_BEAR_SOFT,
            "icon": "▼",
        }

    # ── 5. Highest-quality setup of any proximity ──────────────────
    if pending:
        s = max(pending, key=lambda x: x.quality)
        if s.quality >= 50:
            if s.direction == "BREAKOUT":
                return {
                    "title": "BULLISH BIAS",
                    "bias": "BULLISH-WATCH",
                    "detail": f"Best setup: {s.label}",
                    "level": f"${s.trigger_price:.2f}",
                    "subdetail": f"{s.distance_pct:+.2f}% away, quality {s.quality:.0f}",
                    "color": _COLOR_BULL_SOFT,
                    "icon": "▲",
                }
            return {
                "title": "BEARISH BIAS",
                "bias": "BEARISH-WATCH",
                "detail": f"Best setup: {s.label}",
                "level": f"${s.trigger_price:.2f}",
                "subdetail": f"{s.distance_pct:+.2f}% away, quality {s.quality:.0f}",
                "color": _COLOR_BEAR_SOFT,
                "icon": "▼",
            }

    # ── 6. Fallback ────────────────────────────────────────────────
    return {
        "title": "NO CLEAR BIAS",
        "bias": "NEUTRAL",
        "detail": "No high-conviction setup currently in play",
        "level": f"${last_price:.2f}",
        "subdetail": "Watch the setup map for levels approaching",
        "color": _COLOR_NEUTRAL,
        "icon": "◆",
    }
