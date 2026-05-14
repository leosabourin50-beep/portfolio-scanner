"""Classic chart pattern detection — bull flag, ascending triangle, cup & handle.

Each detector scans the recent N days of OHLCV and returns a structured
result (or None). Results carry the geometric key-points so the dashboard
can overlay them on the candlestick chart.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass
class PatternResult:
    pattern: str
    ticker: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    breakout_level: float
    target: float
    status: str          # "Forming", "Breaking out", "Confirmed"
    confidence: float    # 0-1
    key_points: list     # [(timestamp, price, label), ...]
    notes: str = ""
    # The detector's own "still actionable" distance from breakout. Used by
    # scan_patterns to gate Forming patterns. Defaults to the short-duration
    # window (12%); multi-month patterns override this on construction.
    max_distance: float = 0.12

    def to_row(self) -> dict:
        d = asdict(self)
        d.pop("key_points")
        d.pop("max_distance")
        return d


# ─────────────────────────────────────────────────
# Swing point helpers
# ─────────────────────────────────────────────────

def _swing_highs(highs: pd.Series, distance: int = 5, prominence: float = None) -> np.ndarray:
    """Indices of swing highs using scipy.find_peaks."""
    if prominence is None:
        prominence = highs.std() * 0.3
    peaks, _ = find_peaks(highs.values, distance=distance, prominence=prominence)
    return peaks


def _swing_lows(lows: pd.Series, distance: int = 5, prominence: float = None) -> np.ndarray:
    """Indices of swing lows."""
    if prominence is None:
        prominence = lows.std() * 0.3
    peaks, _ = find_peaks(-lows.values, distance=distance, prominence=prominence)
    return peaks


# ─────────────────────────────────────────────────
# Confidence helpers — shared across all detectors
# ─────────────────────────────────────────────────
# Every detector now scores as:
#     confidence = geometry × proximity × volume_quality
# where geometry is a 0-1 pattern-quality score (no saturation) and the
# two helpers below add the "is this actually setting up to fire?" signal
# that the old multipliers ignored. Patterns far from their breakout get
# heavily penalised; clean geometry on a distant pattern is still scored
# low because it isn't actionable.

def _proximity_factor(last_close: float, breakout_level: float, side: str = "up", max_distance: float = 0.12) -> float:
    """0-1 distance-to-trigger factor.

    side='up' for bullish (breakout above breakout_level), 'down' for bearish
    (breakdown below). 1.0 at/past the level, falls linearly to 0 at
    `max_distance` away. Default 12% suits short-duration patterns; widen
    for multi-month patterns (cup & handle, H&S) where price can sit well
    below the trigger while the pattern is still constructively forming.
    """
    if last_close <= 0:
        return 0.0
    if side == "up":
        dist = (breakout_level - last_close) / last_close
    else:
        dist = (last_close - breakout_level) / last_close
    if dist <= 0:
        return 1.0
    if dist >= max_distance:
        return 0.0
    return 1.0 - dist / max_distance


def _volume_quality_factor(df: pd.DataFrame) -> float:
    """0.75-1.05 multiplier based on 10d/40d volume ratio.

    Volume drying up in the base is constructive for both bullish breakouts
    and bearish distribution patterns (rallies failing on lighter volume).
    Returns ~0.9 when neutral, <0.85 when vol is expanding into the base
    (less constructive), up to 1.05 on heavy dry-up.
    """
    vol = df["Volume"]
    if len(vol) < 40:
        return 0.9
    recent = float(vol.iloc[-10:].mean())
    longer = float(vol.iloc[-40:].mean())
    if longer <= 0:
        return 0.9
    ratio = recent / longer
    if ratio >= 1.10:
        return 0.75
    if ratio <= 0.55:
        return 1.05
    # linear: ratio=1.10 → 0.75, ratio=0.55 → 1.05
    return 0.75 + (1.10 - ratio) / 0.55 * 0.30


def _clip01(x: float) -> float:
    """Clamp to [0, 1]."""
    return max(0.0, min(1.0, x))


# Minimum confidence to publish a pattern. Anything below this is more
# noise than signal — silently dropped in scan_patterns.
MIN_PUBLISH_CONFIDENCE = 0.30


# ─────────────────────────────────────────────────
# Bull Flag
# ─────────────────────────────────────────────────
# A bull flag = a strong, fast up-move ("pole") followed by a tight,
# slightly-down-sloping consolidation ("flag"). Breakout above the flag's
# top continues the prior move.

def detect_bull_flag(df: pd.DataFrame, ticker: str) -> Optional[PatternResult]:
    if len(df) < 30:
        return None

    look = df.tail(40).copy()
    close = look["Close"].values
    high = look["High"].values
    low = look["Low"].values
    n = len(look)

    # Find the strongest 3-10 day pole ending in the last 25 days
    best = None
    for pole_len in range(3, 11):
        for pole_end in range(pole_len, n - 4):
            pole_start_price = close[pole_end - pole_len]
            pole_end_price = close[pole_end]
            pole_gain = (pole_end_price / pole_start_price) - 1
            if pole_gain < 0.10:  # need ≥10% pole
                continue
            # Flag = bars after pole_end through end of window
            flag = look.iloc[pole_end : pole_end + 15]
            if len(flag) < 4:
                continue
            flag_high = flag["High"].max()
            flag_low = flag["Low"].min()
            flag_range = (flag_high - flag_low) / pole_end_price
            if flag_range > 0.10:  # flag too wide
                continue
            # Flag should not retrace more than 50% of the pole
            retrace = (pole_end_price - flag_low) / (pole_end_price - pole_start_price)
            if retrace > 0.50:
                continue
            # Geometry quality — weighted mean of normalised components.
            # Each component is 0-1, scaled so that 1.0 = textbook setup.
            flag_end_idx = pole_end + len(flag) - 1
            pole_strength = _clip01((pole_gain - 0.10) / 0.20)         # 10%→0, 30%+→1
            flag_tightness = _clip01((0.10 - flag_range) / 0.07)       # 10% wide→0, 3% tight→1
            retrace_quality = _clip01((0.50 - retrace) / 0.25)         # 50%→0, 25%→1
            recency = _clip01(1 - (n - 1 - flag_end_idx) / 12)         # today→1, 12+ bars old→0
            score = (0.30 * pole_strength + 0.30 * flag_tightness
                     + 0.20 * retrace_quality + 0.20 * recency)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "pole_start": pole_end - pole_len,
                    "pole_end": pole_end,
                    "flag_end": flag_end_idx,
                    "pole_gain": pole_gain,
                    "flag_high": flag_high,
                    "flag_low": flag_low,
                    "flag_range": flag_range,
                    "retrace": retrace,
                    "pole_start_price": pole_start_price,
                    "pole_end_price": pole_end_price,
                }

    if best is None:
        return None

    last_close = float(df["Close"].iloc[-1])
    breakout_level = best["flag_high"]
    pole_height = best["pole_end_price"] - best["pole_start_price"]
    target = breakout_level + pole_height  # measured-move target

    if last_close > breakout_level * 1.01:
        status = "Confirmed"
    elif last_close > breakout_level:
        status = "Breaking out"
    else:
        status = "Forming"

    proximity = _proximity_factor(last_close, breakout_level, "up")
    vol_q = _volume_quality_factor(df)
    confidence = _clip01(best["score"] * proximity * vol_q)

    key_points = [
        (look.index[best["pole_start"]], best["pole_start_price"], "Pole start"),
        (look.index[best["pole_end"]], best["pole_end_price"], "Pole top"),
        (look.index[best["flag_end"]], best["flag_low"], "Flag low"),
    ]

    return PatternResult(
        pattern="Bull Flag",
        ticker=ticker,
        start_date=look.index[best["pole_start"]],
        end_date=look.index[best["flag_end"]],
        breakout_level=round(breakout_level, 2),
        target=round(target, 2),
        status=status,
        confidence=round(confidence, 2),
        key_points=key_points,
        notes=f"+{best['pole_gain']*100:.1f}% pole, {best['flag_range']*100:.1f}% flag",
    )


# ─────────────────────────────────────────────────
# Ascending Triangle
# ─────────────────────────────────────────────────
# A horizontal resistance (≥3 swing highs at similar levels) + a rising
# support line (higher lows). Bullish continuation; breaks above resistance.

def detect_ascending_triangle(df: pd.DataFrame, ticker: str) -> Optional[PatternResult]:
    if len(df) < 30:
        return None

    look = df.tail(90).copy()
    highs = look["High"]
    lows = look["Low"]
    if len(look) < 30:
        return None

    sh = _swing_highs(highs, distance=4)
    sl = _swing_lows(lows, distance=4)
    if len(sh) < 3 or len(sl) < 2:
        return None

    sh = sh[-5:]
    sl = sl[-5:]

    sh_prices = highs.iloc[sh].values
    resistance = sh_prices.mean()
    spread = sh_prices.std() / resistance
    if spread > 0.025:  # highs must cluster within ~2.5%
        return None

    sl_prices = lows.iloc[sl].values
    if len(sl_prices) < 2 or not all(sl_prices[i] < sl_prices[i + 1] for i in range(len(sl_prices) - 1)):
        return None

    pattern_start = look.index[min(sh[0], sl[0])]
    last_close = float(df["Close"].iloc[-1])

    if last_close > resistance * 1.005:
        status = "Confirmed"
    elif last_close > resistance:
        status = "Breaking out"
    else:
        status = "Forming"

    triangle_height = resistance - sl_prices[0]
    target = resistance + triangle_height

    # Geometry: more touches + tighter resistance cluster + more rising lows
    touch_quality = _clip01((len(sh) - 2) / 3)               # 2→0, 5+→1
    tightness = _clip01((0.025 - spread) / 0.025)            # 2.5%→0, 0%→1
    low_count = _clip01((len(sl) - 1) / 3)                   # 1→0, 4+→1
    geometry = 0.40 * touch_quality + 0.30 * tightness + 0.30 * low_count
    proximity = _proximity_factor(last_close, resistance, "up")
    vol_q = _volume_quality_factor(df)
    confidence = _clip01(geometry * proximity * vol_q)

    key_points = [(look.index[i], highs.iloc[i], "Resistance") for i in sh]
    key_points += [(look.index[i], lows.iloc[i], "Higher low") for i in sl]

    return PatternResult(
        pattern="Ascending Triangle",
        ticker=ticker,
        start_date=pattern_start,
        end_date=look.index[-1],
        breakout_level=round(resistance, 2),
        target=round(target, 2),
        status=status,
        confidence=round(confidence, 2),
        key_points=key_points,
        notes=f"{len(sh)} resistance touches, {len(sl)} rising lows",
    )


# ─────────────────────────────────────────────────
# Cup and Handle
# ─────────────────────────────────────────────────
# A U-shaped recovery (the cup) back near a prior high, followed by a
# small consolidation/pullback (the handle). Breakout above the cup rim
# continues the prior advance.

def detect_cup_and_handle(df: pd.DataFrame, ticker: str) -> Optional[PatternResult]:
    if len(df) < 60:
        return None

    look = df.tail(150).copy()
    close = look["Close"]
    high = look["High"]
    low = look["Low"]
    n = len(look)

    sh = _swing_highs(high, distance=10, prominence=high.std() * 0.5)
    if len(sh) < 2:
        return None

    best = None
    for i in range(len(sh) - 1):
        for j in range(i + 1, len(sh)):
            left_idx, right_idx = sh[i], sh[j]
            cup_duration = right_idx - left_idx
            if cup_duration < 25 or cup_duration > 130:
                continue
            left_price = high.iloc[left_idx]
            right_price = high.iloc[right_idx]
            if abs(right_price - left_price) / left_price > 0.05:  # rims must match within 5%
                continue
            cup_low = low.iloc[left_idx : right_idx + 1].min()
            cup_low_idx = low.iloc[left_idx : right_idx + 1].idxmin()
            depth = (left_price - cup_low) / left_price
            if depth < 0.12 or depth > 0.35:  # 12-35% drop
                continue

            # Handle: 4-25 days after right rim, shallow pullback (<15%)
            # Loosened from 5 → 4 for the single-name analyzer: deeper-data
            # context lets us tolerate slightly earlier handle detection.
            handle = look.iloc[right_idx + 1 : right_idx + 26]
            if len(handle) < 4:
                continue
            handle_low = handle["Low"].min()
            handle_drop = (right_price - handle_low) / right_price
            if handle_drop < 0.02 or handle_drop > 0.15:
                continue
            handle_end = right_idx + len(handle)

            # Geometry: depth in 20-25% zone is ideal, shallow handle, matched rims, recent
            depth_quality = _clip01(1 - abs(depth - 0.225) / 0.125)    # peak at 22.5%, 10/35→0
            handle_quality = _clip01((0.15 - handle_drop) / 0.13)      # 15%→0, 2%→1
            rim_diff = abs(right_price - left_price) / left_price
            rim_match = _clip01(1 - rim_diff / 0.05)                   # 0%→1, 5%→0
            recency = _clip01(1 - (n - 1 - min(handle_end, n - 1)) / 15)
            score = (0.30 * depth_quality + 0.30 * handle_quality
                     + 0.25 * rim_match + 0.15 * recency)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "left_idx": left_idx,
                    "right_idx": right_idx,
                    "cup_low_idx": cup_low_idx,
                    "cup_low": float(cup_low),
                    "left_price": float(left_price),
                    "right_price": float(right_price),
                    "handle_end": min(handle_end, n - 1),
                    "handle_low": float(handle_low),
                    "depth": depth,
                    "handle_drop": handle_drop,
                }

    if best is None:
        return None

    rim = (best["left_price"] + best["right_price"]) / 2
    cup_height = rim - best["cup_low"]
    target = rim + cup_height
    last_close = float(df["Close"].iloc[-1])

    if last_close > rim * 1.01:
        status = "Confirmed"
    elif last_close > rim:
        status = "Breaking out"
    else:
        status = "Forming"

    proximity = _proximity_factor(last_close, rim, "up", max_distance=0.18)
    vol_q = _volume_quality_factor(df)
    confidence = _clip01(best["score"] * proximity * vol_q)

    key_points = [
        (look.index[best["left_idx"]], best["left_price"], "Left rim"),
        (best["cup_low_idx"], best["cup_low"], "Cup low"),
        (look.index[best["right_idx"]], best["right_price"], "Right rim"),
        (look.index[best["handle_end"]], best["handle_low"], "Handle low"),
    ]

    return PatternResult(
        pattern="Cup & Handle",
        ticker=ticker,
        start_date=look.index[best["left_idx"]],
        end_date=look.index[best["handle_end"]],
        breakout_level=round(rim, 2),
        target=round(target, 2),
        status=status,
        confidence=round(confidence, 2),
        key_points=key_points,
        notes=f"{best['depth']*100:.0f}% cup, {best['handle_drop']*100:.1f}% handle",
        max_distance=0.18,
    )


# ─────────────────────────────────────────────────
# VCP (Volatility Contraction Pattern)
# ─────────────────────────────────────────────────
# Mark Minervishi pattern. A series of contractions where each pullback
# is shallower than the last and volume dries up across each contraction.
# The breakout comes on the final, tightest contraction.
#
# Geometry:
#   - 3-4 swing lows, each higher than the last
#   - Each pullback depth (from the preceding swing high) is smaller
#     than the prior pullback depth
#   - Volume declining across contractions
#   - Price near the pivot high (resistance) at pattern completion
#
# This is the highest-conviction breakout pattern because it shows
# sellers exhausting across successive attempts to push price down.

def detect_vcp(df: pd.DataFrame, ticker: str) -> Optional[PatternResult]:
    if len(df) < 60:
        return None

    look = df.tail(160).copy()
    high = look["High"]
    low = look["Low"]
    close = look["Close"]
    volume = look["Volume"]
    n = len(look)

    # Find swing highs and lows
    sh = _swing_highs(high, distance=5, prominence=high.std() * 0.25)
    sl = _swing_lows(low, distance=5, prominence=low.std() * 0.25)

    if len(sh) < 2 or len(sl) < 2:
        return None

    # Use the last 6 swing highs and lows
    sh = sh[-6:]
    sl = sl[-6:]

    sh_prices = high.iloc[sh].values
    sl_prices = low.iloc[sl].values

    # Find the pivot high (highest swing high in the window — the resistance level)
    pivot_idx = sh[np.argmax(sh_prices)]
    pivot_price = float(high.iloc[pivot_idx])

    # Only consider swing lows AFTER the pivot high
    sl_after = sl[sl > pivot_idx]
    if len(sl_after) < 2:
        return None

    sl_after = sl_after[-4:]  # last 4 contractions max
    sl_after_prices = low.iloc[sl_after].values

    # Compute pullback depths: each measured from the pivot high
    pullback_depths = [(pivot_price - p) / pivot_price * 100 for p in sl_after_prices]

    # Requirement 1: each successive pullback must be shallower
    contracting = all(pullback_depths[i] > pullback_depths[i + 1] for i in range(len(pullback_depths) - 1))
    if not contracting:
        return None

    # Requirement 2: first pullback must be meaningful (at least 8%), last must be tight (under 8%)
    if pullback_depths[0] < 8.0:
        return None
    if pullback_depths[-1] > 12.0:
        return None

    # Requirement 3: swing lows should be rising (higher lows)
    rising_lows = all(sl_after_prices[i] < sl_after_prices[i + 1] for i in range(len(sl_after_prices) - 1))
    if not rising_lows:
        return None

    # Requirement 4 (soft): volume should generally decline across contractions
    vol_declining = 0
    for i in range(len(sl_after) - 1):
        # Average volume around each swing low (±3 bars)
        v1_start = max(0, sl_after[i] - 3)
        v1_end = min(n, sl_after[i] + 4)
        v2_start = max(0, sl_after[i + 1] - 3)
        v2_end = min(n, sl_after[i + 1] + 4)
        vol1 = volume.iloc[v1_start:v1_end].mean()
        vol2 = volume.iloc[v2_start:v2_end].mean()
        if vol2 < vol1:
            vol_declining += 1

    vol_score = vol_declining / (len(sl_after) - 1)  # 0-1, higher = better

    # Geometry — VCP integrates volume into its own pattern definition
    # (declining vol across contractions IS part of the pattern), so we
    # include vol_score here and skip the generic volume factor below.
    num_contractions = len(sl_after)
    contraction_ratio = pullback_depths[-1] / pullback_depths[0]
    contractions_score = _clip01((num_contractions - 1) / 3)              # 1→0, 4+→1
    tightness = _clip01(1 - contraction_ratio)                            # ratio=1→0, ratio=0→1
    recency = _clip01(1 - (n - 1 - sl_after[-1]) / 15)
    geometry = (0.30 * contractions_score + 0.35 * tightness
                + 0.20 * vol_score + 0.15 * recency)

    if geometry < 0.20:
        return None

    # Breakout level = pivot high
    breakout_level = pivot_price
    last_close = float(df["Close"].iloc[-1])

    # Target: measured move = pivot high + (pivot high - deepest low)
    deepest_low = float(min(sl_after_prices))
    pattern_height = pivot_price - deepest_low
    target = pivot_price + pattern_height

    # Distance to breakout (VCP requires actionable proximity by definition)
    pct_to_breakout = (breakout_level - last_close) / last_close * 100

    if last_close > breakout_level * 1.01:
        status = "Confirmed"
    elif last_close > breakout_level:
        status = "Breaking out"
    elif pct_to_breakout < 3.0:
        status = "Forming"
    else:
        return None  # too far from breakout to be actionable

    proximity = _proximity_factor(last_close, breakout_level, "up")
    confidence = _clip01(geometry * proximity)

    key_points = [(look.index[pivot_idx], pivot_price, "Pivot high")]
    for i, idx in enumerate(sl_after):
        key_points.append((look.index[idx], float(low.iloc[idx]),
                          f"T{i+1}: -{pullback_depths[i]:.1f}%"))

    return PatternResult(
        pattern="VCP",
        ticker=ticker,
        start_date=look.index[pivot_idx],
        end_date=look.index[-1],
        breakout_level=round(breakout_level, 2),
        target=round(target, 2),
        status=status,
        confidence=round(confidence, 2),
        key_points=key_points,
        notes=f"{num_contractions} contractions: {' → '.join(f'{d:.0f}%' for d in pullback_depths)} | Vol trend: {'declining' if vol_score > 0.5 else 'mixed'}",
    )


# ─────────────────────────────────────────────────
# High Tight Flag
# ─────────────────────────────────────────────────
# A stock doubles (80-100%+ in 4-8 weeks), then pulls back less than
# 20-25% over 3-5 weeks on declining volume. The breakout above the
# flag continues the monster move. Rare but extremely powerful.
#
# Detection:
#   - Pole: 80%+ gain over 15-45 trading days
#   - Flag: <25% pullback from the pole high
#   - Flag duration: 10-30 trading days
#   - Volume should decline during the flag
#   - Breakout = close above flag high

def detect_high_tight_flag(df: pd.DataFrame, ticker: str) -> Optional[PatternResult]:
    if len(df) < 60:
        return None

    look = df.tail(100).copy()
    close = look["Close"].values
    high = look["High"].values
    low = look["Low"].values
    volume = look["Volume"].values
    n = len(look)

    best = None

    # Scan for poles: 80%+ gain over 15-45 days
    for pole_len in range(15, 46):
        for pole_end in range(pole_len, n - 9):  # need at least 10 bars for the flag
            pole_start_price = close[pole_end - pole_len]
            pole_end_price = high[pole_end]  # use high for the pole peak
            pole_gain = (pole_end_price / pole_start_price) - 1

            if pole_gain < 0.80:  # need 80%+ gain
                continue

            # Flag: bars after the pole peak
            flag_start = pole_end + 1
            # Flag can be 10-30 bars
            for flag_len in range(10, min(31, n - flag_start + 1)):
                flag_end = flag_start + flag_len - 1
                if flag_end >= n:
                    break

                flag_slice_low = low[flag_start:flag_end + 1]
                flag_slice_high = high[flag_start:flag_end + 1]
                flag_low = np.min(flag_slice_low)
                flag_high = np.max(flag_slice_high)

                # Flag pullback must be less than 25% from the pole peak
                pullback = (pole_end_price - flag_low) / pole_end_price
                if pullback > 0.25:
                    continue
                if pullback < 0.05:  # need some pullback, not just continuation
                    continue

                # Flag should be tighter than the pole (range < 15% of pole peak)
                flag_range = (flag_high - flag_low) / pole_end_price
                if flag_range > 0.15:
                    continue

                # Volume declining in flag vs pole (soft check, kept for notes)
                pole_vol_avg = np.mean(volume[pole_end - pole_len:pole_end + 1])
                flag_vol_avg = np.mean(volume[flag_start:flag_end + 1])
                vol_decline = flag_vol_avg < pole_vol_avg * 0.8

                # Geometry — pole power, tight pullback, tight flag, freshness
                pole_strength = _clip01((pole_gain - 0.80) / 0.70)         # 80%→0, 150%+→1
                pullback_quality = _clip01((0.25 - pullback) / 0.20)       # 25%→0, 5%→1
                flag_tightness = _clip01((0.15 - flag_range) / 0.12)       # 15%→0, 3%→1
                recency = _clip01(1 - (n - 1 - flag_end) / 15)
                score = (0.30 * pole_strength + 0.25 * pullback_quality
                         + 0.25 * flag_tightness + 0.20 * recency)

                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "pole_start": pole_end - pole_len,
                        "pole_end": pole_end,
                        "flag_start": flag_start,
                        "flag_end": flag_end,
                        "pole_gain": pole_gain,
                        "pole_start_price": float(pole_start_price),
                        "pole_end_price": float(pole_end_price),
                        "flag_high": float(flag_high),
                        "flag_low": float(flag_low),
                        "pullback": pullback,
                        "vol_decline": vol_decline,
                    }

    if best is None:
        return None

    breakout_level = best["flag_high"]
    # Target: measured move = flag high + pole height
    pole_height = best["pole_end_price"] - best["pole_start_price"]
    target = breakout_level + pole_height
    last_close = float(df["Close"].iloc[-1])

    if last_close > breakout_level * 1.01:
        status = "Confirmed"
    elif last_close > breakout_level:
        status = "Breaking out"
    else:
        status = "Forming"

    proximity = _proximity_factor(last_close, breakout_level, "up")
    vol_q = _volume_quality_factor(df)
    confidence = _clip01(best["score"] * proximity * vol_q)

    key_points = [
        (look.index[best["pole_start"]], best["pole_start_price"], "Pole start"),
        (look.index[best["pole_end"]], best["pole_end_price"], "Pole peak"),
        (look.index[best["flag_end"]], best["flag_low"], "Flag low"),
    ]

    return PatternResult(
        pattern="High Tight Flag",
        ticker=ticker,
        start_date=look.index[best["pole_start"]],
        end_date=look.index[best["flag_end"]],
        breakout_level=round(breakout_level, 2),
        target=round(target, 2),
        status=status,
        confidence=round(confidence, 2),
        key_points=key_points,
        notes=f"+{best['pole_gain']*100:.0f}% pole, {best['pullback']*100:.1f}% pullback, vol {'declining' if best['vol_decline'] else 'flat'}",
    )


# ─────────────────────────────────────────────────
# Symmetrical Triangle / Pennant
# ─────────────────────────────────────────────────
# Converging trendlines — lower highs AND higher lows squeezing into
# an apex. Unlike an ascending triangle (flat top), this is neutral
# until it breaks. Direction of the break determines the trade.
#
# Detection:
#   - 3+ swing highs that are sequentially lower
#   - 3+ swing lows that are sequentially higher
#   - The lines converge (would intersect within a reasonable horizon)
#   - Breakout = close beyond either trendline

def detect_symmetrical_triangle(df: pd.DataFrame, ticker: str) -> Optional[PatternResult]:
    if len(df) < 40:
        return None

    look = df.tail(100).copy()
    high = look["High"]
    low = look["Low"]
    close = look["Close"]
    n = len(look)

    sh = _swing_highs(high, distance=4, prominence=high.std() * 0.25)
    sl = _swing_lows(low, distance=4, prominence=low.std() * 0.25)

    if len(sh) < 3 or len(sl) < 3:
        return None

    # Use last 5 of each
    sh = sh[-5:]
    sl = sl[-5:]

    sh_prices = high.iloc[sh].values
    sl_prices = low.iloc[sl].values

    # Requirement: swing highs must be descending (lower highs)
    descending_highs = all(sh_prices[i] > sh_prices[i + 1] for i in range(len(sh_prices) - 1))
    if not descending_highs:
        # Try subsets — at least 3 consecutive descending highs
        found_desc = False
        for start in range(len(sh_prices) - 2):
            subset = sh_prices[start:start + 3]
            if all(subset[i] > subset[i + 1] for i in range(len(subset) - 1)):
                sh = sh[start:start + 3]
                sh_prices = subset
                found_desc = True
                break
        if not found_desc:
            return None

    # Requirement: swing lows must be ascending (higher lows)
    ascending_lows = all(sl_prices[i] < sl_prices[i + 1] for i in range(len(sl_prices) - 1))
    if not ascending_lows:
        found_asc = False
        for start in range(len(sl_prices) - 2):
            subset = sl_prices[start:start + 3]
            if all(subset[i] < subset[i + 1] for i in range(len(subset) - 1)):
                sl = sl[start:start + 3]
                sl_prices = subset
                found_asc = True
                break
        if not found_asc:
            return None

    # Fit trendlines using linear regression
    # Upper trendline through swing highs
    upper_x = sh.astype(float)
    upper_y = sh_prices.astype(float)
    upper_slope, upper_intercept = np.polyfit(upper_x, upper_y, 1)

    # Lower trendline through swing lows
    lower_x = sl.astype(float)
    lower_y = sl_prices.astype(float)
    lower_slope, lower_intercept = np.polyfit(lower_x, lower_y, 1)

    # Upper must be declining, lower must be rising
    if upper_slope >= 0 or lower_slope <= 0:
        return None

    # Lines must converge (they will since slopes have opposite signs)
    # Check they haven't already crossed
    upper_now = upper_slope * (n - 1) + upper_intercept
    lower_now = lower_slope * (n - 1) + lower_intercept
    if lower_now >= upper_now:
        return None  # already crossed, pattern is broken

    # The triangle shouldn't be too wide or too narrow at current bar
    triangle_width = (upper_now - lower_now) / ((upper_now + lower_now) / 2) * 100
    if triangle_width > 20 or triangle_width < 1:
        return None

    # Apex: where the lines would meet
    # upper_slope * x + upper_intercept = lower_slope * x + lower_intercept
    if (upper_slope - lower_slope) != 0:
        apex_x = (lower_intercept - upper_intercept) / (upper_slope - lower_slope)
        bars_to_apex = apex_x - (n - 1)
    else:
        return None

    # Should be within reasonable range (5-40 bars to apex)
    if bars_to_apex < 2 or bars_to_apex > 60:
        return None

    last_close = float(df["Close"].iloc[-1])
    upper_breakout = float(upper_now)
    lower_breakout = float(lower_now)

    # Pattern height at widest point
    pattern_start_idx = min(sh[0], sl[0])
    upper_at_start = upper_slope * pattern_start_idx + upper_intercept
    lower_at_start = lower_slope * pattern_start_idx + lower_intercept
    pattern_height = upper_at_start - lower_at_start

    if last_close > upper_breakout * 1.005:
        status = "Confirmed"
        breakout_level = upper_breakout
        target = upper_breakout + pattern_height
    elif last_close > upper_breakout:
        status = "Breaking out"
        breakout_level = upper_breakout
        target = upper_breakout + pattern_height
    elif last_close < lower_breakout * 0.995:
        status = "Confirmed"  # breakdown confirmed
        breakout_level = lower_breakout
        target = lower_breakout - pattern_height
    elif last_close < lower_breakout:
        status = "Breaking out"
        breakout_level = lower_breakout
        target = lower_breakout - pattern_height
    else:
        status = "Forming"
        breakout_level = upper_breakout  # default to upside
        target = upper_breakout + pattern_height

    # Geometry: R² of trendline fits + touch count + recency
    upper_residuals = upper_y - (upper_slope * upper_x + upper_intercept)
    lower_residuals = lower_y - (lower_slope * lower_x + lower_intercept)
    upper_fit = 1 - np.var(upper_residuals) / max(np.var(upper_y), 1e-10)
    lower_fit = 1 - np.var(lower_residuals) / max(np.var(lower_y), 1e-10)
    fit_score = _clip01((upper_fit + lower_fit) / 2)
    touch_score = _clip01((len(sh) + len(sl) - 4) / 4)             # 4→0, 8+→1
    last_swing_idx = int(max(sh[-1], sl[-1]))
    recency = _clip01(1 - (n - 1 - last_swing_idx) / 15)
    geometry = 0.40 * fit_score + 0.40 * touch_score + 0.20 * recency
    # Either-way breakout: use the closer line for proximity
    prox_up = _proximity_factor(last_close, upper_breakout, "up")
    prox_dn = _proximity_factor(last_close, lower_breakout, "down")
    proximity = max(prox_up, prox_dn)
    vol_q = _volume_quality_factor(df)
    confidence = _clip01(geometry * proximity * vol_q)

    key_points = [(look.index[i], float(high.iloc[i]), "Lower high") for i in sh]
    key_points += [(look.index[i], float(low.iloc[i]), "Higher low") for i in sl]

    direction = "bullish" if last_close > (upper_breakout + lower_breakout) / 2 else "bearish"

    return PatternResult(
        pattern="Symmetrical Triangle",
        ticker=ticker,
        start_date=look.index[pattern_start_idx],
        end_date=look.index[-1],
        breakout_level=round(breakout_level, 2),
        target=round(target, 2),
        status=status,
        confidence=round(confidence, 2),
        key_points=key_points,
        notes=f"{len(sh)} lower highs, {len(sl)} higher lows | {bars_to_apex:.0f} bars to apex | Bias: {direction}",
    )


# ─────────────────────────────────────────────────
# Double Bottom
# ─────────────────────────────────────────────────
# Two lows at roughly the same price separated by a rally. The breakout
# triggers when price clears the peak between the two lows (the neckline).
# A reversal pattern — catches turns off lows.
#
# Detection:
#   - Two swing lows within 3% of each other
#   - Separated by 15-65 bars
#   - A swing high between them defines the neckline
#   - The valley between them must be at least 8% below the neckline
#   - Breakout = close above the neckline

def detect_double_bottom(df: pd.DataFrame, ticker: str) -> Optional[PatternResult]:
    if len(df) < 40:
        return None

    look = df.tail(130).copy()
    high = look["High"]
    low = look["Low"]
    close = look["Close"]
    n = len(look)

    sl = _swing_lows(low, distance=7, prominence=low.std() * 0.3)
    sh = _swing_highs(high, distance=5, prominence=high.std() * 0.3)

    if len(sl) < 2 or len(sh) < 1:
        return None

    best = None

    # Try all pairs of swing lows
    for i in range(len(sl)):
        for j in range(i + 1, len(sl)):
            left_idx = sl[i]
            right_idx = sl[j]
            separation = right_idx - left_idx

            if separation < 15 or separation > 65:
                continue

            left_price = float(low.iloc[left_idx])
            right_price = float(low.iloc[right_idx])

            # Lows must match within 3%
            avg_low = (left_price + right_price) / 2
            if abs(left_price - right_price) / avg_low > 0.03:
                continue

            # Find the highest swing high between the two lows (the neckline)
            between_sh = sh[(sh > left_idx) & (sh < right_idx)]
            if len(between_sh) == 0:
                # No swing high found; use the max high between the two lows
                neckline_idx = left_idx + np.argmax(high.iloc[left_idx:right_idx + 1].values)
                neckline_price = float(high.iloc[neckline_idx])
            else:
                neckline_idx = between_sh[np.argmax(high.iloc[between_sh].values)]
                neckline_price = float(high.iloc[neckline_idx])

            # Depth: neckline to average bottom must be at least 8%
            depth = (neckline_price - avg_low) / neckline_price
            if depth < 0.08 or depth > 0.40:
                continue

            # The right bottom should ideally not be much lower than the left
            # (a lower low suggests continued downtrend, not a double bottom)
            if right_price < left_price * 0.97:
                continue

            # Geometry: rim match, depth, recency — each normalised 0-1
            match_quality = _clip01(1 - abs(left_price - right_price) / avg_low / 0.03)
            depth_quality = _clip01((depth - 0.08) / 0.17)             # 8%→0, 25%+→1
            recency = _clip01(1 - (n - 1 - right_idx) / 20)
            score = 0.35 * match_quality + 0.35 * depth_quality + 0.30 * recency

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "left_idx": left_idx,
                    "right_idx": right_idx,
                    "neckline_idx": neckline_idx,
                    "left_price": left_price,
                    "right_price": right_price,
                    "neckline_price": neckline_price,
                    "depth": depth,
                    "match_quality": match_quality,
                }

    if best is None:
        return None

    breakout_level = best["neckline_price"]
    pattern_height = best["neckline_price"] - (best["left_price"] + best["right_price"]) / 2
    target = breakout_level + pattern_height
    last_close = float(df["Close"].iloc[-1])

    if last_close > breakout_level * 1.01:
        status = "Confirmed"
    elif last_close > breakout_level:
        status = "Breaking out"
    else:
        # Only report if we're within striking distance or recently formed
        pct_to_breakout = (breakout_level - last_close) / last_close * 100
        if pct_to_breakout > 8.0:
            return None
        status = "Forming"

    proximity = _proximity_factor(last_close, breakout_level, "up", max_distance=0.15)
    vol_q = _volume_quality_factor(df)
    confidence = _clip01(best["score"] * proximity * vol_q)

    key_points = [
        (look.index[best["left_idx"]], best["left_price"], "Bottom 1"),
        (look.index[best["neckline_idx"]], best["neckline_price"], "Neckline"),
        (look.index[best["right_idx"]], best["right_price"], "Bottom 2"),
    ]

    return PatternResult(
        pattern="Double Bottom",
        ticker=ticker,
        start_date=look.index[best["left_idx"]],
        end_date=look.index[best["right_idx"]],
        breakout_level=round(breakout_level, 2),
        target=round(target, 2),
        status=status,
        confidence=round(confidence, 2),
        key_points=key_points,
        notes=f"Lows: ${best['left_price']:.2f} / ${best['right_price']:.2f} | Neckline: ${best['neckline_price']:.2f} | Depth: {best['depth']*100:.1f}%",
        max_distance=0.15,
    )


# ─────────────────────────────────────────────────
# Inverse Head and Shoulders
# ─────────────────────────────────────────────────
# Three lows: left shoulder, head (deepest), right shoulder.
# The head is lower than both shoulders. A neckline connects the
# two peaks between the lows. Breakout above the neckline is a
# major reversal signal.
#
# Detection:
#   - Three swing lows where the middle one is the deepest
#   - Left and right shoulder lows should roughly match (within 5%)
#   - Two swing highs between them define the neckline
#   - Head must be at least 5% deeper than the shoulders
#   - Breakout = close above the neckline

def detect_inverse_head_shoulders(df: pd.DataFrame, ticker: str) -> Optional[PatternResult]:
    if len(df) < 60:
        return None

    look = df.tail(150).copy()
    high = look["High"]
    low = look["Low"]
    close = look["Close"]
    n = len(look)

    sl = _swing_lows(low, distance=7, prominence=low.std() * 0.3)
    sh = _swing_highs(high, distance=5, prominence=high.std() * 0.3)

    if len(sl) < 3 or len(sh) < 2:
        return None

    best = None

    # Try all triples of swing lows
    for i in range(len(sl)):
        for j in range(i + 1, len(sl)):
            for k in range(j + 1, len(sl)):
                ls_idx = sl[i]   # left shoulder
                head_idx = sl[j]  # head
                rs_idx = sl[k]   # right shoulder

                ls_price = float(low.iloc[ls_idx])
                head_price = float(low.iloc[head_idx])
                rs_price = float(low.iloc[rs_idx])

                # Head must be the deepest low
                if head_price >= ls_price or head_price >= rs_price:
                    continue

                # Head must be at least 5% deeper than both shoulders
                head_depth_vs_ls = (ls_price - head_price) / ls_price
                head_depth_vs_rs = (rs_price - head_price) / rs_price
                if head_depth_vs_ls < 0.05 or head_depth_vs_rs < 0.05:
                    continue

                # Shoulders should roughly match (within 5%)
                shoulder_avg = (ls_price + rs_price) / 2
                shoulder_spread = abs(ls_price - rs_price) / shoulder_avg
                if shoulder_spread > 0.05:
                    continue

                # Spacing: each segment should be 10-50 bars
                ls_to_head = head_idx - ls_idx
                head_to_rs = rs_idx - head_idx
                if ls_to_head < 10 or ls_to_head > 50:
                    continue
                if head_to_rs < 10 or head_to_rs > 50:
                    continue

                # Symmetry: segments shouldn't differ by more than 2x
                ratio = max(ls_to_head, head_to_rs) / max(min(ls_to_head, head_to_rs), 1)
                if ratio > 2.5:
                    continue

                # Find neckline: swing highs between LS-Head and Head-RS
                sh_left = sh[(sh > ls_idx) & (sh < head_idx)]
                sh_right = sh[(sh > head_idx) & (sh < rs_idx)]

                if len(sh_left) == 0 or len(sh_right) == 0:
                    # Use max high in each segment as the neckline points
                    nl_left_idx = ls_idx + np.argmax(high.iloc[ls_idx:head_idx + 1].values)
                    nl_right_idx = head_idx + np.argmax(high.iloc[head_idx:rs_idx + 1].values)
                else:
                    nl_left_idx = sh_left[np.argmax(high.iloc[sh_left].values)]
                    nl_right_idx = sh_right[np.argmax(high.iloc[sh_right].values)]

                nl_left_price = float(high.iloc[nl_left_idx])
                nl_right_price = float(high.iloc[nl_right_idx])

                # Neckline: interpolate to current bar
                if nl_right_idx != nl_left_idx:
                    nl_slope = (nl_right_price - nl_left_price) / (nl_right_idx - nl_left_idx)
                    neckline_now = nl_right_price + nl_slope * (n - 1 - nl_right_idx)
                else:
                    neckline_now = (nl_left_price + nl_right_price) / 2

                # Pattern height
                neckline_avg = (nl_left_price + nl_right_price) / 2
                pattern_height = neckline_avg - head_price

                # Geometry: symmetry, shoulder match, depth, recency
                symmetry_score = _clip01(1 - (ratio - 1) / 1.5)         # 1.0→1, 2.5→0
                shoulder_match_score = _clip01(1 - shoulder_spread / 0.05)
                avg_depth = (head_depth_vs_ls + head_depth_vs_rs) / 2
                depth_score = _clip01((avg_depth - 0.05) / 0.10)        # 5%→0, 15%+→1
                recency = _clip01(1 - (n - 1 - rs_idx) / 20)
                score = (0.30 * symmetry_score + 0.30 * shoulder_match_score
                         + 0.25 * depth_score + 0.15 * recency)

                if score < 0.20:
                    continue

                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "ls_idx": ls_idx, "head_idx": head_idx, "rs_idx": rs_idx,
                        "ls_price": ls_price, "head_price": head_price, "rs_price": rs_price,
                        "nl_left_idx": nl_left_idx, "nl_right_idx": nl_right_idx,
                        "nl_left_price": nl_left_price, "nl_right_price": nl_right_price,
                        "neckline_now": float(neckline_now),
                        "pattern_height": pattern_height,
                        "symmetry": ratio,
                        "shoulder_spread": shoulder_spread,
                    }

    if best is None:
        return None

    breakout_level = best["neckline_now"]
    target = breakout_level + best["pattern_height"]
    last_close = float(df["Close"].iloc[-1])

    if last_close > breakout_level * 1.01:
        status = "Confirmed"
    elif last_close > breakout_level:
        status = "Breaking out"
    else:
        pct_to_breakout = (breakout_level - last_close) / last_close * 100
        if pct_to_breakout > 10.0:
            return None
        status = "Forming"

    proximity = _proximity_factor(last_close, breakout_level, "up", max_distance=0.15)
    vol_q = _volume_quality_factor(df)
    confidence = _clip01(best["score"] * proximity * vol_q)

    key_points = [
        (look.index[best["ls_idx"]], best["ls_price"], "Left shoulder"),
        (look.index[best["nl_left_idx"]], best["nl_left_price"], "Neckline L"),
        (look.index[best["head_idx"]], best["head_price"], "Head"),
        (look.index[best["nl_right_idx"]], best["nl_right_price"], "Neckline R"),
        (look.index[best["rs_idx"]], best["rs_price"], "Right shoulder"),
    ]

    return PatternResult(
        pattern="Inv H&S",
        ticker=ticker,
        start_date=look.index[best["ls_idx"]],
        end_date=look.index[best["rs_idx"]],
        breakout_level=round(breakout_level, 2),
        target=round(target, 2),
        status=status,
        confidence=round(confidence, 2),
        key_points=key_points,
        notes=f"Shoulders: ${best['ls_price']:.2f} / ${best['rs_price']:.2f} | Head: ${best['head_price']:.2f} | Symmetry: {best['symmetry']:.1f}x",
        max_distance=0.15,
    )


# ─────────────────────────────────────────────────
# Bear Flag
# ─────────────────────────────────────────────────
# Mirror of bull flag. A sharp, fast down-move (the pole) followed by
# a tight, slightly upward-sloping consolidation (the flag). Breakdown
# below the flag's low continues the prior decline.

def detect_bear_flag(df: pd.DataFrame, ticker: str) -> Optional[PatternResult]:
    if len(df) < 30:
        return None

    look = df.tail(40).copy()
    close = look["Close"].values
    high = look["High"].values
    low = look["Low"].values
    n = len(look)

    best = None
    for pole_len in range(3, 11):
        for pole_end in range(pole_len, n - 4):
            pole_start_price = close[pole_end - pole_len]
            pole_end_price = close[pole_end]
            pole_drop = 1 - (pole_end_price / pole_start_price)
            if pole_drop < 0.10:  # need ≥10% drop
                continue

            flag = look.iloc[pole_end : pole_end + 15]
            if len(flag) < 4:
                continue
            flag_high = flag["High"].max()
            flag_low = flag["Low"].min()
            flag_range = (flag_high - flag_low) / pole_end_price
            if flag_range > 0.10:
                continue

            # Flag should not retrace more than 50% of the pole
            retrace = (flag_high - pole_end_price) / (pole_start_price - pole_end_price)
            if retrace > 0.50:
                continue

            flag_end_idx = pole_end + len(flag) - 1
            pole_strength = _clip01((pole_drop - 0.10) / 0.20)         # 10%→0, 30%+→1
            flag_tightness = _clip01((0.10 - flag_range) / 0.07)
            retrace_quality = _clip01((0.50 - retrace) / 0.25)
            recency = _clip01(1 - (n - 1 - flag_end_idx) / 12)
            score = (0.30 * pole_strength + 0.30 * flag_tightness
                     + 0.20 * retrace_quality + 0.20 * recency)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "pole_start": pole_end - pole_len,
                    "pole_end": pole_end,
                    "flag_end": flag_end_idx,
                    "pole_drop": pole_drop,
                    "flag_high": float(flag_high),
                    "flag_low": float(flag_low),
                    "flag_range": flag_range,
                    "retrace": retrace,
                    "pole_start_price": float(pole_start_price),
                    "pole_end_price": float(pole_end_price),
                }

    if best is None:
        return None

    last_close = float(df["Close"].iloc[-1])
    breakout_level = best["flag_low"]  # breakdown below flag low
    pole_height = best["pole_start_price"] - best["pole_end_price"]
    target = breakout_level - pole_height  # measured move DOWN

    if last_close < breakout_level * 0.99:
        status = "Confirmed"
    elif last_close < breakout_level:
        status = "Breaking out"
    else:
        status = "Forming"

    proximity = _proximity_factor(last_close, breakout_level, "down")
    vol_q = _volume_quality_factor(df)
    confidence = _clip01(best["score"] * proximity * vol_q)

    key_points = [
        (look.index[best["pole_start"]], best["pole_start_price"], "Pole top"),
        (look.index[best["pole_end"]], best["pole_end_price"], "Pole bottom"),
        (look.index[best["flag_end"]], best["flag_high"], "Flag high"),
    ]

    return PatternResult(
        pattern="Bear Flag",
        ticker=ticker,
        start_date=look.index[best["pole_start"]],
        end_date=look.index[best["flag_end"]],
        breakout_level=round(breakout_level, 2),
        target=round(target, 2),
        status=status,
        confidence=round(confidence, 2),
        key_points=key_points,
        notes=f"-{best['pole_drop']*100:.1f}% pole, {best['flag_range']*100:.1f}% flag",
    )


# ─────────────────────────────────────────────────
# Descending Triangle
# ─────────────────────────────────────────────────
# Mirror of ascending triangle. Horizontal support (≥3 swing lows at
# similar levels) + declining resistance (lower highs). Bearish
# continuation; breaks below support.

def detect_descending_triangle(df: pd.DataFrame, ticker: str) -> Optional[PatternResult]:
    if len(df) < 30:
        return None

    look = df.tail(90).copy()
    highs = look["High"]
    lows = look["Low"]
    if len(look) < 30:
        return None

    sh = _swing_highs(highs, distance=4)
    sl = _swing_lows(lows, distance=4)
    if len(sh) < 2 or len(sl) < 3:
        return None

    sh = sh[-5:]
    sl = sl[-5:]

    # Support: swing lows must cluster at similar level
    sl_prices = lows.iloc[sl].values
    support = sl_prices.mean()
    spread = sl_prices.std() / support
    if spread > 0.025:
        return None

    # Resistance: swing highs must be descending (lower highs)
    sh_prices = highs.iloc[sh].values
    if len(sh_prices) < 2 or not all(sh_prices[i] > sh_prices[i + 1] for i in range(len(sh_prices) - 1)):
        return None

    pattern_start = look.index[min(sh[0], sl[0])]
    last_close = float(df["Close"].iloc[-1])

    if last_close < support * 0.995:
        status = "Confirmed"
    elif last_close < support:
        status = "Breaking out"
    else:
        status = "Forming"

    triangle_height = sh_prices[0] - support
    target = support - triangle_height  # measured move DOWN

    touch_quality = _clip01((len(sl) - 2) / 3)                # 2→0, 5+→1
    tightness = _clip01((0.025 - spread) / 0.025)
    high_count = _clip01((len(sh) - 1) / 3)                   # 1→0, 4+→1
    geometry = 0.40 * touch_quality + 0.30 * tightness + 0.30 * high_count
    proximity = _proximity_factor(last_close, support, "down")
    vol_q = _volume_quality_factor(df)
    confidence = _clip01(geometry * proximity * vol_q)

    key_points = [(look.index[i], float(lows.iloc[i]), "Support") for i in sl]
    key_points += [(look.index[i], float(highs.iloc[i]), "Lower high") for i in sh]

    return PatternResult(
        pattern="Descending Triangle",
        ticker=ticker,
        start_date=pattern_start,
        end_date=look.index[-1],
        breakout_level=round(support, 2),
        target=round(target, 2),
        status=status,
        confidence=round(confidence, 2),
        key_points=key_points,
        notes=f"{len(sl)} support touches, {len(sh)} lower highs",
    )


# ─────────────────────────────────────────────────
# Inverse Cup and Handle (bearish)
# ─────────────────────────────────────────────────
# An inverted U-shaped rally (the cup) that fails near a prior low,
# followed by a small bounce (the handle). Breakdown below the
# inverted cup rim continues the decline.

def detect_inverse_cup_handle(df: pd.DataFrame, ticker: str) -> Optional[PatternResult]:
    if len(df) < 60:
        return None

    look = df.tail(150).copy()
    close = look["Close"]
    high = look["High"]
    low = look["Low"]
    n = len(look)

    # Find swing lows (the rims of the inverted cup)
    sl = _swing_lows(low, distance=10, prominence=low.std() * 0.5)
    if len(sl) < 2:
        return None

    best = None
    for i in range(len(sl) - 1):
        for j in range(i + 1, len(sl)):
            left_idx, right_idx = sl[i], sl[j]
            cup_duration = right_idx - left_idx
            if cup_duration < 25 or cup_duration > 130:
                continue

            left_price = float(low.iloc[left_idx])
            right_price = float(low.iloc[right_idx])

            # Rims must match within 5%
            if abs(right_price - left_price) / left_price > 0.05:
                continue

            # Cup high (the peak of the inverted U)
            cup_high = float(high.iloc[left_idx:right_idx + 1].max())
            cup_high_idx = high.iloc[left_idx:right_idx + 1].idxmax()

            # Depth: how far price rallied between the two lows (12-35%)
            rim_avg = (left_price + right_price) / 2
            depth = (cup_high - rim_avg) / rim_avg
            if depth < 0.12 or depth > 0.35:
                continue

            # Handle: 4-25 days after right rim, a small bounce (<15%)
            # Matches the bullish Cup & Handle threshold for symmetry.
            handle = look.iloc[right_idx + 1:right_idx + 26]
            if len(handle) < 4:
                continue
            handle_high = float(handle["High"].max())
            handle_bounce = (handle_high - right_price) / right_price
            if handle_bounce < 0.02 or handle_bounce > 0.15:
                continue
            handle_end = right_idx + len(handle)

            depth_quality = _clip01(1 - abs(depth - 0.225) / 0.125)
            handle_quality = _clip01((0.15 - handle_bounce) / 0.13)
            rim_diff = abs(right_price - left_price) / left_price
            rim_match = _clip01(1 - rim_diff / 0.05)
            recency = _clip01(1 - (n - 1 - min(handle_end, n - 1)) / 15)
            score = (0.30 * depth_quality + 0.30 * handle_quality
                     + 0.25 * rim_match + 0.15 * recency)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "left_idx": left_idx,
                    "right_idx": right_idx,
                    "cup_high_idx": cup_high_idx,
                    "cup_high": cup_high,
                    "left_price": left_price,
                    "right_price": right_price,
                    "handle_end": min(handle_end, n - 1),
                    "handle_high": handle_high,
                    "depth": depth,
                    "handle_bounce": handle_bounce,
                }

    if best is None:
        return None

    rim = (best["left_price"] + best["right_price"]) / 2
    cup_height = best["cup_high"] - rim
    target = rim - cup_height  # measured move DOWN
    last_close = float(df["Close"].iloc[-1])

    if last_close < rim * 0.99:
        status = "Confirmed"
    elif last_close < rim:
        status = "Breaking out"
    else:
        status = "Forming"

    proximity = _proximity_factor(last_close, rim, "down")
    vol_q = _volume_quality_factor(df)
    confidence = _clip01(best["score"] * proximity * vol_q)

    key_points = [
        (look.index[best["left_idx"]], best["left_price"], "Left rim"),
        (best["cup_high_idx"], best["cup_high"], "Cup peak"),
        (look.index[best["right_idx"]], best["right_price"], "Right rim"),
        (look.index[best["handle_end"]], best["handle_high"], "Handle high"),
    ]

    return PatternResult(
        pattern="Inv Cup & Handle",
        ticker=ticker,
        start_date=look.index[best["left_idx"]],
        end_date=look.index[best["handle_end"]],
        breakout_level=round(rim, 2),
        target=round(target, 2),
        status=status,
        confidence=round(confidence, 2),
        key_points=key_points,
        notes=f"{best['depth']*100:.0f}% cup, {best['handle_bounce']*100:.1f}% handle bounce",
    )


# ─────────────────────────────────────────────────
# Double Top
# ─────────────────────────────────────────────────
# Mirror of double bottom. Two highs at roughly the same price
# separated by a pullback. Breakdown triggers when price drops
# below the trough between the two highs (the neckline).

def detect_double_top(df: pd.DataFrame, ticker: str) -> Optional[PatternResult]:
    if len(df) < 40:
        return None

    look = df.tail(130).copy()
    high = look["High"]
    low = look["Low"]
    close = look["Close"]
    n = len(look)

    sh = _swing_highs(high, distance=7, prominence=high.std() * 0.3)
    sl = _swing_lows(low, distance=5, prominence=low.std() * 0.3)

    if len(sh) < 2 or len(sl) < 1:
        return None

    best = None

    for i in range(len(sh)):
        for j in range(i + 1, len(sh)):
            left_idx = sh[i]
            right_idx = sh[j]
            separation = right_idx - left_idx

            if separation < 15 or separation > 65:
                continue

            left_price = float(high.iloc[left_idx])
            right_price = float(high.iloc[right_idx])

            # Highs must match within 3%
            avg_high = (left_price + right_price) / 2
            if abs(left_price - right_price) / avg_high > 0.03:
                continue

            # Find the lowest swing low between the two highs (neckline)
            between_sl = sl[(sl > left_idx) & (sl < right_idx)]
            if len(between_sl) == 0:
                neckline_idx = left_idx + np.argmin(low.iloc[left_idx:right_idx + 1].values)
                neckline_price = float(low.iloc[neckline_idx])
            else:
                neckline_idx = between_sl[np.argmin(low.iloc[between_sl].values)]
                neckline_price = float(low.iloc[neckline_idx])

            # Depth: average top to neckline must be at least 8%
            depth = (avg_high - neckline_price) / avg_high
            if depth < 0.08 or depth > 0.40:
                continue

            # Right top shouldn't be much higher than left (higher high = uptrend, not double top)
            if right_price > left_price * 1.03:
                continue

            match_quality = _clip01(1 - abs(left_price - right_price) / avg_high / 0.03)
            depth_quality = _clip01((depth - 0.08) / 0.17)
            recency = _clip01(1 - (n - 1 - right_idx) / 20)
            score = 0.35 * match_quality + 0.35 * depth_quality + 0.30 * recency

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "left_idx": left_idx,
                    "right_idx": right_idx,
                    "neckline_idx": neckline_idx,
                    "left_price": left_price,
                    "right_price": right_price,
                    "neckline_price": neckline_price,
                    "depth": depth,
                    "match_quality": match_quality,
                }

    if best is None:
        return None

    breakout_level = best["neckline_price"]  # breakdown below neckline
    pattern_height = (best["left_price"] + best["right_price"]) / 2 - best["neckline_price"]
    target = breakout_level - pattern_height  # measured move DOWN
    last_close = float(df["Close"].iloc[-1])

    if last_close < breakout_level * 0.99:
        status = "Confirmed"
    elif last_close < breakout_level:
        status = "Breaking out"
    else:
        pct_to_breakdown = (last_close - breakout_level) / last_close * 100
        if pct_to_breakdown > 8.0:
            return None
        status = "Forming"

    proximity = _proximity_factor(last_close, breakout_level, "down")
    vol_q = _volume_quality_factor(df)
    confidence = _clip01(best["score"] * proximity * vol_q)

    key_points = [
        (look.index[best["left_idx"]], best["left_price"], "Top 1"),
        (look.index[best["neckline_idx"]], best["neckline_price"], "Neckline"),
        (look.index[best["right_idx"]], best["right_price"], "Top 2"),
    ]

    return PatternResult(
        pattern="Double Top",
        ticker=ticker,
        start_date=look.index[best["left_idx"]],
        end_date=look.index[best["right_idx"]],
        breakout_level=round(breakout_level, 2),
        target=round(target, 2),
        status=status,
        confidence=round(confidence, 2),
        key_points=key_points,
        notes=f"Tops: ${best['left_price']:.2f} / ${best['right_price']:.2f} | Neckline: ${best['neckline_price']:.2f} | Depth: {best['depth']*100:.1f}%",
    )


# ─────────────────────────────────────────────────
# Head and Shoulders (bearish)
# ─────────────────────────────────────────────────
# Mirror of inverse H&S. Three highs: left shoulder, head (highest),
# right shoulder. Breakdown below the neckline connecting the two
# troughs is a major reversal signal.

def detect_head_shoulders(df: pd.DataFrame, ticker: str) -> Optional[PatternResult]:
    if len(df) < 60:
        return None

    look = df.tail(150).copy()
    high = look["High"]
    low = look["Low"]
    close = look["Close"]
    n = len(look)

    sh = _swing_highs(high, distance=7, prominence=high.std() * 0.3)
    sl = _swing_lows(low, distance=5, prominence=low.std() * 0.3)

    if len(sh) < 3 or len(sl) < 2:
        return None

    best = None

    for i in range(len(sh)):
        for j in range(i + 1, len(sh)):
            for k in range(j + 1, len(sh)):
                ls_idx = sh[i]    # left shoulder
                head_idx = sh[j]  # head
                rs_idx = sh[k]    # right shoulder

                ls_price = float(high.iloc[ls_idx])
                head_price = float(high.iloc[head_idx])
                rs_price = float(high.iloc[rs_idx])

                # Head must be the highest high
                if head_price <= ls_price or head_price <= rs_price:
                    continue

                # Head must be at least 5% higher than both shoulders
                head_height_vs_ls = (head_price - ls_price) / ls_price
                head_height_vs_rs = (head_price - rs_price) / rs_price
                if head_height_vs_ls < 0.05 or head_height_vs_rs < 0.05:
                    continue

                # Shoulders should roughly match (within 5%)
                shoulder_avg = (ls_price + rs_price) / 2
                shoulder_spread = abs(ls_price - rs_price) / shoulder_avg
                if shoulder_spread > 0.05:
                    continue

                # Spacing: 10-50 bars per segment
                ls_to_head = head_idx - ls_idx
                head_to_rs = rs_idx - head_idx
                if ls_to_head < 10 or ls_to_head > 50:
                    continue
                if head_to_rs < 10 or head_to_rs > 50:
                    continue

                # Symmetry
                ratio = max(ls_to_head, head_to_rs) / max(min(ls_to_head, head_to_rs), 1)
                if ratio > 2.5:
                    continue

                # Find neckline: swing lows between LS-Head and Head-RS
                sl_left = sl[(sl > ls_idx) & (sl < head_idx)]
                sl_right = sl[(sl > head_idx) & (sl < rs_idx)]

                if len(sl_left) == 0 or len(sl_right) == 0:
                    nl_left_idx = ls_idx + np.argmin(low.iloc[ls_idx:head_idx + 1].values)
                    nl_right_idx = head_idx + np.argmin(low.iloc[head_idx:rs_idx + 1].values)
                else:
                    nl_left_idx = sl_left[np.argmin(low.iloc[sl_left].values)]
                    nl_right_idx = sl_right[np.argmin(low.iloc[sl_right].values)]

                nl_left_price = float(low.iloc[nl_left_idx])
                nl_right_price = float(low.iloc[nl_right_idx])

                # Neckline interpolated to current bar
                if nl_right_idx != nl_left_idx:
                    nl_slope = (nl_right_price - nl_left_price) / (nl_right_idx - nl_left_idx)
                    neckline_now = nl_right_price + nl_slope * (n - 1 - nl_right_idx)
                else:
                    neckline_now = (nl_left_price + nl_right_price) / 2

                neckline_avg = (nl_left_price + nl_right_price) / 2
                pattern_height = head_price - neckline_avg

                symmetry_score = _clip01(1 - (ratio - 1) / 1.5)
                shoulder_match_score = _clip01(1 - shoulder_spread / 0.05)
                avg_height = (head_height_vs_ls + head_height_vs_rs) / 2
                depth_score = _clip01((avg_height - 0.05) / 0.10)
                recency = _clip01(1 - (n - 1 - rs_idx) / 20)
                score = (0.30 * symmetry_score + 0.30 * shoulder_match_score
                         + 0.25 * depth_score + 0.15 * recency)

                if score < 0.20:
                    continue

                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "ls_idx": ls_idx, "head_idx": head_idx, "rs_idx": rs_idx,
                        "ls_price": ls_price, "head_price": head_price, "rs_price": rs_price,
                        "nl_left_idx": nl_left_idx, "nl_right_idx": nl_right_idx,
                        "nl_left_price": nl_left_price, "nl_right_price": nl_right_price,
                        "neckline_now": float(neckline_now),
                        "pattern_height": pattern_height,
                        "symmetry": ratio,
                        "shoulder_spread": shoulder_spread,
                    }

    if best is None:
        return None

    breakout_level = best["neckline_now"]  # breakdown below neckline
    target = breakout_level - best["pattern_height"]  # measured move DOWN
    last_close = float(df["Close"].iloc[-1])

    if last_close < breakout_level * 0.99:
        status = "Confirmed"
    elif last_close < breakout_level:
        status = "Breaking out"
    else:
        pct_to_breakdown = (last_close - breakout_level) / last_close * 100
        if pct_to_breakdown > 10.0:
            return None
        status = "Forming"

    proximity = _proximity_factor(last_close, breakout_level, "down")
    vol_q = _volume_quality_factor(df)
    confidence = _clip01(best["score"] * proximity * vol_q)

    key_points = [
        (look.index[best["ls_idx"]], best["ls_price"], "Left shoulder"),
        (look.index[best["nl_left_idx"]], best["nl_left_price"], "Neckline L"),
        (look.index[best["head_idx"]], best["head_price"], "Head"),
        (look.index[best["nl_right_idx"]], best["nl_right_price"], "Neckline R"),
        (look.index[best["rs_idx"]], best["rs_price"], "Right shoulder"),
    ]

    return PatternResult(
        pattern="H&S Top",
        ticker=ticker,
        start_date=look.index[best["ls_idx"]],
        end_date=look.index[best["rs_idx"]],
        breakout_level=round(breakout_level, 2),
        target=round(target, 2),
        status=status,
        confidence=round(confidence, 2),
        key_points=key_points,
        notes=f"Shoulders: ${best['ls_price']:.2f} / ${best['rs_price']:.2f} | Head: ${best['head_price']:.2f} | Symmetry: {best['symmetry']:.1f}x",
    )


# ─────────────────────────────────────────────────
# Distribution VCP (bearish VCP)
# ─────────────────────────────────────────────────
# Mirror of VCP. A series of rallies where each bounce is weaker
# than the last — buyers exhausting across successive attempts.
# Breakdown comes on the final, weakest rally.
#
# Geometry:
#   - Pivot low (the support being tested)
#   - 2-4 swing highs after the pivot, each lower than the last
#   - Each rally height (from pivot low) decreases
#   - Volume declining across rallies
#   - Price near the pivot low at pattern completion

def detect_distribution_vcp(df: pd.DataFrame, ticker: str) -> Optional[PatternResult]:
    if len(df) < 60:
        return None

    look = df.tail(160).copy()
    high = look["High"]
    low = look["Low"]
    close = look["Close"]
    volume = look["Volume"]
    n = len(look)

    sh = _swing_highs(high, distance=5, prominence=high.std() * 0.25)
    sl = _swing_lows(low, distance=5, prominence=low.std() * 0.25)

    if len(sh) < 2 or len(sl) < 2:
        return None

    sl = sl[-6:]
    sh = sh[-6:]

    sl_prices = low.iloc[sl].values

    # Pivot low: the lowest swing low in the window (the support)
    pivot_idx = sl[np.argmin(sl_prices)]
    pivot_price = float(low.iloc[pivot_idx])

    # Only consider swing highs AFTER the pivot low
    sh_after = sh[sh > pivot_idx]
    if len(sh_after) < 2:
        return None

    sh_after = sh_after[-4:]
    sh_after_prices = high.iloc[sh_after].values

    # Rally heights: each measured from the pivot low
    rally_heights = [(p - pivot_price) / pivot_price * 100 for p in sh_after_prices]

    # Each successive rally must be weaker
    contracting = all(rally_heights[i] > rally_heights[i + 1] for i in range(len(rally_heights) - 1))
    if not contracting:
        return None

    # First rally must be meaningful (at least 8%), last must be weak (under 12%)
    if rally_heights[0] < 8.0:
        return None
    if rally_heights[-1] > 12.0:
        return None

    # Swing highs should be descending (lower highs)
    lower_highs = all(sh_after_prices[i] > sh_after_prices[i + 1] for i in range(len(sh_after_prices) - 1))
    if not lower_highs:
        return None

    # Volume declining across rallies (soft check)
    vol_declining = 0
    for i in range(len(sh_after) - 1):
        v1_start = max(0, sh_after[i] - 3)
        v1_end = min(n, sh_after[i] + 4)
        v2_start = max(0, sh_after[i + 1] - 3)
        v2_end = min(n, sh_after[i + 1] + 4)
        vol1 = volume.iloc[v1_start:v1_end].mean()
        vol2 = volume.iloc[v2_start:v2_end].mean()
        if vol2 < vol1:
            vol_declining += 1

    vol_score = vol_declining / (len(sh_after) - 1)

    num_contractions = len(sh_after)
    contraction_ratio = rally_heights[-1] / rally_heights[0]
    contractions_score = _clip01((num_contractions - 1) / 3)
    tightness = _clip01(1 - contraction_ratio)
    recency = _clip01(1 - (n - 1 - sh_after[-1]) / 15)
    geometry = (0.30 * contractions_score + 0.35 * tightness
                + 0.20 * vol_score + 0.15 * recency)

    if geometry < 0.20:
        return None

    breakout_level = pivot_price  # breakdown below the pivot
    last_close = float(df["Close"].iloc[-1])

    deepest_high = float(max(sh_after_prices))
    pattern_height = deepest_high - pivot_price
    target = pivot_price - pattern_height  # measured move DOWN

    pct_to_breakdown = (last_close - breakout_level) / last_close * 100

    if last_close < breakout_level * 0.99:
        status = "Confirmed"
    elif last_close < breakout_level:
        status = "Breaking out"
    elif pct_to_breakdown < 3.0:
        status = "Forming"
    else:
        return None

    proximity = _proximity_factor(last_close, breakout_level, "down")
    confidence = _clip01(geometry * proximity)

    key_points = [(look.index[pivot_idx], pivot_price, "Pivot low")]
    for i, idx in enumerate(sh_after):
        key_points.append((look.index[idx], float(high.iloc[idx]),
                          f"R{i+1}: +{rally_heights[i]:.1f}%"))

    return PatternResult(
        pattern="Distribution VCP",
        ticker=ticker,
        start_date=look.index[pivot_idx],
        end_date=look.index[-1],
        breakout_level=round(breakout_level, 2),
        target=round(target, 2),
        status=status,
        confidence=round(confidence, 2),
        key_points=key_points,
        notes=f"{num_contractions} rallies: {' → '.join(f'{r:.0f}%' for r in rally_heights)} | Vol trend: {'declining' if vol_score > 0.5 else 'mixed'}",
    )


# ─────────────────────────────────────────────────
# High Tight Bear Flag
# ─────────────────────────────────────────────────
# Mirror of high tight flag. A stock collapses 40%+ in 3-8 weeks,
# then bounces less than 20-25% over 2-4 weeks on declining volume.
# Breakdown below the flag low continues the collapse.
# Lower threshold than bullish (40% vs 80%) because declines are
# faster and more violent than rallies.

def detect_high_tight_bear_flag(df: pd.DataFrame, ticker: str) -> Optional[PatternResult]:
    if len(df) < 60:
        return None

    look = df.tail(100).copy()
    close = look["Close"].values
    high = look["High"].values
    low = look["Low"].values
    volume = look["Volume"].values
    n = len(look)

    best = None

    for pole_len in range(15, 46):
        for pole_end in range(pole_len, n - 9):
            pole_start_price = close[pole_end - pole_len]
            pole_end_price = low[pole_end]  # use low for the pole bottom
            pole_drop = 1 - (pole_end_price / pole_start_price)

            if pole_drop < 0.40:  # need 40%+ drop
                continue

            flag_start = pole_end + 1
            for flag_len in range(10, min(31, n - flag_start + 1)):
                flag_end = flag_start + flag_len - 1
                if flag_end >= n:
                    break

                flag_slice_high = high[flag_start:flag_end + 1]
                flag_slice_low = low[flag_start:flag_end + 1]
                flag_high = np.max(flag_slice_high)
                flag_low = np.min(flag_slice_low)

                # Bounce must be less than 25% from the pole bottom
                bounce = (flag_high - pole_end_price) / pole_end_price
                if bounce > 0.25:
                    continue
                if bounce < 0.05:
                    continue

                flag_range = (flag_high - flag_low) / pole_end_price
                if flag_range > 0.20:
                    continue

                pole_vol_avg = np.mean(volume[pole_end - pole_len:pole_end + 1])
                flag_vol_avg = np.mean(volume[flag_start:flag_end + 1])
                vol_decline = flag_vol_avg < pole_vol_avg * 0.8

                pole_strength = _clip01((pole_drop - 0.40) / 0.40)         # 40%→0, 80%+→1
                bounce_quality = _clip01((0.25 - bounce) / 0.20)
                flag_tightness = _clip01((0.20 - flag_range) / 0.17)
                recency = _clip01(1 - (n - 1 - flag_end) / 15)
                score = (0.30 * pole_strength + 0.25 * bounce_quality
                         + 0.25 * flag_tightness + 0.20 * recency)

                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "pole_start": pole_end - pole_len,
                        "pole_end": pole_end,
                        "flag_start": flag_start,
                        "flag_end": flag_end,
                        "pole_drop": pole_drop,
                        "pole_start_price": float(pole_start_price),
                        "pole_end_price": float(pole_end_price),
                        "flag_high": float(flag_high),
                        "flag_low": float(flag_low),
                        "bounce": bounce,
                        "vol_decline": vol_decline,
                    }

    if best is None:
        return None

    breakout_level = best["flag_low"]  # breakdown below flag low
    pole_height = best["pole_start_price"] - best["pole_end_price"]
    target = breakout_level - pole_height  # measured move DOWN
    last_close = float(df["Close"].iloc[-1])

    if last_close < breakout_level * 0.99:
        status = "Confirmed"
    elif last_close < breakout_level:
        status = "Breaking out"
    else:
        status = "Forming"

    proximity = _proximity_factor(last_close, breakout_level, "down")
    vol_q = _volume_quality_factor(df)
    confidence = _clip01(best["score"] * proximity * vol_q)

    key_points = [
        (look.index[best["pole_start"]], best["pole_start_price"], "Pole top"),
        (look.index[best["pole_end"]], best["pole_end_price"], "Pole bottom"),
        (look.index[best["flag_end"]], best["flag_high"], "Flag high"),
    ]

    return PatternResult(
        pattern="HT Bear Flag",
        ticker=ticker,
        start_date=look.index[best["pole_start"]],
        end_date=look.index[best["flag_end"]],
        breakout_level=round(breakout_level, 2),
        target=round(target, 2),
        status=status,
        confidence=round(confidence, 2),
        key_points=key_points,
        notes=f"-{best['pole_drop']*100:.0f}% pole, +{best['bounce']*100:.1f}% bounce, vol {'declining' if best['vol_decline'] else 'flat'}",
    )


# ═════════════════════════════════════════════════════════════════════════
# SINGLE-NAME ADDITIONS — short-duration, entry-timing patterns
# ═════════════════════════════════════════════════════════════════════════
# Added for the single-name analyzer. The 15 detectors above are kept in
# sync with the breakout-screener source of truth; the block below is
# project-specific and lives at the bottom so it's easy to extract later.
# ─────────────────────────────────────────────────────────────────────────


def detect_flat_base(df: pd.DataFrame, ticker: str) -> Optional[PatternResult]:
    """Detect a flat base: tight horizontal channel held for 15+ days."""
    if len(df) < 30:
        return None

    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    last_close = close[-1]

    best = None
    best_score = 0.0

    for duration in range(15, min(51, len(df) - 5)):
        window_high = high[-duration:]
        window_low = low[-duration:]

        ceiling = float(window_high.max())
        floor = float(window_low.min())

        if floor <= 0:
            continue

        range_pct = (ceiling - floor) / floor * 100
        if range_pct > 8.0:
            continue

        tightness = max(0.0, (8.0 - range_pct) / 6.0)
        dur_score = min(1.0, 0.5 + (duration - 15) / 50.0)
        geometry = tightness * dur_score

        prox = _proximity_factor(last_close, ceiling, side="up", max_distance=0.10)

        df_window = df.iloc[-duration:]
        vol_q = _volume_quality_factor(df_window) if len(df_window) >= 40 else 0.9

        score = geometry * prox * vol_q

        if score > best_score:
            best_score = score
            breakout_level = round(ceiling, 2)
            base_height = ceiling - floor
            target = round(ceiling + base_height, 2)
            status = "Breaking out" if last_close > ceiling else "Forming"

            best = PatternResult(
                pattern="Flat Base",
                ticker=ticker,
                start_date=df.index[-duration],
                end_date=df.index[-1],
                breakout_level=breakout_level,
                target=target,
                status=status,
                confidence=round(min(1.0, score), 3),
                key_points=[
                    (df.index[-duration], ceiling, "Base ceiling"),
                    (df.index[-duration], floor, "Base floor"),
                    (df.index[-1], last_close, "Current"),
                ],
                notes=f"{duration}d base, {range_pct:.1f}% range",
                max_distance=0.10,
            )

    return best


def detect_falling_wedge(df: pd.DataFrame, ticker: str) -> Optional[PatternResult]:
    """Detect a falling wedge: converging down-sloping trendlines (bullish reversal)."""
    if len(df) < 30:
        return None

    lookback = min(90, len(df) - 5)
    df_window = df.iloc[-lookback:]
    highs_arr = df_window["High"]
    lows_arr = df_window["Low"]
    close = df["Close"]

    sh_idx = _swing_highs(highs_arr, distance=4)
    sl_idx = _swing_lows(lows_arr, distance=4)
    if len(sh_idx) < 3 or len(sl_idx) < 3:
        return None

    sh_idx = sh_idx[-5:]
    sl_idx = sl_idx[-5:]

    sh_vals = highs_arr.iloc[sh_idx].values.astype(float)
    sl_vals = lows_arr.iloc[sl_idx].values.astype(float)
    sh_x = sh_idx.astype(float)
    sl_x = sl_idx.astype(float)

    if len(sh_x) < 2 or len(sl_x) < 2:
        return None

    sh_slope = np.polyfit(sh_x, sh_vals, 1)[0]
    sl_slope = np.polyfit(sl_x, sl_vals, 1)[0]

    if sh_slope >= 0 or sl_slope >= 0:
        return None
    if sl_slope <= sh_slope:
        return None

    first_width = sh_vals[0] - sl_vals[0]
    last_width = sh_vals[-1] - sl_vals[-1]
    if first_width <= 0 or last_width <= 0:
        return None
    if last_width >= first_width * 0.85:
        return None

    n = len(df_window)
    resistance_line = np.polyval(np.polyfit(sh_x, sh_vals, 1), n - 1)
    breakout_level = round(float(resistance_line), 2)

    last_close = float(close.iloc[-1])
    target = round(breakout_level + float(first_width), 2)

    convergence = 1.0 - (last_width / first_width)
    depth = min(1.0, abs(sh_slope) * lookback / sh_vals[0] * 20)
    geometry = convergence * min(1.0, depth + 0.3)

    prox = _proximity_factor(last_close, breakout_level, side="up", max_distance=0.15)
    vol_q = _volume_quality_factor(df_window)
    confidence = round(min(1.0, geometry * prox * vol_q), 3)
    if confidence < 0.05:
        return None

    status = "Breaking out" if last_close > breakout_level else "Forming"

    return PatternResult(
        pattern="Falling Wedge",
        ticker=ticker,
        start_date=df_window.index[int(min(sh_idx[0], sl_idx[0]))],
        end_date=df_window.index[-1],
        breakout_level=breakout_level,
        target=target,
        status=status,
        confidence=confidence,
        key_points=[
            (df_window.index[int(sh_idx[0])], float(sh_vals[0]), "Resistance start"),
            (df_window.index[int(sh_idx[-1])], float(sh_vals[-1]), "Resistance end"),
            (df_window.index[int(sl_idx[0])], float(sl_vals[0]), "Support start"),
            (df_window.index[int(sl_idx[-1])], float(sl_vals[-1]), "Support end"),
        ],
        notes=f"Convergence: {convergence:.0%}, depth: {abs(sh_slope)/sh_vals[0]*100:.1f}%/bar",
        max_distance=0.15,
    )


def detect_rising_wedge(df: pd.DataFrame, ticker: str) -> Optional[PatternResult]:
    """Detect a rising wedge: converging up-sloping trendlines (bearish reversal)."""
    if len(df) < 30:
        return None

    lookback = min(90, len(df) - 5)
    df_window = df.iloc[-lookback:]
    highs_arr = df_window["High"]
    lows_arr = df_window["Low"]
    close = df["Close"]

    sh_idx = _swing_highs(highs_arr, distance=4)
    sl_idx = _swing_lows(lows_arr, distance=4)
    if len(sh_idx) < 3 or len(sl_idx) < 3:
        return None

    sh_idx = sh_idx[-5:]
    sl_idx = sl_idx[-5:]

    sh_vals = highs_arr.iloc[sh_idx].values.astype(float)
    sl_vals = lows_arr.iloc[sl_idx].values.astype(float)
    sh_x = sh_idx.astype(float)
    sl_x = sl_idx.astype(float)

    if len(sh_x) < 2 or len(sl_x) < 2:
        return None

    sh_slope = np.polyfit(sh_x, sh_vals, 1)[0]
    sl_slope = np.polyfit(sl_x, sl_vals, 1)[0]

    if sh_slope <= 0 or sl_slope <= 0:
        return None
    if sh_slope >= sl_slope:
        return None

    first_width = sh_vals[0] - sl_vals[0]
    last_width = sh_vals[-1] - sl_vals[-1]
    if first_width <= 0 or last_width <= 0:
        return None
    if last_width >= first_width * 0.85:
        return None

    n = len(df_window)
    support_line = np.polyval(np.polyfit(sl_x, sl_vals, 1), n - 1)
    breakout_level = round(float(support_line), 2)

    last_close = float(close.iloc[-1])
    target = round(breakout_level - float(first_width), 2)

    convergence = 1.0 - (last_width / first_width)
    depth = min(1.0, abs(sl_slope) * lookback / sl_vals[0] * 20)
    geometry = convergence * min(1.0, depth + 0.3)

    prox = _proximity_factor(last_close, breakout_level, side="down", max_distance=0.15)
    vol_q = _volume_quality_factor(df_window)
    confidence = round(min(1.0, geometry * prox * vol_q), 3)
    if confidence < 0.05:
        return None

    status = "Breaking out" if last_close < breakout_level else "Forming"

    return PatternResult(
        pattern="Rising Wedge",
        ticker=ticker,
        start_date=df_window.index[int(min(sh_idx[0], sl_idx[0]))],
        end_date=df_window.index[-1],
        breakout_level=breakout_level,
        target=target,
        status=status,
        confidence=confidence,
        key_points=[
            (df_window.index[int(sh_idx[0])], float(sh_vals[0]), "Resistance start"),
            (df_window.index[int(sh_idx[-1])], float(sh_vals[-1]), "Resistance end"),
            (df_window.index[int(sl_idx[0])], float(sl_vals[0]), "Support start"),
            (df_window.index[int(sl_idx[-1])], float(sl_vals[-1]), "Support end"),
        ],
        notes=f"Convergence: {convergence:.0%}",
        max_distance=0.15,
    )


def detect_bull_pennant(df: pd.DataFrame, ticker: str) -> Optional[PatternResult]:
    """Detect a bullish pennant: sharp up-move followed by small symmetric contraction."""
    if len(df) < 20:
        return None

    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    volume = df["Volume"].to_numpy(dtype=float)
    last_close = close[-1]

    best = None
    best_score = 0.0
    avg_vol = float(volume[-60:].mean()) if len(volume) >= 60 else float(volume.mean())

    for pole_len in range(3, 9):
        for pennant_len in range(5, 16):
            total = pole_len + pennant_len
            if total > len(df) - 5:
                continue

            pole_start = -total
            pole_end = -pennant_len

            pole_close = close[pole_start:pole_end]
            pole_vol = volume[pole_start:pole_end]
            if len(pole_close) < 2:
                continue

            pole_gain = (pole_close[-1] / pole_close[0] - 1) * 100
            if pole_gain < 5.0:
                continue

            if avg_vol <= 0:
                continue
            vol_ratio = float(pole_vol.mean()) / avg_vol
            if vol_ratio < 1.2:
                continue

            pennant_high = high[pole_end:]
            pennant_low = low[pole_end:]
            if len(pennant_high) < 3:
                continue

            first_range = float(pennant_high[0] - pennant_low[0])
            last_range = float(pennant_high[-1] - pennant_low[-1])
            if first_range <= 0:
                continue

            contraction = last_range / first_range
            if contraction > 0.8:
                continue

            pole_top = float(pole_close[-1])
            pole_bottom = float(pole_close[0])
            pennant_lowest = float(pennant_low.min())
            retrace = (pole_top - pennant_lowest) / (pole_top - pole_bottom) if pole_top > pole_bottom else 1.0
            if retrace > 0.50:
                continue

            pennant_ceiling = float(pennant_high.max())
            breakout_level = round(pennant_ceiling, 2)
            pole_height = pole_top - pole_bottom
            target = round(breakout_level + pole_height, 2)

            pole_quality = min(1.0, pole_gain / 15.0)
            contraction_score = 1.0 - contraction
            geometry = pole_quality * contraction_score * min(1.0, vol_ratio / 2.0)

            prox = _proximity_factor(last_close, breakout_level, side="up", max_distance=0.08)
            score = geometry * prox

            if score > best_score and score > 0.05:
                best_score = score
                status = "Breaking out" if last_close > breakout_level else "Forming"
                best = PatternResult(
                    pattern="Bull Pennant",
                    ticker=ticker,
                    start_date=df.index[pole_start],
                    end_date=df.index[-1],
                    breakout_level=breakout_level,
                    target=target,
                    status=status,
                    confidence=round(min(1.0, score), 3),
                    key_points=[
                        (df.index[pole_start], float(pole_close[0]), "Pole base"),
                        (df.index[pole_end], float(pole_close[-1]), "Pole top"),
                        (df.index[-1], last_close, "Current"),
                    ],
                    notes=f"Pole: +{pole_gain:.1f}% in {pole_len}d, vol {vol_ratio:.1f}x, retrace {retrace:.0%}",
                    max_distance=0.08,
                )

    return best


def detect_bear_pennant(df: pd.DataFrame, ticker: str) -> Optional[PatternResult]:
    """Detect a bearish pennant: sharp down-move followed by small symmetric contraction."""
    if len(df) < 20:
        return None

    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    volume = df["Volume"].to_numpy(dtype=float)
    last_close = close[-1]

    best = None
    best_score = 0.0
    avg_vol = float(volume[-60:].mean()) if len(volume) >= 60 else float(volume.mean())

    for pole_len in range(3, 9):
        for pennant_len in range(5, 16):
            total = pole_len + pennant_len
            if total > len(df) - 5:
                continue

            pole_start = -total
            pole_end = -pennant_len

            pole_close = close[pole_start:pole_end]
            pole_vol = volume[pole_start:pole_end]
            if len(pole_close) < 2:
                continue

            pole_loss = (pole_close[0] / pole_close[-1] - 1) * 100
            if pole_loss < 5.0:
                continue

            if avg_vol <= 0:
                continue
            vol_ratio = float(pole_vol.mean()) / avg_vol
            if vol_ratio < 1.2:
                continue

            pennant_high = high[pole_end:]
            pennant_low = low[pole_end:]
            if len(pennant_high) < 3:
                continue

            first_range = float(pennant_high[0] - pennant_low[0])
            last_range = float(pennant_high[-1] - pennant_low[-1])
            if first_range <= 0:
                continue

            contraction = last_range / first_range
            if contraction > 0.8:
                continue

            pole_bottom = float(pole_close[-1])
            pole_top = float(pole_close[0])
            pennant_highest = float(pennant_high.max())
            retrace = (pennant_highest - pole_bottom) / (pole_top - pole_bottom) if pole_top > pole_bottom else 1.0
            if retrace > 0.50:
                continue

            pennant_floor = float(pennant_low.min())
            breakout_level = round(pennant_floor, 2)
            pole_height = pole_top - pole_bottom
            target = round(breakout_level - pole_height, 2)

            pole_quality = min(1.0, pole_loss / 15.0)
            contraction_score = 1.0 - contraction
            geometry = pole_quality * contraction_score * min(1.0, vol_ratio / 2.0)

            prox = _proximity_factor(last_close, breakout_level, side="down", max_distance=0.08)
            score = geometry * prox

            if score > best_score and score > 0.05:
                best_score = score
                status = "Breaking out" if last_close < breakout_level else "Forming"
                best = PatternResult(
                    pattern="Bear Pennant",
                    ticker=ticker,
                    start_date=df.index[pole_start],
                    end_date=df.index[-1],
                    breakout_level=breakout_level,
                    target=target,
                    status=status,
                    confidence=round(min(1.0, score), 3),
                    key_points=[
                        (df.index[pole_start], float(pole_close[0]), "Pole top"),
                        (df.index[pole_end], float(pole_close[-1]), "Pole bottom"),
                        (df.index[-1], last_close, "Current"),
                    ],
                    notes=f"Pole: -{pole_loss:.1f}% in {pole_len}d, vol {vol_ratio:.1f}x, retrace {retrace:.0%}",
                    max_distance=0.08,
                )

    return best


def detect_three_tight_closes(df: pd.DataFrame, ticker: str) -> Optional[PatternResult]:
    """3+ consecutive closes within 1.5% — Minervini's pre-breakout equilibrium signal."""
    if len(df) < 10:
        return None

    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    last_close = close[-1]

    best = None
    best_score = 0.0

    for n in [5, 4, 3]:
        if len(close) < n + 1:
            continue

        recent = close[-n:]
        mn, mx = float(recent.min()), float(recent.max())
        if mn <= 0:
            continue

        spread_pct = (mx - mn) / mn * 100
        if spread_pct > 1.5:
            continue

        tightness = max(0.0, (1.5 - spread_pct) / 1.5)
        streak_bonus = (n - 3) * 0.15
        geometry = min(1.0, tightness + streak_bonus)

        lookback = min(50, len(close) - 1)
        range_high = float(high[-lookback:].max())
        range_low = float(low[-lookback:].min())

        cluster_center = (mn + mx) / 2.0
        position_in_range = (cluster_center - range_low) / (range_high - range_low) if range_high > range_low else 0.5

        if position_in_range > 0.75:
            breakout_level = round(mx, 2)
            base_height = range_high - range_low
            target = round(breakout_level + base_height * 0.5, 2)
            location_bonus = (position_in_range - 0.75) / 0.25 * 0.3
        elif position_in_range < 0.25:
            breakout_level = round(mn, 2)
            base_height = range_high - range_low
            target = round(breakout_level - base_height * 0.5, 2)
            location_bonus = (0.25 - position_in_range) / 0.25 * 0.3
        else:
            breakout_level = round(mx, 2)
            base_height = range_high - range_low
            target = round(breakout_level + base_height * 0.3, 2)
            location_bonus = 0.0

        score = min(1.0, geometry + location_bonus)
        if score > best_score:
            best_score = score
            best = PatternResult(
                pattern="Three Tight Closes",
                ticker=ticker,
                start_date=df.index[-n],
                end_date=df.index[-1],
                breakout_level=breakout_level,
                target=target,
                status="Forming",
                confidence=round(score, 3),
                key_points=[
                    (df.index[-n], float(recent[0]), "First close"),
                    (df.index[-1], float(recent[-1]), "Last close"),
                ],
                notes=f"{n} closes within {spread_pct:.2f}%, range position: {position_in_range:.0%}",
                max_distance=0.05,
            )

    return best


def detect_inside_day_sequence(df: pd.DataFrame, ticker: str) -> Optional[PatternResult]:
    """2+ consecutive inside days — extreme compression, imminent expansion."""
    if len(df) < 5:
        return None

    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    last_close = close[-1]

    streak = 0
    for i in range(len(high) - 1, 0, -1):
        if high[i] < high[i - 1] and low[i] > low[i - 1]:
            streak += 1
        else:
            break

    if streak < 2:
        return None

    mother_idx = len(high) - 1 - streak
    if mother_idx < 0:
        return None

    mother_high = float(high[mother_idx])
    mother_low = float(low[mother_idx])
    mother_range = mother_high - mother_low
    if mother_range <= 0:
        return None

    inside_ranges = [float(high[i] - low[i]) for i in range(mother_idx + 1, len(high))]
    if len(inside_ranges) < 2:
        return None

    avg_inside_range = sum(inside_ranges) / len(inside_ranges)
    contraction = avg_inside_range / mother_range

    streak_score = min(1.0, (streak - 1) * 0.3 + 0.4)
    contraction_score = max(0.0, 1.0 - contraction)
    geometry = streak_score * (0.5 + 0.5 * contraction_score)

    lookback = min(50, mother_idx)
    range_hi = float(high[mother_idx - lookback:mother_idx + 1].max())
    range_lo = float(low[mother_idx - lookback:mother_idx + 1].min())
    position = (last_close - range_lo) / (range_hi - range_lo) if range_hi > range_lo else 0.5

    breakout_level = round(mother_high, 2)
    target = round(mother_high + mother_range, 2)

    confidence = round(min(1.0, geometry), 3)
    if confidence < 0.1:
        return None

    return PatternResult(
        pattern="Inside Day Sequence",
        ticker=ticker,
        start_date=df.index[mother_idx],
        end_date=df.index[-1],
        breakout_level=breakout_level,
        target=target,
        status="Forming",
        confidence=confidence,
        key_points=[
            (df.index[mother_idx], mother_high, "Mother bar high"),
            (df.index[mother_idx], mother_low, "Mother bar low"),
            (df.index[-1], last_close, "Current (inside)"),
        ],
        notes=f"{streak} inside days, contraction: {contraction:.0%}, range position: {position:.0%}",
        max_distance=0.05,
    )


def detect_pocket_pivot(df: pd.DataFrame, ticker: str) -> Optional[PatternResult]:
    """Pocket pivot: up-close on volume exceeding max down-day volume of prior 10 sessions."""
    if len(df) < 15:
        return None

    close = df["Close"].to_numpy(dtype=float)
    volume = df["Volume"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)

    last_close = close[-1]
    prev_close = close[-2]
    if last_close <= prev_close:
        return None

    down_volumes: list[float] = []
    for i in range(-11, -1):
        if -i > len(close) - 1:
            continue
        if close[i] < close[i - 1]:
            down_volumes.append(float(volume[i]))

    if not down_volumes:
        return None

    max_down_vol = max(down_volumes)
    today_vol = float(volume[-1])
    if today_vol <= max_down_vol:
        return None
    vol_ratio = today_vol / max_down_vol

    near_ma = False
    ma_label = ""
    for period in [10, 20, 50]:
        if len(close) < period + 5:
            continue
        ma_series = pd.Series(close).rolling(period).mean()
        ma = float(ma_series.iloc[-1])
        ma_prev = float(ma_series.iloc[-5])
        ma_rising = ma > ma_prev
        dist_pct = abs(last_close - ma) / ma * 100 if ma > 0 else 99
        if ma_rising and dist_pct < 3.0:
            near_ma = True
            ma_label = f"near rising {period}d MA"
            break

    if not near_ma:
        return None

    bar_range = float(high[-1] - low[-1])
    close_loc = (last_close - float(low[-1])) / bar_range if bar_range > 0 else 0.5
    if close_loc < 0.4:
        return None

    vol_score = min(1.0, (vol_ratio - 1.0) / 1.5)
    close_score = min(1.0, close_loc)
    geometry = vol_score * close_score
    confidence = round(min(0.85, geometry), 3)
    if confidence < 0.15:
        return None

    lookback = min(50, len(high) - 1)
    target = round(float(high[-lookback:].max()), 2)

    return PatternResult(
        pattern="Pocket Pivot",
        ticker=ticker,
        start_date=df.index[-1],
        end_date=df.index[-1],
        breakout_level=round(last_close, 2),
        target=target,
        status="Confirmed",
        confidence=confidence,
        key_points=[
            (df.index[-1], last_close, f"Pocket pivot {ma_label}"),
        ],
        notes=f"Vol {vol_ratio:.1f}x max down-vol, close at {close_loc:.0%} of bar, {ma_label}",
        max_distance=0.05,
    )


# ─────────────────────────────────────────────────
# Top-level scanner
# ─────────────────────────────────────────────────

DETECTORS = [
    # Bullish
    detect_bull_flag,
    detect_ascending_triangle,
    detect_cup_and_handle,
    detect_vcp,
    detect_high_tight_flag,
    detect_symmetrical_triangle,
    detect_double_bottom,
    detect_inverse_head_shoulders,
    # Bearish
    detect_bear_flag,
    detect_descending_triangle,
    detect_inverse_cup_handle,
    detect_double_top,
    detect_head_shoulders,
    detect_distribution_vcp,
    detect_high_tight_bear_flag,
    # Single-name additions — short-duration entry-timing
    detect_flat_base,
    detect_falling_wedge,
    detect_rising_wedge,
    detect_bull_pennant,
    detect_bear_pennant,
    detect_three_tight_closes,
    detect_inside_day_sequence,
    detect_pocket_pivot,
]


def scan_patterns(
    data: dict[str, pd.DataFrame],
    min_confidence: float = MIN_PUBLISH_CONFIDENCE,
    forming_max_distance_pct: float | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, str], PatternResult]]:
    """Run every detector across every ticker. Returns (table, results_dict).

    Filters keep the table focused on actionable charts:
      - `min_confidence`: drop patterns whose final confidence falls below
        this floor. Geometry × proximity × volume; a pattern with clean
        geometry but far from breakout naturally falls under the floor.
      - Per-pattern Forming distance: each detector declares its own
        `max_distance` (e.g. 12% for flags, 18% for cup & handle). Forming
        patterns farther than their detector's window are dropped.
      - `forming_max_distance_pct`: optional hard ceiling (in pct) applied
        on top of per-pattern windows. None = no ceiling.
        Confirmed and Breaking-out patterns are kept regardless of distance.
    """
    rows = []
    results: dict[tuple[str, str], PatternResult] = {}
    for ticker, df in data.items():
        last_close = float(df["Close"].iloc[-1]) if len(df) else 0.0
        for fn in DETECTORS:
            try:
                res = fn(df, ticker)
            except Exception:
                continue
            if res is None:
                continue
            if res.confidence < min_confidence:
                continue
            if res.status == "Forming" and last_close > 0:
                # Distance is direction-aware: bullish patterns need price
                # close to AND below breakout; bearish patterns need price
                # close to AND above the breakdown level.
                dist_pct = abs(res.breakout_level - last_close) / last_close * 100
                cap_pct = res.max_distance * 100
                if forming_max_distance_pct is not None:
                    cap_pct = min(cap_pct, forming_max_distance_pct)
                if dist_pct > cap_pct:
                    continue
            rows.append(res.to_row())
            results[(ticker, res.pattern)] = res
    if not rows:
        return pd.DataFrame(columns=["pattern", "ticker", "status", "breakout_level", "target", "confidence", "notes"]), results

    table = pd.DataFrame(rows)
    status_order = {"Confirmed": 0, "Breaking out": 1, "Forming": 2}
    table["_o"] = table["status"].map(lambda s: status_order.get(s, 9))
    table = table.sort_values(["_o", "confidence"], ascending=[True, False]).drop(columns="_o")
    return table.reset_index(drop=True), results
