"""15-minute bar structure analysis for the single-name analyzer.

Consumes the intraday OHLCV DataFrame from polygon_adapter.fetch_intraday and
returns a structured signals dict that downstream modules (single_name_analyzer,
commentary, grading) consume.

All calculations are deterministic — same input bars produce identical output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


_NEUTRAL_DEFAULTS = {
    "structure": "RANGE_BOUND",
    "proximity_to_daily_high": None,
    "intraday_volume_trend": "FLAT",
    "session_bias": "NEUTRAL",
    "vwap_position": "AT",
    "vwap_value": None,
    "intraday_atr": None,
    "intraday_squeeze": False,
    "last_5d_pattern": "CHOPPY",
    "swing_lows": [],
    "available": False,
}


def neutral_intraday() -> dict:
    """Defaults used when intraday data is unavailable."""
    return dict(_NEUTRAL_DEFAULTS)


def _session_dates(idx: pd.DatetimeIndex) -> list:
    """Return the unique session dates present in the index (using ET-ish date)."""
    # Polygon timestamps are UTC; ET sessions run 14:30–21:00 UTC during EDT.
    # Group by UTC date — close enough to per-session bucketing for the 15m grid
    # since US sessions don't cross UTC midnight in either DST window.
    return sorted({ts.date() for ts in idx})


def _compute_vwap(df_session: pd.DataFrame) -> float:
    """Standard VWAP across the bars of one session."""
    typical = (df_session["High"] + df_session["Low"] + df_session["Close"]) / 3.0
    cum_pv = (typical * df_session["Volume"]).cumsum()
    cum_v = df_session["Volume"].cumsum().replace(0, np.nan)
    vwap = cum_pv / cum_v
    return float(vwap.iloc[-1]) if not vwap.empty else float("nan")


def _find_swing_lows_3bar(lows: np.ndarray) -> list[int]:
    """Indices where lows[i] < lows[i-1] and lows[i] < lows[i+1] (3-bar window)."""
    n = len(lows)
    if n < 3:
        return []
    return [i for i in range(1, n - 1) if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]]


def _find_swing_highs_3bar(highs: np.ndarray) -> list[int]:
    n = len(highs)
    if n < 3:
        return []
    return [i for i in range(1, n - 1) if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]]


def _structure(lows: np.ndarray, highs: np.ndarray, closes: np.ndarray) -> str:
    """Classify intraday structure from swing-point sequencing and SMA slope."""
    swing_low_idx = _find_swing_lows_3bar(lows)
    swing_high_idx = _find_swing_highs_3bar(highs)

    last_lows = [lows[i] for i in swing_low_idx[-3:]] if len(swing_low_idx) >= 3 else []
    last_highs = [highs[i] for i in swing_high_idx[-3:]] if len(swing_high_idx) >= 3 else []

    higher_lows = len(last_lows) >= 3 and all(last_lows[i] > last_lows[i - 1] for i in range(1, len(last_lows)))
    lower_highs = len(last_highs) >= 3 and all(last_highs[i] < last_highs[i - 1] for i in range(1, len(last_highs)))

    if higher_lows and not lower_highs:
        return "HIGHER_LOWS"
    if lower_highs and not higher_lows:
        return "LOWER_HIGHS"

    # Fall back to 20-bar SMA slope to disambiguate trending vs range-bound.
    if len(closes) >= 20:
        sma = pd.Series(closes).rolling(20).mean().to_numpy()
        slope = (sma[-1] - sma[-20]) / sma[-20] if sma[-20] and not np.isnan(sma[-20]) else 0.0
        if slope > 0.005:
            return "TRENDING_UP"
        if slope < -0.005:
            return "TRENDING_DOWN"
    return "RANGE_BOUND"


def _intraday_atr(df: pd.DataFrame, period: int = 14) -> tuple[float, bool]:
    """Return (current_atr, is_compressing) on the intraday timeframe."""
    if len(df) < period + 5:
        return float("nan"), False
    h, l, c = df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy()
    prev_c = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    atr_s = pd.Series(tr).rolling(period).mean()
    last = float(atr_s.iloc[-1]) if not atr_s.empty else float("nan")
    # Compressing = recent ATR < 70% of trailing 50-bar median ATR.
    trailing = atr_s.iloc[-50:] if len(atr_s) >= 50 else atr_s
    median_trailing = float(trailing.median()) if not trailing.empty else float("nan")
    is_compressing = bool(last == last and median_trailing == median_trailing and last < 0.7 * median_trailing)
    return last, is_compressing


def analyze_intraday(df_intraday: pd.DataFrame, daily_signals: dict | None = None) -> dict:
    """Analyze intraday structure relative to the daily setup.

    Args:
        df_intraday: DataFrame from polygon_adapter.fetch_intraday. UTC index.
        daily_signals: Optional daily signals dict (used for proximity_to_daily_high
                       if the daily consolidation range is available).

    Returns:
        Dict with intraday structure signals. If df is empty, returns neutral defaults.
    """
    if df_intraday is None or df_intraday.empty or len(df_intraday) < 10:
        return neutral_intraday()

    # Filter to regular US trading hours. Window 13:30–21:00 UTC covers both
    # EDT (13:30–20:00 UTC) and EST (14:30–21:00 UTC). Pre-market and
    # after-hours bars have wide spreads and thin volume that distort VWAP
    # and swing-point detection.
    df_rth = df_intraday.between_time("13:30", "21:00")
    if df_rth.empty or len(df_rth) < 10:
        return neutral_intraday()

    df = df_rth.copy()
    session_dates = _session_dates(df.index)
    # Restrict to last 5 trading sessions of available data.
    last5 = session_dates[-5:]
    df_last5 = df[df.index.map(lambda ts: ts.date() in last5)]
    if df_last5.empty:
        return neutral_intraday()

    lows = df_last5["Low"].to_numpy(dtype=float)
    highs = df_last5["High"].to_numpy(dtype=float)
    closes = df_last5["Close"].to_numpy(dtype=float)
    volumes = df_last5["Volume"].to_numpy(dtype=float)

    structure = _structure(lows, highs, closes)
    swing_low_idx = _find_swing_lows_3bar(lows)

    # VWAP for the most recent session.
    last_date = session_dates[-1]
    df_today = df[df.index.map(lambda ts: ts.date() == last_date)]
    if not df_today.empty:
        vwap_value = _compute_vwap(df_today)
        last_close = float(df_today["Close"].iloc[-1])
        if vwap_value != vwap_value:  # NaN
            vwap_position = "AT"
        elif last_close > vwap_value * 1.001:
            vwap_position = "ABOVE"
        elif last_close < vwap_value * 0.999:
            vwap_position = "BELOW"
        else:
            vwap_position = "AT"
    else:
        vwap_value = float("nan")
        vwap_position = "AT"

    # Intraday volume trend: avg vol last 2 sessions vs prior 5 sessions.
    if len(session_dates) >= 7:
        recent2 = session_dates[-2:]
        prior5 = session_dates[-7:-2]
        avg_recent = float(df[df.index.map(lambda ts: ts.date() in recent2)]["Volume"].mean() or 0)
        avg_prior = float(df[df.index.map(lambda ts: ts.date() in prior5)]["Volume"].mean() or 0)
        ratio = (avg_recent / avg_prior) if avg_prior > 0 else 1.0
        if ratio > 1.3:
            volume_trend = "INCREASING"
        elif ratio < 0.7:
            volume_trend = "DECREASING"
        else:
            volume_trend = "FLAT"
    else:
        volume_trend = "FLAT"

    # Last-5-day range pattern (tightening vs expanding).
    daily_ranges = []
    for d in last5:
        sess = df[df.index.map(lambda ts: ts.date() == d)]
        if not sess.empty:
            daily_ranges.append(float(sess["High"].max() - sess["Low"].min()))
    if len(daily_ranges) >= 3:
        diffs = [daily_ranges[i] - daily_ranges[i - 1] for i in range(1, len(daily_ranges))]
        if all(d < 0 for d in diffs):
            last_5d_pattern = "TIGHTENING"
        elif all(d > 0 for d in diffs):
            last_5d_pattern = "EXPANDING"
        else:
            last_5d_pattern = "CHOPPY"
    else:
        last_5d_pattern = "CHOPPY"

    intraday_atr, is_compressing = _intraday_atr(df_last5, period=14)

    # Proximity to daily consolidation ceiling, if available.
    proximity_to_daily_high = None
    if daily_signals:
        consol = daily_signals.get("consolidation", {})
        if consol.get("signal", "NONE") != "NONE":
            range_high = consol.get("range_high")
            last_price = float(closes[-1])
            if range_high and range_high > 0:
                proximity_to_daily_high = (range_high - last_price) / range_high * 100

    # Composite session bias.
    if vwap_position == "ABOVE" and structure == "HIGHER_LOWS" and volume_trend in ("INCREASING", "FLAT"):
        session_bias = "BULLISH"
    elif vwap_position == "BELOW" and structure == "LOWER_HIGHS":
        session_bias = "BEARISH"
    elif vwap_position == "ABOVE" and structure in ("HIGHER_LOWS", "TRENDING_UP"):
        session_bias = "BULLISH"
    elif vwap_position == "BELOW" and structure in ("LOWER_HIGHS", "TRENDING_DOWN"):
        session_bias = "BEARISH"
    else:
        session_bias = "NEUTRAL"

    # Persist swing-low timestamps for charting (only if higher lows detected).
    swing_low_times: list = []
    if structure == "HIGHER_LOWS" and swing_low_idx:
        for i in swing_low_idx[-5:]:
            swing_low_times.append((df_last5.index[i], float(lows[i])))

    return {
        "structure": structure,
        "proximity_to_daily_high": proximity_to_daily_high,
        "intraday_volume_trend": volume_trend,
        "session_bias": session_bias,
        "vwap_position": vwap_position,
        "vwap_value": None if vwap_value != vwap_value else round(vwap_value, 2),
        "intraday_atr": None if intraday_atr != intraday_atr else round(intraday_atr, 3),
        "intraday_squeeze": is_compressing,
        "last_5d_pattern": last_5d_pattern,
        "swing_lows": swing_low_times,
        "available": True,
    }
