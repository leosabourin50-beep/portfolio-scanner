"""Single-Name Analyzer orchestrator.

Pulls daily + intraday data, runs the existing detector/pattern engines, layers
on indicators the spec needs (RSI, MACD, multi-period MAs, Bollinger, multi-period
Donchian, RS) and computes the deep analysis layers + entry-quality grade.

The signal engine in detector.py / patterns.py is treated as read-only — this
module wraps it and assembles the structured `signals` dict that commentary.py
and the grading function consume.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

import detector
import patterns
from patterns import PatternResult
import polygon_adapter as pa
import intraday_analysis
import setup_finder


# Bearish detector names — used by the proximity rescore.
# Pattern names emitted by the detectors (exact strings). These are bearish
# setups — long-entry grading should not be boosted by their confidence.
_BEARISH_PATTERNS = {
    "Bear Flag",
    "Descending Triangle",
    "Inv Cup & Handle",
    "Double Top",
    "H&S Top",
    "Distribution VCP",
    "HT Bear Flag",
    # Single-name additions
    "Rising Wedge",
    "Bear Pennant",
}


def _is_bullish_pattern(p) -> bool:
    return p.pattern not in _BEARISH_PATTERNS


CONFIG_PATH = Path(__file__).parent / "config.yaml"

# Optional Streamlit cache for the data pulls. When run outside Streamlit
# (CLI / tests / a scheduled job), fall back to plain function calls.
try:
    import streamlit as _st
    _HAS_ST = True
except Exception:
    _HAS_ST = False

if _HAS_ST:
    @_st.cache_data(ttl=600, show_spinner=False)
    def _cached_fetch_daily(ticker: str, days: int) -> pd.DataFrame:
        return pa.fetch_data(ticker, days=days)

    @_st.cache_data(ttl=600, show_spinner=False)
    def _cached_fetch_intraday(ticker: str, multiplier: int, days: int) -> pd.DataFrame:
        return pa.fetch_intraday(ticker, multiplier=multiplier, days=days)
else:
    def _cached_fetch_daily(ticker: str, days: int) -> pd.DataFrame:
        return pa.fetch_data(ticker, days=days)

    def _cached_fetch_intraday(ticker: str, multiplier: int, days: int) -> pd.DataFrame:
        return pa.fetch_intraday(ticker, multiplier=multiplier, days=days)


# ─────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────
# Indicator helpers (built on top of the raw detector output)
# ─────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _bollinger_state(close: pd.Series, period: int = 20, std: float = 2.0, hist_window: int = 252) -> dict:
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    bbw = (2 * std * sd) / ma
    trailing = bbw.iloc[-hist_window:].dropna() if len(bbw) >= 1 else bbw
    bw_last = float(bbw.iloc[-1]) if not bbw.empty and not pd.isna(bbw.iloc[-1]) else float("nan")
    if not trailing.empty and bw_last == bw_last:
        rank = float((trailing < bw_last).sum() / len(trailing) * 100)
    else:
        rank = float("nan")
    # Recent squeeze if at any time in the last 20 bars BBW was bottom-quartile.
    was_squeezed = False
    if len(trailing) >= 20:
        threshold = float(trailing.quantile(0.25))
        recent_bbw = bbw.iloc[-20:].dropna()
        was_squeezed = bool((recent_bbw <= threshold).any())
    upper = ma + std * sd
    lower = ma - std * sd
    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else last_close
    breaking_up = (
        was_squeezed
        and last_close > float(upper.iloc[-1])
        and prev_close <= float(upper.iloc[-2])
    ) if not pd.isna(upper.iloc[-1]) else False
    breaking_down = (
        was_squeezed
        and last_close < float(lower.iloc[-1])
        and prev_close >= float(lower.iloc[-2])
    ) if not pd.isna(lower.iloc[-1]) else False
    if breaking_up:
        sig = "SQUEEZE_BREAK_UP"
    elif breaking_down:
        sig = "SQUEEZE_BREAK_DOWN"
    elif (rank == rank) and rank < 20:
        sig = "SQUEEZE"
    else:
        sig = "NORMAL"
    return {
        "signal": sig,
        "bandwidth_percentile": None if rank != rank else round(rank, 1),
        "was_squeezed": was_squeezed,
    }


def _ma_signal(close: pd.Series, period: int) -> dict:
    if len(close) < period + 2:
        return {"signal": "INSUFFICIENT", "ma_value": None, "pct_distance": None}
    ma = close.rolling(period).mean()
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    ma_now = float(ma.iloc[-1])
    ma_prev = float(ma.iloc[-2])
    above_now = last > ma_now
    above_prev = prev > ma_prev
    if above_now and not above_prev:
        sig = "CROSS_ABOVE"
    elif above_prev and not above_now:
        sig = "CROSS_BELOW"
    elif above_now:
        sig = "ABOVE"
    else:
        sig = "BELOW"
    return {
        "signal": sig,
        "ma_value": round(ma_now, 2),
        "pct_distance": round((last / ma_now - 1) * 100, 2) if ma_now else 0.0,
    }


def _nday_signal(df: pd.DataFrame, period: int) -> dict:
    """N-day high/low signal in the spec's nday_* shape."""
    if len(df) < period + 1:
        return {"signal": "NEUTRAL", "level": None, "high": None, "low": None, "pct_from_high": None}
    window = df.iloc[-period:]
    hi = float(window["High"].max())
    lo = float(window["Low"].min())
    last = float(df["Close"].iloc[-1])
    if last >= hi:
        sig = "NEW_HIGH"
        level = hi
    elif last <= lo:
        sig = "NEW_LOW"
        level = lo
    else:
        sig = "NEUTRAL"
        level = hi
    return {
        "signal": sig,
        "level": round(level, 2),
        "high": round(hi, 2),
        "low": round(lo, 2),
        "pct_from_high": round((last / hi - 1) * 100, 2),
    }


def _volume_signal(df: pd.DataFrame, avg_period: int = 20) -> dict:
    if len(df) < avg_period + 1:
        return {"signal": "NORMAL", "ratio": None, "today": None, "avg_20d": None}
    avg = float(df["Volume"].iloc[-avg_period - 1:-1].mean())
    today = float(df["Volume"].iloc[-1])
    ratio = today / avg if avg > 0 else 1.0
    if ratio >= 2.0:
        sig = "SURGE"
    elif ratio >= 1.3:
        sig = "ELEVATED"
    elif ratio <= 0.6:
        sig = "DRY_UP"
    else:
        sig = "NORMAL"
    return {
        "signal": sig,
        "ratio": round(ratio, 2),
        "today": round(today, 0),
        "avg_20d": round(avg, 0),
    }


def _rs_signal(df: pd.DataFrame, spy_df: pd.DataFrame) -> dict:
    """20-day and 50-day relative strength changes."""
    if len(df) < 51 or len(spy_df) < 51:
        return {"signal": "NEUTRAL", "rs_20d_chg": None, "rs_50d_chg": None}
    # Align on the intersection of dates.
    joined = pd.DataFrame({"t": df["Close"], "s": spy_df["Close"]}).dropna()
    if len(joined) < 51:
        return {"signal": "NEUTRAL", "rs_20d_chg": None, "rs_50d_chg": None}
    rs = joined["t"] / joined["s"]
    rs_now = float(rs.iloc[-1])
    rs_20 = float(rs.iloc[-21])
    rs_50 = float(rs.iloc[-51])
    chg_20 = (rs_now / rs_20 - 1) * 100
    chg_50 = (rs_now / rs_50 - 1) * 100
    if chg_20 > 2.0 and chg_50 > 0:
        sig = "OUTPERFORMING"
    elif chg_20 < -2.0 and chg_50 < 0:
        sig = "UNDERPERFORMING"
    else:
        sig = "NEUTRAL"
    return {
        "signal": sig,
        "rs_20d_chg": round(chg_20, 2),
        "rs_50d_chg": round(chg_50, 2),
    }


def _rs_series(df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.Series:
    joined = pd.DataFrame({"t": df["Close"], "s": spy_df["Close"]}).dropna()
    if joined.empty:
        return pd.Series(dtype=float)
    return joined["t"] / joined["s"]


# ─────────────────────────────────────────────────
# Consolidation shape: lift detector's flat output into the spec's nested form
# ─────────────────────────────────────────────────

_CONSOL_STATUSES = {"PRIMED", "AT RISK", "COILED ↑", "COILED ↓", "CONSOLIDATING", "TESTING ↑", "TESTING ↓"}


def _consol_signal_label(status: str, breakout: bool, breakdown: bool) -> str:
    if breakout:
        return "BREAKOUT_UP"
    if breakdown:
        return "BREAKOUT_DOWN"
    if status in ("PRIMED", "COILED ↑"):
        return "PRIMED"
    if status in ("AT RISK", "COILED ↓"):
        return "PRIMED_DOWN"
    if status == "CONSOLIDATING":
        return "SQUEEZING"
    if status in ("TESTING ↑", "TESTING ↓"):
        return "TESTING"
    return "NONE"


def _consol_duration_days(df: pd.DataFrame, range_high: float, range_low: float, lookback: int = 100) -> int:
    """How many of the most recent trading bars sit inside the donchian range.

    Tolerates up to 2 consecutive bars outside the range before stopping — a
    single intraday spike or gap-up that reverted same-session shouldn't kill
    the count for an otherwise clean 50-day base.
    """
    tail = df.iloc[-lookback:] if len(df) > lookback else df
    count = 0
    consecutive_outside = 0
    for i in range(len(tail) - 1, -1, -1):
        h = float(tail["High"].iloc[i])
        l = float(tail["Low"].iloc[i])
        if h <= range_high * 1.005 and l >= range_low * 0.995:
            count += 1
            consecutive_outside = 0
        else:
            consecutive_outside += 1
            if consecutive_outside > 2:
                break
            count += 1  # outlier still counts toward the base's duration
    return count


def _higher_lows_in_window(df_window: pd.DataFrame) -> bool:
    """Detect rising swing lows within a window (≥2 sequentially rising lows)."""
    lows = df_window["Low"].to_numpy(dtype=float)
    if len(lows) < 5:
        return False
    swing_idx = [i for i in range(1, len(lows) - 1) if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]]
    if len(swing_idx) < 2:
        return False
    swing_lows = [lows[i] for i in swing_idx]
    return all(swing_lows[i] > swing_lows[i - 1] for i in range(1, len(swing_lows)))


# ─────────────────────────────────────────────────
# Build the structured signals dict on top of detector.analyze()
# ─────────────────────────────────────────────────

def compute_extended_signals(
    df: pd.DataFrame,
    spy_df: pd.DataFrame,
    cfg: dict,
) -> tuple[dict, dict]:
    """Run detector.analyze and assemble the spec's nested signals dict.

    Returns:
        (signals, raw_detector_dict)
    """
    params = cfg["params"]
    ind = cfg["indicators"]

    if len(spy_df) >= 22:
        spy_ret_21d = round(
            (float(spy_df["Close"].iloc[-1]) / float(spy_df["Close"].iloc[-22]) - 1) * 100,
            2,
        )
    else:
        spy_ret_21d = None

    raw = detector.analyze(df, params, spy_ret_21d=spy_ret_21d)
    close = df["Close"]
    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else last_close

    # Consolidation block (lifted from detector's flat output).
    range_high = float(raw["donchian_high"])
    range_low = float(raw["donchian_low"])
    consol_label = _consol_signal_label(raw["status"], raw["breakout"], raw["breakdown"])
    # consol_bars measures how many recent bars sit inside the Donchian range.
    # This is a property of the price data, not today's status label — a name
    # that's TRENDING UP today can still have a clean 50-day base behind it.
    # Compute it independently of consol_label.
    consol_bars = _consol_duration_days(df, range_high, range_low)
    consolidation = {
        "signal": consol_label,
        "range_high": round(range_high, 2),
        "range_low": round(range_low, 2),
        "atr_percentile": float(raw["atr_pct"]),
        "had_squeeze": bool(raw["squeezing"]),
        "consol_bars": consol_bars,
    }

    # MA signals at each requested period.
    ma_signals = {f"ma_{p}": _ma_signal(close, p) for p in ind["ma_periods"]}

    # Multi-period Donchian/N-day signals.
    nday_signals = {f"nday_{p}": _nday_signal(df, p) for p in ind["donchian_periods"]}

    # Bollinger band squeeze state.
    boll = _bollinger_state(close, period=ind.get("bb_period", params["bb_period"]),
                             std=params["bb_std"], hist_window=params["pct_window"])

    # RSI.
    rsi_s = _rsi(close, period=ind["rsi_period"])
    rsi_val = float(rsi_s.iloc[-1]) if not rsi_s.empty and not pd.isna(rsi_s.iloc[-1]) else 50.0
    if rsi_val > 70:
        rsi_sig = "OVERBOUGHT"
    elif rsi_val < 30:
        rsi_sig = "OVERSOLD"
    elif rsi_val > 55:
        rsi_sig = "BULLISH_MOMENTUM"
    elif rsi_val < 45:
        rsi_sig = "BEARISH_MOMENTUM"
    else:
        rsi_sig = "NEUTRAL"

    # MACD.
    macd_line, sig_line, hist = _macd(close, ind["macd_fast"], ind["macd_slow"], ind["macd_signal"])
    m_now = float(macd_line.iloc[-1])
    s_now = float(sig_line.iloc[-1])
    h_now = float(hist.iloc[-1])
    m_prev = float(macd_line.iloc[-2]) if len(macd_line) >= 2 else m_now
    s_prev = float(sig_line.iloc[-2]) if len(sig_line) >= 2 else s_now
    if m_now > s_now and m_prev <= s_prev:
        macd_sig = "BULLISH_CROSS"
    elif m_now < s_now and m_prev >= s_prev:
        macd_sig = "BEARISH_CROSS"
    elif m_now > s_now:
        macd_sig = "BULLISH"
    else:
        macd_sig = "BEARISH"

    # Relative strength.
    rs = _rs_signal(df, spy_df)

    # Volume.
    vol = _volume_signal(df, avg_period=20)

    # Price action.
    pa_sig = {
        "signal": raw["confirm_up"] if raw["confirm_up"] != "Unconfirmed" else raw["confirm_dn"],
        "gap_pct": round((float(df["Open"].iloc[-1]) / prev_close - 1) * 100, 2) if prev_close else 0.0,
        "range_expansion": float(raw["range_10d_pct"]),
        "close_loc": float(raw["close_loc"]),
    }

    # Score and verdict translation from detector's setup scores.
    score_up = float(raw["setup_score_up"])
    score_dn = float(raw["setup_score_dn"])
    net = round(score_up - score_dn, 1)
    if raw["breakout"]:
        verdict = "STRONG_BREAKOUT" if raw["confirm_up"] == "Strong" else "BREAKOUT"
    elif raw["breakdown"]:
        verdict = "STRONG_BREAKDOWN" if raw["confirm_dn"] == "Strong" else "BREAKDOWN"
    elif net > 15:
        verdict = "LEANING_BULL"
    elif net < -15:
        verdict = "LEANING_BEAR"
    else:
        verdict = "NEUTRAL"

    signals = {
        "consolidation": consolidation,
        **nday_signals,
        "volume": vol,
        **ma_signals,
        "bollinger": boll,
        "rsi": {"signal": rsi_sig, "value": round(rsi_val, 1)},
        "macd": {
            "signal": macd_sig,
            "macd": round(m_now, 3),
            "signal_line": round(s_now, 3),
            "histogram": round(h_now, 3),
        },
        "rel_strength": rs,
        "price_action": pa_sig,
        "_score": {
            "bull": round(score_up, 1),
            "bear": round(score_dn, 1),
            "net": net,
            "verdict": verdict,
        },
        "_price": {
            "last": round(last_close, 2),
            "chg_pct": round(raw["pct_1d"], 2),
        },
        "_status": raw["status"],
        "_setup_score_up": round(score_up, 1),
    }
    return signals, raw


# ─────────────────────────────────────────────────
# Deep analysis layers
# ─────────────────────────────────────────────────

def compute_deep_layers(df: pd.DataFrame, spy_df: pd.DataFrame, signals: dict) -> dict:
    last_close = float(df["Close"].iloc[-1])
    consol = signals["consolidation"]

    # ── Proximity ──
    if consol["signal"] != "NONE":
        range_high = float(consol["range_high"])
    else:
        range_high = float(signals["nday_50"]["high"]) if signals["nday_50"]["high"] else last_close
    proximity_pct = (range_high - last_close) / range_high * 100 if range_high else 0.0
    if last_close > range_high:
        prox_zone = "ABOVE"
    elif proximity_pct < 1:
        prox_zone = "AT_LEVEL"
    elif proximity_pct <= 3:
        prox_zone = "APPROACHING"
    elif proximity_pct <= 7:
        prox_zone = "MID_RANGE"
    else:
        prox_zone = "FAR"

    # ── Base volume profile + recent tightness ──
    # consol_bars is duration (bars inside the 50d Donchian). For TIGHTNESS we
    # measure the actual range of the LAST ~25 bars, not the full Donchian —
    # otherwise a stock that ran from $50 to $130 over the donchian window
    # gets credit for a "base" just because every bar is inside its own range.
    base_volume_trend = "NO_BASE"
    base_volume_ratio = None
    higher_lows_in_base = False
    consol_range_pct = 0.0
    consol_quality = "NONE"
    if consol["consol_bars"] >= 5:
        base_n = consol["consol_bars"]
        pre_n = 20
        base_window = df.iloc[-base_n:]
        pre_window = df.iloc[-(base_n + pre_n):-base_n] if len(df) >= base_n + pre_n else None
        if pre_window is not None and not pre_window.empty:
            base_vol = float(base_window["Volume"].mean())
            pre_vol = float(pre_window["Volume"].mean())
            if pre_vol > 0:
                base_volume_ratio = base_vol / pre_vol
                if base_volume_ratio < 0.8:
                    base_volume_trend = "DECLINING"
                elif base_volume_ratio > 1.2:
                    base_volume_trend = "RISING"
                else:
                    base_volume_trend = "FLAT"
        higher_lows_in_base = _higher_lows_in_window(base_window)

        # Tightness measured over a capped recent window so trending names
        # with wide Donchian ranges don't get mis-classified as tight bases.
        tight_n = min(int(base_n), 25)
        tight_window = df.iloc[-tight_n:]
        recent_hi = float(tight_window["High"].max())
        recent_lo = float(tight_window["Low"].min())
        if recent_lo > 0:
            consol_range_pct = (recent_hi - recent_lo) / recent_lo * 100
        if consol_range_pct < 10:
            consol_quality = "TIGHT"
        elif consol_range_pct < 20:
            consol_quality = "MODERATE"
        elif consol_range_pct < 30:
            consol_quality = "WIDE"
        else:
            # Beyond 30% over 25 bars isn't a base — it's a trending move.
            consol_quality = "NONE"

    # ── MA stack ──
    ma20 = signals["ma_20"]["ma_value"]
    ma50 = signals["ma_50"]["ma_value"]
    ma200 = signals["ma_200"]["ma_value"]
    if all(v is not None for v in (ma20, ma50, ma200)):
        if ma20 > ma50 > ma200 and last_close > ma20:
            ma_stack = "BULLISH_ALIGNED"
        elif ma200 > ma50 > ma20 and last_close < ma20:
            ma_stack = "BEARISH_ALIGNED"
        elif last_close > ma20 and last_close > ma50 and last_close < ma200:
            ma_stack = "PARTIALLY_ALIGNED"
        else:
            ma_stack = "MIXED"
    else:
        ma_stack = "MIXED"

    # ── RS trend ──
    rs_series = _rs_series(df, spy_df)
    rs_trend_20d = "FLAT"
    rs_new_20d_high = False
    rs_leading = False
    if len(rs_series) >= 21:
        rs_now = float(rs_series.iloc[-1])
        rs_20 = float(rs_series.iloc[-21])
        chg = (rs_now / rs_20 - 1) * 100 if rs_20 else 0
        if chg > 2:
            rs_trend_20d = "UP"
        elif chg < -2:
            rs_trend_20d = "DOWN"
        rs_20d_window = rs_series.iloc[-20:]
        rs_new_20d_high = bool(rs_now >= float(rs_20d_window.max()))
        # Leading = RS trending up while price still in/below the consolidation range.
        price_in_or_below_base = (
            consol["signal"] in ("SQUEEZING", "PRIMED", "TESTING", "PRIMED_DOWN", "NONE")
            and last_close < float(consol["range_high"])
        )
        rs_leading = bool(rs_trend_20d == "UP" and rs_new_20d_high and price_in_or_below_base)

    # ── Close location (last bar) ──
    last_bar = df.iloc[-1]
    bar_range = float(last_bar["High"]) - float(last_bar["Low"])
    close_loc = (float(last_bar["Close"]) - float(last_bar["Low"])) / bar_range if bar_range > 0 else 0.5

    return {
        "proximity_to_breakout_pct": round(proximity_pct, 2),
        "proximity_zone": prox_zone,
        "base_volume_trend": base_volume_trend,
        "base_volume_ratio": None if base_volume_ratio is None else round(base_volume_ratio, 2),
        "ma_stack": ma_stack,
        "ma_20_value": ma20,
        "ma_50_value": ma50,
        "ma_200_value": ma200,
        "consol_range_pct": round(consol_range_pct, 2),
        "consol_duration_days": int(consol["consol_bars"]),
        "consol_quality": consol_quality,
        "higher_lows_in_base": higher_lows_in_base,
        "rs_trend_20d": rs_trend_20d,
        "rs_new_20d_high": rs_new_20d_high,
        "rs_leading": rs_leading,
        "close_location": round(close_loc, 2),
    }


# ─────────────────────────────────────────────────
# Setup type classification
# ─────────────────────────────────────────────────

def classify_setup_type(signals: dict, deep: dict, pattern_results: list) -> str:
    """Classify the technical setup into a named type for downstream routing.

    A name can only be in one setup type at a time; precedence is roughly
    "most actionable first" — fresh breakouts and squeezes ahead of pullbacks,
    pullbacks ahead of pattern formations, pattern formations ahead of generic
    base-building.
    """
    consol = signals["consolidation"]

    # 1. Classic squeeze near breakout — the highest-conviction setup
    if consol["signal"] in ("PRIMED", "SQUEEZING", "PRIMED_DOWN", "TESTING"):
        if deep["proximity_zone"] in ("AT_LEVEL", "APPROACHING"):
            return "SQUEEZE_NEAR_BREAKOUT"

    # 2. Active breakout or breakout retest
    if consol["signal"] == "BREAKOUT_UP":
        if deep["proximity_zone"] == "AT_LEVEL":
            return "BREAKOUT_RETEST"
        return "ACTIVE_BREAKOUT"

    # 3. Trend pullback to a rising MA — only valid in a bullish-aligned stack
    #    and only when price is at or just above the MA (not broken below it).
    if deep["ma_stack"] == "BULLISH_ALIGNED":
        ma20_dist = signals.get("ma_20", {}).get("pct_distance")
        if ma20_dist is not None and -0.5 <= ma20_dist <= 2.5:
            return "TREND_PULLBACK_TO_20MA"
        ma50_dist = signals.get("ma_50", {}).get("pct_distance")
        if ma50_dist is not None and -0.5 <= ma50_dist <= 2.5:
            return "TREND_PULLBACK_TO_50MA"

    # 4. Chart pattern formation (no squeeze required)
    if pattern_results:
        best = max(pattern_results, key=lambda p: p.confidence)
        if best.confidence >= 0.5 and best.status in ("Forming", "Breaking out"):
            return "PATTERN_FORMATION"

    # 5. Consolidation exists but price is still mid-range
    if consol["signal"] != "NONE":
        return "BASE_BUILDING"

    return "NO_DEFINED_SETUP"


# ─────────────────────────────────────────────────
# Entry-quality grading
# ─────────────────────────────────────────────────

def _score_consolidation(deep: dict, signals: dict) -> int:
    """Score consolidation quality (0-20 pts).

    Width quality and ATR compression are scored independently and summed —
    the prior elif chain only covered paired combinations and silently
    dropped a TIGHT/MODERATE base with a normal ATR percentile to 0, which
    is the same score as "no base at all".
    """
    q = deep["consol_quality"]
    atr_pct = signals["consolidation"]["atr_percentile"]
    if q == "NONE":
        return 0

    quality_pts = {"TIGHT": 12, "MODERATE": 8, "WIDE": 4}.get(q, 0)

    if atr_pct < 15:
        atr_pts = 8
    elif atr_pct < 25:
        atr_pts = 6
    elif atr_pct < 40:
        atr_pts = 4
    elif atr_pct < 60:
        atr_pts = 2
    else:
        atr_pts = 0

    pts = min(20, quality_pts + atr_pts)
    if deep["higher_lows_in_base"]:
        pts = min(20, pts + 2)
    return pts


def _score_proximity(deep: dict, signals: dict) -> int:
    status = signals.get("_status", "")
    zone = deep["proximity_zone"]
    base = {
        "AT_LEVEL": 20,
        "APPROACHING": 14,
        "ABOVE": 12,
        "MID_RANGE": 6,
        "FAR": 2,
    }.get(zone, 0)
    # Status is a single-bar signal — penalise but don't zero out an
    # otherwise-valid multi-week setup just because today went red.
    if status == "EXTENDED":
        base = max(0, base - 8)
    elif status in ("BREAKDOWN", "WEAK", "TRENDING DOWN"):
        base = max(0, base - 4)
    return base


def _score_ma_stack(deep: dict) -> int:
    return {
        "BULLISH_ALIGNED": 15,
        "PARTIALLY_ALIGNED": 10,
        "MIXED": 5,
        "BEARISH_ALIGNED": 0,
    }.get(deep["ma_stack"], 0)


def _score_rs(deep: dict) -> int:
    if deep["rs_leading"] and deep["rs_trend_20d"] == "UP":
        return 15
    if deep["rs_trend_20d"] == "UP":
        return 10
    if deep["rs_trend_20d"] == "FLAT":
        return 5
    return 0


def _score_volume(deep: dict) -> int:
    return {
        "DECLINING": 10,
        "FLAT": 6,
        "RISING": 2,
        "NO_BASE": 4,
    }.get(deep["base_volume_trend"], 0)


def _score_intraday(intraday: dict) -> int:
    s = intraday["structure"]
    bias = intraday["session_bias"]
    if s == "HIGHER_LOWS" and bias == "BULLISH":
        return 10
    if s == "HIGHER_LOWS" and bias == "NEUTRAL":
        return 7
    if s == "RANGE_BOUND" and bias == "BULLISH":
        return 5
    if s == "TRENDING_UP":
        return 6
    if s == "RANGE_BOUND" and bias == "NEUTRAL":
        return 4
    # Floor at 3: a single bearish 15-min session shouldn't kill a valid
    # multi-week setup. Intraday confirms, it doesn't disqualify.
    return 3


def _effective_confidence(p) -> float:
    """Weekly patterns get a 20% boost — same setup on a longer timeframe is
    inherently higher conviction (weeks/months of price action vs days)."""
    base = float(p.confidence)
    if "[weekly]" in (p.notes or ""):
        base = min(1.0, base * 1.2)
    return base


def _score_patterns(pattern_results: list) -> tuple[int, Any]:
    if not pattern_results:
        return 0, None
    best = max(pattern_results, key=_effective_confidence)
    eff = _effective_confidence(best)
    if eff >= 0.7 and best.status in ("Forming", "Breaking out"):
        return 10, best
    if eff >= 0.5:
        return 7, best
    if eff >= 0.3:
        return 4, best
    if eff >= 0.15:
        return 2, best
    return 0, best


def grade_entry_quality(
    signals: dict,
    pattern_results: list,
    intraday: dict,
    deep: dict,
    cfg: dict,
) -> dict:
    scores = {
        "consolidation": _score_consolidation(deep, signals),
        "proximity": _score_proximity(deep, signals),
        "ma_stack": _score_ma_stack(deep),
        "rs": _score_rs(deep),
        "volume": _score_volume(deep),
        "intraday": _score_intraday(intraday),
    }
    pattern_score, best_pattern = _score_patterns(pattern_results)
    scores["patterns"] = pattern_score
    total = sum(scores.values())

    setup_type = classify_setup_type(signals, deep, pattern_results)
    floor_applied: str | None = None

    # ── Pattern-based floor ──
    # A high-confidence BULLISH pattern at or near its breakout level is a valid
    # long-entry setup even if the consolidation detector didn't fire. Use
    # effective confidence so weekly patterns get the same floor treatment.
    # Bearish patterns (H&S Top, Rising Wedge, etc.) intentionally don't
    # trigger this — they're short setups, not long entries.
    best_bullish = max(
        (p for p in pattern_results or [] if _is_bullish_pattern(p)),
        key=_effective_confidence,
        default=None,
    )
    if (best_bullish
            and _effective_confidence(best_bullish) >= 0.6
            and deep["proximity_zone"] in ("AT_LEVEL", "APPROACHING")
            and total < 55):
        total = 55
        floor_applied = "pattern"

    # ── Strong-setup floor ──
    # Clean consolidation + aligned MAs is a real setup — one bad session
    # shouldn't drag it to AVOID.
    if scores["consolidation"] >= 12 and scores["ma_stack"] >= 10 and total < 45:
        total = 45
        floor_applied = floor_applied or "strong-setup"

    # ── Trend-pullback floor ──
    if setup_type in ("TREND_PULLBACK_TO_20MA", "TREND_PULLBACK_TO_50MA"):
        if scores["ma_stack"] >= 10 and scores["rs"] >= 10 and total < 60:
            total = 60
            floor_applied = floor_applied or "trend-pullback-strong"
        elif scores["ma_stack"] >= 10 and scores["rs"] >= 5 and total < 50:
            total = 50
            floor_applied = floor_applied or "trend-pullback"

    # ── Breakout-retest floor ──
    if setup_type == "BREAKOUT_RETEST" and scores["ma_stack"] >= 10 and total < 60:
        total = 60
        floor_applied = floor_applied or "breakout-retest"

    # ── Weekly-pattern floor ──
    # High-conviction weekly setups should be flagged even if the DAILY
    # proximity zone says we're not near a daily Donchian level. The weekly
    # pattern has its own breakout level; if price is near or just through
    # it, the name is worth watching regardless of daily structure.
    #
    # Two tiers:
    #   - Confirmed/Breaking-out weekly pattern, effective conf ≥ 0.7 → 55
    #     (KLAC's VCP at 0.84 eff=1.0 falls here even with RS declining)
    #   - Effective conf ≥ 0.6, any status, price within ±10% of pattern
    #     breakout level → 50 (early signal, forming or low-conviction)
    weekly_floor_pat = None
    last_price = signals["_price"]["last"]
    for p in pattern_results or []:
        if "[weekly]" not in (p.notes or ""):
            continue
        if not _is_bullish_pattern(p):
            continue
        eff = _effective_confidence(p)
        bk = p.breakout_level
        if not bk or not last_price:
            continue
        dist_pct = abs(bk - last_price) / last_price * 100
        if eff >= 0.7 and p.status in ("Confirmed", "Breaking out") and dist_pct <= 12:
            if total < 55:
                total = 55
                floor_applied = floor_applied or "weekly-pattern-confirmed"
            weekly_floor_pat = p
            break
        if eff >= 0.6 and dist_pct <= 10 and total < 50:
            total = 50
            floor_applied = floor_applied or "weekly-pattern"
            weekly_floor_pat = weekly_floor_pat or p

    th = cfg["grading"]
    if total >= th["excellent_min"]:
        grade = "EXCELLENT"
    elif total >= th["good_min"]:
        grade = "GOOD"
    elif total >= th["wait_min"]:
        grade = "WAIT"
    else:
        grade = "AVOID"

    # Reasons for/against.
    reasons_for: list[str] = []
    reasons_against: list[str] = []
    if scores["consolidation"] >= 12:
        reasons_for.append(f"{deep['consol_quality'].title()} consolidation, ATR pctile {signals['consolidation']['atr_percentile']:.0f}")
    elif deep["consol_quality"] == "NONE" and setup_type not in (
        "TREND_PULLBACK_TO_20MA", "TREND_PULLBACK_TO_50MA",
        "ACTIVE_BREAKOUT", "BREAKOUT_RETEST", "PATTERN_FORMATION",
    ):
        reasons_against.append("No defined consolidation range")
    if scores["proximity"] >= 14:
        reasons_for.append(f"Price {deep['proximity_to_breakout_pct']:.1f}% from breakout level")
    elif scores["proximity"] <= 2 and setup_type not in (
        "TREND_PULLBACK_TO_20MA", "TREND_PULLBACK_TO_50MA",
    ):
        reasons_against.append("Far from breakout level or already extended")
    if scores["ma_stack"] >= 10:
        reasons_for.append(f"MA stack: {deep['ma_stack'].replace('_', ' ').lower()}")
    elif scores["ma_stack"] == 0:
        reasons_against.append("MA stack bearish")
    if scores["rs"] >= 10:
        reasons_for.append(f"RS trend {deep['rs_trend_20d'].lower()}{', leading price' if deep['rs_leading'] else ''}")
    elif scores["rs"] == 0:
        reasons_against.append("RS underperforming SPY over 20 days")
    if scores["volume"] == 10:
        reasons_for.append(f"Base volume declining ({deep['base_volume_ratio']:.2f}x pre-base)")
    elif scores["volume"] == 2:
        reasons_against.append(f"Base volume rising ({deep['base_volume_ratio']:.2f}x pre-base) — potential distribution")
    if scores["intraday"] >= 7:
        reasons_for.append(f"Intraday: {intraday['structure'].replace('_', ' ').lower()}, session {intraday['session_bias'].lower()}")
    if best_pattern and pattern_score >= 7 and _is_bullish_pattern(best_pattern):
        reasons_for.append(f"{best_pattern.pattern} pattern ({best_pattern.confidence:.0%} confidence, {best_pattern.status.lower()})")
    elif best_pattern and pattern_score >= 7 and not _is_bullish_pattern(best_pattern):
        reasons_against.append(f"Bearish {best_pattern.pattern} pattern ({best_pattern.confidence:.0%} confidence, {best_pattern.status.lower()})")
    if setup_type == "TREND_PULLBACK_TO_20MA":
        reasons_for.append("Trend pullback to rising 20-day MA")
    elif setup_type == "TREND_PULLBACK_TO_50MA":
        reasons_for.append("Trend pullback to rising 50-day MA")
    elif setup_type == "BREAKOUT_RETEST":
        reasons_for.append("Breakout retest in progress")
    if weekly_floor_pat is not None:
        eff = _effective_confidence(weekly_floor_pat)
        reasons_for.append(
            f"Weekly {weekly_floor_pat.pattern.lower()} "
            f"{weekly_floor_pat.status.lower()} ({eff:.0%} eff. conf, "
            f"breakout ${weekly_floor_pat.breakout_level:.2f})"
        )

    # Trigger / Risk / Target — base logic (squeeze/pattern), then setup-type overrides.
    # For trigger/target we use the highest-confidence BULLISH pattern, since
    # the grade represents long-entry quality. Bearish patterns surface in
    # commentary but don't define the trigger.
    consol = signals["consolidation"]
    range_high = consol.get("range_high")
    range_low = consol.get("range_low")
    if consol["signal"] != "NONE" and range_high:
        trigger = f"Break above ${range_high:.2f} on 2x+ volume with close in upper third of bar"
    elif best_bullish is not None:
        trigger = f"Break above ${best_bullish.breakout_level:.2f} ({best_bullish.pattern} trigger)"
    else:
        trigger = "No defined trigger level — trending, not basing."

    if consol["signal"] != "NONE" and range_low:
        risk = f"Below ${range_low:.2f} — consolidation floor"
    elif deep.get("ma_50_value"):
        risk = f"Below ${deep['ma_50_value']:.2f} — 50-day MA"
    else:
        risk = "No defined stop reference"

    target = None
    if best_bullish is not None and best_bullish.target:
        target = f"${best_bullish.target:.2f} — measured move from {best_bullish.pattern}"
    elif consol["signal"] != "NONE" and range_high and range_low:
        measured = range_high + (range_high - range_low)
        target = f"${measured:.2f} — measured move from consolidation"
    else:
        last = signals["_price"]["last"]
        h52 = signals["nday_252"]["high"]
        if h52 and last and last < h52:
            target = f"${h52:.2f} — 52-week high"
        else:
            target = "No defined target"

    # Setup-type overrides for non-squeeze entries.
    ma20_v = deep.get("ma_20_value")
    ma50_v = deep.get("ma_50_value")
    ma200_v = deep.get("ma_200_value")
    nday50_hi = signals.get("nday_50", {}).get("high")
    nday252_hi = signals.get("nday_252", {}).get("high")
    if setup_type == "TREND_PULLBACK_TO_20MA" and ma20_v:
        trigger = f"Bounce off 20-day MA (${ma20_v:.2f}) with a close in upper half of bar"
        if ma50_v:
            risk = f"Below ${ma50_v:.2f} — 50-day MA (trend invalidation)"
        if not target or target.startswith("No defined") or "—" not in target:
            if nday252_hi:
                target = f"${nday252_hi:.2f} — 52-week high"
            elif nday50_hi:
                target = f"${nday50_hi:.2f} — 50-day high"
    elif setup_type == "TREND_PULLBACK_TO_50MA" and ma50_v:
        trigger = f"Bounce off 50-day MA (${ma50_v:.2f}) with a close in upper half of bar"
        if ma200_v:
            risk = f"Below ${ma200_v:.2f} — 200-day MA"
        else:
            risk = f"Below ${ma50_v:.2f} — 50-day MA"
        if not target or target.startswith("No defined"):
            if nday252_hi:
                target = f"${nday252_hi:.2f} — 52-week high"
            elif nday50_hi:
                target = f"${nday50_hi:.2f} — 50-day high"
    elif setup_type == "BREAKOUT_RETEST" and range_high:
        trigger = f"Hold above ${range_high:.2f} on the retest with volume"
        if range_low:
            risk = f"Below ${range_low:.2f} — failed retest"
    elif setup_type == "ACTIVE_BREAKOUT" and range_high:
        trigger = f"Already triggered — watch for close above ${range_high:.2f} to hold"
        risk = f"Below ${range_high:.2f} — failed breakout"

    return {
        "grade": grade,
        "grade_score": int(round(total)),
        "score_breakdown": scores,
        "reasons_for": reasons_for,
        "reasons_against": reasons_against,
        "trigger": trigger,
        "risk_level": risk,
        "target": target,
        "setup_type": setup_type,
        "floor_applied": floor_applied,
    }


# ─────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────

def _pattern_results_for(df: pd.DataFrame, ticker: str) -> list:
    _, results = patterns.scan_patterns({ticker: df}, min_confidence=0.0)
    return list(results.values())


def _rescore_patterns_for_single_name(pattern_results: list, df: pd.DataFrame) -> list:
    """Widen proximity tolerance for single-name deep analysis.

    The broad screener uses max_distance=0.12 which crushes confidence on
    patterns that are geometrically clean but currently far from the trigger.
    For single-name work, we widen to 0.25 so forming patterns surface earlier.
    Patterns inside the old 12% window get their confidence scaled back up;
    patterns 12%-25% out get a modest floor based on the fact that the
    detector accepted the geometry at all.
    """
    if not pattern_results or df.empty:
        return pattern_results
    last_close = float(df["Close"].iloc[-1])
    if last_close <= 0:
        return pattern_results
    old_max = 0.12
    new_max = 0.25
    out: list = []
    for p in pattern_results:
        side = "down" if p.pattern in _BEARISH_PATTERNS else "up"
        if side == "up":
            dist = (p.breakout_level - last_close) / last_close
        else:
            dist = (last_close - p.breakout_level) / last_close
        if dist <= 0:
            new_prox = 1.0
        elif dist >= new_max:
            new_prox = 0.0
        else:
            new_prox = 1.0 - dist / new_max
        if dist > 0 and dist <= new_max:
            if dist <= old_max:
                # Recoverable: the detector's old proximity factor merely
                # suppressed the score; strip it out and re-apply ours.
                old_prox = max(1e-6, 1.0 - dist / old_max)
                base = p.confidence / old_prox
                new_conf = round(min(1.0, base * new_prox), 3)
                out.append(PatternResult(
                    pattern=p.pattern, ticker=p.ticker,
                    start_date=p.start_date, end_date=p.end_date,
                    breakout_level=p.breakout_level, target=p.target,
                    status=p.status, confidence=new_conf,
                    key_points=p.key_points, notes=p.notes,
                    max_distance=new_max,
                ))
            else:
                # 12-25%: the old proximity zero'd out the confidence so we
                # can't recover the geometry score. Use a modest floor that
                # signals "early-forming, watch it" without overpromoting.
                floor_conf = round(new_prox * 0.5, 3)
                tag = " (rescored: distant)" if "rescored" not in (p.notes or "") else ""
                out.append(PatternResult(
                    pattern=p.pattern, ticker=p.ticker,
                    start_date=p.start_date, end_date=p.end_date,
                    breakout_level=p.breakout_level, target=p.target,
                    status="Forming", confidence=floor_conf,
                    key_points=p.key_points, notes=(p.notes or "") + tag,
                    max_distance=new_max,
                ))
        else:
            # Inside the level already, or too distant — leave as-is.
            out.append(p)
    return out


def _tag_weekly(pattern_results: list) -> list:
    """Append [weekly] to notes so downstream can distinguish timeframes."""
    out = []
    for p in pattern_results:
        notes = p.notes or ""
        if "[weekly]" not in notes:
            notes = (notes + " [weekly]").strip()
        out.append(PatternResult(
            pattern=p.pattern, ticker=p.ticker,
            start_date=p.start_date, end_date=p.end_date,
            breakout_level=p.breakout_level, target=p.target,
            status=p.status, confidence=p.confidence,
            key_points=p.key_points, notes=notes,
            max_distance=p.max_distance,
        ))
    return out


def _resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to Friday-anchored weekly bars."""
    if df is None or df.empty:
        return pd.DataFrame()
    w = df.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()
    return w


def _sync_live_close(df_daily: pd.DataFrame, df_intraday: pd.DataFrame) -> dict:
    """Sync today's daily-bar OHLC from the latest intraday data.

    The daily aggregate from Polygon /aggs/day lags the live tape by minutes
    (sometimes 15+) even on real-time plans. We patch today's daily bar using
    the latest 15-min intraday data so setup detection reflects live price.

    Three cases:
      1. df_daily ends at today AND df_intraday has today's bars
           → update today's daily Close/High/Low in place
      2. df_daily ends before today AND df_intraday has today's bars
           → append a synthetic today bar built from today's intraday OHLCV
      3. Otherwise (market closed, no fresh intraday)
           → no-op

    Returns a `live` dict for downstream UI display:
        {enabled, last_price, source_ts, synthetic}
    """
    info = {"enabled": False, "last_price": None, "source_ts": None, "synthetic": False}
    if df_intraday is None or df_intraday.empty or df_daily is None or df_daily.empty:
        return info

    intraday_dates = pd.DatetimeIndex(df_intraday.index).date
    latest_intraday_date = intraday_dates[-1]
    latest_daily_date = df_daily.index[-1].date()

    today_intraday = df_intraday[intraday_dates == latest_intraday_date]
    if today_intraday.empty:
        return info

    live_open = float(today_intraday["Open"].iloc[0])
    live_high = float(today_intraday["High"].max())
    live_low = float(today_intraday["Low"].min())
    live_close = float(today_intraday["Close"].iloc[-1])
    live_volume = float(today_intraday["Volume"].sum())
    live_ts = today_intraday.index[-1]

    if latest_intraday_date == latest_daily_date:
        # Case 1: update today's existing daily bar in place
        idx = df_daily.index[-1]
        df_daily.loc[idx, "Close"] = live_close
        df_daily.loc[idx, "High"] = max(float(df_daily.loc[idx, "High"]), live_high)
        df_daily.loc[idx, "Low"] = min(float(df_daily.loc[idx, "Low"]), live_low)
        info.update({
            "enabled": True,
            "last_price": round(live_close, 2),
            "source_ts": live_ts,
            "synthetic": False,
        })
    elif latest_intraday_date > latest_daily_date:
        # Case 2: Polygon hasn't built today's daily yet — synthesize it.
        # Anchor at 04:00 to match Polygon's daily-bar timestamp convention
        # (00:00 ET = 04:00 UTC) so the index keeps its meaning.
        new_ts = pd.Timestamp(f"{latest_intraday_date} 04:00:00")
        df_daily.loc[new_ts, "Open"] = live_open
        df_daily.loc[new_ts, "High"] = live_high
        df_daily.loc[new_ts, "Low"] = live_low
        df_daily.loc[new_ts, "Close"] = live_close
        df_daily.loc[new_ts, "Volume"] = live_volume
        df_daily.sort_index(inplace=True)
        info.update({
            "enabled": True,
            "last_price": round(live_close, 2),
            "source_ts": live_ts,
            "synthetic": True,
        })

    return info


def analyze_ticker(ticker: str, cfg: dict | None = None) -> dict:
    """Run the full single-name analysis for one ticker."""
    ticker = ticker.strip().upper()
    cfg = cfg or load_config()

    df_cached = _cached_fetch_daily(ticker, days=cfg["data"]["daily_bars_days"])
    if df_cached is None or df_cached.empty:
        return {
            "ticker": ticker,
            "error": f"No data found for {ticker}. Check the symbol and try again.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    # Copy before any mutation — _cached_fetch_daily returns a shared reference
    # held by Streamlit's cache; we don't want our live-sync edits to bleed
    # into other concurrent reads of the same ticker.
    df = df_cached.copy()

    spy_df = _cached_fetch_daily(cfg["data"]["benchmark"], days=cfg["data"]["daily_bars_days"])
    df_intraday = _cached_fetch_intraday(
        ticker,
        multiplier=cfg["data"]["intraday_multiplier"],
        days=cfg["data"]["intraday_bars_days"],
    )

    # Patch today's daily bar from intraday so setup detection runs on the
    # live tape instead of Polygon's lagged daily aggregate.
    live_info = _sync_live_close(df, df_intraday)

    signals, raw_detector = compute_extended_signals(df, spy_df, cfg)

    # Daily patterns — rescored with wider 25% proximity for single-name work.
    daily_patterns = _rescore_patterns_for_single_name(
        _pattern_results_for(df, ticker), df
    )

    # Weekly patterns — same scanner, weekly bars, tagged [weekly] in notes.
    df_weekly = _resample_weekly(df)
    if len(df_weekly) >= 30:
        weekly_patterns = _rescore_patterns_for_single_name(
            _pattern_results_for(df_weekly, ticker), df_weekly
        )
        weekly_patterns = _tag_weekly(weekly_patterns)
    else:
        weekly_patterns = []
    pattern_results = daily_patterns + weekly_patterns

    intraday = intraday_analysis.analyze_intraday(df_intraday, signals)
    deep = compute_deep_layers(df, spy_df, signals)
    entry_quality = grade_entry_quality(signals, pattern_results, intraday, deep, cfg)

    # New: enumerate every plausible future breakout/breakdown setup.
    setups = setup_finder.find_all_setups(
        df=df,
        df_weekly=df_weekly,
        pattern_results=pattern_results,
        signals=signals,
        deep=deep,
        cfg=cfg,
    )
    breakouts, breakdowns = setup_finder.split_by_direction(setups)
    triggered = setup_finder.triggered_recently(setups)
    headline = setup_finder.headline_takeaway(setups, signals["_price"]["last"])

    return {
        "ticker": ticker,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "price": signals["_price"],
        "status": signals["_status"],
        "score": signals["_score"],
        "setup_score": signals["_setup_score_up"],
        "signals": signals,
        "raw_detector": raw_detector,
        "patterns": pattern_results,
        "daily_patterns": daily_patterns,
        "weekly_patterns": weekly_patterns,
        "intraday": intraday,
        "deep": deep,
        "entry_quality": entry_quality,
        "setup_type": entry_quality.get("setup_type", "NO_DEFINED_SETUP"),
        "setups": setups,
        "breakouts": breakouts,
        "breakdowns": breakdowns,
        "triggered": triggered,
        "headline": headline,
        "live": live_info,
        "df_daily": df,
        "df_weekly": df_weekly,
        "df_intraday": df_intraday,
        "df_spy": spy_df,
    }
