"""Breakout / consolidation detection on OHLCV data.

v2 — Pre-breakout detection upgrade:
  - ATR compression velocity (how fast the squeeze is forming)
  - Volume trend slope (gradient, not binary)
  - Higher lows / lower highs within the base (slope-gated linreg)
  - Pattern integration (VCP/flag/triangle boost via scan_patterns)
  - Reweighted scoring: proximity 30 pts (was 50), new inputs get 30 pts
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift()
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def fetch_universe(tickers: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from polygon_adapter import fetch_data
    days = max(lookback_days, 365)
    out: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fetch_data, t, days): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                df = fut.result()
            except Exception:
                continue
            if not df.empty:
                out[t] = df
    return out


def _percentile_rank(series: pd.Series, value: float) -> float:
    """Where does `value` sit in `series`? 0 = lowest, 100 = highest."""
    s = series.dropna()
    if len(s) == 0:
        return float("nan")
    return float((s <= value).sum() / len(s) * 100)


def _return_n_days(close: pd.Series, n: int) -> float | None:
    if len(close) < n + 1:
        return None
    return float(close.iloc[-1] / close.iloc[-n - 1] - 1) * 100


# ─────────────────────────────────────────────────
# Swing-point structure: higher lows / lower highs
# ─────────────────────────────────────────────────
# These run on the base/consolidation window only (default last 25 bars,
# not the full 50-bar Donchian window) — running over a 50-bar window that
# includes a trending leg makes a trend look like rising lows. The slope
# is gated: micro-noise (|pct_slope| < 0.1) returns 0 to avoid scoring
# chop as structure.

_BASE_WINDOW = 25       # bars to use for the basing analysis
_BASE_MIN_BARS = 15     # need at least this many bars in the segment
_SLOPE_MIN = 0.10       # min |%-slope per swing-point| to count as real
_SWING_DISTANCE = 4     # min bars between detected swing points


def _find_swing_lows(lows: np.ndarray, distance: int = _SWING_DISTANCE) -> list[int]:
    out: list[int] = []
    n = len(lows)
    for i in range(distance, n - distance):
        window = lows[i - distance: i + distance + 1]
        if lows[i] == np.min(window):
            out.append(i)
    return out


def _find_swing_highs(highs: np.ndarray, distance: int = _SWING_DISTANCE) -> list[int]:
    out: list[int] = []
    n = len(highs)
    for i in range(distance, n - distance):
        window = highs[i - distance: i + distance + 1]
        if highs[i] == np.max(window):
            out.append(i)
    return out


def _linreg_pct_slope(prices: list[float]) -> float:
    """% slope per index step, normalised by mean. 0 if too few / undefined."""
    if len(prices) < 2:
        return 0.0
    x = np.arange(len(prices), dtype=float)
    y = np.asarray(prices, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = float(np.sum((x - x_mean) ** 2))
    if denom < 1e-9 or y_mean <= 0:
        return 0.0
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    return slope / y_mean * 100


def _higher_lows_score(lows: np.ndarray) -> float:
    """0-1 score for rising swing lows in the base. Slope-gated."""
    if len(lows) < _BASE_MIN_BARS:
        return 0.0
    idxs = _find_swing_lows(lows)
    if len(idxs) < 2:
        return 0.0
    prices = [float(lows[i]) for i in idxs]
    slope = _linreg_pct_slope(prices)
    if slope <= _SLOPE_MIN:
        return 0.0
    rising_pairs = sum(1 for i in range(len(prices) - 1) if prices[i + 1] > prices[i])
    rising_ratio = rising_pairs / max(len(prices) - 1, 1)
    return min(1.0, 0.5 + rising_ratio * 0.5)


def _lower_highs_score(highs: np.ndarray) -> float:
    """0-1 score for declining swing highs in the base. Slope-gated."""
    if len(highs) < _BASE_MIN_BARS:
        return 0.0
    idxs = _find_swing_highs(highs)
    if len(idxs) < 2:
        return 0.0
    prices = [float(highs[i]) for i in idxs]
    slope = _linreg_pct_slope(prices)
    if slope >= -_SLOPE_MIN:
        return 0.0
    declining_pairs = sum(1 for i in range(len(prices) - 1) if prices[i + 1] < prices[i])
    declining_ratio = declining_pairs / max(len(prices) - 1, 1)
    return min(1.0, 0.5 + declining_ratio * 0.5)


# ─────────────────────────────────────────────────
# ATR compression velocity
# ─────────────────────────────────────────────────

def _atr_compression_velocity(
    atr_ratio_series: pd.Series, pct_window: int, lookback: int = 10,
) -> float:
    """How many percentile points has ATR percentile dropped over `lookback` bars?
    Positive = compressing (squeeze tightening). Returns 0 on either no drop or
    insufficient history.

    Note: the two percentile windows used here overlap heavily (offset by
    `lookback` over a 252-bar window) so the magnitude in practice is modest —
    a 15 pctile-point drop in 10 days is a strong reading, not extreme.
    """
    needed = pct_window + lookback
    if len(atr_ratio_series) < needed:
        return 0.0
    current_val = float(atr_ratio_series.iloc[-1])
    current_pct = _percentile_rank(atr_ratio_series.iloc[-pct_window:], current_val)
    past_val = float(atr_ratio_series.iloc[-1 - lookback])
    past_pct = _percentile_rank(
        atr_ratio_series.iloc[-pct_window - lookback:-lookback], past_val,
    )
    return max(0.0, past_pct - current_pct)


# ─────────────────────────────────────────────────
# Volume trend ratio
# ─────────────────────────────────────────────────

def _volume_trend_ratio(volume: pd.Series, short: int = 10, long: int = 40) -> float:
    """short-window mean volume / long-window mean volume.
    Below 1.0 = drying up. Calibrated so 0.85→0 pts and 0.55→10 pts upstream,
    which keeps the scoring range realistic for liquid US names.
    """
    if len(volume) < long:
        return 1.0
    short_avg = float(volume.iloc[-short:].mean())
    long_avg = float(volume.iloc[-long:].mean())
    if long_avg <= 0:
        return 1.0
    return short_avg / long_avg


def _vol_dryup_points(ratio: float) -> float:
    """0-10 score from volume trend ratio. Recalibrated tighter than v2-draft:
    0.85+ scores 0 (no meaningful dry-up); 0.55- scores full 10."""
    if ratio >= 0.85:
        return 0.0
    if ratio <= 0.55:
        return 10.0
    return (0.85 - ratio) / 0.30 * 10.0


# ─────────────────────────────────────────────────
# Setup scoring (v2 weights)
# ─────────────────────────────────────────────────
# Total = 100. Component caps:
#   Proximity         30   (was 50 — proximity alone shouldn't dominate)
#   Squeeze level     15   (8 ATR + 7 BBW)
#   Trend regime      10
#   Bar shape         10
#   RS vs SPY          5
#   ATR velocity      10   NEW — squeeze forming, not just formed
#   Volume dry-up     10   NEW — gradient, replaces old binary band
#   Higher/lower X     5   NEW — base structure
#   Pattern boost      5   NEW — VCP / flag / triangle confirmation


def _setup_score_up(
    *, dist_pct: float, atr_pct: float, bbw_pct: float,
    above_ma: bool, slope_20d: float, vol_ratio: float,
    rs_21d: float | None, close_loc: float, pct_1d: float,
    atr_velocity: float, vol_trend_ratio: float,
    higher_lows: float, pattern_forming: bool,
) -> float:
    if dist_pct <= 0 or dist_pct > 15:
        return 0.0

    prox = (1 - dist_pct / 15) * 30
    atr_s = max(0.0, (50 - min(atr_pct, 100)) / 50) * 8
    bbw_s = max(0.0, (50 - min(bbw_pct, 100)) / 50) * 7

    trend_s = 0.0
    if above_ma:
        trend_s += 5
    if slope_20d > 0:
        trend_s += 5

    rs_s = 0.0
    if rs_21d is not None and rs_21d > 0:
        rs_s = min(5.0, rs_21d / 2.0)

    bar_s = 0.0
    if close_loc >= 0.6:
        bar_s += 5.0
    if pct_1d > 0:
        bar_s += 5.0

    atr_vel_s = min(10.0, atr_velocity / 15 * 10)
    vol_dry_s = _vol_dryup_points(vol_trend_ratio)
    hl_s = higher_lows * 5.0
    pat_s = 5.0 if pattern_forming else 0.0

    return min(100.0, prox + atr_s + bbw_s + trend_s + rs_s + bar_s
               + atr_vel_s + vol_dry_s + hl_s + pat_s)


def _setup_score_dn(
    *, dist_pct: float, atr_pct: float, bbw_pct: float,
    above_ma: bool, slope_20d: float, vol_ratio: float,
    rs_21d: float | None, close_loc: float, pct_1d: float,
    atr_velocity: float, vol_trend_ratio: float,
    lower_highs: float, pattern_forming: bool,
) -> float:
    if dist_pct <= 0 or dist_pct > 15:
        return 0.0

    prox = (1 - dist_pct / 15) * 30
    atr_s = max(0.0, (50 - min(atr_pct, 100)) / 50) * 8
    bbw_s = max(0.0, (50 - min(bbw_pct, 100)) / 50) * 7

    trend_s = 0.0
    if not above_ma:
        trend_s += 5
    if slope_20d < 0:
        trend_s += 5

    rs_s = 0.0
    if rs_21d is not None and rs_21d < 0:
        rs_s = min(5.0, abs(rs_21d) / 2.0)

    bar_s = 0.0
    if close_loc <= 0.4:
        bar_s += 5.0
    if pct_1d < 0:
        bar_s += 5.0

    atr_vel_s = min(10.0, atr_velocity / 15 * 10)
    vol_dry_s = _vol_dryup_points(vol_trend_ratio)
    lh_s = lower_highs * 5.0
    pat_s = 5.0 if pattern_forming else 0.0

    return min(100.0, prox + atr_s + bbw_s + trend_s + rs_s + bar_s
               + atr_vel_s + vol_dry_s + lh_s + pat_s)


# ─────────────────────────────────────────────────
# Main analyze()
# ─────────────────────────────────────────────────

def analyze(
    df: pd.DataFrame,
    params: dict,
    spy_ret_21d: float | None = None,
    bullish_pattern: str | None = None,
    bearish_pattern: str | None = None,
) -> dict:
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    tr = _true_range(high, low, close)
    atr_short = tr.rolling(params["atr_short"]).mean()
    atr_long = tr.rolling(params["atr_long"]).mean()
    atr_ratio_s = atr_short / atr_long

    bb_ma = close.rolling(params["bb_period"]).mean()
    bb_sd = close.rolling(params["bb_period"]).std()
    bbw_s = (2 * params["bb_std"] * bb_sd) / bb_ma

    range_window = params["range_window"]
    recent_range = (high.rolling(range_window).max() - low.rolling(range_window).min()) / close

    donchian = high.rolling(params["donchian_period"]).max().shift(1)
    donchian_lo = low.rolling(params["donchian_period"]).min().shift(1)
    vol_avg = vol.rolling(params["volume_avg_period"]).mean()
    ma_50 = close.rolling(50).mean()
    high_20 = high.rolling(20).max()
    low_20 = low.rolling(20).min()

    last = -1
    last_close = float(close.iloc[last])
    last_high = float(high.iloc[last])
    last_low = float(low.iloc[last])
    prev_close = float(close.iloc[-2])
    atr_ratio = float(atr_ratio_s.iloc[last])
    bbw = float(bbw_s.iloc[last])
    range_pct = float(recent_range.iloc[last])
    donchian_high = float(donchian.iloc[last])
    donchian_low = float(donchian_lo.iloc[last])
    prev_donchian_high = float(donchian.iloc[-2])
    prev_donchian_low = float(donchian_lo.iloc[-2])
    vol_ratio = float(vol.iloc[last] / vol_avg.iloc[last])
    pct_1d = (last_close / prev_close - 1) * 100

    bar_range = last_high - last_low
    close_loc = (last_close - last_low) / bar_range if bar_range > 0 else 0.5

    pct_window = params["pct_window"]
    atr_pct = _percentile_rank(atr_ratio_s.iloc[-pct_window:], atr_ratio)
    bbw_pct = _percentile_rank(bbw_s.iloc[-pct_window:], bbw)

    squeezing = atr_pct < params["atr_pct_max"] and bbw_pct < params["bbw_pct_max"]

    fresh_cross_up = last_close > donchian_high and prev_close <= prev_donchian_high
    fresh_cross_down = last_close < donchian_low and prev_close >= prev_donchian_low

    strong_up = vol_ratio >= params["strong_vol_mult"] and close_loc >= params["strong_close_loc"]
    moderate_up_volume = vol_ratio >= params["moderate_vol_mult"] and close_loc >= params["moderate_close_loc"]
    magnitude_up = pct_1d >= params["magnitude_pct"] and close_loc >= params["magnitude_close_loc"]
    moderate_up = moderate_up_volume or magnitude_up
    strong_dn = vol_ratio >= params["strong_vol_mult"] and (1 - close_loc) >= params["strong_close_loc"]
    moderate_dn_volume = vol_ratio >= params["moderate_vol_mult"] and (1 - close_loc) >= params["moderate_close_loc"]
    magnitude_dn = pct_1d <= -params["magnitude_pct"] and (1 - close_loc) >= params["magnitude_close_loc"]
    moderate_dn = moderate_dn_volume or magnitude_dn

    if strong_up:
        confirm_up_tier = "Strong"
    elif moderate_up_volume:
        confirm_up_tier = "Moderate"
    elif magnitude_up:
        confirm_up_tier = "Magnitude"
    else:
        confirm_up_tier = "Unconfirmed"

    if strong_dn:
        confirm_dn_tier = "Strong"
    elif moderate_dn_volume:
        confirm_dn_tier = "Moderate"
    elif magnitude_dn:
        confirm_dn_tier = "Magnitude"
    else:
        confirm_dn_tier = "Unconfirmed"

    breakout = fresh_cross_up and (strong_up or moderate_up)
    breakdown = fresh_cross_down and (strong_dn or moderate_dn)
    testing_up = fresh_cross_up and not breakout
    testing_dn = fresh_cross_down and not breakdown

    high_52w = float(high.tail(252).max())
    pct_from_52w = last_close / high_52w - 1

    ma50 = float(ma_50.iloc[last]) if pd.notna(ma_50.iloc[last]) else last_close
    above_ma = last_close > ma50
    close_20d_ago = float(close.iloc[-21]) if len(close) >= 21 else last_close
    slope_20 = (last_close / close_20d_ago - 1) * 100 if close_20d_ago else 0
    h20 = float(high_20.iloc[last])
    l20 = float(low_20.iloc[last])
    pct_from_20d_high = (last_close / h20 - 1) * 100
    pct_from_20d_low = (last_close / l20 - 1) * 100

    pct_to_breakout_above = (donchian_high - last_close) / last_close
    pct_to_breakdown_below = (last_close - donchian_low) / last_close

    # ── New v2 inputs ──
    atr_velocity = _atr_compression_velocity(atr_ratio_s, pct_window, lookback=10)
    vol_trend = _volume_trend_ratio(vol, short=10, long=40)

    # Base/consolidation window: last 25 bars excluding today. Anchored short
    # so the swing-point regression measures the base, not a trending leg.
    base_end = len(low) - 1
    base_start = max(0, base_end - _BASE_WINDOW)
    base_lows = low.iloc[base_start:base_end].to_numpy(dtype=float)
    base_highs = high.iloc[base_start:base_end].to_numpy(dtype=float)
    hl_score = _higher_lows_score(base_lows)
    lh_score = _lower_highs_score(base_highs)

    bull_pattern_forming = bullish_pattern is not None
    bear_pattern_forming = bearish_pattern is not None

    # ── Status classification ──
    if breakout:
        status = "BREAKOUT"
    elif breakdown:
        status = "BREAKDOWN"
    elif testing_up:
        status = "TESTING ↑"
    elif testing_dn:
        status = "TESTING ↓"
    elif squeezing and pct_to_breakout_above < 0.03:
        status = "PRIMED"
    elif squeezing and pct_to_breakdown_below < 0.03:
        status = "AT RISK"
    elif (0 < pct_to_breakout_above <= 0.05
          and (atr_pct < 50 or bbw_pct < 50)
          and above_ma):
        status = "COILED ↑"
    elif (0 < pct_to_breakdown_below <= 0.05
          and (atr_pct < 50 or bbw_pct < 50)
          and not above_ma):
        status = "COILED ↓"
    elif last_close > donchian_high:
        status = "EXTENDED"
    elif last_close < donchian_low:
        status = "WEAK"
    elif squeezing:
        status = "CONSOLIDATING"
    elif above_ma and -15 < pct_from_20d_high < -5:
        status = "PULLBACK"
    elif (not above_ma) and pct_from_20d_low > 5 and not magnitude_dn and not strong_dn and pct_1d > -2:
        status = "BOUNCE"
    elif above_ma and slope_20 > 2:
        status = "TRENDING UP"
    elif (not above_ma) and slope_20 < -2:
        status = "TRENDING DOWN"
    else:
        status = "NEUTRAL"

    stock_ret_21d = _return_n_days(close, 21)
    if stock_ret_21d is not None and spy_ret_21d is not None:
        rs_21d = round(stock_ret_21d - spy_ret_21d, 2)
    else:
        rs_21d = None

    dist_up_pct = (donchian_high - last_close) / last_close * 100
    dist_dn_pct = (last_close - donchian_low) / last_close * 100

    setup_score_up = _setup_score_up(
        dist_pct=dist_up_pct, atr_pct=atr_pct, bbw_pct=bbw_pct,
        above_ma=above_ma, slope_20d=slope_20, vol_ratio=vol_ratio,
        rs_21d=rs_21d, close_loc=close_loc, pct_1d=pct_1d,
        atr_velocity=atr_velocity, vol_trend_ratio=vol_trend,
        higher_lows=hl_score, pattern_forming=bull_pattern_forming,
    )
    setup_score_dn = _setup_score_dn(
        dist_pct=dist_dn_pct, atr_pct=atr_pct, bbw_pct=bbw_pct,
        above_ma=above_ma, slope_20d=slope_20, vol_ratio=vol_ratio,
        rs_21d=rs_21d, close_loc=close_loc, pct_1d=pct_1d,
        atr_velocity=atr_velocity, vol_trend_ratio=vol_trend,
        lower_highs=lh_score, pattern_forming=bear_pattern_forming,
    )

    return {
        "close": round(last_close, 2),
        "pct_1d": round(pct_1d, 2),
        "donchian_high": round(donchian_high, 2),
        "donchian_low": round(donchian_low, 2),
        "pct_to_breakout": round((donchian_high / last_close - 1) * 100, 2),
        "pct_to_breakdown": round((last_close / donchian_low - 1) * 100, 2),
        "atr_contraction": round(atr_ratio, 2),
        "atr_pct": round(atr_pct, 1),
        "bbw_pct": round(bbw_pct, 1),
        "range_10d_pct": round(range_pct * 100, 2),
        "vol_ratio": round(vol_ratio, 2),
        "close_loc": round(close_loc, 2),
        "confirm_up": confirm_up_tier,
        "confirm_dn": confirm_dn_tier,
        "slope_20d": round(slope_20, 2),
        "vs_ma50": round((last_close / ma50 - 1) * 100, 2),
        "pct_from_52w_high": round(pct_from_52w * 100, 2),
        "rs_21d": rs_21d,
        "setup_score_up": round(setup_score_up, 1),
        "setup_score_dn": round(setup_score_dn, 1),
        "atr_velocity": round(atr_velocity, 1),
        "vol_trend_ratio": round(vol_trend, 2),
        "higher_lows_score": round(hl_score, 2),
        "lower_highs_score": round(lh_score, 2),
        "pattern_boost_up": bullish_pattern or "",
        "pattern_boost_dn": bearish_pattern or "",
        "status": status,
        "squeezing": squeezing,
        "breakout": breakout,
        "breakdown": breakdown,
        # ── CONFLUENCE GATES ─────────────────────────────────────────
        # Fire ONLY when every independent signal aligns. Designed for
        # 0-2 hits per scan, not 8. Bullish uses v2 base-structure +
        # volume-dry-up signals (the old close_loc check pushed this
        # toward TESTING ↑ names that were already moving, not toward
        # genuine pre-breakout coils — see commit history).
        "confluence": bool(
            0 < pct_to_breakout_above < 0.05         # within 5% of Donchian high
            and atr_pct < 30                          # squeeze formed
            and atr_velocity > 5                      # squeeze still tightening
            and vol_trend < 0.85                      # 10d vol drying vs 40d
            and (rs_21d is not None and rs_21d > 0)   # outperforming SPY
            and above_ma                              # above 50d MA
            and slope_20 > 0                          # 20d slope up
            and pct_1d >= -0.5                        # not getting sold today
            and hl_score >= 0.5                       # rising swing lows in base
        ),
        # Bearish mirror — same scarcity profile.
        "confluence_dn": bool(
            0 < pct_to_breakdown_below < 0.05         # within 5% of Donchian low
            and atr_pct < 30                          # squeeze formed
            and atr_velocity > 5                      # squeeze still tightening
            and vol_trend < 0.85                      # vol drying up
            and (rs_21d is not None and rs_21d < 0)   # underperforming SPY
            and not above_ma                          # below 50d MA
            and slope_20 < 0                          # 20d slope down
            and pct_1d <= 0.5                         # not getting bought today
            and lh_score >= 0.5                       # falling swing highs in base
        ),
    }


def latest_bar_date(data: dict[str, pd.DataFrame]) -> pd.Timestamp | None:
    if not data:
        return None
    return max(df.index[-1] for df in data.values() if len(df))


# ─────────────────────────────────────────────────
# Pattern lookup: derive per-ticker best bull/bear pattern from
# the single scan_patterns() call, so we don't run detectors twice.
# ─────────────────────────────────────────────────

def _best_patterns_per_ticker(
    pat_table: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return ({ticker: bull_pattern_name}, {ticker: bear_pattern_name}).

    A pattern counts as "actionable" for setup-score boost if its status is
    Forming or Breaking out — Confirmed means the level already broke, which
    the Donchian engine handles separately.

    Direction is derived from target vs breakout_level (handles
    detect_symmetrical_triangle, which can fire either way).

    Tie-break: higher confidence wins.
    """
    bull: dict[str, str] = {}
    bear: dict[str, str] = {}
    if pat_table is None or len(pat_table) == 0:
        return bull, bear
    if not {"ticker", "pattern", "status", "breakout_level",
            "target", "confidence"}.issubset(pat_table.columns):
        return bull, bear

    actionable = pat_table[pat_table["status"].isin(["Forming", "Breaking out"])]
    if len(actionable) == 0:
        return bull, bear

    direction = (actionable["target"] > actionable["breakout_level"]).map(
        {True: "bull", False: "bear"}
    )
    actionable = actionable.assign(_dir=direction.values)

    for tkr, grp in actionable.groupby("ticker"):
        bull_pick = grp[grp["_dir"] == "bull"].nlargest(1, "confidence")
        bear_pick = grp[grp["_dir"] == "bear"].nlargest(1, "confidence")
        if len(bull_pick):
            bull[tkr] = str(bull_pick["pattern"].iloc[0])
        if len(bear_pick):
            bear[tkr] = str(bear_pick["pattern"].iloc[0])
    return bull, bear


def scan(tickers: list[str], params: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Full scan: fetch once, run patterns once, then analyze() per ticker
    with pattern info passed in. patterns.scan_patterns() already runs all 15
    detectors across every ticker — we reuse its output here instead of
    re-running detectors inside this loop.
    """
    fetch_list = list(tickers)
    spy_added = "SPY" not in {t.upper() for t in fetch_list}
    if spy_added:
        fetch_list.append("SPY")

    data = fetch_universe(fetch_list, params["lookback_days"])

    spy_df = data.get("SPY")
    spy_ret_21d = _return_n_days(spy_df["Close"], 21) if spy_df is not None and len(spy_df) >= 22 else None

    # One pattern scan for the whole universe, reused by every analyze() call.
    bull_lookup: dict[str, str] = {}
    bear_lookup: dict[str, str] = {}
    try:
        from patterns import scan_patterns
        pat_table, _ = scan_patterns(data)
        bull_lookup, bear_lookup = _best_patterns_per_ticker(pat_table)
    except Exception:
        pass

    rows = []
    for t in tickers:
        df = data.get(t)
        if df is None or len(df) < params["atr_long"] + 5:
            rows.append({"ticker": t, "status": "NO DATA"})
            continue
        try:
            result = analyze(
                df, params,
                spy_ret_21d=spy_ret_21d,
                bullish_pattern=bull_lookup.get(t),
                bearish_pattern=bear_lookup.get(t),
            )
        except Exception as e:
            rows.append({"ticker": t, "status": f"ERROR: {e}"})
            continue
        rows.append({"ticker": t, **result})

    table = pd.DataFrame(rows)
    status_order = {
        "BREAKOUT": 0, "PRIMED": 1, "COILED ↑": 2, "TESTING ↑": 3,
        "EXTENDED": 4, "CONSOLIDATING": 5, "PULLBACK": 6, "TRENDING UP": 7,
        "BOUNCE": 8, "NEUTRAL": 9, "TRENDING DOWN": 10, "COILED ↓": 11,
        "AT RISK": 12, "TESTING ↓": 13, "WEAK": 14, "BREAKDOWN": 15,
        "NO DATA": 16,
    }
    table["_order"] = table["status"].map(lambda s: status_order.get(s, 99))
    table = table.sort_values(["_order", "pct_to_breakout"], na_position="last").drop(columns="_order")
    return table.reset_index(drop=True), data
