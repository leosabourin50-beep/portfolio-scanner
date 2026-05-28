"""Render an analyzer result into a PNG chart for Telegram alerts.

Uses **matplotlib (Agg backend)** — no browser, no Chromium, no kaleido. A
single render is ~50ms and a few tens of MB, so it runs comfortably on the
512MB Fly watcher VM (kaleido's headless Chromium OOM-killed it). The styling
mirrors the single-name-screener daily chart (phosphor/amber/cyan on
near-black) so an alert chart looks like the app.

Self-contained: does NOT import `app.py` (Streamlit). Charts are best-effort —
`render_daily_png` returns None on ANY failure and never raises, so the caller
falls back to a plain-text alert.
"""
from __future__ import annotations

import io
import sys

# Mirror the app's terminal palette.
BG_0 = "#050608"
TEXT_1 = "#9098a3"
TEXT_2 = "#5b6470"
AMBER = "#ffb000"
CYAN = "#00d9ff"
PHOSPHOR = "#00ff8c"
WARN = "#ff3b3b"
GRID = "#161a20"
MA_COLORS = {20: "#3b82f6", 50: "#f97316", 200: "#ef4444"}

# Default alert chart window: ~8 months of trading days reads well on a phone.
DEFAULT_SHOW_DAYS = 170


def _safe_float(v, default=0.0) -> float:
    try:
        f = float(v)
        return f if f == f else default  # reject NaN
    except (TypeError, ValueError):
        return default


def render_daily_png(
    result: dict,
    cfg: dict | None = None,
    highlight: dict | None = None,
    show_days: int = DEFAULT_SHOW_DAYS,
    width: int = 1100,
    height: int = 620,
    title: str | None = None,
) -> bytes | None:
    """Daily candles + MA20/50/200 + volume, with the firing level emphasized.

    Args:
        result:    an analyze_ticker() result dict (needs df_daily).
        cfg:       optional config (for log-scale thresholds); defaults applied.
        highlight: optional {"price": float, "label": str, "direction": str}
                   — the level that fired, drawn as a bold line + label.
        title:     optional chart title; defaults to "TICKER — headline".

    Returns PNG bytes, or None on any failure.
    """
    fig = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        df = result.get("df_daily")
        if df is None or getattr(df, "empty", True) or len(df) < 30:
            return None

        chart_cfg = (cfg or {}).get("chart", {}) if cfg else {}
        show_days = int(chart_cfg.get("alert_show_days", show_days) or show_days)
        df_view = df.iloc[-show_days:].copy()
        n = len(df_view)
        ticker = result.get("ticker", "")
        x = np.arange(n)

        o = df_view["Open"].to_numpy(dtype=float)
        h = df_view["High"].to_numpy(dtype=float)
        low = df_view["Low"].to_numpy(dtype=float)
        c = df_view["Close"].to_numpy(dtype=float)
        vol = df_view["Volume"].to_numpy(dtype=float)
        up = c >= o
        colors = np.where(up, PHOSPHOR, WARN)

        dpi = 110
        fig, (ax, axv) = plt.subplots(
            2, 1, sharex=True, figsize=(width / dpi, height / dpi), dpi=dpi,
            gridspec_kw={"height_ratios": [3.6, 1], "hspace": 0.04},
        )
        fig.patch.set_facecolor(BG_0)

        # Candlesticks: wicks via vlines, bodies via bars (vectorized, no loop).
        ax.vlines(x, low, h, color=colors, linewidth=0.7)
        body_bottom = np.minimum(o, c)
        body_h = np.abs(c - o)
        # Give doji bars a hair of height so they're visible.
        body_h = np.where(body_h < (h - low) * 0.02 + 1e-9, (h - low) * 0.05 + 1e-9, body_h)
        ax.bar(x, body_h, bottom=body_bottom, width=0.7, color=colors, linewidth=0)

        # Moving averages (computed on full history, sliced to the view).
        for period in (20, 50, 200):
            if len(df) < period:
                continue
            ma = df["Close"].rolling(period).mean().to_numpy(dtype=float)[-n:]
            ax.plot(x, ma, color=MA_COLORS[period], linewidth=1.0, label=f"MA{period}")

        candle_low = float(np.nanmin(low))
        candle_high = float(np.nanmax(h))
        y_lo = candle_low * 0.96
        y_hi = candle_high * 1.04

        # Emphasize the level that triggered the alert.
        hl_price = _safe_float((highlight or {}).get("price")) if highlight else 0.0
        if highlight and hl_price > 0:
            direction = (highlight.get("direction") or "").upper()
            hl_color = PHOSPHOR if direction == "BREAKOUT" else (
                WARN if direction == "BREAKDOWN" else AMBER)
            label = (highlight.get("label") or "LEVEL").upper()
            y_lo = min(y_lo, hl_price * 0.98)
            y_hi = max(y_hi, hl_price * 1.02)
            ax.axhline(hl_price, color=hl_color, linewidth=1.7)
            ax.text(0.4, hl_price, f"  ${hl_price:.2f} {label}", color=hl_color,
                    fontsize=9, fontfamily="monospace", va="bottom", ha="left")

        # Log scale when the visible range is wide.
        thr = chart_cfg.get("log_scale_threshold_pct", 50)
        if chart_cfg.get("auto_log_scale", True) and candle_low > 0 and \
                (candle_high - candle_low) / candle_low * 100 >= thr:
            ax.set_yscale("log")
        ax.set_ylim(y_lo, y_hi)

        # Volume row, colored by up/down close.
        axv.bar(x, vol, width=0.8, color=colors, linewidth=0, alpha=0.55)

        # Title.
        headline = result.get("headline") or {}
        if title is None:
            ht = headline.get("title") or ""
            title = f"{ticker} — {ht}" if ht else ticker
        ax.set_title(title, color=TEXT_1, loc="left", fontsize=12,
                     fontfamily="monospace", pad=8)

        # X tick labels: a handful of evenly-spaced dates (positional x means
        # no weekend/holiday gaps).
        if n:
            ticks = np.linspace(0, n - 1, min(7, n), dtype=int)
            axv.set_xticks(ticks)
            axv.set_xticklabels([df_view.index[i].strftime("%b %y") for i in ticks])
            ax.set_xlim(-1, n)

        # Dark terminal styling for both axes.
        for a in (ax, axv):
            a.set_facecolor(BG_0)
            a.grid(True, color=GRID, linewidth=0.5)
            a.tick_params(colors=TEXT_1, labelsize=8)
            for spine in a.spines.values():
                spine.set_color(TEXT_2)
                spine.set_linewidth(0.5)
        # Upper-right keeps the legend clear of the (left-anchored) trigger label.
        ax.legend(loc="upper right", fontsize=7, facecolor=BG_0,
                  edgecolor=TEXT_2, labelcolor=TEXT_1, framealpha=0.5)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=BG_0, bbox_inches="tight")
        return buf.getvalue()
    except Exception as e:  # noqa: BLE001 — charts must never break an alert
        print(f"[warn] chart render failed: {e}", file=sys.stderr)
        return None
    finally:
        if fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass
