"""Render an analyzer result into a PNG chart for Telegram alerts.

Self-contained on purpose: it does NOT import `app.py` (a Streamlit module
that pulls in the whole UI), so the Fly worker and the GitHub Actions cron can
render a chart without importing Streamlit. The styling mirrors the
single-name-screener daily chart (phosphor/amber/cyan on near-black) so an
alert chart looks like the app.

Charts are a best-effort enhancement: `render_daily_png` returns None on ANY
failure (missing data, kaleido not installed / broken in the container, etc.)
and the caller falls back to a plain-text alert. It must never raise.

PNG export needs `kaleido` (added to requirements.txt). kaleido is imported
lazily by plotly only at `to_image` time, so importing this module never fails
even if kaleido is absent.
"""
from __future__ import annotations

import math

# Mirror the app's terminal palette.
BG_0 = "#050608"
TEXT_1 = "#9098a3"
TEXT_2 = "#5b6470"
AMBER = "#ffb000"
CYAN = "#00d9ff"
PHOSPHOR = "#00ff8c"
LIME = "#9aff5a"
WARN = "#ff3b3b"
GRID = "rgba(255,255,255,0.05)"

# Default alert chart window: ~8 months of trading days reads well on a phone,
# unlike the app's full 2-year view.
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
                   — the level that fired, drawn as a bold line + annotation.
        title:     optional chart title; defaults to "TICKER — headline".

    Returns PNG bytes, or None on any failure.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd

        df = result.get("df_daily")
        if df is None or getattr(df, "empty", True) or len(df) < 30:
            return None

        chart_cfg = (cfg or {}).get("chart", {}) if cfg else {}
        show_days = int(chart_cfg.get("alert_show_days", show_days) or show_days)
        df_view = df.iloc[-show_days:].copy()
        ticker = result.get("ticker", "")

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.78, 0.22], vertical_spacing=0.03,
        )

        fig.add_trace(
            go.Candlestick(
                x=df_view.index, open=df_view["Open"], high=df_view["High"],
                low=df_view["Low"], close=df_view["Close"], name=ticker,
                increasing_line_color=PHOSPHOR, decreasing_line_color=WARN,
                increasing_fillcolor=PHOSPHOR, decreasing_fillcolor=WARN,
                showlegend=False,
            ),
            row=1, col=1,
        )

        ma_colors = {20: "#3b82f6", 50: "#f97316", 200: "#ef4444"}
        for period in (20, 50, 200):
            if len(df) < period:
                continue
            ma = df["Close"].rolling(period).mean().iloc[-show_days:]
            fig.add_trace(
                go.Scatter(
                    x=ma.index, y=ma.values, mode="lines",
                    name=f"MA{period}",
                    line=dict(color=ma_colors[period], width=1.1),
                ),
                row=1, col=1,
            )

        candle_low = float(df_view["Low"].min())
        candle_high = float(df_view["High"].max())
        y_pad_lo = candle_low * 0.96
        y_pad_hi = candle_high * 1.04

        # Emphasize the level that triggered the alert.
        hl_price = _safe_float((highlight or {}).get("price")) if highlight else 0.0
        if highlight and hl_price > 0:
            direction = (highlight.get("direction") or "").upper()
            color = PHOSPHOR if direction == "BREAKOUT" else (
                WARN if direction == "BREAKDOWN" else AMBER)
            label = (highlight.get("label") or "LEVEL").upper()
            # Make sure the level is in frame even if price ran away from it.
            y_pad_lo = min(y_pad_lo, hl_price * 0.98)
            y_pad_hi = max(y_pad_hi, hl_price * 1.02)
            fig.add_hline(
                y=hl_price, line=dict(color=color, width=1.8, dash="solid"),
                annotation_text=f"${hl_price:.2f} {label}",
                annotation_position="top left",
                annotation_font=dict(color=color, size=12, family="JetBrains Mono"),
                row=1, col=1,
            )

        # Volume row, colored by up/down close.
        vol_colors = [PHOSPHOR if c >= o else WARN
                      for o, c in zip(df_view["Open"], df_view["Close"])]
        fig.add_trace(
            go.Bar(x=df_view.index, y=df_view["Volume"], marker_color=vol_colors,
                   name="Volume", showlegend=False, opacity=0.5),
            row=2, col=1,
        )

        # Title.
        headline = result.get("headline") or {}
        if title is None:
            ht = headline.get("title") or ""
            title = f"{ticker} — {ht}" if ht else ticker

        fig.update_layout(
            title=dict(text=title, font=dict(family="JetBrains Mono", color=TEXT_1, size=14), x=0.02),
            height=height, width=width, template="plotly_dark",
            paper_bgcolor=BG_0, plot_bgcolor=BG_0,
            font=dict(family="JetBrains Mono", color=TEXT_1, size=11),
            margin=dict(l=10, r=10, t=44, b=10),
            xaxis_rangeslider_visible=False,
            showlegend=False,
        )

        # Collapse weekend / holiday gaps so candles are contiguous.
        try:
            visible = pd.DatetimeIndex([ts.normalize() for ts in df_view.index]).unique()
            rangebreaks = [dict(bounds=["sat", "mon"])]
            if len(visible):
                full = pd.bdate_range(visible.min(), visible.max())
                missing = sorted(set(full) - set(visible))
                if missing:
                    rangebreaks.append(dict(values=[d.strftime("%Y-%m-%d") for d in missing]))
            fig.update_xaxes(rangebreaks=rangebreaks)
        except Exception:
            pass

        # Log scale when the visible range is wide.
        use_log = False
        thr = chart_cfg.get("log_scale_threshold_pct", 50)
        if chart_cfg.get("auto_log_scale", True) and candle_low > 0:
            if (candle_high - candle_low) / candle_low * 100 >= thr:
                use_log = True
        if use_log and y_pad_lo > 0:
            fig.update_yaxes(type="log", range=[math.log10(y_pad_lo), math.log10(y_pad_hi)],
                             row=1, col=1, gridcolor=GRID)
        else:
            fig.update_yaxes(range=[y_pad_lo, y_pad_hi], row=1, col=1, gridcolor=GRID)
        fig.update_yaxes(gridcolor=GRID, row=2, col=1)
        fig.update_xaxes(gridcolor=GRID)

        return fig.to_image(format="png", width=width, height=height, scale=2)
    except Exception as e:  # noqa: BLE001 — charts must never break an alert
        import sys
        print(f"[warn] chart render failed: {e}", file=sys.stderr)
        return None
