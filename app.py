"""Portfolio Scanner — Streamlit dashboard.

Drops a portfolio of tickers in and surfaces the top-3 most actionable moves
across the names, with a full table of every ticker's headline read. Reuses
the single_name_analyzer engine — every ticker can be drilled into for the
full single-name view (charts + setup map + commentary).

Layout:
  • Top of page: portfolio input textarea (persists in session_state) + Scan
  • After scan:
      ▸ TOP MOVES section — 3 big actionable cards
      ▸ PORTFOLIO TABLE — every ticker, sortable
      ▸ DRILL-DOWN — click any ticker → full single-name analyzer view
"""
from __future__ import annotations

import html as html_mod
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

import commentary as commentary_mod
import single_name_analyzer as sna
import portfolio_scanner as ps


# ─────────────────────────────────────────────────
# Terminal palette (mirrors single-name-screener)
# ─────────────────────────────────────────────────

BG_0 = "#050608"
BG_1 = "#0c0e12"
BG_2 = "#14171c"
TEXT_0 = "#e6e8eb"
TEXT_1 = "#9098a3"
TEXT_2 = "#5b6470"
AMBER = "#ffb000"
CYAN = "#00d9ff"
PHOSPHOR = "#00ff8c"
LIME = "#9aff5a"
WARN = "#ff3b3b"
GRID = "rgba(255,255,255,0.04)"

GRADE_COLORS = {
    "EXCELLENT": PHOSPHOR,
    "GOOD": LIME,
    "WAIT": AMBER,
    "AVOID": WARN,
}


# ─────────────────────────────────────────────────
# CSS — Bloomberg-terminal aesthetic (matches single-name-screener)
# ─────────────────────────────────────────────────

TERMINAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-0: #050608;
    --bg-1: #0c0e12;
    --bg-2: #14171c;
    --border: rgba(255,255,255,0.06);
    --border-strong: rgba(255,255,255,0.12);
    --text-0: #e6e8eb;
    --text-1: #9098a3;
    --text-2: #5b6470;
    --amber: #ffb000;
    --cyan: #00d9ff;
    --phosphor: #00ff8c;
    --warning: #ff3b3b;
    --mono: 'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace;
    --sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp { background: var(--bg-0); }
.block-container { padding-top: 3.25rem !important; padding-bottom: 2rem; max-width: 1500px; }
[data-testid="stHeader"] { background: var(--bg-0); height: 2.5rem; }
[data-testid="stToolbar"] { right: 1rem; }
body, .stApp, p, span, div, label { color: var(--text-0); font-family: var(--sans); }

h1, h2, h3, h4, h5, h6 {
    font-family: var(--mono) !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase;
    color: var(--text-0);
}
h1 { font-size: 1.6rem !important; font-weight: 600 !important; letter-spacing: 0.18em !important; }
h3 { font-size: 0.85rem !important; color: var(--text-1) !important; margin-top: 0.5rem !important; }
h4 { font-size: 0.75rem !important; color: var(--text-2) !important; }

.stButton button {
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    background: var(--bg-2) !important;
    border: 1px solid var(--border-strong) !important;
    color: var(--text-0) !important;
    border-radius: 2px !important;
    transition: all 0.15s ease;
}
.stButton button:hover {
    border-color: var(--amber) !important;
    color: var(--amber) !important;
}
.stButton button[kind="primary"] {
    background: var(--amber) !important;
    color: var(--bg-0) !important;
    border-color: var(--amber) !important;
    font-weight: 600 !important;
}
.stButton button[kind="primary"]:hover {
    background: #ffc933 !important;
    color: var(--bg-0) !important;
}

.stTextInput input, .stTextArea textarea {
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
    background: var(--bg-1) !important;
    border: 1px solid var(--border-strong) !important;
    color: var(--text-0) !important;
    border-radius: 2px !important;
    letter-spacing: 0.05em;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--amber) !important;
    box-shadow: none !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 2px;
}
.stDataFrame, [data-testid="stDataFrame"] * {
    font-family: var(--mono) !important;
}

hr { border: none; height: 1px; background: var(--border); margin: 1.5rem 0 !important; }

.terminal-header {
    font-family: var(--mono);
    font-size: 1.7rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    color: var(--text-0);
    text-transform: uppercase;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--amber);
    margin-bottom: 0.35rem;
}
.terminal-sub {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text-2);
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.section-label {
    font-family: var(--mono);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-1);
    padding: 14px 0 8px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
}
.section-label::before {
    content: "";
    display: inline-block;
    width: 4px;
    height: 11px;
    background: var(--accent, var(--amber));
    margin-right: 10px;
}

/* ── Top moves cards ─────────────────────────────────────────── */
.move-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 1.25rem;
}
@media (max-width: 1000px) {
    .move-grid { grid-template-columns: 1fr; }
}
.move-card {
    background: linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
    border: 1px solid var(--border-strong);
    border-left: 5px solid var(--amber);
    padding: 16px 18px;
    border-radius: 2px;
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.move-rank {
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: var(--text-2);
}
.move-head {
    display: flex;
    align-items: baseline;
    gap: 10px;
    flex-wrap: wrap;
}
.move-ticker {
    font-family: var(--mono);
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-0);
    letter-spacing: 0.08em;
}
.move-price {
    font-family: var(--mono);
    font-size: 0.85rem;
    color: var(--text-1);
}
.move-chg-pos { color: var(--phosphor); }
.move-chg-neg { color: var(--warning); }
.move-headline {
    font-family: var(--mono);
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.move-detail {
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--text-0);
}
.move-level {
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--text-0);
    margin-top: 4px;
}
.move-level .k {
    color: var(--text-2);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-size: 0.62rem;
    margin-right: 6px;
}
.move-tier {
    display: inline-block;
    margin-top: 6px;
    padding: 2px 8px;
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-2);
    background: rgba(255,255,255,0.04);
    border-radius: 2px;
}

/* ── Portfolio table ─────────────────────────────────────────── */
.port-row {
    display: grid;
    grid-template-columns: 70px 90px 70px 1fr 100px 80px;
    align-items: center;
    gap: 12px;
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 0.78rem;
}
.port-header {
    background: var(--bg-2);
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border-strong);
    color: var(--text-2);
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}
.port-tick { font-weight: 600; color: var(--text-0); }
.port-price { color: var(--text-0); }
.port-chg-pos { color: var(--phosphor); }
.port-chg-neg { color: var(--warning); }
.port-head { color: var(--text-0); }
.port-tier { color: var(--text-2); font-size: 0.66rem; letter-spacing: 0.12em; }
.port-level { color: var(--text-1); text-align: right; }

/* ── Clickable portfolio rows (Streamlit columns layout) ────────── */
.port-grid-row .stButton { margin: 0 !important; }
.port-grid-row .stButton button {
    width: 100%;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-bottom: 1px solid var(--border) !important;
    border-radius: 0 !important;
    color: var(--text-0) !important;
    font-family: var(--mono) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 8px 10px !important;
    height: 36px !important;
    text-transform: none !important;
}
.port-grid-row .stButton button:hover {
    background: rgba(255, 176, 0, 0.06) !important;
    border-color: var(--amber) !important;
    color: var(--amber) !important;
}
.port-grid-row .stButton button:focus {
    box-shadow: none !important;
    outline: none !important;
}
.port-grid-row [data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
    padding: 9px 8px !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    border-bottom: 1px solid var(--border) !important;
    line-height: 18px !important;
}
.port-grid-header [data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
    padding: 10px 8px !important;
    font-family: var(--mono) !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    color: var(--text-2) !important;
    background: var(--bg-2);
    border-bottom: 1px solid var(--border-strong) !important;
}
.cell-chg-pos { color: var(--phosphor); }
.cell-chg-neg { color: var(--warning); }
.cell-right   { text-align: right; }

/* ── Live pill ───────────────────────────────────────────────── */
.live-pill {
    display: inline-block;
    margin-left: 10px;
    padding: 2px 8px;
    font-family: var(--mono);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: var(--phosphor);
    background: rgba(0, 255, 140, 0.08);
    border: 1px solid rgba(0, 255, 140, 0.35);
    border-radius: 2px;
}
.live-dot {
    color: var(--phosphor);
    margin-right: 4px;
    animation: live-pulse 1.6s ease-in-out infinite;
}
@keyframes live-pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.35; }
}

[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    background: var(--bg-1);
}
[data-testid="stExpander"] summary {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-1) !important;
}

[data-testid="stAlert"] {
    border-radius: 2px !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
}

.js-plotly-plot { background: var(--bg-0) !important; }
</style>
"""


# ─────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────

def _parse_portfolio(text: str) -> list[str]:
    """Accept comma, space, semicolon, or newline-separated tickers. Dedup."""
    if not text:
        return []
    raw = re.split(r"[,;\s]+", text)
    out: list[str] = []
    seen: set[str] = set()
    for t in raw:
        t = t.strip().upper()
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _section_label(text: str, accent: str = AMBER) -> None:
    st.markdown(
        f"<div class='section-label' style='--accent:{accent}'>{text}</div>",
        unsafe_allow_html=True,
    )


def _render_move_card(read: ps.ActionableRead, rank: int) -> str:
    accent = read.color
    chg_class = "move-chg-pos" if read.chg_pct >= 0 else "move-chg-neg"
    headline = html_mod.escape(read.headline_title)
    detail = html_mod.escape(read.headline_detail or "")
    level = html_mod.escape(read.headline_level or "")
    tier = html_mod.escape(read.tier_label)
    setup_label = html_mod.escape(read.setup_label or "")
    return (
        f"<div class='move-card' style='border-left-color:{accent};'>"
        f"  <div class='move-rank'>RANK #{rank}</div>"
        f"  <div class='move-head'>"
        f"    <span class='move-ticker'>{read.ticker}</span>"
        f"    <span class='move-price'>${read.last_price:.2f}</span>"
        f"    <span class='{chg_class}'>{read.chg_pct:+.2f}%</span>"
        f"  </div>"
        f"  <div class='move-headline' style='color:{accent};'>{read.icon} {headline}</div>"
        f"  <div class='move-detail'>{detail}</div>"
        f"  <div class='move-level'><span class='k'>Level</span>{level}</div>"
        f"  <div class='move-level'><span class='k'>Setup</span>{setup_label}</div>"
        f"  <div class='move-tier'>{tier}</div>"
        f"</div>"
    )


def _render_top_moves(reads: list[ps.ActionableRead]) -> None:
    top = ps.top_n(reads, n=3)
    if not top:
        st.info("No actionable moves in this portfolio right now — every name shows a neutral / no-clear-bias read.")
        return
    cards = "".join(_render_move_card(r, i + 1) for i, r in enumerate(top))
    st.markdown(f"<div class='move-grid'>{cards}</div>", unsafe_allow_html=True)


_PORT_COL_WIDTHS = [1.0, 1.0, 0.9, 3.0, 1.4, 1.1]
_PORT_HEADERS = ["Ticker", "Price", "Chg", "Headline", "Tier", "Level"]


def _render_portfolio_table(reads: list[ps.ActionableRead]) -> None:
    """Render the full portfolio as a clickable grid. The Ticker cell is a
    Streamlit button per row — clicking it sets `drill_ticker` in session
    state and reruns, which populates the drill-down section below."""
    # Header row
    st.markdown("<div class='port-grid-header'>", unsafe_allow_html=True)
    hcols = st.columns(_PORT_COL_WIDTHS)
    for c, label in zip(hcols, _PORT_HEADERS):
        c.markdown(label)
    st.markdown("</div>", unsafe_allow_html=True)

    # Data rows — ticker is a real button so the click sets session state
    for r in reads:
        st.markdown("<div class='port-grid-row'>", unsafe_allow_html=True)
        cols = st.columns(_PORT_COL_WIDTHS)
        with cols[0]:
            if st.button(r.ticker, key=f"row_btn_{r.ticker}", use_container_width=True):
                st.session_state.drill_ticker = r.ticker
                st.rerun()
        chg_cls = "cell-chg-pos" if r.chg_pct >= 0 else "cell-chg-neg"
        cols[1].markdown(f"${r.last_price:.2f}")
        cols[2].markdown(f"<span class='{chg_cls}'>{r.chg_pct:+.2f}%</span>", unsafe_allow_html=True)
        cols[3].markdown(
            f"<span style='color:{r.color};'>{r.icon} {html_mod.escape(r.headline_title)}</span>",
            unsafe_allow_html=True,
        )
        cols[4].markdown(
            f"<span style='color:var(--text-2); font-size:0.7rem; letter-spacing:0.12em;'>"
            f"{html_mod.escape(r.tier_label)}</span>",
            unsafe_allow_html=True,
        )
        cols[5].markdown(
            f"<span class='cell-right' style='color:var(--text-1); display:block;'>"
            f"{html_mod.escape(r.headline_level or '')}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────
# Drill-down view — borrowed from single-name app
# ─────────────────────────────────────────────────

def _terminal_layout(height: int = 560) -> dict:
    return dict(
        height=height, template="plotly_dark",
        paper_bgcolor=BG_0, plot_bgcolor=BG_0,
        font=dict(family="JetBrains Mono", color=TEXT_1, size=11),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", y=1.05, x=0.0,
            font=dict(family="JetBrains Mono", color=TEXT_1, size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
    )


def _build_daily_chart(result: dict, cfg: dict) -> go.Figure:
    """Annotated daily chart — candles, MAs, RS row, full setup-map overlay
    (horizontal + diagonal trendlines + channel rails). Mirrors the single-
    name-screener daily chart exactly so the drill-down visual matches."""
    df = result["df_daily"]
    spy_df = result.get("df_spy")
    show_days = cfg["chart"]["daily_show_days"]
    df_view = df.iloc[-show_days:].copy()

    signals = result.get("signals") or {}
    consol = (signals.get("consolidation") or {})
    ma_colors = cfg["chart"]["ma_colors"]

    has_rs = spy_df is not None and not spy_df.empty
    if has_rs:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.66, 0.17, 0.17], vertical_spacing=0.03,
        )
    else:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.78, 0.22], vertical_spacing=0.03,
        )

    fig.add_trace(
        go.Candlestick(
            x=df_view.index, open=df_view["Open"], high=df_view["High"],
            low=df_view["Low"], close=df_view["Close"], name=result["ticker"],
            increasing_line_color=PHOSPHOR, decreasing_line_color=WARN,
            increasing_fillcolor=PHOSPHOR, decreasing_fillcolor=WARN,
            showlegend=False,
        ),
        row=1, col=1,
    )

    for period, key in [(20, "ma_20"), (50, "ma_50"), (200, "ma_200")]:
        ma = df["Close"].rolling(period).mean()
        ma_view = ma.iloc[-show_days:]
        fig.add_trace(
            go.Scatter(
                x=ma_view.index, y=ma_view.values, mode="lines",
                name=f"MA{period}", line=dict(color=ma_colors[key], width=1.2),
            ),
            row=1, col=1,
        )

    if consol.get("signal") and consol["signal"] != "NONE":
        fig.add_hrect(
            y0=consol["range_low"], y1=consol["range_high"],
            fillcolor="rgba(0, 217, 255, 0.04)", line_width=0, row=1, col=1,
        )

    breakouts = list(result.get("breakouts", []))[:6]
    breakdowns = list(result.get("breakdowns", []))[:6]
    triggered = list(result.get("triggered", []))[:4]
    setup_level_lines: list[tuple[float, str, str, str, float, str]] = []
    diagonal_lines: list[tuple[dict, str, str, str, float]] = []

    def _style_for(s):
        if s.direction == "BREAKOUT":
            if s.quality_label == "STRONG":
                return PHOSPHOR, "solid", 1.4
            if s.quality_label == "MODERATE":
                return LIME, "dash", 1.1
            return AMBER, "dot", 0.9
        if s.quality_label == "STRONG":
            return WARN, "solid", 1.4
        if s.quality_label == "MODERATE":
            return "#ff7a7a", "dash", 1.1
        return "#ffd166", "dot", 0.9

    for s in breakouts + breakdowns:
        if s.kind.startswith("CHANNEL_WEEKLY_"):
            continue
        color, dash, width = _style_for(s)
        if s.line_meta:
            diagonal_lines.append((s.line_meta, color, dash, s.label, width))
        else:
            setup_level_lines.append((s.trigger_price, color, dash, s.label, width, "right"))
    for s in triggered:
        if s.kind.startswith("CHANNEL_WEEKLY_"):
            continue
        muted = "rgba(0,255,140,0.45)" if s.direction == "BREAKOUT" else "rgba(255,59,59,0.45)"
        if s.line_meta:
            diagonal_lines.append((s.line_meta, muted, "dot", f"{s.label} (crossed)", 0.8))
        else:
            setup_level_lines.append((
                s.trigger_price, muted, "dot", f"{s.label} (crossed)", 0.8, "left",
            ))

    candle_low = float(df_view["Low"].min())
    candle_high = float(df_view["High"].max())
    y_pad_lo = candle_low * 0.95
    y_pad_hi = candle_high * 1.05
    for s in breakouts:
        if candle_high < s.trigger_price <= candle_high * 1.15:
            y_pad_hi = max(y_pad_hi, s.trigger_price * 1.02)
    for s in breakdowns:
        if candle_low * 0.85 <= s.trigger_price < candle_low:
            y_pad_lo = min(y_pad_lo, s.trigger_price * 0.98)
    off_chart_labels: list[str] = []

    for y, color, dash, label, width, position in setup_level_lines:
        if y_pad_lo <= y <= y_pad_hi:
            fig.add_hline(
                y=y, line=dict(color=color, width=width, dash=dash),
                annotation_text=f"${y:.2f} {label.upper()}",
                annotation_position=position,
                annotation_font=dict(color=color, size=10, family="JetBrains Mono"),
                row=1, col=1,
            )
        else:
            if y > candle_high:
                pct = (y / candle_high - 1) * 100
            else:
                pct = (y / candle_low - 1) * 100
            off_chart_labels.append(f"{label.upper()} ${y:.2f} ({pct:+.0f}%)")

    view_start = df_view.index[0]
    for lm, color, dash, label, width in diagonal_lines:
        try:
            x0, x1 = lm["x0"], lm["x1"]
            y0, y1 = float(lm["y0"]), float(lm["y1"])
        except (KeyError, TypeError, ValueError):
            continue
        x0_clamped = x0 if x0 >= view_start else view_start
        if x0 < view_start and x1 != x0:
            try:
                total = (x1 - x0).total_seconds()
                frac = (view_start - x0).total_seconds() / total if total else 0
                y0_clamped = y0 + (y1 - y0) * frac
            except Exception:
                y0_clamped = y0
        else:
            y0_clamped = y0
        fig.add_shape(
            type="line",
            x0=x0_clamped, y0=y0_clamped, x1=x1, y1=y1,
            line=dict(color=color, width=width, dash=dash),
            xref="x", yref="y", row=1, col=1,
        )
        fig.add_annotation(
            x=x1, y=y1,
            text=f"{label.upper()} ${y1:.2f}",
            showarrow=False,
            font=dict(color=color, size=10, family="JetBrains Mono"),
            xanchor="right", yanchor="bottom",
            row=1, col=1,
        )

    colors = [PHOSPHOR if c >= o else WARN for o, c in zip(df_view["Open"], df_view["Close"])]
    fig.add_trace(
        go.Bar(
            x=df_view.index, y=df_view["Volume"], marker_color=colors,
            name="Volume", showlegend=False, opacity=0.55,
        ),
        row=2, col=1,
    )

    if has_rs:
        joined = pd.DataFrame({"t": df["Close"], "s": spy_df["Close"]}).dropna()
        if not joined.empty:
            rs = joined["t"] / joined["s"]
            rs_norm = rs / float(rs.iloc[max(0, len(rs) - show_days)])
            rs_view = rs_norm.iloc[-show_days:]
            rs_ma20 = rs_norm.rolling(20).mean().iloc[-show_days:]
            fig.add_trace(
                go.Scatter(
                    x=rs_view.index, y=rs_view.values, mode="lines",
                    name="RS vs SPY", line=dict(color=CYAN, width=1.2),
                ),
                row=3, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=rs_ma20.index, y=rs_ma20.values, mode="lines",
                    name="RS MA(20)", line=dict(color=CYAN, width=0.8, dash="dot"),
                ),
                row=3, col=1,
            )
            fig.add_hline(
                y=1.0, line=dict(color=TEXT_2, width=0.8, dash="dash"),
                row=3, col=1,
            )

    eq = result.get("entry_quality") or {}
    grade = eq.get("grade")
    score = eq.get("grade_score")
    if grade is not None:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.99, y=0.99,
            text=f"<b>{grade}</b>  ·  {score}",
            showarrow=False,
            bgcolor=GRADE_COLORS.get(grade, AMBER),
            bordercolor="rgba(0,0,0,0)",
            font=dict(color=BG_0, size=12, family="JetBrains Mono"),
            align="right",
        )

    fresh_today = [
        s for s in triggered
        if s.bars_since_trigger == 0 and s.timeframe == "daily"
    ]
    if fresh_today:
        s = max(fresh_today, key=lambda x: x.quality)
        if s.direction == "BREAKOUT":
            stamp_text = "▲ BREAKOUT CLEARED"
            stamp_color = PHOSPHOR
            stamp_bg = "rgba(0, 255, 140, 0.18)"
        else:
            stamp_text = "▼ BREAKDOWN TRIGGERED"
            stamp_color = WARN
            stamp_bg = "rgba(255, 59, 59, 0.18)"
        fig.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.99,
            text=(
                f"<b>{stamp_text}</b><br>"
                f"<span style='font-size:11px;'>"
                f"{html_mod.escape(s.label)} · ${s.trigger_price:.2f}"
                f"</span>"
            ),
            showarrow=False,
            bgcolor=stamp_bg,
            bordercolor=stamp_color,
            borderwidth=1, borderpad=8,
            font=dict(color=stamp_color, size=16, family="JetBrains Mono"),
            align="left", xanchor="left", yanchor="top",
        )

    if off_chart_labels:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.99, y=0.92,
            text="<br>".join(off_chart_labels),
            showarrow=False,
            bgcolor="rgba(168, 85, 247, 0.18)",
            bordercolor="#a855f7",
            borderwidth=1, borderpad=4,
            font=dict(color="#c084fc", size=10, family="JetBrains Mono"),
            align="right", xanchor="right", yanchor="top",
        )

    fig.update_layout(**_terminal_layout(height=720 if has_rs else 620))

    visible_dates = pd.DatetimeIndex([ts.normalize() for ts in df_view.index]).unique()
    if len(visible_dates):
        full_weekdays = pd.bdate_range(visible_dates.min(), visible_dates.max())
        missing = sorted(set(full_weekdays) - set(visible_dates))
        rangebreaks = [dict(bounds=["sat", "mon"])]
        if missing:
            rangebreaks.append(dict(values=[d.strftime("%Y-%m-%d") for d in missing]))
        fig.update_xaxes(rangebreaks=rangebreaks)
    else:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    use_log = False
    if cfg["chart"].get("auto_log_scale", True) and candle_low > 0:
        range_pct = (candle_high - candle_low) / candle_low * 100
        use_log = range_pct >= cfg["chart"].get("log_scale_threshold_pct", 50)

    y_range = [math.log10(y_pad_lo), math.log10(y_pad_hi)] if use_log else [y_pad_lo, y_pad_hi]

    fig.update_yaxes(
        dict(
            title=dict(text="PRICE", font=dict(family="JetBrains Mono", size=10, color=TEXT_2)),
            type="log" if use_log else "linear",
            range=y_range,
        ),
        row=1, col=1,
    )
    fig.update_yaxes(title=dict(text="VOL", font=dict(family="JetBrains Mono", size=10, color=TEXT_2)), row=2, col=1)
    if has_rs:
        fig.update_yaxes(title=dict(text="RS", font=dict(family="JetBrains Mono", size=10, color=TEXT_2)), row=3, col=1)
    return fig


def _build_weekly_chart(result: dict, cfg: dict) -> go.Figure | None:
    """Weekly candlestick chart with 10W/40W MAs, weekly-pattern overlays, and
    weekly-channel rails. Mirrors the single-name-screener weekly chart."""
    df_weekly = result.get("df_weekly")
    if df_weekly is None or df_weekly.empty:
        return None

    show_weeks = cfg["chart"].get("weekly_show_weeks", 104)
    df_view = df_weekly.iloc[-show_weeks:].copy()
    if df_view.empty:
        return None

    weekly_pats = result.get("weekly_patterns") or []

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.03,
    )

    fig.add_trace(
        go.Candlestick(
            x=df_view.index, open=df_view["Open"], high=df_view["High"],
            low=df_view["Low"], close=df_view["Close"], name=result["ticker"],
            increasing_line_color=PHOSPHOR, decreasing_line_color=WARN,
            increasing_fillcolor=PHOSPHOR, decreasing_fillcolor=WARN,
            showlegend=False,
        ),
        row=1, col=1,
    )

    for period, label, color in [
        (10, "10W MA", "#3b82f6"),
        (40, "40W MA", "#ef4444"),
    ]:
        ma = df_weekly["Close"].rolling(period).mean()
        ma_view = ma.iloc[-show_weeks:]
        fig.add_trace(
            go.Scatter(
                x=ma_view.index, y=ma_view.values, mode="lines",
                name=label, line=dict(color=color, width=1.2),
            ),
            row=1, col=1,
        )

    candle_low = float(df_view["Low"].min())
    candle_high = float(df_view["High"].max())
    y_pad_lo = candle_low * 0.97
    y_pad_hi = candle_high * 1.05
    pats_sorted = sorted(weekly_pats, key=lambda p: p.confidence, reverse=True)
    if pats_sorted and pats_sorted[0].confidence >= 0.15 and pats_sorted[0].target:
        tg = float(pats_sorted[0].target)
        if candle_high < tg <= candle_high * 1.20:
            y_pad_hi = max(y_pad_hi, tg * 1.02)

    off_chart_labels: list[str] = []
    for p in pats_sorted[:2]:
        if p.confidence < 0.15:
            continue
        if y_pad_lo <= p.breakout_level <= y_pad_hi:
            fig.add_hline(
                y=p.breakout_level,
                line=dict(color="#a855f7", width=1.2, dash="dash"),
                annotation_text=f"{p.pattern.upper()} ${p.breakout_level:.2f}",
                annotation_position="left",
                annotation_font=dict(color="#c084fc", size=10, family="JetBrains Mono"),
                row=1, col=1,
            )
        else:
            pct = (p.breakout_level / candle_high - 1) * 100
            off_chart_labels.append(
                f"{p.pattern.upper()} ${p.breakout_level:.2f} ({pct:+.0f}% off-chart)"
            )
        if p.target:
            if y_pad_lo <= p.target <= y_pad_hi:
                fig.add_hline(
                    y=p.target,
                    line=dict(color="#a855f7", width=1, dash="dot"),
                    annotation_text=f"TGT ${p.target:.2f}",
                    annotation_position="left",
                    annotation_font=dict(color="#c084fc", size=10, family="JetBrains Mono"),
                    row=1, col=1,
                )
            else:
                pct = (p.target / candle_high - 1) * 100
                off_chart_labels.append(f"TGT ${p.target:.2f} ({pct:+.0f}% off-chart)")

    colors = [PHOSPHOR if c >= o else WARN for o, c in zip(df_view["Open"], df_view["Close"])]
    fig.add_trace(
        go.Bar(
            x=df_view.index, y=df_view["Volume"], marker_color=colors,
            name="Volume", showlegend=False, opacity=0.55,
        ),
        row=2, col=1,
    )

    view_start = df_view.index[0]
    for s in (
        list(result.get("breakouts", []))
        + list(result.get("breakdowns", []))
        + list(result.get("triggered", []))
    ):
        if not s.kind.startswith("CHANNEL_WEEKLY_"):
            continue
        if not s.line_meta:
            continue
        is_triggered = getattr(s, "is_triggered", False)
        if s.direction == "BREAKOUT":
            color = PHOSPHOR if s.quality_label == "STRONG" else LIME
        else:
            color = WARN if s.quality_label == "STRONG" else "#ff7a7a"
        if is_triggered:
            color = "rgba(0,255,140,0.45)" if s.direction == "BREAKOUT" else "rgba(255,59,59,0.45)"
            dash = "dot"
            width = 0.9
            label_suffix = " (crossed)"
        elif s.quality_label == "STRONG":
            dash, width = "solid", 1.5
            label_suffix = ""
        elif s.quality_label == "MODERATE":
            dash, width = "dash", 1.2
            label_suffix = ""
        else:
            dash, width = "dot", 1.0
            label_suffix = ""
        lm = s.line_meta
        try:
            x0 = lm["x0"]; x1 = lm["x1"]
            y0 = float(lm["y0"]); y1 = float(lm["y1"])
        except (KeyError, TypeError, ValueError):
            continue
        x0_clamped = x0 if x0 >= view_start else view_start
        if x0 < view_start and x1 != x0:
            try:
                total = (x1 - x0).total_seconds()
                frac = (view_start - x0).total_seconds() / total if total else 0
                y0_clamped = y0 + (y1 - y0) * frac
            except Exception:
                y0_clamped = y0
        else:
            y0_clamped = y0
        fig.add_shape(
            type="line",
            x0=x0_clamped, y0=y0_clamped, x1=x1, y1=y1,
            line=dict(color=color, width=width, dash=dash),
            xref="x", yref="y", row=1, col=1,
        )
        fig.add_annotation(
            x=x1, y=y1,
            text=f"{s.label.upper()}{label_suffix} ${y1:.2f}",
            showarrow=False,
            font=dict(color=color, size=10, family="JetBrains Mono"),
            xanchor="right", yanchor="bottom",
            row=1, col=1,
        )

    if off_chart_labels:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.99, y=0.99,
            text="<br>".join(off_chart_labels),
            showarrow=False,
            bgcolor="rgba(168, 85, 247, 0.18)",
            bordercolor="#a855f7",
            borderwidth=1, borderpad=4,
            font=dict(color="#c084fc", size=10, family="JetBrains Mono"),
            align="right", xanchor="right", yanchor="top",
        )

    fig.update_layout(**_terminal_layout(height=520))

    use_log = False
    if cfg["chart"].get("auto_log_scale", True) and candle_low > 0:
        range_pct = (candle_high - candle_low) / candle_low * 100
        use_log = range_pct >= cfg["chart"].get("log_scale_threshold_pct", 50)
    y_range = [math.log10(y_pad_lo), math.log10(y_pad_hi)] if use_log else [y_pad_lo, y_pad_hi]

    fig.update_yaxes(
        dict(
            title=dict(text="PRICE", font=dict(family="JetBrains Mono", size=10, color=TEXT_2)),
            type="log" if use_log else "linear",
            range=y_range,
        ),
        row=1, col=1,
    )
    fig.update_yaxes(
        title=dict(text="VOL", font=dict(family="JetBrains Mono", size=10, color=TEXT_2)),
        row=2, col=1,
    )
    return fig


def _render_drilldown(read: ps.ActionableRead, cfg: dict) -> None:
    r = read.result
    _section_label(f"DRILL-DOWN · {read.ticker}", accent=read.color)
    st.markdown(
        f"<div style='font-family:var(--mono); color:var(--text-1); font-size:0.85rem; margin-bottom:8px;'>"
        f"<span style='color:{read.color}; font-weight:600;'>{read.icon} {read.headline_title}</span>"
        f" · {html_mod.escape(read.headline_detail or '')} · {html_mod.escape(read.headline_level or '')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-family:var(--mono); font-size:0.68rem; letter-spacing:0.14em; "
        "color:var(--text-2); text-transform:uppercase; margin-top:6px;'>Daily</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(_build_daily_chart(r, cfg), use_container_width=True)

    weekly_fig = _build_weekly_chart(r, cfg)
    if weekly_fig is not None:
        st.markdown(
            "<div style='font-family:var(--mono); font-size:0.68rem; letter-spacing:0.14em; "
            "color:var(--text-2); text-transform:uppercase; margin-top:14px;'>Weekly</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(weekly_fig, use_container_width=True)

    with st.expander("COMMENTARY · FULL NARRATIVE", expanded=False):
        md = commentary_mod.generate_commentary(r)
        st.markdown(md)


# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────

DEFAULT_PORTFOLIO = (
    "AMZN, CRH, GEV, GOOGL, KLAC, LITE, LRCX, NVDA, PH, RYCEY, STM, TSLA, URI"
)


def main() -> None:
    st.set_page_config(
        page_title="Portfolio Scanner",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

    if "portfolio_text" not in st.session_state:
        st.session_state.portfolio_text = DEFAULT_PORTFOLIO
    if "reads" not in st.session_state:
        st.session_state.reads = None
    if "drill_ticker" not in st.session_state:
        st.session_state.drill_ticker = None

    if not os.environ.get("POLYGON_API_KEY") and not Path(".env").exists():
        st.error(
            "POLYGON_API_KEY not found. Set it as an environment variable "
            "or add it as a Hugging Face Space secret."
        )

    cfg = sna.load_config()

    # Header
    h1, h2 = st.columns([5, 2])
    with h1:
        st.markdown("<div class='terminal-header'>Portfolio Scanner</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='terminal-sub'>Portfolio of tickers · top-3 actionable moves · full breakout/breakdown engine per name</div>",
            unsafe_allow_html=True,
        )
    with h2:
        scan = st.button("SCAN PORTFOLIO", use_container_width=True, type="primary")

    # Portfolio input
    portfolio_text = st.text_area(
        "Portfolio (comma, space, or newline-separated tickers)",
        value=st.session_state.portfolio_text,
        height=80,
        key="portfolio_input",
        label_visibility="visible",
    )
    st.session_state.portfolio_text = portfolio_text
    tickers = _parse_portfolio(portfolio_text)
    if tickers:
        st.caption(f"{len(tickers)} tickers parsed: {', '.join(tickers)}")

    # Scan
    if scan and tickers:
        prog = st.progress(0.0, text=f"Scanning {len(tickers)} tickers...")

        def _on_progress(done: int, total: int, tk: str) -> None:
            prog.progress(done / max(total, 1), text=f"[{done}/{total}] {tk} done")

        try:
            reads = ps.scan_portfolio(tickers, cfg=cfg, max_workers=8, progress_callback=_on_progress)
            st.session_state.reads = reads
            st.session_state.drill_ticker = None
        except Exception as e:
            st.error(f"Scan failed: {e}")
            return
        prog.empty()

    reads = st.session_state.reads
    if reads is None:
        st.markdown(
            "<div style='font-family:var(--mono); color:var(--text-2); font-size:0.85rem; "
            "padding:1.5rem 0;'>Enter tickers above and press SCAN PORTFOLIO. "
            "Each ticker runs the full single-name analyzer (Donchian / channels / patterns / weekly / "
            "AVWAP / trendlines / reversal candles) in parallel; the top 3 most actionable reads "
            "surface here.</div>",
            unsafe_allow_html=True,
        )
        return

    # Top moves
    _section_label("TOP MOVES · 3 MOST ACTIONABLE", accent=PHOSPHOR)
    _render_top_moves(reads)

    # Full portfolio table
    _section_label("PORTFOLIO · ALL NAMES", accent=AMBER)
    _render_portfolio_table(reads)

    # Drill-down (set by clicking a ticker in the portfolio table above)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    drill_ticker = st.session_state.get("drill_ticker")
    if drill_ticker:
        read = next((r for r in reads if r.ticker == drill_ticker), None)
        if read is None:
            st.session_state.drill_ticker = None
        else:
            head_col, btn_col = st.columns([5, 1])
            with btn_col:
                if st.button("CLEAR", key="clear_drill", use_container_width=True):
                    st.session_state.drill_ticker = None
                    st.rerun()
            if read.result.get("error"):
                st.error(f"{drill_ticker} — analysis error: {read.result.get('error')}")
            else:
                _render_drilldown(read, cfg)
    else:
        _section_label("DRILL-DOWN", accent=CYAN)
        st.markdown(
            "<div style='font-family:var(--mono); color:var(--text-2); font-size:0.78rem; "
            "padding:0.5rem 0;'>Click any ticker in the portfolio table above to load "
            "the full daily + weekly chart and narrative.</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
