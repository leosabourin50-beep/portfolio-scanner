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


def _render_portfolio_table(reads: list[ps.ActionableRead]) -> None:
    rows = ["<div class='port-row port-header'>"
            "<div>Ticker</div><div>Price</div><div>Chg</div>"
            "<div>Headline</div><div>Tier</div><div>Level</div>"
            "</div>"]
    for r in reads:
        chg_cls = "port-chg-pos" if r.chg_pct >= 0 else "port-chg-neg"
        rows.append(
            f"<div class='port-row'>"
            f"<div class='port-tick'>{r.ticker}</div>"
            f"<div class='port-price'>${r.last_price:.2f}</div>"
            f"<div class='{chg_cls}'>{r.chg_pct:+.2f}%</div>"
            f"<div class='port-head' style='color:{r.color};'>{r.icon} {html_mod.escape(r.headline_title)}</div>"
            f"<div class='port-tier'>{html_mod.escape(r.tier_label)}</div>"
            f"<div class='port-level'>{html_mod.escape(r.headline_level or '')}</div>"
            f"</div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


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


def _build_simple_daily_chart(result: dict, show_days: int = 252) -> go.Figure:
    """A simpler daily chart for the drill-down view — candles + MAs +
    setup levels drawn as horizontal lines (with diagonal channel rails)."""
    df = result["df_daily"]
    df_view = df.iloc[-show_days:].copy()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.03,
    )
    fig.add_trace(go.Candlestick(
        x=df_view.index, open=df_view["Open"], high=df_view["High"],
        low=df_view["Low"], close=df_view["Close"], name=result["ticker"],
        increasing_line_color=PHOSPHOR, decreasing_line_color=WARN,
        increasing_fillcolor=PHOSPHOR, decreasing_fillcolor=WARN,
        showlegend=False,
    ), row=1, col=1)

    for period, color in [(20, "#3b82f6"), (50, "#f97316"), (200, "#ef4444")]:
        ma = df["Close"].rolling(period).mean().iloc[-show_days:]
        fig.add_trace(go.Scatter(
            x=ma.index, y=ma.values, mode="lines",
            name=f"MA{period}", line=dict(color=color, width=1.2),
        ), row=1, col=1)

    # Draw top setup levels (skip weekly channels — those belong on weekly chart)
    candle_low = float(df_view["Low"].min())
    candle_high = float(df_view["High"].max())
    y_pad_lo, y_pad_hi = candle_low * 0.95, candle_high * 1.05

    for s in (result.get("breakouts") or [])[:5] + (result.get("breakdowns") or [])[:5]:
        if s.kind.startswith("CHANNEL_WEEKLY_"):
            continue
        if s.line_meta:
            try:
                fig.add_shape(
                    type="line",
                    x0=s.line_meta["x0"], y0=float(s.line_meta["y0"]),
                    x1=s.line_meta["x1"], y1=float(s.line_meta["y1"]),
                    line=dict(
                        color=PHOSPHOR if s.direction == "BREAKOUT" else WARN,
                        width=1.2, dash="dash"
                    ),
                    xref="x", yref="y", row=1, col=1,
                )
            except Exception:
                pass
        elif y_pad_lo <= s.trigger_price <= y_pad_hi:
            color = PHOSPHOR if s.direction == "BREAKOUT" else WARN
            fig.add_hline(
                y=s.trigger_price,
                line=dict(color=color, width=1.0, dash="dash"),
                annotation_text=f"${s.trigger_price:.2f}",
                annotation_position="right",
                annotation_font=dict(color=color, size=10, family="JetBrains Mono"),
                row=1, col=1,
            )

    colors = [PHOSPHOR if c >= o else WARN for o, c in zip(df_view["Open"], df_view["Close"])]
    fig.add_trace(go.Bar(
        x=df_view.index, y=df_view["Volume"], marker_color=colors,
        name="Volume", showlegend=False, opacity=0.55,
    ), row=2, col=1)

    fig.update_layout(**_terminal_layout(height=560))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


def _render_drilldown(read: ps.ActionableRead) -> None:
    r = read.result
    _section_label(f"DRILL-DOWN · {read.ticker}", accent=read.color)
    st.markdown(
        f"<div style='font-family:var(--mono); color:var(--text-1); font-size:0.85rem; margin-bottom:8px;'>"
        f"<span style='color:{read.color}; font-weight:600;'>{read.icon} {read.headline_title}</span>"
        f" · {html_mod.escape(read.headline_detail or '')} · {html_mod.escape(read.headline_level or '')}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(_build_simple_daily_chart(r), use_container_width=True)

    with st.expander("COMMENTARY · FULL NARRATIVE", expanded=False):
        md = commentary_mod.generate_commentary(r)
        st.markdown(md)


# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────

DEFAULT_PORTFOLIO = (
    "NVDA, AAPL, MSFT, GOOGL, AMZN, META, TSLA, AVGO, AMD, NFLX"
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

    # Drill-down picker
    st.markdown("&nbsp;", unsafe_allow_html=True)
    _section_label("DRILL-DOWN · SELECT A TICKER", accent=CYAN)
    ticker_options = [r.ticker for r in reads]
    selected = st.selectbox(
        "Drill into a ticker for charts + narrative",
        options=["—"] + ticker_options,
        index=0,
        label_visibility="collapsed",
    )
    if selected and selected != "—":
        st.session_state.drill_ticker = selected
        read = next((r for r in reads if r.ticker == selected), None)
        if read is not None and not read.result.get("error"):
            _render_drilldown(read)
        elif read is not None:
            st.error(f"{selected} — analysis error: {read.result.get('error')}")


if __name__ == "__main__":
    main()
