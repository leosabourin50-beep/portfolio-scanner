---
title: Portfolio Scanner
emoji: 🐢
colorFrom: gray
colorTo: gray
sdk: docker
pinned: false
---

# Portfolio Scanner

A portfolio-wide breakout/breakdown scanner. Drop in a list of tickers — the
tool runs the full single-name analyzer engine across every name in parallel
and surfaces the **top-3 most actionable moves**.

Each name gets the same multi-source setup detection used by
[single-name-screener](https://github.com/leosabourin50-beep/single-name-screener):

- Donchian highs/lows at 20/50/100/252 days (with fresh-trigger detection)
- Moving-average reclaims and losses (20/50/200)
- Channel detection (daily + weekly, confirmed + forming tiers)
- Diagonal trendlines through multi-touch swings
- Pattern recognition (15 bullish + 9 bearish patterns, daily + weekly)
- Anchored VWAP from major swing highs/lows
- Bollinger-squeeze upper/lower bands
- Unfilled gaps as magnet targets
- Reversal candles at structural S/R
- Horizontal pivot clusters

Each name receives a **headline takeaway** (FRESH BREAKOUT / WEEKLY ASCENDING
CHANNEL / PRIMED FOR BREAKOUT / etc.) and an actionability score. The
scanner ranks the portfolio by score and shows the top 3 in big cards at the
top of the page, with the full table underneath and a per-ticker drill-down
showing the daily chart + commentary.

## Actionability Tiers (highest → lowest)

| Tier | Description |
|---|---|
| FRESH_TRIGGER     | A breakout/breakdown fired today |
| RECENT_TRIGGER    | Triggered within the last few bars |
| IMMEDIATE_SETUP   | At trigger level + STRONG quality (about to fire) |
| NEAR_SETUP        | Within 3% of trigger + STRONG quality |
| BEST_PENDING      | Any pending setup with quality ≥ 50 |
| NEUTRAL           | No clear bias |

## Quick Start

```bash
pip install -r requirements.txt
export POLYGON_API_KEY=your_key_here
streamlit run app.py
```

Open http://localhost:8501, paste a list of tickers, click SCAN PORTFOLIO.

## Related

- [single-name-screener](https://github.com/leosabourin50-beep/single-name-screener) —
  the deep-dive single-ticker analyzer this project forks from. The two
  projects share their engine files (`detector.py`, `patterns.py`,
  `setup_finder.py`, `single_name_analyzer.py`, etc.) intentionally.
