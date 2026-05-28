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

## Telegram alerts

Two delivery paths, each on its **own** bot token:

- **`scanner_cron.py`** — daily-tier alerts (fresh/recent triggers, primed
  setups) run by GitHub Actions every 30 min during market hours. Uses
  `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`. Dedup state and the alert log are
  persisted between runs via the Actions cache (no more git commit-back).
- **`intraday_watcher.py`** — a long-running Fly.io worker that pings the first
  time price closes through a watched level intraday, plus violent-move alerts.
  Uses `TELEGRAM_WATCHER_BOT_TOKEN` / `TELEGRAM_WATCHER_CHAT_ID` (falls back to
  the cron vars if unset). Hosts the interactive command bot.

Every alert carries the **daily chart** with the firing level highlighted
(rendered via kaleido). If chart export isn't available the alert falls back to
text automatically.

### Interactive commands (watcher bot)

The watcher long-polls its own bot for commands (authorized to your chat id):

| Command | Does |
|---|---|
| `/status` | watcher liveness + portfolio summary |
| `/list` | show the portfolio (positions + mutes) |
| `/positions` | owned positions with live P/L |
| `/scan TICKER` | on-demand read + chart |
| `/add TICKER [N @ PRICE]` | add a ticker (optionally as a holding) |
| `/remove TICKER` | remove a ticker |
| `/mute TICKER` · `/unmute TICKER` | silence / restore a name |
| `/snooze TICKER 30m\|2h\|1d` | temporarily silence (default 1h) |

Edits made via Telegram persist on the Fly volume (`portfolio_runtime.txt`),
taking precedence over the repo `portfolio.txt`.

## Positions (owned vs watched)

`portfolio.txt` (and the live runtime list) accept an optional position:

```
NVDA                 # watch-only — opportunity alerts
TSLA 100 @ 240.50    # owned: 100 shares, entry 240.50 → risk-framed alerts
GOOGL 50 @ 150
```

Alerts on a name you **hold** include a HELD line with current P/L, so a
breakdown on a position reads as a risk alert, not just an opportunity.

## Signal scorecard

Every delivered alert is logged to `alert_log.jsonl`. `scorecard.py` grades
each by its direction-adjusted forward return (1d / 5d) and the watcher sends a
**weekly scorecard** every Friday after the close — so you can tune the
noisy-vs-quiet filters with data. Run `python scorecard.py alert_log.jsonl` to
print it on demand.

## Deploying the watcher (Fly)

```bash
fly secrets set \
  POLYGON_API_KEY=... \
  TELEGRAM_WATCHER_BOT_TOKEN=... \
  TELEGRAM_WATCHER_CHAT_ID=...
fly deploy
```

The watcher uses a one-call Polygon snapshot per poll as a pre-filter (set
`USE_SNAPSHOT=0` to disable), fetching 5-min bars only for tickers in play —
so the portfolio can grow well past a dozen names without linear API cost.

## Related

- [single-name-screener](https://github.com/leosabourin50-beep/single-name-screener) —
  the deep-dive single-ticker analyzer this project forks from. The two
  projects share their engine files (`detector.py`, `patterns.py`,
  `setup_finder.py`, `single_name_analyzer.py`, etc.) intentionally.
