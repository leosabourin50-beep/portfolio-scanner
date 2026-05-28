# CLAUDE.md — Portfolio Scanner

## What This Project Does

A portfolio-wide breakout/breakdown scanner. The user enters a portfolio of
tickers (paste a list, save in session) and the tool runs the full
single-name analyzer engine across every name in parallel, then surfaces the
**top-3 most actionable moves** — the names with fresh triggers, primed
setups at current price, or active channel context.

This is a fork of `single-name-screener` that **reuses every engine file
verbatim** (`detector.py`, `patterns.py`, `polygon_adapter.py`,
`setup_finder.py`, `single_name_analyzer.py`, `intraday_analysis.py`,
`commentary.py`). The two projects intentionally share these so improvements
to the detection / setup-map logic flow into both. Only the UI and the new
`portfolio_scanner.py` orchestrator differ.

## Key Rules — Read Before Writing Any Code

1. **Engine files are shared with single-name-screener.** Treat
   `detector.py`, `patterns.py`, `polygon_adapter.py`, `setup_finder.py`,
   `single_name_analyzer.py`, `intraday_analysis.py`, and `commentary.py` as
   the source of truth — if you change one of them here, mirror the change
   to single-name-screener (or vice versa) so the projects don't drift.

2. **Actionability ranking uses the headline-takeaway hierarchy.** The
   priority tiers (FRESH_TRIGGER > RECENT_TRIGGER > IMMEDIATE_SETUP >
   NEAR_SETUP > BEST_PENDING > NEUTRAL) reflect the same logic as the
   single-name headline. The `score_actionability` function in
   `portfolio_scanner.py` is the single source of truth for cross-portfolio
   ranking; don't sneak in ad-hoc weighting in `app.py`.

3. **Parallel scan via ThreadPoolExecutor.** `scan_portfolio` runs each
   ticker through `analyze_ticker` in a thread pool (default 8 workers).
   Polygon's connection pool is sized to 20 in `polygon_adapter.py`, so
   bumping max_workers is safe up to that limit.

4. **No LLM calls.** Commentary is deterministic (inherited from
   single-name-screener's `commentary.py`). Every claim in the UI maps to a
   specific number in the analyzer result dict.

5. **Same Bloomberg-terminal aesthetic** as single-name-screener (phosphor
   green / amber / cyan / warning-red on near-black, JetBrains Mono +
   Inter). The CSS is mirrored from the single-name app. `chart_render.py`
   re-mirrors the daily-chart palette so alert PNGs match — it deliberately
   does NOT import `app.py` (that would pull Streamlit into the worker).

6. **Telegram delivery is shared via `notify.py`.** Don't reintroduce a local
   `_send_telegram`. The daily cron and the intraday watcher run on SEPARATE
   bot tokens (`TELEGRAM_BOT_TOKEN` vs `TELEGRAM_WATCHER_BOT_TOKEN`); the
   watcher binds a `notify.Notifier` to its own token. Charts are best-effort:
   `notify.send_alert` falls back to text if the PNG is None or fails.

7. **The interactive command bot lives in the watcher only.** Telegram allows
   one `getUpdates` consumer per token, and the cron is ephemeral — so
   `telegram_commands.run_command_loop` runs as a daemon thread inside
   `intraday_watcher.py`. Don't add a second long-poller.

8. **Portfolio + positions go through `portfolio.py`.** It parses the
   `TICKER [N @ ENTRY]` format, prefers the volume-backed runtime file once
   edited via Telegram, and owns the mute/snooze store. Both the cron and the
   watcher import it instead of re-reading `portfolio.txt`.

9. **Dedup state is no longer committed to git.** The cron persists
   `.alerts_state.json` + `.alert_log.jsonl` via the Actions cache; the watcher
   keeps its state on the Fly `/data` volume. The committed `.alerts_state.json`
   in the repo is legacy and can be deleted.

## Architecture

```
portfolio-scanner/
├── CLAUDE.md                   # This file
├── README.md                   # User-facing readme
├── requirements.txt            # Python deps
├── config.yaml                 # Inherited from single-name-screener
├── detector.py                 # SHARED — engine, don't modify here
├── patterns.py                 # SHARED
├── polygon_adapter.py          # SHARED
├── setup_finder.py             # SHARED
├── single_name_analyzer.py     # SHARED — analyze_ticker called per name
├── intraday_analysis.py        # SHARED
├── commentary.py               # SHARED — used in drill-down view
├── portfolio_scanner.py        # NEW — parallel scan + actionability ranking
├── app.py                      # NEW — portfolio-first Streamlit UI
├── scanner_cron.py             # Daily-tier Telegram alerts (GitHub Actions cron)
├── intraday_watcher.py         # Long-running Fly worker — intraday level/violent alerts
├── notify.py                   # Shared Telegram delivery (message/photo) + alert log
├── chart_render.py             # Analyzer result -> PNG (best-effort, kaleido)
├── portfolio.py                # Portfolio + positions + mute/snooze state
├── telegram_commands.py        # Interactive command bot (runs in the watcher)
├── scorecard.py                # Weekly forward-return signal scorecard
├── Dockerfile                  # Inherited
├── Dockerfile.watcher          # Fly worker image (runs intraday_watcher.py)
└── .streamlit/config.toml      # Inherited
```

## Build Order

Build and test modules in this exact order:

1. Verify the shared engine files run as-is — `import single_name_analyzer`
   should work without changes.
2. `portfolio_scanner.py` — verify `scan_portfolio(["NVDA", "AAPL"])`
   returns a list of `ActionableRead` records sorted by score.
3. `app.py` — wire up the Streamlit UI.

## Key Constraints

- Python 3.9+
- Polygon API key required (env var `POLYGON_API_KEY` or `.env` file)
- All charts use Plotly
- Dark theme Streamlit (set in .streamlit/config.toml)
- Polygon Stocks Starter or higher recommended for real-time data

## Relationship to Single-Name-Screener

- **Single-name-screener** is for one ticker at a time — deep technical
  read, multiple charts (daily / weekly / intraday), full setup map,
  full narrative commentary.
- **Portfolio-scanner** is for many tickers — surfaces the few most
  actionable, with drill-down into any individual name (which shows the
  daily chart + headline + narrative, but a slimmer view than the
  single-name app).
