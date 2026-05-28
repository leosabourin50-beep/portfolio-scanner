"""Send a one-off chart-bearing test alert to the watcher's Telegram bot.

Useful for verifying chart rendering + delivery end-to-end after a deploy:

    python send_test_ping.py            # uses first portfolio ticker (or NVDA)
    python send_test_ping.py NVDA AAPL  # one ping per ticker

Uses the same Notifier / chart pipeline as the live watcher, so a green result
here means real alerts will render too. Honors DRY_RUN=1 (prints instead).
"""
from __future__ import annotations

import sys

import portfolio as pf
import portfolio_scanner as ps
import single_name_analyzer as sna
import chart_render
import notify


def main() -> int:
    tickers = [t.upper() for t in sys.argv[1:]]
    if not tickers:
        loaded = pf.load_portfolio()
        tickers = [loaded[0]] if loaded else ["NVDA"]

    notifier = notify.Notifier.from_env(
        "TELEGRAM_WATCHER_BOT_TOKEN", "TELEGRAM_WATCHER_CHAT_ID")
    if not notifier.token or not notifier.chat_id:
        print("[fatal] no watcher bot token/chat in env "
              "(TELEGRAM_WATCHER_BOT_TOKEN/CHAT_ID or TELEGRAM_BOT_TOKEN/CHAT_ID)",
              file=sys.stderr)
        return 1

    cfg = sna.load_config()
    sent = 0
    for ticker in tickers:
        print(f"[test] analyzing {ticker}…")
        result = sna.analyze_ticker(ticker, cfg=cfg)
        if result.get("error"):
            print(f"[skip] {ticker}: {result['error']}", file=sys.stderr)
            continue
        headline = result.get("headline") or {}
        price = result.get("price") or {}
        last = float(price.get("last") or 0.0)
        chg = float(price.get("chg_pct") or 0.0)
        _score, tier, anchor = ps.score_actionability(result)
        highlight = None
        if anchor is not None:
            highlight = {"price": float(anchor.trigger_price),
                         "label": anchor.label, "direction": anchor.direction}
        png = chart_render.render_daily_png(result, cfg, highlight=highlight)
        caption = (
            f"🧪 <b>TEST PING — {ticker}</b>\n"
            f"<code>${last:.2f}  ({'+' if chg >= 0 else ''}{chg:.2f}%)</code>\n"
            f"{headline.get('icon', '◆')} {headline.get('title', '')}\n"
            f"<i>chart render {'OK' if png else 'unavailable (text fallback)'}</i>"
        )
        if notifier.alert(caption, png_bytes=png):
            sent += 1
            print(f"[test] sent {ticker} (chart: {'yes' if png else 'no'})")
    print(f"[done] {sent}/{len(tickers)} test pings sent")
    return 0 if sent else 1


if __name__ == "__main__":
    sys.exit(main())
