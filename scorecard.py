"""Signal scorecard — closes the feedback loop on alert quality.

Reads the alert JSONL log (written by scanner_cron.py and intraday_watcher.py)
and grades each alert by its forward return over the following trading days,
direction-adjusted (a breakdown that then falls is a "hit"). Produces a weekly
Telegram digest so the noisy-vs-quiet filters can be tuned with data instead of
vibes.

Forward returns are anchored to the daily close on the alert's trading date and
measured N trading bars later, using Polygon daily closes.

Usage:
  python scorecard.py [path/to/alert_log.jsonl]    # prints the digest
  from scorecard import weekly_scorecard            # returns HTML or None
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polygon_adapter as poly

HORIZONS = (1, 5)          # trading days
LOOKBACK_DAYS = 10         # alerts within this many calendar days are graded
MAX_RECENT_ROWS = 12       # cap the per-alert list in the digest


def _read_log(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict] = []
    try:
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return rows


def _dedupe(rows: list[dict]) -> list[dict]:
    """Collapse duplicate alerts (cron + watcher can both log the same name on
    the same day) keyed by ticker|kind|direction|trading_date."""
    seen: set[tuple] = set()
    out: list[dict] = []
    for r in rows:
        key = (r.get("ticker"), r.get("kind"), r.get("direction"), r.get("trading_date"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _signed_returns(ticker: str, trading_date: str, direction: str,
                    daily_cache: dict) -> dict[int, float | None]:
    """Direction-adjusted forward returns per horizon (positive = signal was
    'right'). None when not enough forward bars exist yet."""
    df = daily_cache.get(ticker)
    if df is None:
        df = poly.fetch_data(ticker, days=120)
        daily_cache[ticker] = df
    out: dict[int, float | None] = {h: None for h in HORIZONS}
    if df is None or df.empty:
        return out
    try:
        dates = [d.date() for d in df.index]
        target = datetime.strptime(trading_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return out
    # First bar on/after the alert's trading date.
    idx = next((i for i, d in enumerate(dates) if d >= target), None)
    if idx is None:
        return out
    closes = df["Close"].tolist()
    base = closes[idx]
    if not base:
        return out
    flip = -1.0 if direction == "BREAKDOWN" else 1.0
    for h in HORIZONS:
        j = idx + h
        if j < len(closes) and closes[j]:
            out[h] = flip * (closes[j] / base - 1.0) * 100.0
    return out


def weekly_scorecard(log_path: str | Path,
                     lookback_days: int = LOOKBACK_DAYS) -> str | None:
    """Build the HTML digest, or None if there are no alerts to grade."""
    rows = _dedupe(_read_log(log_path))
    if not rows:
        return None

    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date()
    recent: list[dict] = []
    for r in rows:
        td = r.get("trading_date")
        try:
            d = datetime.strptime(td, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        if d >= cutoff:
            recent.append(r)
    if not recent:
        return None

    daily_cache: dict = {}
    graded: list[tuple[dict, dict[int, float | None]]] = []
    for r in recent:
        rets = _signed_returns(r.get("ticker", ""), r.get("trading_date", ""),
                               r.get("direction", "BREAKOUT"), daily_cache)
        graded.append((r, rets))

    # Aggregate on the longest horizon that has matured data.
    grade_h = HORIZONS[-1]
    matured = [(r, rets) for r, rets in graded if rets.get(grade_h) is not None]
    n_total = len(graded)

    lines = ["<b>📊 WEEKLY SIGNAL SCORECARD</b>"]
    if matured:
        vals = [rets[grade_h] for _, rets in matured]
        hits = sum(1 for v in vals if v > 0)
        avg = sum(vals) / len(vals)
        best_r, best = max(matured, key=lambda x: x[1][grade_h])
        worst_r, worst = min(matured, key=lambda x: x[1][grade_h])
        pending = n_total - len(matured)
        lines.append(
            f"<i>{n_total} alerts · {len(matured)} matured"
            + (f" · {pending} pending" if pending else "") + "</i>"
        )
        lines.append(
            f"Hit rate ({grade_h}d, dir-adj): <b>{hits/len(matured)*100:.0f}%</b> "
            f"({hits}/{len(matured)})"
        )
        lines.append(f"Avg {grade_h}d move: <b>{avg:+.1f}%</b>")
        lines.append(f"Best: {best_r.get('ticker')} {best[grade_h]:+.1f}% · "
                     f"Worst: {worst_r.get('ticker')} {worst[grade_h]:+.1f}%")
    else:
        lines.append(f"<i>{n_total} alerts this week — none matured to {grade_h}d yet</i>")

    # Per-alert rows (most recent first).
    graded.sort(key=lambda x: x[0].get("trading_date", ""), reverse=True)
    lines.append("— recent —")
    for r, rets in graded[:MAX_RECENT_ROWS]:
        arrow = "▲" if r.get("direction") != "BREAKDOWN" else "▼"
        parts = []
        for h in HORIZONS:
            v = rets.get(h)
            parts.append(f"{h}d {v:+.1f}%" if v is not None else f"{h}d —")
        kind = (r.get("kind") or "").replace("_", " ").lower()
        lines.append(f"{arrow} <b>{r.get('ticker')}</b> {kind} · " + " · ".join(parts))

    return "\n".join(lines)


def main() -> int:
    log_path = sys.argv[1] if len(sys.argv) > 1 else "alert_log.jsonl"
    msg = weekly_scorecard(log_path)
    if not msg:
        print("[scorecard] no alerts to grade")
        return 0
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
