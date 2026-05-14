"""Portfolio Scanner — runs the single-name analyzer across a portfolio in
parallel and ranks each name by actionability.

The actionability score derives from the same headline-takeaway priority
hierarchy used in the single-name analyzer:

    Tier 1: Fresh trigger today        (just fired, watch for hold)
    Tier 2: Triggered recently          (within last few bars)
    Tier 3: Primed — IMMEDIATE + STRONG (about to fire)
    Tier 4: Approaching — NEAR + STRONG
    Tier 5: Best pending setup (any proximity, quality ≥ 50)
    Tier 6: No clear bias

The score is `tier_base + quality_bonus` so within a tier, higher-quality
setups rank above lower-quality ones. The top-N most actionable names get
surfaced in the portfolio dashboard's "Top Moves" section.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import single_name_analyzer as sna


# Per-tier score bases. Within each tier, the named setup's `quality` (0-100)
# is added so the strongest setup wins within a tier.
TIER_BASE = {
    "FRESH_TRIGGER":      1000,
    "RECENT_TRIGGER":      900,
    "IMMEDIATE_SETUP":     800,
    "NEAR_SETUP":          700,
    "BEST_PENDING":        500,
    "NEUTRAL":               0,
}

TIER_LABELS = {
    "FRESH_TRIGGER":   "Fresh trigger today",
    "RECENT_TRIGGER":  "Triggered recently",
    "IMMEDIATE_SETUP": "Primed at trigger",
    "NEAR_SETUP":      "Approaching trigger",
    "BEST_PENDING":    "Best pending setup",
    "NEUTRAL":         "No clear bias",
}


@dataclass
class ActionableRead:
    ticker: str
    score: float
    tier: str
    tier_label: str
    direction: str            # "BREAKOUT" / "BREAKDOWN" / "NEUTRAL"
    setup_kind: str
    setup_label: str
    trigger_price: float
    distance_pct: float
    proximity_class: str
    headline_title: str
    headline_detail: str
    headline_level: str
    color: str
    icon: str
    last_price: float
    chg_pct: float
    result: dict              # full analyzer result for drill-down


# ─────────────────────────────────────────────────
# Actionability scoring per ticker
# ─────────────────────────────────────────────────

def score_actionability(result: dict) -> tuple[float, str, Optional[object]]:
    """Return (score, tier, anchor_setup) for one ticker's analysis result.

    The anchor_setup is the single Setup object that defines why this name is
    actionable — passed through for the UI to show its trigger/target/risk.
    """
    if result.get("error"):
        return 0.0, "NEUTRAL", None

    triggered = result.get("triggered") or []
    pending = (result.get("breakouts") or []) + (result.get("breakdowns") or [])

    # ── Tier 1: Fresh trigger today ─────────────────────────────────
    fresh_today = [s for s in triggered if s.bars_since_trigger == 0]
    if fresh_today:
        best = max(fresh_today, key=lambda s: s.quality)
        return TIER_BASE["FRESH_TRIGGER"] + best.quality, "FRESH_TRIGGER", best

    # ── Tier 2: Triggered within last few bars ──────────────────────
    if triggered:
        # Most recent + highest-quality among recent
        best = max(triggered, key=lambda s: (-s.bars_since_trigger, s.quality))
        # Decay penalty: each bar since trigger drops score by 20
        decay = (best.bars_since_trigger or 0) * 20
        return TIER_BASE["RECENT_TRIGGER"] + best.quality - decay, "RECENT_TRIGGER", best

    # ── Tier 3: IMMEDIATE proximity with STRONG quality ─────────────
    imm_strong = [
        s for s in pending
        if s.proximity_class == "IMMEDIATE" and s.quality_label == "STRONG"
    ]
    if imm_strong:
        best = max(imm_strong, key=lambda s: s.quality)
        return TIER_BASE["IMMEDIATE_SETUP"] + best.quality, "IMMEDIATE_SETUP", best

    # ── Tier 4: NEAR proximity with STRONG quality ──────────────────
    near_strong = [
        s for s in pending
        if s.proximity_class == "NEAR" and s.quality_label == "STRONG"
    ]
    if near_strong:
        best = max(near_strong, key=lambda s: s.quality)
        return TIER_BASE["NEAR_SETUP"] + best.quality, "NEAR_SETUP", best

    # ── Tier 5: Highest-quality pending setup (any proximity) ───────
    if pending:
        best = max(pending, key=lambda s: s.quality)
        if best.quality >= 50:
            return TIER_BASE["BEST_PENDING"] + best.quality / 2, "BEST_PENDING", best

    # ── Tier 6: fallback ────────────────────────────────────────────
    return TIER_BASE["NEUTRAL"], "NEUTRAL", None


def _build_read(ticker: str, result: dict) -> ActionableRead:
    """Build the ActionableRead record the UI consumes."""
    score, tier, anchor = score_actionability(result)
    headline = result.get("headline") or {}
    price = result.get("price") or {}

    direction = "NEUTRAL"
    setup_kind = ""
    setup_label = ""
    trigger_price = 0.0
    distance_pct = 0.0
    proximity_class = ""
    if anchor is not None:
        direction = anchor.direction
        setup_kind = anchor.kind
        setup_label = anchor.label
        trigger_price = float(anchor.trigger_price)
        distance_pct = float(anchor.distance_pct)
        proximity_class = anchor.proximity_class

    return ActionableRead(
        ticker=ticker,
        score=score,
        tier=tier,
        tier_label=TIER_LABELS[tier],
        direction=direction,
        setup_kind=setup_kind,
        setup_label=setup_label,
        trigger_price=trigger_price,
        distance_pct=distance_pct,
        proximity_class=proximity_class,
        headline_title=headline.get("title", ""),
        headline_detail=headline.get("detail", ""),
        headline_level=headline.get("level", ""),
        color=headline.get("color", "#ffb000"),
        icon=headline.get("icon", "◆"),
        last_price=float(price.get("last") or 0.0),
        chg_pct=float(price.get("chg_pct") or 0.0),
        result=result,
    )


# ─────────────────────────────────────────────────
# Parallel portfolio scan
# ─────────────────────────────────────────────────

def scan_portfolio(
    tickers: list[str],
    cfg: dict | None = None,
    max_workers: int = 8,
    progress_callback=None,
) -> list[ActionableRead]:
    """Analyze every ticker in the portfolio (parallel) and return ranked reads.

    `progress_callback(done, total, ticker)` is invoked after each ticker
    completes so the UI can drive a progress bar.
    """
    cfg = cfg or sna.load_config()
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    tickers = list(dict.fromkeys(tickers))  # de-dupe, keep order
    if not tickers:
        return []

    reads: list[ActionableRead] = []
    total = len(tickers)
    done = 0

    def _one(tk: str) -> ActionableRead:
        try:
            result = sna.analyze_ticker(tk, cfg=cfg)
        except Exception as e:
            result = {
                "ticker": tk,
                "error": f"{type(e).__name__}: {e}",
                "headline": {"title": "ERROR", "detail": str(e), "level": "—",
                             "color": "#ff3b3b", "icon": "✕", "bias": "ERROR"},
                "price": {"last": 0.0, "chg_pct": 0.0},
            }
        return _build_read(tk, result)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_one, tk): tk for tk in tickers}
        for fut in as_completed(futures):
            read = fut.result()
            reads.append(read)
            done += 1
            if progress_callback is not None:
                try:
                    progress_callback(done, total, read.ticker)
                except Exception:
                    pass

    # Stable sort: by score desc, then by ticker for tie-break determinism
    reads.sort(key=lambda r: (-r.score, r.ticker))
    return reads


# ─────────────────────────────────────────────────
# Top-N selection
# ─────────────────────────────────────────────────

def top_n(reads: list[ActionableRead], n: int = 3,
          require_tier: tuple[str, ...] | None = None) -> list[ActionableRead]:
    """Top-N most actionable reads from the scanned portfolio.

    If `require_tier` is set, only reads in those tiers qualify (e.g. limit
    to fresh triggers + primed setups, exclude pure "best pending").
    """
    if require_tier is not None:
        candidates = [r for r in reads if r.tier in require_tier]
    else:
        # Default: exclude pure NEUTRAL — nothing actionable about it
        candidates = [r for r in reads if r.tier != "NEUTRAL"]
    return candidates[:n]


def by_tier(reads: list[ActionableRead]) -> dict[str, list[ActionableRead]]:
    """Group reads by tier, preserving sort order within each."""
    out: dict[str, list[ActionableRead]] = {t: [] for t in TIER_BASE}
    for r in reads:
        out.setdefault(r.tier, []).append(r)
    return out
