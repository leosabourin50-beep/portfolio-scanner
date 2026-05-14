"""Deterministic commentary generator for the single-name analyzer.

Reads a result dict from single_name_analyzer.analyze_ticker() and emits a
multi-section markdown narrative. No LLM. Every claim maps to a specific
numeric field in result["signals"], result["deep"], result["intraday"], or
result["patterns"]. Same inputs → identical output.

Sections are omitted if their data isn't applicable — no "N/A" filler.
"""
from __future__ import annotations


# Display names for the pattern keys returned by patterns.scan_patterns.
# Keys must match the exact strings emitted by detectors in patterns.py.
PATTERN_NAMES = {
    "Bull Flag": "bull flag",
    "Ascending Triangle": "ascending triangle",
    "Cup & Handle": "cup and handle",
    "VCP": "volatility contraction pattern (VCP)",
    "High Tight Flag": "high tight flag",
    "Symmetrical Triangle": "symmetrical triangle",
    "Double Bottom": "double bottom",
    "Inv H&S": "inverse head and shoulders",
    "Bear Flag": "bear flag",
    "Descending Triangle": "descending triangle",
    "Inv Cup & Handle": "inverse cup and handle",
    "Double Top": "double top",
    "H&S Top": "head and shoulders top",
    "Distribution VCP": "distribution VCP",
    "HT Bear Flag": "high tight bear flag",
    # Single-name additions — short-duration entry-timing patterns
    "Flat Base": "flat base",
    "Falling Wedge": "falling wedge",
    "Rising Wedge": "rising wedge",
    "Bull Pennant": "bullish pennant",
    "Bear Pennant": "bearish pennant",
    "Three Tight Closes": "three tight closes",
    "Inside Day Sequence": "inside day sequence",
    "Pocket Pivot": "pocket pivot",
}


def _pretty_pattern(name: str) -> str:
    return PATTERN_NAMES.get(name, name.lower())


def _consolidation_paragraph(ticker: str, signals: dict, deep: dict) -> str:
    """The classic squeeze-near-breakout paragraph (shared by multiple setup types)."""
    consol = signals["consolidation"]
    atr_pct = consol["atr_percentile"]
    s = (
        f"**Setup** — {ticker} is in a {deep['consol_duration_days']}-day consolidation "
        f"between ${consol['range_low']:.2f} and ${consol['range_high']:.2f}"
    )
    if atr_pct < 15:
        s += f", with ATR in the {atr_pct:.0f}th percentile of its trailing range — that's an extreme squeeze."
    elif atr_pct < 25:
        s += f", with ATR in the {atr_pct:.0f}th percentile — a legitimate squeeze."
    elif atr_pct < 40:
        s += f", with ATR in the {atr_pct:.0f}th percentile — moderate compression, not yet a full squeeze."
    else:
        s += f", with ATR at the {atr_pct:.0f}th percentile — no squeeze yet, volatility is still elevated."

    if deep["consol_quality"] == "TIGHT":
        s += f" The range is tight at {deep['consol_range_pct']:.1f}%."
    elif deep["consol_quality"] == "MODERATE":
        s += f" The range spans {deep['consol_range_pct']:.1f}% — moderate width."
    elif deep["consol_quality"] == "WIDE":
        s += f" The range is wide at {deep['consol_range_pct']:.1f}% — this isn't a clean base."

    if deep["higher_lows_in_base"]:
        s += " Swing lows within the base are rising — internal structure is bullish."
    return s


def _commentary_setup(result: dict) -> str:
    signals = result["signals"]
    deep = result["deep"]
    consol = signals["consolidation"]
    ticker = result["ticker"]
    status = signals.get("_status", "")
    setup_type = result.get("setup_type", "NO_DEFINED_SETUP")

    if setup_type == "SQUEEZE_NEAR_BREAKOUT":
        return _consolidation_paragraph(ticker, signals, deep)

    if setup_type == "BASE_BUILDING":
        body = _consolidation_paragraph(ticker, signals, deep)
        body += " Price is still mid-range — the base is building but not yet actionable."
        return body

    if setup_type == "TREND_PULLBACK_TO_20MA":
        ma_val = deep["ma_20_value"]
        dist = signals.get("ma_20", {}).get("pct_distance", 0)
        return (
            f"**Setup** — {ticker} is pulling back to its rising 20-day MA "
            f"(${ma_val:.2f}, {abs(dist):.1f}% away) in a bullish-aligned trend — "
            "a standard trend-continuation entry point."
        )

    if setup_type == "TREND_PULLBACK_TO_50MA":
        ma_val = deep["ma_50_value"]
        dist = signals.get("ma_50", {}).get("pct_distance", 0)
        return (
            f"**Setup** — {ticker} is pulling back to its rising 50-day MA "
            f"(${ma_val:.2f}, {abs(dist):.1f}% away) with MAs fully aligned — "
            "a deeper pullback entry in an established trend."
        )

    if setup_type == "ACTIVE_BREAKOUT" and consol.get("range_high"):
        return (
            f"**Setup** — {ticker} has broken above its prior range. "
            "The breakout is active — watch for follow-through volume and "
            f"whether price holds above ${consol['range_high']:.2f}."
        )

    if setup_type == "BREAKOUT_RETEST" and consol.get("range_high"):
        return (
            f"**Setup** — {ticker} broke out and is now retesting the breakout level "
            f"near ${consol['range_high']:.2f}. A successful hold here with volume "
            "confirms the breakout — this is a second-chance entry."
        )

    if setup_type == "PATTERN_FORMATION":
        pats = result.get("patterns") or []
        if pats:
            best = max(pats, key=lambda p: p.confidence)
            name = _pretty_pattern(best.pattern)
            return (
                f"**Setup** — {ticker} doesn't have a classic squeeze, but a "
                f"{name} is forming with {best.confidence:.0%} confidence. "
                f"Breakout level: ${best.breakout_level:.2f}."
            )

    # Fallbacks for NO_DEFINED_SETUP and edge cases.
    if "TRENDING" in status or status == "EXTENDED":
        return (
            f"**Setup** — {ticker} is trending, not basing. There's no defined "
            "consolidation or pullback level to act against here."
        )
    if status in ("WEAK", "BREAKDOWN", "TRENDING DOWN"):
        return f"**Setup** — {ticker} is in a downtrend — no entry structure on this side."
    return f"**Setup** — {ticker} is not in a recognizable consolidation, pullback, or pattern setup."


def _commentary_position(result: dict) -> str:
    signals = result["signals"]
    deep = result["deep"]
    last = signals["_price"]["last"]
    prox = deep["proximity_to_breakout_pct"]
    zone = deep["proximity_zone"]

    s = f"**Position** — Price is at ${last:.2f}"
    if zone == "AT_LEVEL":
        s += f", sitting {prox:.1f}% below the range ceiling — right at the trigger."
    elif zone == "APPROACHING":
        s += f", {prox:.1f}% below the range ceiling — approaching the breakout level."
    elif zone == "ABOVE":
        s += ", which is above the prior range high. The breakout has already triggered."
    elif zone == "MID_RANGE":
        s += ", sitting mid-range in the consolidation — not yet at the trigger level."
    elif zone == "FAR":
        s += f", {prox:.1f}% below the range ceiling — still distant from the breakout level."

    cl = deep["close_location"]
    if cl > 0.7:
        s += f" Today's close is at the {cl*100:.0f}th percentile of the day's range — strong close."
    elif cl < 0.3:
        s += f" Today's close is at the {cl*100:.0f}th percentile of the day's range — weak close."
    return s


def _commentary_trend(result: dict) -> str:
    deep = result["deep"]
    stack = deep["ma_stack"]
    ma20, ma50, ma200 = deep["ma_20_value"], deep["ma_50_value"], deep["ma_200_value"]

    if stack == "BULLISH_ALIGNED" and None not in (ma20, ma50, ma200):
        s = (
            f"**Trend** — The MA stack is aligned (20 > 50 > 200 at "
            f"${ma20:.2f} / ${ma50:.2f} / ${ma200:.2f}), price is above all three — full trend support."
        )
    elif stack == "PARTIALLY_ALIGNED" and ma200 is not None:
        s = (
            f"**Trend** — Price is above the 20 and 50-day MAs but below the 200-day at "
            f"${ma200:.2f} — there's overhead resistance from the long-term average."
        )
    elif stack == "BEARISH_ALIGNED":
        s = "**Trend** — All MAs are above price (200 > 50 > 20) — this is a downtrend structure."
    else:
        s = "**Trend** — MAs are mixed — no clean trend alignment."

    if deep["rs_leading"]:
        s += " RS vs SPY is making new highs while price is still in the base — institutional accumulation is leading price."
    elif deep["rs_trend_20d"] == "UP":
        s += " RS vs SPY has been trending up over 20 days — outperforming the market."
    elif deep["rs_trend_20d"] == "DOWN":
        s += " RS vs SPY is declining — underperforming the market, which weakens the setup."
    else:
        s += " RS vs SPY is flat — no clear relative momentum."
    return s


def _commentary_volume(result: dict) -> str:
    deep = result["deep"]
    signals = result["signals"]
    bvt = deep["base_volume_trend"]
    bvr = deep["base_volume_ratio"]

    parts: list[str] = []
    if bvt == "DECLINING" and bvr is not None:
        parts.append(
            f"Volume has been declining during the base (ratio: {bvr:.2f}x pre-base average) — healthy drying up of supply."
        )
    elif bvt == "RISING" and bvr is not None:
        parts.append(
            f"Volume has been rising during the base without a breakout (ratio: {bvr:.2f}x) — watch for distribution."
        )
    elif bvt == "FLAT" and bvr is not None:
        parts.append(f"Volume during the base is roughly in line with prior levels ({bvr:.2f}x).")

    vol = signals["volume"]
    if vol.get("ratio"):
        if vol["signal"] == "SURGE":
            parts.append(f"Current volume is {vol['ratio']:.1f}x the 20-day average — that's a surge.")
        elif vol["signal"] == "ELEVATED":
            parts.append(f"Current volume is elevated at {vol['ratio']:.1f}x average.")
        elif vol["signal"] == "DRY_UP":
            parts.append(f"Current volume is dried up at {vol['ratio']:.1f}x average.")

    if not parts:
        return ""
    return "**Volume** — " + " ".join(parts)


def _commentary_momentum(result: dict) -> str:
    signals = result["signals"]
    rsi = signals["rsi"]["value"]
    macd_sig = signals["macd"]["signal"]
    hist = signals["macd"]["histogram"]

    s = f"**Momentum** — RSI is at {rsi:.0f}"
    if 45 <= rsi <= 65:
        s += " — healthy for a base, not overbought."
    elif rsi > 70:
        s += (
            " — overbought territory. If this is a fresh breakout, momentum names can live above 70 for weeks. "
            "If it's not breaking out, this is extended."
        )
    elif rsi < 30:
        s += " — oversold. This is a mean-reversion zone, not a momentum entry."
    elif rsi < 45:
        s += " — below midline, momentum is weak."
    else:
        s += " — bullish momentum building above the midline."

    if macd_sig == "BULLISH_CROSS":
        s += " MACD just crossed bullish — momentum is turning up."
    elif macd_sig == "BEARISH_CROSS":
        s += " MACD just crossed bearish — momentum is fading."
    elif macd_sig == "BULLISH" and hist > 0:
        s += " MACD histogram is expanding — momentum is accelerating."
    return s


def _commentary_patterns(result: dict) -> str:
    pats = result.get("patterns", [])
    if not pats:
        return ""
    pats_sorted = sorted(pats, key=lambda p: p.confidence, reverse=True)
    lines: list[str] = []
    for p in pats_sorted[:4]:
        if p.confidence < 0.15:
            continue
        name = _pretty_pattern(p.pattern)
        timeframe = " on the weekly chart" if "[weekly]" in (p.notes or "") else ""
        if p.confidence >= 0.5:
            if p.status == "Forming":
                lines.append(
                    f"A {name} is forming{timeframe} with {p.confidence:.0%} confidence. "
                    f"Breakout level: ${p.breakout_level:.2f}. Measured-move target: ${p.target:.2f}."
                )
            elif p.status == "Breaking out":
                lines.append(
                    f"A {name} is breaking out{timeframe} ({p.confidence:.0%} confidence). Target: ${p.target:.2f}."
                )
            elif p.status == "Confirmed":
                lines.append(
                    f"A confirmed {name}{timeframe} ({p.confidence:.0%} confidence) targets ${p.target:.2f}."
                )
        else:
            lines.append(
                f"An early-stage {name} may be forming{timeframe} "
                f"({p.confidence:.0%} confidence, breakout at ${p.breakout_level:.2f}). "
                "Geometry is rough — monitor, don't act on this alone."
            )
    if not lines:
        return ""
    return "**Chart Patterns** — " + " ".join(lines)


def _commentary_intraday(result: dict) -> str:
    intraday = result.get("intraday", {})
    if not intraday.get("available"):
        return ""

    parts: list[str] = ["On the 15-minute chart"]
    structure = intraday["structure"]
    if structure == "HIGHER_LOWS":
        parts.append(
            ", price is forming higher lows as it approaches the daily resistance — "
            "buyers are stepping up at progressively higher levels."
        )
    elif structure == "LOWER_HIGHS":
        parts.append(", price is making lower highs — intraday sellers are in control.")
    elif structure == "TRENDING_UP":
        parts.append(", price is trending up through the session.")
    elif structure == "TRENDING_DOWN":
        parts.append(", price is trending down through the session.")
    else:
        parts.append(", price is range-bound.")

    tail_parts: list[str] = []
    if intraday["intraday_squeeze"]:
        tail_parts.append(
            "The 15-minute ATR is compressing — an intraday squeeze is forming within the daily squeeze."
        )
    if intraday["vwap_position"] == "ABOVE" and intraday["vwap_value"] is not None:
        tail_parts.append(f"Price is above VWAP (${intraday['vwap_value']:.2f}) — institutional flow is supportive.")
    elif intraday["vwap_position"] == "BELOW" and intraday["vwap_value"] is not None:
        tail_parts.append(f"Price is below VWAP (${intraday['vwap_value']:.2f}) — session flow is negative.")

    if intraday["last_5d_pattern"] == "TIGHTENING":
        tail_parts.append("Daily ranges have been contracting over the last 5 sessions — the squeeze is intensifying.")
    elif intraday["last_5d_pattern"] == "EXPANDING":
        tail_parts.append("Daily ranges have been expanding over the last 5 sessions — volatility is rising.")

    header = "**Intraday Read** — " + "".join(parts).lstrip()
    if tail_parts:
        header += " " + " ".join(tail_parts)
    return header


def _commentary_entry(result: dict) -> str:
    eq = result["entry_quality"]
    grade = eq["grade"]
    if grade == "EXCELLENT":
        verdict = "Entry quality: **EXCELLENT**. The setup is fully formed and the trigger is imminent or active."
    elif grade == "GOOD":
        verdict = "Entry quality: **GOOD**. The setup is strong — watch for the trigger."
    elif grade == "WAIT":
        verdict = "Entry quality: **WAIT**. The setup is building but not ready yet."
    else:
        verdict = "Entry quality: **AVOID**. The technicals do not support an entry here."

    parts = [
        f"**Entry Assessment** — {verdict}",
        f"_Trigger:_ {eq['trigger']}",
        f"_Risk:_ {eq['risk_level']}",
        f"_Target:_ {eq['target']}",
    ]
    if eq.get("reasons_for"):
        parts.append("_Supporting:_ " + " · ".join(eq["reasons_for"]))
    if eq.get("reasons_against"):
        parts.append("_Against:_ " + " · ".join(eq["reasons_against"]))
    return "  \n".join(parts)


def _format_setup_line(s) -> str:
    """One bullet for a Setup record."""
    arrow = "▲" if s.direction == "BREAKOUT" else "▼"
    dist = s.distance_pct
    dist_str = f"{dist:+.2f}%"
    if s.distance_atr is not None:
        dist_str += f" ({s.distance_atr:.1f} ATR)"
    parts = [
        f"{arrow} **${s.trigger_price:.2f}** — {s.label}  ",
        f"_{s.proximity_class.title()}_ · {dist_str} · quality {s.quality:.0f} ({s.quality_label.title()})  ",
        f"_Trigger:_ {s.trigger_condition}",
    ]
    if s.invalidation_price is not None:
        parts.append(f"_Invalidation:_ ${s.invalidation_price:.2f}")
    if s.target_price is not None:
        parts.append(f"_Target:_ ${s.target_price:.2f}")
    if s.rationale:
        parts.append("_Why:_ " + "; ".join(s.rationale))
    if s.confluence_with:
        parts.append("_Confluence:_ " + ", ".join(s.confluence_with))
    return "  \n".join(parts)


def _format_triggered_line(s) -> str:
    """One bullet for a recently-triggered Setup record."""
    arrow = "▲" if s.direction == "BREAKOUT" else "▼"
    b = s.bars_since_trigger
    unit = "wk" if s.timeframe == "weekly" else "d"
    when = "today" if b == 0 else (
        f"1 {unit} ago" if b == 1 else f"{b} {unit} ago"
    )
    parts = [
        f"{arrow} **${s.trigger_price:.2f}** — {s.label} _(crossed {when})_  ",
        f"_{s.trigger_condition}_",
    ]
    if s.target_price is not None:
        parts.append(f"_Next target:_ ${s.target_price:.2f}")
    if s.rationale:
        parts.append("_Why:_ " + "; ".join(s.rationale))
    return "  \n".join(parts)


def _commentary_setup_map(result: dict, max_per_side: int = 5) -> str:
    """The headline output — every plausible future trigger, grouped by direction.

    Also surfaces recently-triggered levels at the top so the user knows what
    just fired (vs. what's still pending).
    """
    breakouts = result.get("breakouts", [])[:max_per_side]
    breakdowns = result.get("breakdowns", [])[:max_per_side]
    triggered = result.get("triggered", [])[:4]
    last = result["price"]["last"]
    ticker = result["ticker"]

    head = (
        f"### {ticker} setup map — current price ${last:.2f}\n\n"
        f"_All plausible future trigger levels ranked by proximity and quality._"
    )
    sections = [head]

    if triggered:
        lines = ["#### Recently triggered — watch for hold/follow-through"]
        for s in triggered:
            lines.append(_format_triggered_line(s))
        sections.append("\n\n".join(lines))

    if breakouts:
        lines = ["#### Breakout setups above"]
        for s in breakouts:
            lines.append(_format_setup_line(s))
        sections.append("\n\n".join(lines))
    else:
        sections.append("#### Breakout setups above\n\n_No actionable breakout levels within 20% above current price._")

    if breakdowns:
        lines = ["#### Breakdown setups below"]
        for s in breakdowns:
            lines.append(_format_setup_line(s))
        sections.append("\n\n".join(lines))
    else:
        sections.append("#### Breakdown setups below\n\n_No actionable breakdown levels within 20% below current price._")

    return "\n\n".join(sections)


def _commentary_current_state(result: dict) -> str:
    """One-paragraph 'where are we right now' summary — derived from existing
    deep layers but framed as context for the setup map, not as a verdict."""
    deep = result["deep"]
    signals = result["signals"]
    eq = result["entry_quality"]
    ticker = result["ticker"]
    last = result["price"]["last"]
    chg = result["price"]["chg_pct"]
    status = result.get("status", "")
    ma_stack = deep.get("ma_stack", "MIXED").replace("_", " ").lower()
    rs_trend = deep.get("rs_trend_20d", "FLAT").lower()
    consol = signals["consolidation"]

    parts = [
        f"**Current state** — {ticker} at ${last:.2f} ({chg:+.2f}% today). "
        f"MA stack {ma_stack}; 20-day RS trend {rs_trend}; detector status: {status}."
    ]
    if consol["signal"] != "NONE":
        parts.append(
            f"Daily Donchian range: ${consol['range_low']:.2f}-${consol['range_high']:.2f} "
            f"(ATR pctile {consol['atr_percentile']:.0f})."
        )
    parts.append(f"Setup-quality readout (legacy grade): **{eq['grade']}** ({eq['grade_score']}/100).")
    return "  \n".join(parts)


def generate_commentary(result: dict) -> str:
    """Build the full multi-section commentary for the analysis result dict.

    The headline is the setup map — every plausible future trigger, both
    directions, ranked. Supporting sections (position, trend, volume, momentum,
    patterns, intraday) follow as context.
    """
    if result.get("error"):
        return f"**Error** — {result['error']}"

    sections: list[str] = []
    sections.append(_commentary_setup_map(result))
    sections.append(_commentary_current_state(result))
    for fn in (
        _commentary_position,
        _commentary_trend,
        _commentary_volume,
        _commentary_momentum,
        _commentary_patterns,
        _commentary_intraday,
    ):
        s = fn(result)
        if s:
            sections.append(s)
    return "\n\n".join(sections)
