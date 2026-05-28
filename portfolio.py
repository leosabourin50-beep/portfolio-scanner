"""Unified portfolio + positions + mute state, shared by the cron and the
intraday watcher.

Portfolio file format (one entry per line, in portfolio.txt or the runtime
file). Both forms are valid and may be mixed:

    NVDA                 # watch-only — opportunity alerts
    TSLA 100 @ 240.50    # owned — 100 shares, entry 240.50 (risk alerts)
    GOOGL 50 @ 150       # owned

Blank lines and `#` comments are ignored. This is fully backward compatible
with the old plain-ticker-per-line portfolio.txt.

Two-tier file resolution:
  • portfolio.txt (repo, baked into the image) is the SEED list.
  • <DATA_DIR>/portfolio_runtime.txt is the live list edited via Telegram
    commands. It only exists once you mutate the portfolio at runtime, and it
    then takes precedence (so live edits persist across restarts on the Fly
    volume). Delete it to fall back to portfolio.txt.

The SCAN_PORTFOLIO env var (comma/space list) overrides everything as a
watch-only list — handy for one-off local runs.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Persist runtime state on the Fly volume if mounted, else the working dir.
DATA_DIR = Path("/data") if Path("/data").is_dir() else Path(".")

SEED_PORTFOLIO_FILE = Path(os.environ.get("PORTFOLIO_FILE", "portfolio.txt"))
RUNTIME_PORTFOLIO_FILE = Path(os.environ.get("RUNTIME_PORTFOLIO_PATH",
                                             DATA_DIR / "portfolio_runtime.txt"))
MUTES_FILE = Path(os.environ.get("MUTES_PATH", DATA_DIR / "mutes.json"))

# "100 @ 240.50", "100@240.5", "100 @ $240.50"
_POS_RE = re.compile(r"^([0-9]*\.?[0-9]+)\s*@\s*\$?([0-9]*\.?[0-9]+)$")


@dataclass
class Position:
    ticker: str
    shares: float | None = None
    entry: float | None = None

    @property
    def is_owned(self) -> bool:
        return self.shares is not None and self.entry is not None

    def to_line(self) -> str:
        if self.is_owned:
            return f"{self.ticker} {self.shares:g} @ {self.entry:g}"
        return self.ticker


# ─────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────

def _parse_line(raw: str) -> Position | None:
    line = raw.split("#", 1)[0].strip()
    if not line:
        return None
    line = line.replace(",", " ")
    parts = line.split(None, 1)
    ticker = parts[0].upper()
    if not ticker:
        return None
    if len(parts) == 1:
        return Position(ticker=ticker)
    m = _POS_RE.match(parts[1].strip())
    if not m:
        # Unrecognized trailing junk — treat as watch-only rather than drop it.
        return Position(ticker=ticker)
    try:
        return Position(ticker=ticker, shares=float(m.group(1)), entry=float(m.group(2)))
    except ValueError:
        return Position(ticker=ticker)


def _parse_text(text: str) -> list[Position]:
    out: list[Position] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        pos = _parse_line(raw)
        if pos is None or pos.ticker in seen:
            continue
        seen.add(pos.ticker)
        out.append(pos)
    return out


def _active_file() -> Path | None:
    """The file the live portfolio is read from: runtime file if present,
    else the seed file, else None."""
    if RUNTIME_PORTFOLIO_FILE.exists():
        return RUNTIME_PORTFOLIO_FILE
    if SEED_PORTFOLIO_FILE.exists():
        return SEED_PORTFOLIO_FILE
    return None


# ─────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────

def load_positions() -> list[Position]:
    """Full portfolio as Position records (preserving file order)."""
    env = os.environ.get("SCAN_PORTFOLIO", "").strip()
    if env:
        toks = [t.strip().upper() for t in env.replace(",", " ").split() if t.strip()]
        seen: set[str] = set()
        out: list[Position] = []
        for t in toks:
            if t not in seen:
                seen.add(t)
                out.append(Position(ticker=t))
        return out
    f = _active_file()
    if f is None:
        return []
    try:
        return _parse_text(f.read_text())
    except OSError as e:
        print(f"[warn] could not read portfolio file {f}: {e}", file=sys.stderr)
        return []


def load_portfolio() -> list[str]:
    """Tickers only — backward-compatible drop-in for the old load_portfolio()."""
    return [p.ticker for p in load_positions()]


def positions_map() -> dict[str, Position]:
    return {p.ticker: p for p in load_positions()}


# ─────────────────────────────────────────────────────────────
# Mutation (writes the runtime file on the volume)
# ─────────────────────────────────────────────────────────────

def _write_positions(positions: list[Position]) -> None:
    RUNTIME_PORTFOLIO_FILE.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Live portfolio — edited via Telegram commands.\n"
        "# Format: TICKER  or  TICKER SHARES @ ENTRY\n"
    )
    body = "\n".join(p.to_line() for p in positions)
    RUNTIME_PORTFOLIO_FILE.write_text(header + body + "\n")


def add_entry(ticker: str, shares: float | None = None,
              entry: float | None = None) -> bool:
    """Add or update a ticker. Returns True if newly added, False if it was
    already present (in which case position data, if given, is updated)."""
    ticker = ticker.strip().upper()
    positions = load_positions()
    for p in positions:
        if p.ticker == ticker:
            if shares is not None:
                p.shares = shares
            if entry is not None:
                p.entry = entry
            _write_positions(positions)
            return False
    positions.append(Position(ticker=ticker, shares=shares, entry=entry))
    _write_positions(positions)
    return True


def remove_ticker(ticker: str) -> bool:
    """Remove a ticker. Returns True if it was present."""
    ticker = ticker.strip().upper()
    positions = load_positions()
    kept = [p for p in positions if p.ticker != ticker]
    if len(kept) == len(positions):
        return False
    _write_positions(kept)
    return True


# ─────────────────────────────────────────────────────────────
# Mute / snooze
# ─────────────────────────────────────────────────────────────

def load_mutes() -> dict[str, str]:
    """ticker -> "muted" (indefinite) or an ISO-8601 UTC timestamp (snooze
    until). Expired snoozes are pruned on load."""
    if not MUTES_FILE.exists():
        return {}
    try:
        data = json.loads(MUTES_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    now = datetime.now(timezone.utc)
    pruned = {}
    changed = False
    for tk, val in data.items():
        if val == "muted":
            pruned[tk] = val
            continue
        try:
            until = datetime.fromisoformat(val)
            if until.tzinfo is None:
                until = until.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            changed = True
            continue
        if until > now:
            pruned[tk] = val
        else:
            changed = True
    if changed:
        _write_mutes(pruned)
    return pruned


def _write_mutes(mutes: dict[str, str]) -> None:
    MUTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    MUTES_FILE.write_text(json.dumps(mutes, indent=2))


def is_muted(ticker: str, now: datetime | None = None) -> bool:
    ticker = ticker.upper()
    mutes = load_mutes()  # already prunes expired
    val = mutes.get(ticker)
    if val is None:
        return False
    if val == "muted":
        return True
    # Still-pending snooze (expired ones were pruned in load_mutes).
    return True


def set_mute(ticker: str) -> None:
    mutes = load_mutes()
    mutes[ticker.upper()] = "muted"
    _write_mutes(mutes)


def set_snooze(ticker: str, until: datetime) -> None:
    if until.tzinfo is None:
        until = until.replace(tzinfo=timezone.utc)
    mutes = load_mutes()
    mutes[ticker.upper()] = until.astimezone(timezone.utc).isoformat()
    _write_mutes(mutes)


def clear_mute(ticker: str) -> bool:
    mutes = load_mutes()
    if ticker.upper() in mutes:
        del mutes[ticker.upper()]
        _write_mutes(mutes)
        return True
    return False
