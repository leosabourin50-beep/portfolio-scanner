"""
Polygon.io data adapter — drop-in replacement for the yfinance fetch in app.py

SETUP:
  1. pip install polygon-api-client
  2. Get your API key at https://polygon.io/dashboard/api-keys
  3. Set it as an environment variable:
       export POLYGON_API_KEY="your_key_here"
     Or create a .env file in the project root:
       POLYGON_API_KEY=your_key_here
"""

import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from polygon import RESTClient


_CLIENT: RESTClient | None = None


def get_client() -> RESTClient:
    """Cached Polygon client — single TLS connection pool reused across calls."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("POLYGON_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
    if not api_key:
        raise ValueError(
            "POLYGON_API_KEY not found. Set it as an env var or in a .env file.\n"
            "Get your key at https://polygon.io/dashboard/api-keys"
        )
    # Aggressive timeouts + zero internal retries.
    # The polygon-api-client retries 3x by default on every error — so a 5s
    # timeout becomes 4 × 5s = 20s. We do our own retry at fetch_data level.
    _CLIENT = RESTClient(
        api_key, connect_timeout=3.0, read_timeout=5.0,
        retries=0, num_pools=20,
    )
    return _CLIENT


def fetch_data(ticker: str, days: int = 365) -> pd.DataFrame:
    """
    Fetch daily OHLCV from Polygon.io.
    Index: DatetimeIndex; Columns: Open, High, Low, Close, Volume.
    Retries once on transient timeout/network failure.
    """
    client = get_client()
    end = datetime.now()
    start = end - timedelta(days=days)
    last_err = None
    for attempt in range(2):
        try:
            aggs = client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=start.strftime("%Y-%m-%d"),
                to=end.strftime("%Y-%m-%d"),
                limit=50000,
                adjusted=True,
            )
            if not aggs:
                return pd.DataFrame()
            rows = [{
                "Date": pd.Timestamp(bar.timestamp, unit="ms"),
                "Open": bar.open, "High": bar.high, "Low": bar.low,
                "Close": bar.close, "Volume": bar.volume,
            } for bar in aggs]
            df = pd.DataFrame(rows).set_index("Date").sort_index()
            df.index = pd.to_datetime(df.index)
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = df[col].astype(float)
            return df
        except Exception as e:
            last_err = e
            continue
    print(f"[Polygon] {ticker} failed after retry: {last_err}")
    return pd.DataFrame()


def fetch_benchmark(days: int = 365) -> pd.DataFrame:
    return fetch_data("SPY", days)


def warm_up(days: int = 500) -> None:
    """Pre-pay the polygon cold-start cost so the first user-triggered scan is fast.
    Fetches a small set of tickers with the same `days` parameter the real scan uses,
    so the connection pool is fully warm including for large payloads.
    """
    from concurrent.futures import ThreadPoolExecutor
    seeds = ["SPY", "AAPL", "MSFT", "NVDA", "AMZN"]
    with ThreadPoolExecutor(max_workers=5) as ex:
        list(ex.map(lambda t: fetch_data(t, days=days), seeds))


def fetch_intraday(
    ticker: str,
    multiplier: int = 15,
    timespan: str = "minute",
    days: int = 20,
) -> pd.DataFrame:
    """
    Fetch intraday bars from Polygon.

    Args:
        ticker: Stock symbol
        multiplier: Bar size (15 = 15-minute bars)
        timespan: "minute", "hour"
        days: How many calendar days back to pull

    Returns:
        DataFrame with DatetimeIndex (UTC), columns: Open, High, Low, Close, Volume.
        Empty DataFrame on error.
    """
    try:
        client = get_client()
        end = datetime.now()
        start = end - timedelta(days=days)

        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=start.strftime("%Y-%m-%d"),
            to=end.strftime("%Y-%m-%d"),
            limit=50000,
            adjusted=True,
        )

        if not aggs:
            return pd.DataFrame()

        rows = [{
            "Date": pd.Timestamp(bar.timestamp, unit="ms", tz="UTC"),
            "Open": bar.open, "High": bar.high, "Low": bar.low,
            "Close": bar.close, "Volume": bar.volume,
        } for bar in aggs]

        df = pd.DataFrame(rows).set_index("Date").sort_index()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = df[col].astype(float)
        return df

    except Exception as e:
        print(f"[Polygon] Error fetching intraday {ticker}: {e}")
        return pd.DataFrame()


def get_snapshot_all(tickers: list[str]) -> dict[str, dict]:
    """One grouped snapshot call for the whole portfolio's current state.

    Returns ticker -> {last, day_open, day_high, day_low, day_close,
    day_volume, prev_close, change_pct, min_close, min_volume, min_ts}. The
    intraday watcher uses this as a cheap per-poll pre-filter: instead of N
    per-ticker 5-min fetches, it pulls every ticker's price/volume in one
    request and only fetches 5-min bars for the names actually in play.

    Returns {} on any failure (e.g. plan without snapshot access) so callers
    fall back to the per-ticker path.
    """
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not tickers:
        return {}
    client = get_client()
    try:
        snaps = client.get_snapshot_all("stocks", tickers=tickers)
    except Exception as e:
        print(f"[Polygon] snapshot_all failed: {e}")
        return {}

    out: dict[str, dict] = {}
    for s in snaps or []:
        try:
            tk = getattr(s, "ticker", None)
            if not tk:
                continue
            day = getattr(s, "day", None)
            minute = getattr(s, "min", None)
            prev = getattr(s, "prev_day", None)
            last_trade = getattr(s, "last_trade", None)

            def _g(obj, *names):
                for n in names:
                    v = getattr(obj, n, None) if obj is not None else None
                    if v is not None:
                        try:
                            return float(v)
                        except (TypeError, ValueError):
                            return None
                return None

            day_close = _g(day, "close")
            min_close = _g(minute, "close")
            last_px = _g(last_trade, "price", "p") or min_close or day_close
            min_ts_raw = getattr(minute, "timestamp", None) if minute is not None else None
            min_ts = None
            if min_ts_raw:
                try:
                    # Polygon minute snapshot timestamps are ms since epoch.
                    min_ts = datetime.fromtimestamp(float(min_ts_raw) / 1000.0, tz=timezone.utc)
                except (TypeError, ValueError, OSError):
                    min_ts = None

            out[tk.upper()] = {
                "last": last_px,
                "day_open": _g(day, "open"),
                "day_high": _g(day, "high"),
                "day_low": _g(day, "low"),
                "day_close": day_close,
                "day_volume": _g(day, "volume"),
                "prev_close": _g(prev, "close"),
                "change_pct": _g(s, "todays_change_percent"),
                "min_close": min_close,
                "min_volume": _g(minute, "volume"),
                "min_ts": min_ts,
            }
        except Exception:
            continue
    return out
