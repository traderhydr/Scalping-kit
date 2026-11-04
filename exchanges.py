"""Public market-data clients for Binance and BloFin (no API keys needed)."""
from __future__ import annotations

import re
import time

import pandas as pd
import requests

TF_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400,
}
BLOFIN_BAR = {"1h": "1H", "2h": "2H", "4h": "4H", "1d": "1D"}  # others are identical

BINANCE_HOSTS = {
    "futures": "https://fapi.binance.com/fapi/v1/klines",
    "spot": "https://api.binance.com/api/v3/klines",
}
BLOFIN_URL = "https://openapi.blofin.com/api/v1/market/candles"
QUOTES = ("USDT", "USDC", "BUSD", "USD")


class ExchangeError(Exception):
    pass


def normalize_symbol(raw: str) -> tuple[str, str]:
    """'btc/usdt', 'BTC-USDT', 'BTCUSDT', 'BTCUSDT.P' -> ('BTC', 'USDT')."""
    s = raw.upper().strip()
    s = re.sub(r"\.P$|PERP$", "", s)
    parts = [x for x in re.split(r"[-/_:]", s) if x]
    if len(parts) == 2:
        return parts[0], parts[1]
    s = "".join(parts)
    for q in QUOTES:
        if s.endswith(q) and len(s) > len(q):
            return s[: -len(q)], q
    raise ExchangeError(f"Can't parse symbol '{raw}'. Use e.g. BTCUSDT.")


def _frame(rows: list[list[float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"]).astype(float)
    df["ts"] = df["ts"].astype("int64")
    return df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)


def _drop_unclosed(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    now_ms = int(time.time() * 1000)
    return df[df["ts"] + TF_SECONDS[tf] * 1000 <= now_ms].reset_index(drop=True)


def fetch_binance(symbol: str, tf: str, limit: int = 1000, market: str = "futures") -> pd.DataFrame:
    base, quote = normalize_symbol(symbol)
    order = [market] + [m for m in BINANCE_HOSTS if m != market]
    last_err = None
    for m in order:  # fall back to the other market if blocked / symbol missing
        try:
            r = requests.get(
                BINANCE_HOSTS[m],
                params={"symbol": base + quote, "interval": tf, "limit": limit},
                timeout=15,
            )
            if r.status_code != 200:
                raise ExchangeError(f"Binance {m} HTTP {r.status_code}: {r.text[:120]}")
            rows = [[k[0], k[1], k[2], k[3], k[4]] for k in r.json()]
            if rows:
                return _drop_unclosed(_frame(rows), tf)
        except (requests.RequestException, ExchangeError, ValueError) as e:
            last_err = e
    raise ExchangeError(str(last_err))


def fetch_blofin(symbol: str, tf: str, limit: int = 1000) -> pd.DataFrame:
    base, quote = normalize_symbol(symbol)
    try:
        r = requests.get(
            BLOFIN_URL,
            params={"instId": f"{base}-{quote}", "bar": BLOFIN_BAR.get(tf, tf), "limit": limit},
            timeout=15,
        )
        body = r.json()
    except (requests.RequestException, ValueError) as e:
        raise ExchangeError(f"BloFin request failed: {e}")
    if body.get("code") != "0":
        raise ExchangeError(f"BloFin error: {body.get('msg') or body}")
    rows = [[c[0], c[1], c[2], c[3], c[4]] for c in body.get("data", [])]
    if not rows:
        raise ExchangeError("BloFin returned no candles")
    return _drop_unclosed(_frame(rows), tf)


def fetch(exchange: str, symbol: str, tf: str, limit: int = 1000, binance_market: str = "futures") -> pd.DataFrame:
    if tf not in TF_SECONDS:
        raise ExchangeError(f"Unsupported timeframe '{tf}'. Use one of: {', '.join(TF_SECONDS)}")
    if exchange == "binance":
        return fetch_binance(symbol, tf, limit, binance_market)
    if exchange == "blofin":
        return fetch_blofin(symbol, tf, limit)
    raise ExchangeError(f"Unknown exchange '{exchange}' (use binance or blofin)")
