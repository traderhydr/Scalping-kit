"""Python port of the "Scalping PullBack Tool v2.1" Pine Script (scalping_kit_v2.1.pine).

Only the logic that drives the Buy/Sell alert arrows is ported (plus fractals,
which are used for context). Plotting-only parts of the script are skipped.

Input DataFrame columns: ts (ms, bar open time), open, high, low, close.
Rows must be CLOSED candles, oldest first.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Params:
    hilo_len: int = 34          # High Low PAC channel Length
    fast_ema: int = 89
    medium_ema: int = 200
    lookback: int = 3           # Pullback Lookback for PAC Cross Check
    use_ha: bool = True         # Use Heikin Ashi candles in Algo calculations
    filter_bw: bool = False     # Show Ideal Fractals Only (regular fractals)


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty_like(ha_c)
    ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(ha_c)):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    ha_h = np.maximum.reduce([h, ha_o, ha_c])
    ha_l = np.minimum.reduce([l, ha_o, ha_c])
    return pd.DataFrame({"open": ha_o, "high": ha_h, "low": ha_l, "close": ha_c}, index=df.index)


def ema(s: pd.Series, length: int) -> pd.Series:
    # Pine ta.ema: seeded with the first value, alpha = 2/(len+1).
    return s.ewm(span=length, adjust=False).mean()


def _fractals(high: np.ndarray, low: np.ndarray, regular: bool):
    """Fractals flagged on the bar where they are *confirmed* (centre bar is t-2)."""
    n = len(high)
    top = np.zeros(n, bool)
    bot = np.zeros(n, bool)
    for t in range(4, n):
        h4, h3, h2, h1, h0 = high[t - 4], high[t - 3], high[t - 2], high[t - 1], high[t]
        l4, l3, l2, l1, l0 = low[t - 4], low[t - 3], low[t - 2], low[t - 1], low[t]
        if regular:
            top[t] = h4 < h3 < h2 > h1 > h0
            bot[t] = l4 > l3 > l2 < l1 < l0
        else:
            top[t] = h4 < h2 and h3 <= h2 and h2 >= h1 and h2 > h0
            bot[t] = l4 > l2 and l3 >= l2 and l2 <= l1 and l2 < l0
    return top, bot


def compute(df: pd.DataFrame, p: Params | None = None) -> pd.DataFrame:
    """Return df plus indicator columns.

    New columns: pacC, pacL, pacU, fastEMA, mediumEMA, trend (1/-1/0),
    state (TradeDirection 1/-1/0), signal (+1 new Buy, -1 new Sell, 0 none),
    top_fractal, bot_fractal.
    """
    p = p or Params()
    out = df.reset_index(drop=True).copy()
    src = heikin_ashi(out) if p.use_ha else out[["open", "high", "low", "close"]]
    ha_o, ha_c = src["open"].to_numpy(), src["close"].to_numpy()

    out["fastEMA"] = ema(src["close"], p.fast_ema)
    out["mediumEMA"] = ema(src["close"], p.medium_ema)
    out["pacC"] = ema(src["close"], p.hilo_len)
    out["pacL"] = ema(src["low"], p.hilo_len)
    out["pacU"] = ema(src["high"], p.hilo_len)

    fast, med = out["fastEMA"].to_numpy(), out["mediumEMA"].to_numpy()
    pacC, pacL, pacU = (out[k].to_numpy() for k in ("pacC", "pacL", "pacU"))

    trend = np.where((fast > med) & (pacL > med), 1, np.where((fast < med) & (pacU < med), -1, 0))
    out["trend"] = trend

    top, bot = _fractals(out["high"].to_numpy(float), out["low"].to_numpy(float), p.filter_bw)
    out["top_fractal"], out["bot_fractal"] = top, bot

    n = len(out)
    state = np.zeros(n, int)
    signal = np.zeros(n, int)
    last_below = None  # last index where haClose < pacC   (ta.barssince)
    last_above = None  # last index where haClose > pacC
    prev = 0
    for i in range(n):
        if ha_c[i] < pacC[i]:
            last_below = i
        if ha_c[i] > pacC[i]:
            last_above = i
        bs_below = (i - last_below) if last_below is not None else None
        bs_above = (i - last_above) if last_above is not None else None

        exit_u = ha_o[i] < pacU[i] and ha_c[i] > pacU[i] and bs_below is not None and bs_below <= p.lookback
        exit_l = ha_o[i] > pacL[i] and ha_c[i] < pacL[i] and bs_above is not None and bs_above <= p.lookback
        buy = trend[i] == 1 and exit_u
        sell = trend[i] == -1 and exit_l

        cur = prev
        if cur == 1 and ha_c[i] < pacC[i]:
            cur = 0
        elif cur == -1 and ha_c[i] > pacC[i]:
            cur = 0
        elif cur == 0 and buy:
            cur = 1
        elif cur == 0 and sell:
            cur = -1

        if prev == 0 and cur != 0:
            signal[i] = cur
        state[i] = cur
        prev = cur

    out["state"] = state
    out["signal"] = signal
    return out
