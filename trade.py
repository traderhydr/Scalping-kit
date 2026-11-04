"""Turn an indicator signal into a trade plan (entry / stop / targets)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from indicator import Params


@dataclass
class TradePlan:
    side: str            # "LONG" | "SHORT"
    entries: list[float]  # [market entry, better-price scale-in 2, scale-in 3]
    stop: float
    targets: list[float]
    rr: list[float]
    bar_ts: int          # open time (ms) of the signal candle

    @property
    def entry(self) -> float:
        return self.entries[0]


def build_plan(df: pd.DataFrame, i: int, p: Params, rr: tuple[float, ...] = (0.5, 1.0, 1.5, 2.5), entry_step_r: float = 0.15,
               atr_len: int = 14, atr_buffer: float = 0.1) -> TradePlan:
    """df is the output of indicator.compute; i is the index of the signal bar.

    Entry = close of the signal candle (real price, not Heikin Ashi).
    Stop  = beyond the pullback extreme (lowest low / highest high over the
            lookback window) with a small ATR buffer.
    """
    side = int(df["signal"].iloc[i])
    assert side != 0, "bar has no signal"
    tr = np.maximum.reduce([
        (df["high"] - df["low"]).to_numpy(),
        (df["high"] - df["close"].shift()).abs().fillna(0).to_numpy(),
        (df["low"] - df["close"].shift()).abs().fillna(0).to_numpy(),
    ])
    atr = float(pd.Series(tr).rolling(atr_len, min_periods=1).mean().iloc[i])
    window = df.iloc[max(0, i - p.lookback - 1): i + 1]
    entry = float(df["close"].iloc[i])
    # Entry 1 = signal close; entries 2 and 3 are limit orders a fraction of the risk
    # deeper into the pullback (lower for LONG, higher for SHORT).
    if side == 1:
        stop = float(window["low"].min()) - atr_buffer * atr
        risk = entry - stop
        targets = [entry + r * risk for r in rr]
        entries = [entry - k * entry_step_r * risk for k in range(3)]
    else:
        stop = float(window["high"].max()) + atr_buffer * atr
        risk = stop - entry
        targets = [entry - r * risk for r in rr]
        entries = [entry + k * entry_step_r * risk for k in range(3)]
    return TradePlan("LONG" if side == 1 else "SHORT", entries, stop, targets, list(rr), int(df["ts"].iloc[i]))
