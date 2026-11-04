import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exchanges import normalize_symbol  # noqa: E402
from indicator import Params, compute  # noqa: E402
from trade import build_plan  # noqa: E402


def synthetic(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    # trending random walk with oscillation -> produces pullbacks
    close = 100 + np.cumsum(rng.normal(0.05, 0.6, n)) + 4 * np.sin(np.arange(n) / 25)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.random(n) * 0.4
    low = np.minimum(open_, close) - rng.random(n) * 0.4
    return pd.DataFrame({"ts": np.arange(n) * 300_000, "open": open_, "high": high, "low": low, "close": close})


def test_signals_are_valid_and_alternate_through_flat():
    out = compute(synthetic())
    assert (out["signal"] != 0).sum() > 0
    # a signal only occurs when previous state was flat and new state is non-flat
    prev = out["state"].shift().fillna(0)
    sig = out["signal"] != 0
    assert (prev[sig] == 0).all() and (out["state"][sig] == out["signal"][sig]).all()
    # buys only in uptrend, sells only in downtrend
    assert (out.loc[out["signal"] == 1, "trend"] == 1).all()
    assert (out.loc[out["signal"] == -1, "trend"] == -1).all()


def test_trade_plan_geometry():
    df = compute(synthetic())
    for i in df.index[df["signal"] != 0]:
        plan = build_plan(df, int(i), Params())
        assert len(plan.entries) == 3 and len(plan.targets) == 4
        if plan.side == "LONG":
            assert plan.stop < plan.entries[2] < plan.entries[1] < plan.entry < plan.targets[0] < plan.targets[-1]
        else:
            assert plan.stop > plan.entries[2] > plan.entries[1] > plan.entry > plan.targets[0] > plan.targets[-1]


def test_symbol_normalisation():
    for s in ("BTCUSDT", "btc/usdt", "BTC-USDT", "BTCUSDT.P"):
        assert normalize_symbol(s) == ("BTC", "USDT")
