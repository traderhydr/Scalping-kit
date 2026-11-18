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


def _plan(side="LONG"):
    from trade import TradePlan
    if side == "LONG":
        return TradePlan("LONG", [100, 99.9, 99.8], 98.0, [101, 102, 103, 105], [1] * 4, 0)
    return TradePlan("SHORT", [100, 100.1, 100.2], 102.0, [99, 98, 97, 95], [1] * 4, 0)


def _run(rows, side="LONG", be_after=1, fee=0.0, slip=0.0):
    import backtest
    df = pd.DataFrame([{"ts": k, "high": h, "low": l, "close": (h + l) / 2} for k, (h, l) in enumerate([(100, 100)] + rows)])
    return backtest.simulate(df, 0, _plan(side), be_after, 100, 15, fee, slip)


def test_backtest_all_targets_hit():
    r = _run([(106, 100.5)])
    assert r["outcome"] == "ALL_TP" and r["tps"] == 4
    assert abs(r["gross_pct"] - (1 + 2 + 3 + 5) / 4) < 1e-9          # 25% slice at each TP


def test_backtest_breakeven_after_tp1():
    r = _run([(101.5, 100.2), (100.5, 99.9)])                          # TP1, then back to entry
    assert r["outcome"] == "BE" and r["tps"] == 1
    assert abs(r["gross_pct"] - 0.25 * 1.0) < 1e-9                     # only the TP1 slice banked


def test_backtest_stop_and_same_candle_rule():
    assert _run([(100.5, 97.9)])["outcome"] == "SL"
    assert _run([(101.5, 97.5)])["outcome"] == "SL"                    # stop wins if a candle touches both


def test_backtest_short_and_fees():
    r = _run([(100.2, 98.9)], side="SHORT", fee=0.05)
    assert r["tps"] == 1 and abs(r["net_pct"] - (0.25 - 0.10)) < 1e-9
