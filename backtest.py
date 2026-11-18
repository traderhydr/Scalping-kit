#!/usr/bin/env python3
"""Backtest the Scalping Kit v2.1 signals with the bot's trade rules.

Replays the same logic the bot uses live: signal on a closed candle -> entry 1 at that close,
stop beyond the pullback extreme, 4 take-profits (equal 25% slices), stop to breakeven after TP1.

Examples:
  python backtest.py --exchange binance --symbols BTCUSDT,ETHUSDT --tf 5m --days 30
  python backtest.py --exchange blofin --top 15 --tf 15m --days 60 --fee 0.06 --csv trades.csv
  python backtest.py --symbols BTCUSDT --tf 5m --days 30 --min-stop-pct 0.3   # test wider stops
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import exchanges
from indicator import Params, compute
from trade import build_plan


def simulate(df: pd.DataFrame, i: int, plan, be_after: int, max_candles: int, lev: float,
             fee_pct: float, slip_pct: float) -> dict:
    """Walk candles after signal bar i. Same rules as the live tracker (stop wins if a candle touches both)."""
    long_ = plan.side == "LONG"
    sign = 1 if long_ else -1
    entry, n = plan.entry, len(plan.targets)
    w = 1.0 / n
    hi, lo, cl = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    stop, be, idx, rem = plan.stop, False, 0, 1.0
    exits: list[tuple[float, float]] = []
    outcome, closed = None, None
    last = min(len(df) - 1, i + max_candles)
    for j in range(i + 1, last + 1):
        if (lo[j] <= stop) if long_ else (hi[j] >= stop):
            px = stop * (1 - slip_pct / 100 * sign)  # slippage makes the exit worse
            exits.append((rem, px))
            outcome, closed, rem = ("BE" if be else "SL"), j, 0.0
            break
        while idx < n and ((hi[j] >= plan.targets[idx]) if long_ else (lo[j] <= plan.targets[idx])):
            exits.append((w, plan.targets[idx]))
            rem -= w
            idx += 1
        if idx >= n:
            outcome, closed, rem = "ALL_TP", j, 0.0
            break
        if be_after and not be and idx >= be_after:
            be, stop = True, entry
    if rem > 1e-9:  # data ended or trade too old: close at last close
        exits.append((rem, cl[last]))
        outcome, closed = "TIMEOUT", last
    gross = sum(wt * sign * (px - entry) / entry * 100 for wt, px in exits)
    net = gross - 2 * fee_pct               # entry + exit fees on full notional, % of position
    risk_pct = abs(entry - plan.stop) / entry * 100
    return {"outcome": outcome, "tps": idx, "gross_pct": gross, "net_pct": net, "roi": net * lev,
            "risk_pct": risk_pct, "tp1_pct": abs(plan.targets[0] - entry) / entry * 100,
            "bars": closed - i, "close_i": closed}


def backtest_symbol(df: pd.DataFrame, a: argparse.Namespace, eval_from_ts: int) -> list[dict]:
    p = Params()
    ind = compute(df, p)
    trades, busy_until, last_sig_ts = [], {}, {}
    sec_ms = exchanges.TF_SECONDS[a.tf] * 1000
    for i in ind.index[ind["signal"] != 0]:
        i = int(i)
        if ind["ts"].iloc[i] < eval_from_ts or i >= len(ind) - 1:
            continue
        side = "LONG" if ind["signal"].iloc[i] == 1 else "SHORT"
        ts = int(ind["ts"].iloc[i])
        if busy_until.get(side, -1) >= i:                       # same coin+direction already open
            continue
        if ts - last_sig_ts.get(side, -10**18) < a.cooldown * 60_000:  # cooldown
            continue
        plan = build_plan(ind, i, p, tuple(a.rr), a.entry_step, min_stop_pct=a.min_stop_pct)
        r = simulate(ind, i, plan, a.be_after, a.max_candles, a.lev, a.fee, a.slip)
        busy_until[side] = r["close_i"]
        last_sig_ts[side] = ts
        trades.append({"ts": ts, "side": side, "entry": plan.entry, "stop": plan.stop, **r})
    return trades


def summarize(tr: pd.DataFrame, label: str) -> str:
    if tr.empty:
        return f"{label}: no trades"
    win = tr["net_pct"] > 0
    gp, gl = tr.loc[win, "roi"].sum(), -tr.loc[~win, "roi"].sum()
    cum = tr.sort_values("ts")["roi"].cumsum()
    dd = (cum.cummax() - cum).max()
    oc = tr["outcome"].value_counts().to_dict()
    return (f"{label}: {len(tr)} trades | win {win.mean() * 100:.0f}% | "
            f"outcomes {oc} | total ROI {tr['roi'].sum():+.1f}% | avg {tr['roi'].mean():+.2f}%/trade | "
            f"PF {gp / gl if gl else float('inf'):.2f} | max DD {dd:.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exchange", default="binance", choices=["binance", "blofin"])
    ap.add_argument("--symbols", default="", help="comma list, e.g. BTCUSDT,ETHUSDT")
    ap.add_argument("--top", type=int, default=0, help="use top-N coins by 24h volume instead")
    ap.add_argument("--tf", default="5m")
    ap.add_argument("--days", type=float, default=30)
    ap.add_argument("--lev", type=float, default=15)
    ap.add_argument("--fee", type=float, default=0.05, help="taker fee %% per side (Binance futures 0.05, BloFin 0.06)")
    ap.add_argument("--slip", type=float, default=0.02, help="slippage %% on stop exits")
    ap.add_argument("--rr", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.5])
    ap.add_argument("--entry-step", type=float, default=0.15)
    ap.add_argument("--be-after", type=int, default=1, help="move stop to breakeven after this TP (0=never)")
    ap.add_argument("--min-stop-pct", type=float, default=0.0, help="widen stops tighter than this %% from entry")
    ap.add_argument("--cooldown", type=int, default=60, help="minutes between signals, same coin+direction")
    ap.add_argument("--max-candles", type=int, default=500)
    ap.add_argument("--csv", default="", help="write all trades to this CSV")
    a = ap.parse_args()

    if a.top:
        pool = exchanges.top_symbols(a.exchange, a.top + 60)
        symbols = [s for s in pool if s[:-4] not in exchanges.DEFAULT_EXCLUDE][: a.top]
    else:
        symbols = [exchanges.normalize_symbol(s) for s in a.symbols.split(",") if s.strip()]
        symbols = [b + q for b, q in symbols]
    if not symbols:
        ap.error("give --symbols or --top")

    now_ms = pd.Timestamp.now("UTC").value // 1_000_000
    eval_from = now_ms - int(a.days * 86400_000)
    print(f"Backtest {a.exchange} {a.tf} last {a.days:g}d | lev {a.lev:g}x fee {a.fee}%/side slip {a.slip}% "
          f"| TP {a.rr} | BE after TP{a.be_after} | min stop {a.min_stop_pct}%\n")
    frames = []
    for sym in symbols:
        try:
            df = exchanges.fetch_history(a.exchange, sym, a.tf, a.days)
        except Exception as e:  # noqa: BLE001
            print(f"{sym}: data error: {e}", file=sys.stderr)
            continue
        if len(df) < Params().medium_ema + 50:
            print(f"{sym}: only {len(df)} candles, skipped", file=sys.stderr)
            continue
        t = pd.DataFrame(backtest_symbol(df, a, eval_from))
        if not t.empty:
            t.insert(0, "symbol", sym)
            frames.append(t)
        print(summarize(t, f"{sym:<12}"))
    if not frames:
        print("\nNo trades.")
        return 0
    allt = pd.concat(frames, ignore_index=True)
    print("\n" + "=" * 100)
    print(summarize(allt, "ALL"))
    for side in ("LONG", "SHORT"):
        print(summarize(allt[allt["side"] == side], f"  {side}"))
    fee_rt = 2 * a.fee
    print(f"\nMedian stop distance: {allt['risk_pct'].median():.2f}% of price "
          f"(= {allt['risk_pct'].median() * a.lev:.0f}% of margin at {a.lev:g}x)")
    print(f"Median TP1 distance:  {allt['tp1_pct'].median():.3f}%  vs round-trip fees {fee_rt:.2f}% "
          f"-> {(allt['tp1_pct'] < fee_rt).mean() * 100:.0f}% of trades have TP1 smaller than fees")
    print("ROI = % of margin at the chosen leverage, fixed margin per trade, not compounded.")
    if a.csv:
        allt.to_csv(a.csv, index=False)
        print(f"Trades written to {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
