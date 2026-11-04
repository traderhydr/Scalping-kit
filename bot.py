"""Telegram bot that scans Binance / BloFin candles with the Scalping PullBack Tool v2.1
logic and posts trade signals (entry, stop, targets).

Signals only - the bot never places orders.
"""
from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import time
from pathlib import Path

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

import exchanges
from indicator import Params, compute
from trade import TradePlan, build_plan

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
log = logging.getLogger("scalpbot")

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
STATE_FILE = Path(os.environ.get("STATE_FILE", "state.json"))
BINANCE_MARKET = os.environ.get("BINANCE_MARKET", "futures")  # futures | spot
ALLOWED = {int(x) for x in os.environ.get("ALLOWED_CHAT_IDS", "").split(",") if x.strip()}
RR = tuple(float(x) for x in os.environ.get("TAKE_PROFIT_RR", "1,1.5,2").split(","))
PARAMS = Params(
    hilo_len=int(os.environ.get("PAC_LENGTH", 34)),
    fast_ema=int(os.environ.get("FAST_EMA", 89)),
    medium_ema=int(os.environ.get("MEDIUM_EMA", 200)),
    lookback=int(os.environ.get("PULLBACK_LOOKBACK", 3)),
    use_ha=os.environ.get("USE_HEIKIN_ASHI", "true").lower() == "true",
)
SCAN_EVERY = 15      # seconds between scheduler ticks
CLOSE_GRACE = 3      # seconds to wait after a candle closes before fetching

# ---------------------------------------------------------------- state
state = {"watch": {}, "last_sig": {}}  # watch: chat_id -> [[exchange, symbol, tf]]; last_sig: key -> bar_ts
checked: dict[str, int] = {}           # key -> last closed bar_ts already evaluated (in-memory)


def load_state() -> None:
    if STATE_FILE.exists():
        state.update(json.loads(STATE_FILE.read_text()))


def save_state() -> None:
    STATE_FILE.write_text(json.dumps(state, indent=1))


def key(ex: str, sym: str, tf: str) -> str:
    return f"{ex}:{sym}:{tf}"


def canon(sym: str) -> str:
    b, q = exchanges.normalize_symbol(sym)
    return f"{b}{q}"


# ---------------------------------------------------------------- formatting
def fmt(x: float) -> str:
    return f"{x:.6g}" if abs(x) < 1 else f"{x:,.4f}".rstrip("0").rstrip(".")


def signal_text(ex: str, sym: str, tf: str, plan: TradePlan, trend: int) -> str:
    icon = "🟢" if plan.side == "LONG" else "🔴"
    tn = {1: "Uptrend", -1: "Downtrend", 0: "Transition"}[trend]
    tps = "\n".join(f"  TP{n + 1}: <code>{fmt(t)}</code> ({r}R)" for n, (t, r) in enumerate(zip(plan.targets, plan.rr)))
    risk_pct = abs(plan.entry - plan.stop) / plan.entry * 100
    return (
        f"{icon} <b>{plan.side}</b> {html.escape(sym)} · {tf} · {ex.upper()}\n"
        f"Pullback recovery · {tn}\n\n"
        f"Entry: <code>{fmt(plan.entry)}</code>\n"
        f"Stop:  <code>{fmt(plan.stop)}</code> ({risk_pct:.2f}%)\n{tps}\n\n"
        f"<i>Signal only, not financial advice.</i>"
    )


# ---------------------------------------------------------------- core
def analyse(ex: str, sym: str, tf: str):
    df = exchanges.fetch(ex, sym, tf, 1000, BINANCE_MARKET)
    if len(df) < PARAMS.medium_ema + 10:
        raise exchanges.ExchangeError(f"Only {len(df)} candles available, need {PARAMS.medium_ema + 10}+")
    return compute(df, PARAMS)


async def scan_job(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    now = time.time()
    subs: dict[str, list[int]] = {}
    for chat, items in state["watch"].items():
        for ex, sym, tf in items:
            subs.setdefault(key(ex, sym, tf), []).append(int(chat))

    for k, chats in subs.items():
        ex, sym, tf = k.split(":")
        sec = exchanges.TF_SECONDS[tf]
        last_closed_open = int((now - CLOSE_GRACE) // sec) * sec - sec  # open time (s) of newest closed bar
        if checked.get(k) == last_closed_open * 1000:
            continue
        try:
            df = await asyncio.to_thread(analyse, ex, sym, tf)
        except Exception as e:  # noqa: BLE001
            log.warning("fetch %s failed: %s", k, e)
            continue
        if int(df["ts"].iloc[-1]) < last_closed_open * 1000:
            continue  # exchange hasn't published the new candle yet, retry next tick
        checked[k] = int(df["ts"].iloc[-1])
        i = len(df) - 1
        if df["signal"].iloc[i] == 0 or state["last_sig"].get(k) == checked[k]:
            continue
        plan = build_plan(df, i, PARAMS, RR)
        state["last_sig"][k] = checked[k]
        save_state()
        text = signal_text(ex, sym, tf, plan, int(df["trend"].iloc[i]))
        for chat in chats:
            await ctx.bot.send_message(chat, text, parse_mode="HTML")
        log.info("signal %s %s", k, plan.side)


# ---------------------------------------------------------------- commands
def authorised(update: Update) -> bool:
    return not ALLOWED or (update.effective_chat and update.effective_chat.id in ALLOWED)


def parse_args(args: list[str]):
    """<symbol> <tf> [exchange]"""
    if len(args) < 2:
        raise ValueError("Usage: <symbol> <timeframe> [binance|blofin]")
    sym, tf = canon(args[0]), args[1].lower()
    ex = args[2].lower() if len(args) > 2 else "binance"
    if tf not in exchanges.TF_SECONDS:
        raise ValueError(f"Timeframe must be one of: {', '.join(exchanges.TF_SECONDS)}")
    if ex not in ("binance", "blofin"):
        raise ValueError("Exchange must be binance or blofin")
    return ex, sym, tf


HELP = (
    "<b>Scalping Kit v2.1 bot</b>\n"
    "Pullback-recovery signals from the PAC(34) / EMA 89 / EMA 200 Pine script.\n\n"
    "/add &lt;symbol&gt; &lt;tf&gt; [binance|blofin] – watch a market\n"
    "/remove &lt;symbol&gt; &lt;tf&gt; [exchange] – stop watching\n"
    "/list – show watchlist\n"
    "/scan &lt;symbol&gt; &lt;tf&gt; [exchange] – analyse now\n"
    "/chatid – show this chat's id\n\n"
    "Example: <code>/add BTCUSDT 5m blofin</code>\n"
    f"Timeframes: {', '.join(exchanges.TF_SECONDS)}\n"
    "Signals fire when a candle closes. Signal only – no orders are placed."
)


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if authorised(update):
        await update.message.reply_html(HELP)


async def cmd_chatid(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(str(update.effective_chat.id))


async def cmd_add(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorised(update):
        return
    try:
        ex, sym, tf = parse_args(ctx.args)
        await asyncio.to_thread(exchanges.fetch, ex, sym, tf, 5, BINANCE_MARKET)  # validate market exists
    except Exception as e:  # noqa: BLE001
        await update.message.reply_text(f"❌ {e}")
        return
    items = state["watch"].setdefault(str(update.effective_chat.id), [])
    if [ex, sym, tf] not in items:
        items.append([ex, sym, tf])
        save_state()
    await update.message.reply_text(f"✅ Watching {sym} {tf} on {ex}")


async def cmd_remove(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorised(update):
        return
    try:
        ex, sym, tf = parse_args(ctx.args)
    except Exception as e:  # noqa: BLE001
        await update.message.reply_text(f"❌ {e}")
        return
    items = state["watch"].get(str(update.effective_chat.id), [])
    if [ex, sym, tf] in items:
        items.remove([ex, sym, tf])
        save_state()
        await update.message.reply_text(f"🗑 Removed {sym} {tf} on {ex}")
    else:
        await update.message.reply_text("Not in your watchlist.")


async def cmd_list(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorised(update):
        return
    items = state["watch"].get(str(update.effective_chat.id), [])
    await update.message.reply_text(
        "\n".join(f"• {s} {t} ({e})" for e, s, t in items) if items else "Watchlist is empty. Use /add."
    )


async def cmd_scan(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorised(update):
        return
    try:
        ex, sym, tf = parse_args(ctx.args)
        df = await asyncio.to_thread(analyse, ex, sym, tf)
    except Exception as e:  # noqa: BLE001
        await update.message.reply_text(f"❌ {e}")
        return
    i = len(df) - 1
    trend = {1: "Uptrend", -1: "Downtrend", 0: "Transition"}[int(df["trend"].iloc[i])]
    pos = "above PAC" if df["close"].iloc[i] > df["pacU"].iloc[i] else "below PAC" if df["close"].iloc[i] < df["pacL"].iloc[i] else "inside PAC"
    head = f"<b>{html.escape(sym)} {tf} · {ex.upper()}</b>\nTrend: {trend} · Price {pos} · Last close {fmt(float(df['close'].iloc[i]))}\n"
    recent = df.index[df["signal"] != 0]
    if len(recent):
        j = int(recent[-1])
        plan = build_plan(df, j, PARAMS, RR)
        ago = i - j
        head += f"\nLast signal: {ago} candle(s) ago\n"
        await update.message.reply_html(head + signal_text(ex, sym, tf, plan, int(df["trend"].iloc[j])))
    else:
        await update.message.reply_html(head + "\nNo signal in the loaded history.")


def main() -> None:
    if not TOKEN:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN (get one from @BotFather).")
    load_state()
    app = Application.builder().token(TOKEN).build()
    for name, fn in [("start", cmd_start), ("help", cmd_start), ("chatid", cmd_chatid), ("add", cmd_add),
                     ("remove", cmd_remove), ("list", cmd_list), ("scan", cmd_scan)]:
        app.add_handler(CommandHandler(name, fn))
    app.job_queue.run_repeating(scan_job, interval=SCAN_EVERY, first=5)
    log.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
