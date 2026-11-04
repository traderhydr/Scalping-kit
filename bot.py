"""Telegram bot that scans Binance / BloFin candles with the Scalping PullBack Tool v2.1
logic and posts trade signals (entry, stop, targets).

Signals only - the bot never places orders.
"""
from __future__ import annotations

import asyncio
import html
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

import exchanges
from indicator import Params, compute
from trade import TradePlan, build_plan

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)  # its request URLs contain the bot token
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("scalpbot")

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
STATE_FILE = Path(os.environ.get("STATE_FILE", "state.json"))
BINANCE_MARKET = os.environ.get("BINANCE_MARKET", "futures")  # futures | spot
CHANNEL_ID = os.environ.get("CHANNEL_ID", "").strip()  # e.g. -1001234567890 or @mychannel
ALLOWED = {int(x) for x in os.environ.get("ALLOWED_CHAT_IDS", "").split(",") if x.strip()}
LEVERAGE = int(os.environ.get("LEVERAGE", 15))
MARGIN_MODE = os.environ.get("MARGIN_MODE", "Cross").capitalize()  # Cross | Isolated
ENTRY_STEP_R = float(os.environ.get("ENTRY_STEP_R", 0.15))  # spacing of entries 2/3, in units of risk
RR = tuple(float(x) for x in os.environ.get("TAKE_PROFIT_RR", "0.5,1,1.5,2.5").split(","))
PARAMS = Params(
    hilo_len=int(os.environ.get("PAC_LENGTH", 34)),
    fast_ema=int(os.environ.get("FAST_EMA", 89)),
    medium_ema=int(os.environ.get("MEDIUM_EMA", 200)),
    lookback=int(os.environ.get("PULLBACK_LOOKBACK", 3)),
    use_ha=os.environ.get("USE_HEIKIN_ASHI", "true").lower() == "true",
)
AUTO_COUNT = int(os.environ.get("AUTO_SCAN_COUNT", 0))            # 0 = off; e.g. 15
AUTO_TFS = [x.strip() for x in os.environ.get("AUTO_SCAN_TIMEFRAMES", "5m").split(",") if x.strip()]
AUTO_EXCHANGES = [x.strip() for x in os.environ.get("AUTO_SCAN_EXCHANGES", "binance").split(",") if x.strip()]
# Non-crypto / tokenised-stock / commodity perps that rank high on volume; override with AUTO_SCAN_EXCLUDE
AUTO_EXCLUDE = {x.strip().upper() for x in os.environ.get(
    "AUTO_SCAN_EXCLUDE", "SOXL,SOXS,SNDK,XAU,XAG,SKHYNIX,KORU,SPCX,MU,CL,NATGAS,BZ,TSLA,NVDA,AAPL,MSTR,COIN").split(",") if x.strip()}
QUIET_ENV = os.environ.get("QUIET_HOURS", "").strip()   # e.g. 22:00-06:00 (no new signals in this window)
QUIET_TZ_ENV = os.environ.get("QUIET_TZ", "UTC").strip()
AUTO_REFRESH = 3600  # seconds between re-ranking the top coins
SCAN_EVERY = 15      # seconds between scheduler ticks
CLOSE_GRACE = 3      # seconds to wait after a candle closes before fetching

# ---------------------------------------------------------------- state
state = {"watch": {}, "last_sig": {}, "open": [], "results": {"tp_hits": 0, "sl_hits": 0, "completed": 0}}  # watch: chat_id -> [[exchange, symbol, tf]]; last_sig: key -> bar_ts
stats = {"rounds": 0, "last_scan": 0.0, "last_scanned": 0, "last_errors": 0, "signals": 0, "markets": 0}
checked: dict[str, int] = {}           # key -> last closed bar_ts already evaluated (in-memory)


def load_state() -> None:
    if STATE_FILE.exists():
        state.update(json.loads(STATE_FILE.read_text()))
    if "quiet" not in state:  # first run: take the environment default
        try:
            a, b = parse_window(QUIET_ENV) if QUIET_ENV else (None, None)
            get_tz(QUIET_TZ_ENV)
            state["quiet"] = {"start": a, "end": b, "tz": QUIET_TZ_ENV} if a else None
        except Exception:  # noqa: BLE001
            log.warning("Invalid QUIET_HOURS/QUIET_TZ, ignoring")
            state["quiet"] = None
    state.setdefault("open", [])
    state.setdefault("results", {"tp_hits": 0, "sl_hits": 0, "completed": 0})


def save_state() -> None:
    STATE_FILE.write_text(json.dumps(state, indent=1))


auto_cache: dict[str, tuple[float, list[str]]] = {}  # exchange -> (fetched_at, symbols)


async def auto_markets() -> list[tuple[str, str, str]]:
    """Top-N coins by 24h volume on each configured exchange x timeframe."""
    out = []
    for ex in AUTO_EXCHANGES:
        ts, syms = auto_cache.get(ex, (0.0, []))
        if time.time() - ts > AUTO_REFRESH:
            try:
                pool = await asyncio.to_thread(exchanges.top_symbols, ex, AUTO_COUNT + 60, BINANCE_MARKET)
                syms = [x for x in pool if x[:-4] not in AUTO_EXCLUDE][:AUTO_COUNT]
                auto_cache[ex] = (time.time(), syms)
                log.info("auto-scan %s top %d: %s", ex, AUTO_COUNT, ",".join(syms))
            except Exception as e:  # noqa: BLE001
                log.warning("top symbols %s failed: %s", ex, e)
        out += [(ex, s, tf) for s in syms for tf in AUTO_TFS]
    return out


def parse_window(text: str) -> tuple[str, str]:
    a, b = text.split("-")
    for x in (a, b):
        datetime.strptime(x.strip(), "%H:%M")
    return a.strip(), b.strip()


def get_tz(name: str):
    return timezone.utc if name.upper() == "UTC" else ZoneInfo(name)


def in_quiet(now: float) -> bool:
    q = state.get("quiet")
    if not q:
        return False
    t = datetime.fromtimestamp(now, get_tz(q["tz"])).strftime("%H:%M")
    return q["start"] <= t < q["end"] if q["start"] <= q["end"] else (t >= q["start"] or t < q["end"])


def quiet_text() -> str:
    q = state.get("quiet")
    return f"{q['start']}-{q['end']} {q['tz']}" if q else "off"


def key(ex: str, sym: str, tf: str) -> str:
    return f"{ex}:{sym}:{tf}"


def canon(sym: str) -> str:
    b, q = exchanges.normalize_symbol(sym)
    return f"{b}{q}"


# ---------------------------------------------------------------- formatting
def fmt(x: float) -> str:
    """Plain decimal price, never scientific notation (UNI 7.5490, BTC 67123.40, PEPE 0.0000050551)."""
    a = abs(x)
    if a >= 1000:
        d = 2
    elif a >= 100:
        d = 3
    elif a >= 1:
        d = 4
    elif a > 0:
        d = min(12, int(-math.floor(math.log10(a))) + 4)  # keep 5 significant digits for sub-1 prices
    else:
        d = 4
    return f"{x:.{d}f}"


def signal_text(ex: str, sym: str, tf: str, plan: TradePlan, trend: int) -> str:
    base, quote = exchanges.normalize_symbol(sym)
    num = lambda xs: "\n".join(f"{n}) {fmt(v)}" for n, v in enumerate(xs, 1))  # noqa: E731
    return (
        f"#{html.escape(base)}/{html.escape(quote)}\n"
        f"{plan.side}\n"
        f"Timeframe: {tf}\n"
        f"Entry Targets:\n{num(plan.entries)}\n"
        f"Take-Profit Targets:\n{num(plan.targets)}\n"
        f"Stop Targets:\n{num([plan.stop])}\n"
        f"Leverage: {MARGIN_MODE} ({LEVERAGE}X)"
    )


# ---------------------------------------------------------------- core
def analyse(ex: str, sym: str, tf: str):
    df = exchanges.fetch(ex, sym, tf, 1000, BINANCE_MARKET)
    if len(df) < PARAMS.medium_ema + 10:
        raise exchanges.ExchangeError(f"Only {len(df)} candles available, need {PARAMS.medium_ema + 10}+")
    return compute(df, PARAMS)


MAX_TRADE_CANDLES = 500  # stop tracking a trade that has hit nothing after this many candles


def pct(trade: dict, price: float) -> float:
    e = trade["entries"][0]
    return (price - e) / e * 100 * (1 if trade["side"] == "LONG" else -1)


def roi_text(trade: dict, price: float) -> str:
    p = pct(trade, price)
    return f"{p:+.2f}% price · {p * LEVERAGE:+.1f}% ROI at {MARGIN_MODE} {LEVERAGE}X"


async def track_trades(ctx: ContextTypes.DEFAULT_TYPE, k: str, df) -> None:
    """Check candles after each open trade's signal for TP / SL hits and post updates as replies."""
    keep = []
    changed = False
    for t in state["open"]:
        if t["key"] != k:
            keep.append(t)
            continue
        base, quote = exchanges.normalize_symbol(t["symbol"])
        head = f"#{base}/{quote} {t['side']} · {k.split(':')[2]}"
        long_ = t["side"] == "LONG"
        closed = False
        for row in df[df["ts"] > t["last_ts"]].itertuples():
            t["last_ts"] = int(row.ts)
            t["candles"] += 1
            sl_hit = row.low <= t["stop"] if long_ else row.high >= t["stop"]
            new_hits = []
            idx = t["tps_hit"]
            while idx < len(t["targets"]) and (
                row.high >= t["targets"][idx] if long_ else row.low <= t["targets"][idx]
            ):
                new_hits.append(idx)
                idx += 1
            if not sl_hit:  # SL and TP in the same candle: can't know the order, assume the stop (conservative)
                t["tps_hit"] = idx
            else:
                new_hits = []
            msgs = []
            for n in new_hits:
                state["results"]["tp_hits"] += 1
                msgs.append(f"✅ {head}\nTP{n + 1} hit 🎯\n{roi_text(t, t['targets'][n])}")
            if sl_hit:
                state["results"]["sl_hits"] += 1
                after = f" (after TP{t['tps_hit']})" if t["tps_hit"] else ""
                msgs.append(f"🛑 {head}\nStop loss hit{after}\n{roi_text(t, t['stop'])}")
                closed = True
            elif t["tps_hit"] >= len(t["targets"]):
                msgs.append(f"🏆 {head}\nAll {len(t['targets'])} targets hit")
                state["results"]["completed"] += 1
                closed = True
            for m in msgs:
                for chat, mid in t["msgs"].items():
                    try:
                        await ctx.bot.send_message(chat, m, reply_to_message_id=mid, allow_sending_without_reply=True)
                    except Exception as e:  # noqa: BLE001
                        log.warning("tp/sl update to %s failed: %s", chat, e)
            if msgs:
                log.info("trade update %s %s: %s", k, t["side"], msgs[-1].splitlines()[1])
            changed = True
            if closed:
                break
        if not closed and t["candles"] < MAX_TRADE_CANDLES:
            keep.append(t)
        elif not closed:
            log.info("dropped stale trade %s", k)
    if changed or len(keep) != len(state["open"]):
        state["open"] = keep
        save_state()


async def scan_job(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    now = time.time()
    subs: dict[str, list[int]] = {}
    for chat, items in state["watch"].items():
        for ex, sym, tf in items:
            subs.setdefault(key(ex, sym, tf), []).append(int(chat))

    if AUTO_COUNT:
        recipients = [CHANNEL_ID] if CHANNEL_ID else [str(c) for c in ALLOWED]
        for ex, sym, tf in await auto_markets():
            lst = subs.setdefault(key(ex, sym, tf), [])
            for r in recipients:
                if r and r not in map(str, lst):
                    lst.append(r)

    watched = set(subs)
    for t in state["open"]:  # keep tracking trades even if the coin left the top-N list
        subs.setdefault(t["key"], [])
    stats["markets"] = len(watched)
    scanned = errors = 0
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
            errors += 1
            continue
        if int(df["ts"].iloc[-1]) < last_closed_open * 1000:
            continue  # exchange hasn't published the new candle yet, retry next tick
        checked[k] = int(df["ts"].iloc[-1])
        scanned += 1
        await track_trades(ctx, k, df)
        i = len(df) - 1
        if in_quiet(now):
            continue  # quiet hours: no new signals (open trades above are still tracked)
        if k not in watched or df["signal"].iloc[i] == 0 or state["last_sig"].get(k) == checked[k]:
            continue
        plan = build_plan(df, i, PARAMS, RR, ENTRY_STEP_R)
        state["last_sig"][k] = checked[k]
        save_state()
        text = signal_text(ex, sym, tf, plan, int(df["trend"].iloc[i]))
        targets = list(chats)
        if CHANNEL_ID and CHANNEL_ID not in map(str, targets):
            targets.append(CHANNEL_ID)  # every watched market's signals are also posted to the channel
        msg_ids = {}
        for chat in targets:
            try:
                m = await ctx.bot.send_message(chat, text, parse_mode="HTML")
                msg_ids[str(chat)] = m.message_id
            except Exception as e:  # noqa: BLE001
                log.warning("send to %s failed: %s", chat, e)
        state["open"].append({
            "key": k, "symbol": sym, "side": plan.side, "entries": plan.entries, "stop": plan.stop,
            "targets": plan.targets, "tps_hit": 0, "candles": 0, "last_ts": plan.bar_ts, "msgs": msg_ids,
        })
        save_state()
        stats["signals"] += 1
        log.info("signal %s %s", k, plan.side)

    stats["last_scan"], stats["last_errors"] = time.time(), errors
    if scanned or errors:
        stats["rounds"] += 1
        log.info("scan round: %d candles evaluated, %d errors, %d markets watched", scanned, errors, len(subs))


# ---------------------------------------------------------------- commands
async def cmd_quiet(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorised(update):
        return
    if not ctx.args:
        await update.message.reply_text(
            f"Quiet hours: {quiet_text()}\nSet: /quiet 22:00-06:00 [Timezone, e.g. Asia/Karachi]\nTurn off: /quiet off")
        return
    if ctx.args[0].lower() == "off":
        state["quiet"] = None
    else:
        try:
            a, b = parse_window(ctx.args[0])
            tz = ctx.args[1] if len(ctx.args) > 1 else (state.get("quiet") or {}).get("tz", QUIET_TZ_ENV)
            get_tz(tz)
        except Exception:  # noqa: BLE001
            await update.message.reply_text("❌ Use: /quiet 22:00-06:00 [Timezone]  (24h time, e.g. Asia/Dubai, Europe/London, UTC)")
            return
        state["quiet"] = {"start": a, "end": b, "tz": tz}
    save_state()
    await update.message.reply_text(f"✅ Quiet hours: {quiet_text()}")


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorised(update):
        return
    ago = f"{time.time() - stats['last_scan']:.0f}s ago" if stats["last_scan"] else "never"
    newest = max(checked.values(), default=0)
    qline = f"Quiet hours: {quiet_text()}" + (" (quiet now - no new signals)" if in_quiet(time.time()) else "") + "\n"
    auto = (f"Auto-scan: {AUTO_COUNT} coins x {','.join(AUTO_TFS)} on {','.join(AUTO_EXCHANGES)}"
            if AUTO_COUNT else "Auto-scan: off")
    await update.message.reply_text(
        f"Markets watched: {stats['markets']}\n"
        f"Candles evaluated: {len(checked)} markets up to date\n"
        f"Last candle processed: {time.strftime('%H:%M UTC', time.gmtime(newest / 1000)) if newest else 'none yet'}\n"
        f"Last scheduler tick: {ago}\n"
        f"Fetch errors last round: {stats['last_errors']}\n"
        f"Signals sent since start: {stats['signals']}\n"
        + qline +
        f"Open trades: {len(state['open'])} · TP hits: {state['results']['tp_hits']} · "
        f"SL hits: {state['results']['sl_hits']} · Fully completed: {state['results']['completed']}\n"
        + auto
    )


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
    "/status – is the scanner running\n"
    "/quiet 22:00-06:00 [tz] – no new signals in this window (/quiet off)\n"
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
        plan = build_plan(df, j, PARAMS, RR, ENTRY_STEP_R)
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
                     ("remove", cmd_remove), ("list", cmd_list), ("scan", cmd_scan), ("status", cmd_status), ("quiet", cmd_quiet)]:
        app.add_handler(CommandHandler(name, fn))
    app.job_queue.run_repeating(scan_job, interval=SCAN_EVERY, first=5)
    if AUTO_COUNT and not (CHANNEL_ID or ALLOWED):
        log.warning("AUTO_SCAN_COUNT set but neither CHANNEL_ID nor ALLOWED_CHAT_IDS - no one to send to")
    log.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
