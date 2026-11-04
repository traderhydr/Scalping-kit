# Scalping Kit Telegram Bot

Telegram bot that runs a Python port of the **Scalping PullBack Tool v2.1** Pine Script
(`scalping_kit_v2.1.pine`) on Binance and BloFin candles and posts trade signals
(entry, stop, TP1–3). It is **signal only** – it never places orders and needs no exchange API keys
(public market data only).

## How it works
- `exchanges.py` – public klines from Binance (futures, falls back to spot) and BloFin.
- `indicator.py` – port of the script: Heikin Ashi, PAC(34) EMA of high/low/close, EMA 89/200 trend filter,
  pullback-recovery Buy/Sell state machine (same as the green/maroon arrows).
- `trade.py` – entry = signal candle close; stop = pullback extreme ± 0.1 ATR; targets at 1R / 1.5R / 2R.
- `bot.py` – Telegram commands + a scheduler that checks each watched market right after every candle close
  (signals are only generated on closed candles, like the script's "closed candles only" option).

## Run
```bash
pip install -r requirements.txt
export TELEGRAM_BOT_TOKEN=...   # create one with @BotFather
python bot.py
```
Set `CHANNEL_ID` to also post every signal to a Telegram channel (add the bot as channel admin with "Post messages").
Auto-scan: set `AUTO_SCAN_COUNT=15` (plus `AUTO_SCAN_TIMEFRAMES`, `AUTO_SCAN_EXCHANGES`) to scan the top coins by 24h volume; signals go to `CHANNEL_ID`.
Commands: `/add BTCUSDT 5m blofin`, `/remove`, `/list`, `/scan ETHUSDT 15m binance`, `/chatid`.
Optional settings are in `.env.example` (export them as env vars).

## Test
```bash
pytest
```

## Notes
- Verify the port against TradingView on a few historical signals before trusting it; small differences can
  come from EMA warm-up and differing Heikin Ashi seed bars.
- Backtest the stop/target rules yourself – they are sensible defaults, not part of the original script.
- Binance futures may be geo-blocked on some hosts; the bot then falls back to spot data.
