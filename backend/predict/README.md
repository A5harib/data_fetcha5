# Short-horizon price prediction

Predicts where the mid-price will be N seconds ahead. Default 30s.

## What it outputs

```json
{
  "ready": true,
  "direction": "UP",
  "score": 0.42,
  "expected_move_pct": 0.018,
  "confidence": 0.21,
  "horizon_sec": 30,
  "model_ic": 0.031
}
```

`score` is in units of 60-second realized volatility. `expected_move_pct` is
that converted back into a percentage at the current volatility, which is the
number worth reading. `direction` is `FLAT` when the model has no opinion, and
`ready` is false when there is no trained model or not enough live history.

## Set expectations first

At a 30-second horizon, price is close to a coin flip. A real information
coefficient here is 0.02 to 0.06. That is a genuine result, not a weak one.
An IC above 0.15 almost always means data leaked from the future into the
features, and `evaluate.py` will say so in its verdict rather than
congratulating you.

Fees are the other wall. Binance taker is 4bp a side, so a round trip costs
about 10bp with slippage. A signal has to clear that before it is worth
trading, and most signals at this horizon do not. `evaluate.py` subtracts
costs so this is visible instead of assumed.

## Running it

Install what is missing first. Only numpy is required for the default ridge
model and every self-check; xgboost is needed only for `--kind gbm`.

```bash
pip install -r ../requirements.txt
```

**1. Record.** Historical klines carry no order book, so training data has to
be captured live.

```bash
python -m predict.recorder --symbol BTCUSDT --hours 24
```

Writes one snapshot per second to `backend/data/BTCUSDT_snapshots.jsonl`.
24 hours gives about 86,000 samples. A few hours is enough for a first look.

**2. Train and score.**

```bash
python -m predict.train --symbol BTCUSDT --horizon 30
```

Prints recording health, then a walk-forward scoreboard against three
baselines, then a verdict. The model is only saved if walk-forward IC is
positive and it beats the incumbent on held-out recent data.

Score without saving anything:

```bash
python -m predict.train --symbol BTCUSDT --horizon 30 --no-save
```

**3. Serve.** In `main.py`:

```python
from predict.api import router as predict_router, attach_recorder
app.include_router(predict_router)
```

Then `GET /api/predict/BTCUSDT`, `/status`, `/features`, or the once-per-second
websocket at `/api/predict/ws/BTCUSDT`.

## Checking it yourself

```bash
python -m predict.test_features   # no future data reaches the features
python -m predict.evaluate        # scoreboard finds planted edges, not noise
python -m predict.model           # weights, frozen scaler, swap guard, round-trip
python -m predict.test_pipeline   # end to end on synthetic markets
```

`test_pipeline` is the one that matters most. It runs a market with a real
planted edge and a pure random walk through the same code, and requires the
pipeline to find the first and report nothing on the second.

## Stream names, and a bug worth knowing about

Binance's `@aggTrade` stream returns **no messages** on the futures combined
endpoint used here, while `@depth5@100ms` works normally. Verified live:
`@aggTrade` alone delivered 0 messages in 15 seconds, `@trade` delivered 840.

That combination fails in the worst way. Depth keeps arriving, so the
connection looks healthy, but trade history stays empty and `build_snapshot`
returns None forever. The recorder runs all night and writes nothing, with no
error in any log.

Both this recorder and `backend/engine/binance_stream.py` now use `@trade`.
The recorder also carries a watchdog that logs an explicit error if 500 depth
messages arrive with zero trades, and `test_pipeline.py` pins the behaviour so
depth-only input can never silently produce fabricated features.

If you see zero snapshots, check the trade stream first.

## How this differs from `backend/ml/`

The older reversal model has problems this design exists to avoid.

| Old | Here |
|---|---|
| Label built from chained `shift`/`rolling`, hard to verify | One explicit lookup, with a test that rebuilds every feature from past-only data |
| Scaler refit on every retrain, so thresholds drifted | Scaler fit once and frozen into the model file |
| Sample weights by row position | Weights by wall-clock age, 6h half-life |
| Retrained on the last 2 days only | Historical base stays in the mix |
| Any retrained model swapped in and overwrote the file | Candidate must beat the incumbent on holdout, or it is rejected |
| Save in place, so a crash corrupted the model | Write temp, then rename |
| Trained on synthetic data if Binance was unreachable | Refuses to train, reports `NO_MODEL` |
| Trained on a volume proxy, predicted on real book data | Same feature code for training and live |
| No backtest | Walk-forward with embargo, baselines, and fees |

The old system still runs. Nothing here modifies it.

## When the verdict says there is no signal

That is an answer. These 15 features may not predict 30-second returns, and
finding that out in a day of recording beats discovering it in live trading.
Options, roughly in order of expected value:

- Try a longer horizon (`--horizon 60` or `120`). Short horizons are dominated
  by bid-ask bounce.
- Record more data. A few hours spans one market regime.
- Add features with real microstructure content: full book depth rather than
  top 5, trade size distribution, queue position, funding rate.
- Accept the result. Most short-horizon signals do not survive fees, and a
  display-only indicator is still useful.
