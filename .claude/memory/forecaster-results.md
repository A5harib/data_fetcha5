---
name: forecaster-results
description: Out-of-sample results of the backend/forecast models (BTC trained 2026-10-15, XAU eval only)
metadata:
  type: project
---

Every number below comes from walk-forward testing on data the model never saw. BTC has 16 quarterly test blocks, 2023-01 to 2026-10, each trained only on data before it.

## BTCUSDT, next 60 minutes (saved model)
| What | Result |
|---|---|
| Up/down calls | 18,414 independent calls, **right 55.5%**. Always-up scores 50.7%. |
| Up/down by quarter | Beat always-up in **16/16 quarters**. Worst quarter 52.3%. |
| Up/down trend over time | 57-59% in 2023, about 53-56% in 2025-26 |
| Size of wins vs losses | Avg win +0.32%, avg loss -0.39%, so about +0.006% per call before fees |
| Fees | Taker round trip is about 0.12% and maker about 0.04%, so the calls lose money if traded |
| 80% price range | Caught the real price 77-81% of the time in every quarter, calm or wild markets and weekends |
| Range model vs textbook (HAR) | Explains 44% of how big the next hour's move is, against 33% for HAR. Better in 16/16 quarters. |
| 9-in-10 floor | Price ended above it 88-90% of the time, every quarter |

## XAUUSDT, next 60 minutes (evaluated, not saved)
- **History:** about 10 months, since the 2025-12-11 listing. Folds used 3 months of training and 1-month tests, 7 folds in total.
- **Range model:** explains 59% against 50% for HAR, and won 5 of 7 folds. It lost the first two, which had the least training data.
- **Up/down:** the old tree model failed. The snap-back rule was never run on XAU before the pause. Note that the states study found nothing that holds for gold.

## The old xgboost up/down model (replaced)
It was right 50.5% overall and 51.3% on its most confident 10%, which is a coin flip. Trees and ridge regression fit the *average* move, and that has no stable sign. See [[research-findings]].

## Live example (2026-10-14 20:51 UTC)
BTC was at $81,624. The 80% range was $81,289 to $81,961 (±0.41%), and there was no up/down call because price was not stretched.
