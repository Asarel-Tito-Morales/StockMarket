# Stock Sentiment Agent

Production-style Python agent that:
1. pulls stock news from a configurable provider
2. scores sentiment
3. pulls stock prices from Yahoo Finance
4. aligns news to 30-minute stock buckets
5. trains a deep learning classifier for post-news price direction
6. stores outputs to CSV
7. optionally runs continuously on a schedule

## Install
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run once
```bash
python main.py --config json.cfg
```

## Run continuously
Set `CONTINUOUS_RUN` to `true` and `RUN_EVERY_MINUTES` to `5` in `json.cfg`, then run:
```bash
python main.py --config json.cfg
```

## Notes
- For NewsAPI, set `NEWS_SOURCE` = `newsapi`.
- For Alpha Vantage news, set `NEWS_SOURCE` = `alphavantage`.
- Retraining cadence is controlled separately by `MODEL_RETRAIN_EVERY_HOURS`.
- Timestamp alignment uses the `MATCHING_RULE` field:
  - `floor`: 10:07 -> 10:00, 10:31 -> 10:30
  - `nearest`: 10:07 -> 10:00, 10:22 -> 10:30
