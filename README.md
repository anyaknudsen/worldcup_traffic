# Traffic Prediction System

A small Python project for generating mock traffic data, engineering time-series
features, training traffic prediction models, and evaluating them with
expanding-window walk-forward backtesting. The backtest reports model
performance against a naive "last observed value" baseline for
`congestion_score` and `travel_time_mins`.

## Files

- `traffic_client.py` - mock traffic data generation and placeholder API client.
- `model.py` - time features, lag features, model pipelines, train/predict
  helpers, naive baseline predictions, metric comparison, and feature
  importances for tree models.
- `backtester.py` - expanding-window walk-forward backtesting.
- `main.py` - command-line orchestration for simple prediction and backtesting.
- `tests/` - pytest coverage for generation, feature engineering, training, and
  backtesting.

## Installation

```bash
pip install -r requirements.txt
```

## Run tests

```bash
pytest
```

## Usage

Run the default walk-forward backtest with generated mock data:

```bash
python main.py
```

Run only the simple chronological train/test prediction:

```bash
python main.py --mode simple --days 10
```

Run a smaller backtest:

```bash
python main.py --mode backtest --days 10 --initial-train 5 --test-window 2
```

Backtest a single target:

```bash
python main.py --mode backtest --targets congestion_score
```

Save output CSV files:

```bash
python main.py --save
```

## Notes

By default, `TrafficAPIClient` uses the configured TomTom key to fetch a live
traffic snapshot. You can override it with the `TRAFFIC_API_KEY` environment
variable, or pass `api_key=""` to `TrafficAPIClient` to use mock traffic data.
Live TomTom data is current-state data, so the client repeats that snapshot
across the requested hourly timestamps for compatibility with the prediction
pipeline.
