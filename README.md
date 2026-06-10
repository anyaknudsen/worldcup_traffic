# Traffic Prediction System

A small Python project for generating mock traffic data, engineering time-series
features, training a traffic prediction model, and evaluating it with
expanding-window walk-forward backtesting.

## Files

- `traffic_client.py` - mock traffic data generation and placeholder API client.
- `model.py` - time features, lag features, model pipelines, train/predict
  helpers, and metrics.
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

Run both simple prediction and backtesting with generated mock data:

```bash
python main.py
```

Run only the simple train/test prediction:

```bash
python main.py --mode simple --days 10
```

Run a smaller backtest:

```bash
python main.py --mode backtest --days 10 --initial-train 5 --test-window 2
```

Save output CSV files:

```bash
python main.py --save
```

## Notes

When no `TRAFFIC_API_KEY` environment variable is set, the project uses mock
traffic data. Live API access is intentionally not implemented; no external
credentials or APIs are required to run the project or its tests.
