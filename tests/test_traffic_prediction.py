import numpy as np
import pandas as pd
import pytest

from backtester import WalkForwardBacktester
from main import run_simple_prediction
from model import LagFeatureEngineer, TimeFeatureEngineer
from traffic_client import TrafficAPIClient


def synthetic_traffic_data(periods=72):
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="h")
    hours = np.arange(periods)
    congestion = 35 + 10 * np.sin(2 * np.pi * hours / 24) + 0.1 * hours

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "location_id": "test_location",
            "congestion_score": congestion,
            "travel_time_mins": 20 + congestion * 0.8,
            "speed_kph": 60 - congestion * 0.4,
            "incident_count": (hours % 31 == 0).astype(int),
        }
    )


def test_mock_traffic_generation_returns_expected_columns_and_ranges():
    np.random.seed(42)
    client = TrafficAPIClient()

    df = client.fetch_traffic_data(
        {"city": "Doha", "country": "QA"},
        pd.Timestamp("2026-01-01 00:00:00"),
        pd.Timestamp("2026-01-01 05:00:00"),
    )

    assert list(df.columns) == [
        "timestamp",
        "location_id",
        "congestion_score",
        "travel_time_mins",
        "speed_kph",
        "incident_count",
    ]
    assert len(df) == 6
    assert df["location_id"].unique().tolist() == ["Doha_QA"]
    assert df["congestion_score"].between(0, 100).all()
    assert df["speed_kph"].between(20, 60).all()
    assert set(df["incident_count"].unique()).issubset({0, 1})


def test_time_feature_engineering_adds_expected_time_columns():
    df = pd.DataFrame({"timestamp": ["2026-01-05 06:00:00"]})

    transformed = TimeFeatureEngineer().fit_transform(df)

    assert transformed.loc[0, "hour"] == 6
    assert transformed.loc[0, "day_of_week"] == 0
    assert transformed.loc[0, "is_weekend"] == 0
    assert transformed.loc[0, "hour_sin"] == pytest.approx(1.0)
    assert transformed.loc[0, "hour_cos"] == pytest.approx(0.0, abs=1e-12)
    assert transformed.loc[0, "day_of_week_sin"] == pytest.approx(0.0)
    assert transformed.loc[0, "day_of_week_cos"] == pytest.approx(1.0)


def test_lag_feature_engineering_sorts_by_timestamp_and_adds_lags():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01 02:00:00", "2026-01-01 00:00:00", "2026-01-01 01:00:00"]
            ),
            "congestion_score": [30, 10, 20],
        }
    )

    transformed = LagFeatureEngineer(
        lag_columns=["congestion_score"],
        lag_hours=[1, 2],
    ).fit_transform(df)

    assert transformed["congestion_score"].tolist() == [10, 20, 30]
    assert np.isnan(transformed.loc[0, "congestion_score_lag_1"])
    assert transformed.loc[1, "congestion_score_lag_1"] == 10
    assert transformed.loc[2, "congestion_score_lag_1"] == 20
    assert np.isnan(transformed.loc[1, "congestion_score_lag_2"])
    assert transformed.loc[2, "congestion_score_lag_2"] == 10


def test_simple_train_predict_flow_returns_predictions_and_metrics():
    data = synthetic_traffic_data(periods=60)

    result = run_simple_prediction(data)

    assert set(result["metrics"]) == {"mae", "rmse"}
    assert result["metrics"]["mae"] >= 0
    assert result["metrics"]["rmse"] >= 0
    assert len(result["predictions"]) == len(result["actuals"]) == 12
    assert np.isfinite(result["predictions"]).all()


def test_backtesting_runs_on_small_synthetic_dataset():
    data = synthetic_traffic_data(periods=96)
    backtester = WalkForwardBacktester(initial_train_days=2, test_days=1)

    summary = backtester.backtest(data)

    assert summary["total_folds"] == 2
    assert summary["mae_mean"] >= 0
    assert summary["rmse_mean"] >= 0
    assert len(summary["fold_results"]) == 2
    assert all(len(fold["predictions"]) == 24 for fold in summary["fold_results"])
    assert all(np.isfinite(fold["predictions"]).all() for fold in summary["fold_results"])
