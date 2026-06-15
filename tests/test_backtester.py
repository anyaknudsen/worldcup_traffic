import numpy as np
import pandas as pd
import pytest

import backtester as backtester_module
from backtester import WalkForwardBacktester
from traffic_client import TrafficAPIClient


def make_hourly_traffic(hours):
    timestamps = pd.date_range("2026-01-01", periods=hours, freq="h")
    congestion = np.linspace(10.0, 80.0, hours)
    return pd.DataFrame(
        {
            "timestamp": timestamps[::-1],
            "location_id": "Doha_QA",
            "congestion_score": congestion[::-1],
            "travel_time_mins": 20 + congestion[::-1] * 0.8,
            "speed_kph": 60 - congestion[::-1] * 0.4,
            "incident_count": np.zeros(hours, dtype=int),
        }
    )


def test_backtest_runs_expanding_windows_and_summarizes_metrics(monkeypatch):
    def fake_create_model_pipeline_from_features(model_type):
        return {"model_type": model_type}

    def fake_train_model(pipeline, X_train, y_train):
        pipeline["mean_target"] = float(y_train.mean())
        pipeline["train_columns"] = list(X_train.columns)
        return pipeline

    def fake_predict_model(pipeline, X):
        return np.full(len(X), pipeline["mean_target"])

    monkeypatch.setattr(
        backtester_module,
        "create_model_pipeline_from_features",
        fake_create_model_pipeline_from_features,
    )
    monkeypatch.setattr(backtester_module, "train_model", fake_train_model)
    monkeypatch.setattr(backtester_module, "predict_model", fake_predict_model)

    summary = WalkForwardBacktester(
        initial_train_days=1, test_days=1, model_type="gradient_boosting"
    ).backtest(make_hourly_traffic(72))

    assert summary["total_folds"] == 2
    assert summary["target_columns"] == ["congestion_score"]
    assert "model_metrics" in summary
    assert "baseline_metrics" in summary
    assert "comparison" in summary
    assert summary["mae_mean"] == pytest.approx(
        np.mean([fold["mae"] for fold in summary["fold_results"]])
    )
    assert summary["rmse_mean"] == pytest.approx(
        np.mean([fold["rmse"] for fold in summary["fold_results"]])
    )

    first_fold = summary["fold_results"][0]
    second_fold = summary["fold_results"][1]
    assert first_fold["fold"] == 1
    assert first_fold["train_size"] == 24
    assert first_fold["test_size"] == 24
    assert len(first_fold["predictions"]) == 24
    assert len(first_fold["baseline_predictions"]) == 24
    assert first_fold["baseline_predictions"][0] == pytest.approx(
        make_hourly_traffic(72)
        .sort_values("timestamp")
        .reset_index(drop=True)
        .loc[23, "congestion_score"]
    )
    assert second_fold["train_size"] == 48
    assert second_fold["test_size"] == 24
    assert first_fold["train_start"] < first_fold["train_end"] < first_fold["test_start"]


def test_backtest_reports_comparison_for_multiple_targets(monkeypatch):
    def fake_create_model_pipeline_from_features(model_type):
        return {"model_type": model_type}

    def fake_train_model(pipeline, X_train, y_train):
        pipeline["last_target"] = float(y_train.iloc[-1])
        return pipeline

    def fake_predict_model(pipeline, X):
        return np.full(len(X), pipeline["last_target"])

    monkeypatch.setattr(
        backtester_module,
        "create_model_pipeline_from_features",
        fake_create_model_pipeline_from_features,
    )
    monkeypatch.setattr(backtester_module, "train_model", fake_train_model)
    monkeypatch.setattr(backtester_module, "predict_model", fake_predict_model)

    summary = WalkForwardBacktester(initial_train_days=1, test_days=1).backtest(
        make_hourly_traffic(48),
        target_columns=["congestion_score", "travel_time_mins"],
    )

    assert summary["target_columns"] == ["congestion_score", "travel_time_mins"]
    assert set(summary["model_metrics"]) == {"congestion_score", "travel_time_mins"}
    assert set(summary["baseline_metrics"]) == {"congestion_score", "travel_time_mins"}
    assert set(summary["comparison"]) == {"congestion_score", "travel_time_mins"}

    first_fold = summary["fold_results"][0]
    assert list(first_fold["predictions"].columns) == [
        "congestion_score",
        "travel_time_mins",
    ]
    assert list(first_fold["baseline_predictions"].columns) == [
        "congestion_score",
        "travel_time_mins",
    ]
    assert first_fold["baseline_predictions"].iloc[0]["congestion_score"] == pytest.approx(
        make_hourly_traffic(48)
        .sort_values("timestamp")
        .reset_index(drop=True)
        .loc[23, "congestion_score"]
    )
    assert "improvement" in summary["comparison"]["congestion_score"]["mae"]
    assert "improvement_pct" in summary["comparison"]["travel_time_mins"]["rmse"]


def test_backtest_rejects_data_shorter_than_train_plus_test_window():
    backtester = WalkForwardBacktester(initial_train_days=1, test_days=1)

    with pytest.raises(ValueError, match="Need at least 48 hours"):
        backtester.backtest(make_hourly_traffic(47))


def test_backtest_produces_fold_on_ninety_days_of_hourly_mock_data(monkeypatch):
    monkeypatch.delenv("TRAFFIC_API_KEY", raising=False)
    np.random.seed(42)

    def fake_create_model_pipeline_from_features(model_type):
        return {"model_type": model_type}

    def fake_train_model(pipeline, X_train, y_train):
        pipeline["mean_target"] = float(y_train.mean())
        return pipeline

    def fake_predict_model(pipeline, X):
        return np.full(len(X), pipeline["mean_target"])

    monkeypatch.setattr(
        backtester_module,
        "create_model_pipeline_from_features",
        fake_create_model_pipeline_from_features,
    )
    monkeypatch.setattr(backtester_module, "train_model", fake_train_model)
    monkeypatch.setattr(backtester_module, "predict_model", fake_predict_model)

    start_time = pd.Timestamp("2026-01-01 00:00:00")
    end_time = start_time + pd.Timedelta(hours=90 * 24 - 1)
    data = TrafficAPIClient(api_key="").fetch_traffic_data(
        {"city": "Doha", "country": "QA"},
        start_time,
        end_time,
    )

    summary = WalkForwardBacktester(
        initial_train_days=30, test_days=7
    ).backtest(data)

    assert len(data) == 90 * 24
    assert summary["total_folds"] >= 1
    assert len(summary["fold_results"]) == summary["total_folds"]
    assert all(fold["train_size"] >= 30 * 24 for fold in summary["fold_results"])
    assert all(fold["test_size"] == 7 * 24 for fold in summary["fold_results"])
