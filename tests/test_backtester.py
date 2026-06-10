import numpy as np
import pandas as pd
import pytest

import backtester as backtester_module
from backtester import WalkForwardBacktester


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
    assert second_fold["train_size"] == 48
    assert second_fold["test_size"] == 24
    assert first_fold["train_start"] < first_fold["train_end"] < first_fold["test_start"]


def test_backtest_rejects_data_shorter_than_train_plus_test_window():
    backtester = WalkForwardBacktester(initial_train_days=1, test_days=1)

    with pytest.raises(ValueError, match="Need at least 48 hours"):
        backtester.backtest(make_hourly_traffic(47))


def test_backtest_does_not_use_test_targets_as_features(monkeypatch):
    data = make_hourly_traffic(48)
    data = data.sort_values("timestamp").reset_index(drop=True)
    data.loc[24:, "congestion_score"] = np.arange(1000.0, 1024.0)
    data.loc[24:, "travel_time_mins"] = 20 + data.loc[24:, "congestion_score"] * 0.8
    test_targets = set(data.loc[24:, "congestion_score"])
    captured_X_test = []

    def fake_create_model_pipeline_from_features(model_type):
        return {"model_type": model_type}

    def fake_train_model(pipeline, X_train, y_train):
        pipeline["mean_target"] = float(y_train.mean())
        return pipeline

    def fake_predict_model(pipeline, X):
        captured_X_test.append(X.copy())
        return np.full(len(X), pipeline["mean_target"])

    monkeypatch.setattr(
        backtester_module,
        "create_model_pipeline_from_features",
        fake_create_model_pipeline_from_features,
    )
    monkeypatch.setattr(backtester_module, "train_model", fake_train_model)
    monkeypatch.setattr(backtester_module, "predict_model", fake_predict_model)

    summary = WalkForwardBacktester(initial_train_days=1, test_days=1).backtest(data)

    assert summary["total_folds"] == 1
    X_test = captured_X_test[0]
    feature_values = set(pd.Series(X_test.to_numpy().ravel()).dropna())
    assert not test_targets.intersection(feature_values)
    assert "congestion_score" not in X_test.columns
    assert "travel_time_mins" not in X_test.columns
    assert "speed_kph" not in X_test.columns
