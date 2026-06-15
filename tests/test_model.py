import numpy as np
import pandas as pd
import pytest

from model import (
    LagFeatureEngineer,
    NumericFeatureSelector,
    TimeFeatureEngineer,
    compare_metrics,
    create_model_pipeline,
    create_model_pipeline_from_features,
    evaluate_predictions,
    predict_naive_last_value,
)


def test_time_feature_engineer_adds_expected_time_features_without_mutating_input():
    source = pd.DataFrame({"timestamp": ["2026-01-03 06:00:00"]})

    transformed = TimeFeatureEngineer().fit_transform(source)

    assert "hour" not in source.columns
    assert {
        "hour",
        "day_of_week",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
    }.issubset(transformed.columns)
    assert transformed.loc[0, "hour"] == 6
    assert transformed.loc[0, "day_of_week"] == 5
    assert transformed.loc[0, "is_weekend"] == 1
    assert np.isclose(transformed.loc[0, "hour_sin"], 1.0)
    assert np.isclose(transformed.loc[0, "hour_cos"], 0.0)
    assert np.isclose(transformed.loc[0, "day_of_week_sin"], np.sin(2 * np.pi * 5 / 7))
    assert np.isclose(transformed.loc[0, "day_of_week_cos"], np.cos(2 * np.pi * 5 / 7))


def test_lag_feature_engineer_preserves_order_and_adds_chronological_lags():
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=26, freq="h")
    source = pd.DataFrame(
        {
            "timestamp": timestamps,
            "congestion_score": np.arange(26, dtype=float),
        }
    ).sample(frac=1, random_state=42)

    transformed = LagFeatureEngineer(
        lag_columns=["congestion_score"], lag_hours=[1, 24]
    ).fit_transform(source)

    chronological_source = source.sort_values("timestamp").reset_index(drop=True)
    expected_by_timestamp = {
        row.timestamp: {
            "lag_1": np.nan if idx < 1 else float(idx - 1),
            "lag_24": np.nan if idx < 24 else float(idx - 24),
        }
        for idx, row in chronological_source.iterrows()
    }

    assert transformed["timestamp"].tolist() == source["timestamp"].tolist()
    assert transformed["congestion_score"].tolist() == source["congestion_score"].tolist()
    assert {"congestion_score_lag_1", "congestion_score_lag_24"}.issubset(
        transformed.columns
    )
    for row in transformed.itertuples():
        expected = expected_by_timestamp[row.timestamp]
        if np.isnan(expected["lag_1"]):
            assert np.isnan(row.congestion_score_lag_1)
        else:
            assert row.congestion_score_lag_1 == expected["lag_1"]
        if np.isnan(expected["lag_24"]):
            assert np.isnan(row.congestion_score_lag_24)
        else:
            assert row.congestion_score_lag_24 == expected["lag_24"]


def test_evaluate_predictions_returns_mae_and_rmse():
    metrics = evaluate_predictions(np.array([1.0, 2.0, 4.0]), np.array([1.0, 4.0, 1.0]))

    assert metrics["mae"] == pytest.approx(5 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(13 / 3))


def test_naive_last_value_uses_most_recent_available_value():
    train = pd.DataFrame({"congestion_score": [10.0, 20.0]})
    test = pd.DataFrame({"congestion_score": [25.0, 35.0, 40.0]})

    predictions = predict_naive_last_value(train, test)

    assert predictions["congestion_score"].tolist() == [20.0, 25.0, 35.0]


def test_compare_metrics_reports_model_vs_baseline_improvement():
    model_metrics = {"congestion_score": {"mae": 2.0, "rmse": 3.0}}
    baseline_metrics = {"congestion_score": {"mae": 5.0, "rmse": 4.0}}

    comparison = compare_metrics(model_metrics, baseline_metrics)

    mae_comparison = comparison["congestion_score"]["mae"]
    assert mae_comparison["model"] == 2.0
    assert mae_comparison["baseline"] == 5.0
    assert mae_comparison["improvement"] == 3.0
    assert mae_comparison["improvement_pct"] == pytest.approx(60.0)
    assert mae_comparison["model_better"] is True


def test_create_model_pipeline_from_features_rejects_unknown_model_type():
    with pytest.raises(ValueError, match="Unsupported model type"):
        create_model_pipeline_from_features("unsupported")
