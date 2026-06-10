import numpy as np
import pandas as pd
import pytest

from model import (
    LagFeatureEngineer,
    NumericFeatureSelector,
    TimeFeatureEngineer,
    create_model_pipeline,
    create_model_pipeline_from_features,
    evaluate_predictions,
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


def test_lag_feature_engineer_adds_one_and_twenty_four_hour_lags_chronologically():
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

    assert transformed["timestamp"].tolist() == timestamps.tolist()
    assert transformed["congestion_score"].tolist() == list(np.arange(26, dtype=float))
    assert {"congestion_score_lag_1", "congestion_score_lag_24"}.issubset(
        transformed.columns
    )
    assert np.isnan(transformed.loc[0, "congestion_score_lag_1"])
    assert transformed.loc[1, "congestion_score_lag_1"] == 0.0
    assert transformed.loc[25, "congestion_score_lag_1"] == 24.0
    assert transformed.loc[:23, "congestion_score_lag_24"].isna().all()
    assert transformed.loc[24, "congestion_score_lag_24"] == 0.0
    assert transformed.loc[25, "congestion_score_lag_24"] == 1.0


def test_evaluate_predictions_returns_mae_and_rmse():
    metrics = evaluate_predictions(np.array([1.0, 2.0, 4.0]), np.array([1.0, 4.0, 1.0]))

    assert metrics["mae"] == pytest.approx(5 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(13 / 3))


def test_create_model_pipeline_from_features_rejects_unknown_model_type():
    with pytest.raises(ValueError, match="Unsupported model type"):
        create_model_pipeline_from_features("unsupported")
