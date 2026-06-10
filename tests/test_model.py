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
    assert transformed.loc[0, "hour"] == 6
    assert transformed.loc[0, "day_of_week"] == 5
    assert transformed.loc[0, "is_weekend"] == 1
    assert np.isclose(transformed.loc[0, "hour_sin"], 1.0)
    assert np.isclose(transformed.loc[0, "hour_cos"], 0.0)
    assert np.isclose(transformed.loc[0, "day_of_week_sin"], np.sin(2 * np.pi * 5 / 7))
    assert np.isclose(transformed.loc[0, "day_of_week_cos"], np.cos(2 * np.pi * 5 / 7))


def test_lag_feature_engineer_preserves_input_order_and_adds_chronological_lags():
    source = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01 02:00", "2026-01-01 00:00", "2026-01-01 01:00"]
            ),
            "congestion_score": [30.0, 10.0, 20.0],
        }
    )

    transformed = LagFeatureEngineer(
        lag_columns=["congestion_score"], lag_hours=[1, 2]
    ).fit_transform(source)

    assert transformed["timestamp"].tolist() == source["timestamp"].tolist()
    assert transformed["congestion_score"].tolist() == [30.0, 10.0, 20.0]
    assert transformed.loc[0, "congestion_score_lag_1"] == 20.0
    assert np.isnan(transformed.loc[1, "congestion_score_lag_1"])
    assert transformed.loc[2, "congestion_score_lag_1"] == 10.0
    assert transformed.loc[0, "congestion_score_lag_2"] == 10.0
    assert np.isnan(transformed.loc[2, "congestion_score_lag_2"])


def test_lag_feature_engineer_transform_uses_history_without_test_targets():
    train = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="h"),
            "congestion_score": [10.0, 20.0, 30.0],
        }
    )
    test = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 03:00", periods=2, freq="h"),
            "congestion_score": [1000.0, 2000.0],
        }
    )

    engineer = LagFeatureEngineer(
        lag_columns=["congestion_score"], lag_hours=[1]
    )
    engineer.fit_transform(train)
    transformed = engineer.transform(test)

    assert transformed.loc[0, "congestion_score_lag_1"] == 30.0
    assert np.isnan(transformed.loc[1, "congestion_score_lag_1"])
    assert transformed["congestion_score"].tolist() == [1000.0, 2000.0]


def test_lag_feature_engineer_handles_timezone_aware_unsorted_timestamps():
    source = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 02:00:00+03:00",
                    "2026-01-01 00:00:00+03:00",
                    "2026-01-01 01:00:00+03:00",
                ]
            ),
            "congestion_score": [30.0, 10.0, 20.0],
        }
    )

    transformed = LagFeatureEngineer(
        lag_columns=["congestion_score"], lag_hours=[1]
    ).fit_transform(source)

    assert transformed["timestamp"].tolist() == source["timestamp"].tolist()
    assert transformed["congestion_score_lag_1"].tolist()[0] == 20.0
    assert np.isnan(transformed.loc[1, "congestion_score_lag_1"])
    assert transformed.loc[2, "congestion_score_lag_1"] == 10.0


def test_numeric_feature_selector_aligns_prediction_columns_to_training_columns():
    selector = NumericFeatureSelector()
    train = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    predict = pd.DataFrame({"b": [5.0], "c": [6.0]})

    selector.fit(train)
    transformed = selector.transform(predict)

    assert transformed.columns.tolist() == ["a", "b"]
    assert np.isnan(transformed.loc[0, "a"])
    assert transformed.loc[0, "b"] == 5.0


def test_model_pipeline_predicts_when_target_lag_columns_are_missing_and_nan():
    train = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=8, freq="h")[::-1],
            "location_id": "Doha_QA",
            "congestion_score": np.linspace(10.0, 30.0, 8)[::-1],
            "travel_time_mins": np.linspace(20.0, 30.0, 8)[::-1],
        }
    )
    predict = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 08:00", periods=2, freq="h"),
            "location_id": "Doha_QA",
            "travel_time_mins": [999.0, 1000.0],
        }
    )
    pipeline = create_model_pipeline("gradient_boosting")

    pipeline.fit(train, train["congestion_score"])
    predictions = pipeline.predict(predict)

    assert len(predictions) == 2
    assert np.isfinite(predictions).all()


def test_evaluate_predictions_returns_mae_and_rmse():
    metrics = evaluate_predictions(np.array([1.0, 2.0, 4.0]), np.array([1.0, 4.0, 1.0]))

    assert metrics["mae"] == pytest.approx(5 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(13 / 3))


def test_create_model_pipeline_from_features_rejects_unknown_model_type():
    with pytest.raises(ValueError, match="Unsupported model type"):
        create_model_pipeline_from_features("unsupported")
