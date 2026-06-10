"""Expanding-window walk-forward backtesting for traffic prediction."""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from model import (
    create_train_test_feature_sets,
    create_model_pipeline_from_features,
    evaluate_predictions_by_target,
    get_feature_importances,
    predict_naive_last_value,
    predict_model,
    select_forecast_features,
    sort_time_series,
    train_model,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class WalkForwardBacktester:
    """Run expanding-window walk-forward backtests."""

    def __init__(
        self,
        initial_train_days: int = 30,
        test_days: int = 7,
        model_type: str = "random_forest",
    ):
        """
        Initialize the backtester.

        Args:
            initial_train_days: Initial training window size in days.
            test_days: Number of days to predict in each iteration.
            model_type: Model type to use (``random_forest`` or
                ``gradient_boosting``).
        """
        self.initial_train_days = initial_train_days
        self.test_days = test_days
        self.model_type = model_type
        self.history = []

    def backtest(
        self,
        data: pd.DataFrame,
        target_column: str = "congestion_score",
        target_columns=None,
        timestamp_column: str = "timestamp",
    ) -> Dict:
        """
        Run expanding-window walk-forward backtesting.

        Args:
            data: Complete dataset with timestamp and features.
            target_column: Name of the target column to predict.
            target_columns: Optional list of target columns to predict. When
                provided, this takes precedence over ``target_column``.
            timestamp_column: Name of the timestamp column.

        Returns:
            Dictionary containing fold results and summary metrics.
        """
        data = sort_time_series(data, timestamp_column=timestamp_column)

        initial_train_hours = self.initial_train_days * 24
        test_hours = self.test_days * 24

        if len(data) < initial_train_hours + test_hours:
            raise ValueError(
                "Not enough data. Need at least "
                f"{initial_train_hours + test_hours} hours, but got {len(data)} hours."
            )

        end_train_idx = initial_train_hours
        fold_results = []
        fold = 0

        while end_train_idx + test_hours <= len(data):
            fold += 1
            logger.info("--- Fold %s ---", fold)

            combined_end_idx = end_train_idx + test_hours
            train_data = data.iloc[:end_train_idx].reset_index(drop=True)
            test_data = data.iloc[end_train_idx:combined_end_idx].reset_index(drop=True)
            logger.info(
                "Train/test data period: %s to %s",
                train_data[timestamp_column].iloc[0],
                test_data[timestamp_column].iloc[-1],
            )

            train_featured, test_featured = create_train_test_feature_sets(
                train_data,
                test_data,
                timestamp_column=timestamp_column,
            )
            logger.info(
                "Featured train/test shapes: %s / %s",
                train_featured.shape,
                test_featured.shape,
            )

            logger.info(
                "Training period: %s to %s",
                train_featured[timestamp_column].iloc[0],
                train_featured[timestamp_column].iloc[-1],
            )
            logger.info(
                "Test period: %s to %s",
                test_featured[timestamp_column].iloc[0],
                test_featured[timestamp_column].iloc[-1],
            )
            logger.info(
                "Training samples: %s, Test samples: %s",
                len(train_featured),
                len(test_featured),
            )

            X_train = select_forecast_features(
                train_featured,
                target_column=target_column,
                timestamp_column=timestamp_column,
            )
            y_train = train_featured[target_column]
            X_test = select_forecast_features(
                test_featured,
                target_column=target_column,
                timestamp_column=timestamp_column,
            )
            y_test = test_featured[target_column]

            model_pipeline = create_model_pipeline_from_features(self.model_type)
            trained_pipeline = train_model(model_pipeline, X_train, y_train)
            y_pred = predict_model(trained_pipeline, X_test)

            logger.info(
                "Fold %s - %s MAE: %.2f (baseline %.2f), RMSE: %.2f (baseline %.2f)",
                fold,
                target_columns[0],
                metrics[target_columns[0]]["mae"],
                baseline_metrics[target_columns[0]]["mae"],
                metrics[target_columns[0]]["rmse"],
                baseline_metrics[target_columns[0]]["rmse"],
            )

            primary_target = target_columns[0]
            fold_result = {
                "fold": fold,
                "target_columns": target_columns,
                "train_start": train_featured[timestamp_column].iloc[0],
                "train_end": train_featured[timestamp_column].iloc[-1],
                "test_start": test_featured[timestamp_column].iloc[0],
                "test_end": test_featured[timestamp_column].iloc[-1],
                "train_size": len(train_featured),
                "test_size": len(test_featured),
                "metrics": metrics,
                "baseline_metrics": baseline_metrics,
                "comparison": comparison,
                "feature_importances": feature_importances,
                "mae": metrics[primary_target]["mae"],
                "rmse": metrics[primary_target]["rmse"],
                "baseline_mae": baseline_metrics[primary_target]["mae"],
                "baseline_rmse": baseline_metrics[primary_target]["rmse"],
                "mae_improvement": comparison[primary_target]["mae"]["improvement"],
                "rmse_improvement": comparison[primary_target]["rmse"]["improvement"],
                "predictions": self._format_target_output(predictions, target_columns),
                "baseline_predictions": self._format_target_output(
                    baseline_predictions,
                    target_columns,
                ),
                "actuals": self._format_target_output(y_test, target_columns),
            }
            fold_results.append(fold_result)
            end_train_idx = combined_end_idx

        self.history = fold_results
        return self._summarize(fold_results, target_columns)

    def _resolve_target_columns(self, target_column, target_columns):
        if target_columns is None:
            if target_column is None:
                target_columns = DEFAULT_TARGET_COLUMNS
            elif isinstance(target_column, (list, tuple)):
                target_columns = list(target_column)
            else:
                target_columns = [target_column]
        else:
            target_columns = list(target_columns)

        if not target_columns:
            raise ValueError("At least one target column is required.")
        return target_columns

    def _format_target_output(self, values, target_columns):
        if len(target_columns) == 1:
            return values[target_columns[0]].values
        return values.reset_index(drop=True)

    def _summarize_metric_group(self, fold_results, group_name, target_columns):
        summary = {}
        for target in target_columns:
            summary[target] = {}
            for metric_name in ["mae", "rmse"]:
                values = [
                    result[group_name][target][metric_name] for result in fold_results
                ]
                summary[target][f"{metric_name}_mean"] = np.mean(values)
                summary[target][f"{metric_name}_std"] = np.std(values)
                summary[target][f"{metric_name}_min"] = np.min(values)
                summary[target][f"{metric_name}_max"] = np.max(values)
        return summary

    def _summarize_feature_importances(self, fold_results, target_columns):
        summary = {}
        for target in target_columns:
            importances_by_feature = {}
            for result in fold_results:
                for item in result["feature_importances"].get(target, []):
                    importances_by_feature.setdefault(item["feature"], []).append(
                        item["importance"]
                    )

            ranked = sorted(
                (
                    {"feature": feature, "importance": float(np.mean(values))}
                    for feature, values in importances_by_feature.items()
                ),
                key=lambda item: item["importance"],
                reverse=True,
            )
            summary[target] = ranked[:10]
        return summary

    def _summarize(self, fold_results, target_columns):
        if not fold_results:
            return {
                "total_folds": 0,
                "target_columns": target_columns,
                "mae_mean": None,
                "mae_std": None,
                "mae_min": None,
                "mae_max": None,
                "rmse_mean": None,
                "rmse_std": None,
                "rmse_min": None,
                "rmse_max": None,
                "model_metrics": {},
                "baseline_metrics": {},
                "comparison": {},
                "feature_importances": {},
                "fold_results": [],
            }

        model_metrics = self._summarize_metric_group(
            fold_results,
            "metrics",
            target_columns,
        )
        baseline_metrics = self._summarize_metric_group(
            fold_results,
            "baseline_metrics",
            target_columns,
        )
        comparison = compare_metrics(
            {
                target: {
                    "mae": model_metrics[target]["mae_mean"],
                    "rmse": model_metrics[target]["rmse_mean"],
                }
                for target in target_columns
            },
            {
                target: {
                    "mae": baseline_metrics[target]["mae_mean"],
                    "rmse": baseline_metrics[target]["rmse_mean"],
                }
                for target in target_columns
            },
        )
        feature_importances = self._summarize_feature_importances(
            fold_results,
            target_columns,
        )
        primary_target = target_columns[0]
        primary_metrics = model_metrics[primary_target]
        primary_baseline = baseline_metrics[primary_target]
        primary_comparison = comparison[primary_target]

        return {
            "total_folds": len(fold_results),
            "target_columns": target_columns,
            "model_metrics": model_metrics,
            "baseline_metrics": baseline_metrics,
            "comparison": comparison,
            "feature_importances": feature_importances,
            "mae_mean": primary_metrics["mae_mean"],
            "mae_std": primary_metrics["mae_std"],
            "mae_min": primary_metrics["mae_min"],
            "mae_max": primary_metrics["mae_max"],
            "rmse_mean": primary_metrics["rmse_mean"],
            "rmse_std": primary_metrics["rmse_std"],
            "rmse_min": primary_metrics["rmse_min"],
            "rmse_max": primary_metrics["rmse_max"],
            "baseline_mae_mean": primary_baseline["mae_mean"],
            "baseline_rmse_mean": primary_baseline["rmse_mean"],
            "mae_improvement": primary_comparison["mae"]["improvement"],
            "rmse_improvement": primary_comparison["rmse"]["improvement"],
            "fold_results": fold_results,
        }

    def print_summary(self, summary: Dict):
        """Log a formatted summary of backtesting results."""
        logger.info("\n%s", "=" * 60)
        logger.info("WALK-FORWARD BACKTESTING SUMMARY")
        logger.info("=" * 60)

        if summary["total_folds"] == 0:
            logger.info("No backtesting folds completed. Not enough data.")
            return

        logger.info("Number of folds: %s", summary["total_folds"])
        for target in summary["target_columns"]:
            model_metrics = summary["model_metrics"][target]
            baseline_metrics = summary["baseline_metrics"][target]
            comparison = summary["comparison"][target]

            logger.info("\nTarget: %s", target)
            logger.info("MAE (Mean Absolute Error):")
            logger.info("  Model mean:    %.2f", model_metrics["mae_mean"])
            logger.info("  Baseline mean: %.2f", baseline_metrics["mae_mean"])
            logger.info(
                "  Improvement:   %.2f (%.1f%%)",
                comparison["mae"]["improvement"],
                comparison["mae"]["improvement_pct"],
            )
            logger.info("RMSE (Root Mean Squared Error):")
            logger.info("  Model mean:    %.2f", model_metrics["rmse_mean"])
            logger.info("  Baseline mean: %.2f", baseline_metrics["rmse_mean"])
            logger.info(
                "  Improvement:   %.2f (%.1f%%)",
                comparison["rmse"]["improvement"],
                comparison["rmse"]["improvement_pct"],
            )

            importances = summary["feature_importances"].get(target, [])
            if importances:
                logger.info("Top feature importances:")
                for item in importances[:5]:
                    logger.info("  %s: %.3f", item["feature"], item["importance"])

        if summary["total_folds"] >= 3:
            first_three_mae = [fold["mae"] for fold in summary["fold_results"][:3]]
            last_three_mae = [fold["mae"] for fold in summary["fold_results"][-3:]]
            first_avg = np.mean(first_three_mae)
            last_avg = np.mean(last_three_mae)
            trend = (
                "improving"
                if last_avg < first_avg
                else "degrading"
                if last_avg > first_avg
                else "stable"
            )
            logger.info("\nPerformance trend (first 3 vs last 3 folds): %s", trend)
            logger.info("  First 3 folds avg MAE: %.2f", first_avg)
            logger.info("  Last 3 folds avg MAE:  %.2f", last_avg)


def run_backtest_example():
    """Run a demonstration backtest using generated mock traffic data."""
    from traffic_client import TrafficAPIClient

    logger.info("Generating mock data for backtesting example...")
    client = TrafficAPIClient()
    location = {"lat": 25.4850, "lng": 51.4475}
    end_time = pd.Timestamp.now()
    start_time = end_time - pd.Timedelta(days=90)

    data = client.fetch_traffic_data(location, start_time, end_time)
    logger.info("Generated %s hourly records", len(data))

    backtester = WalkForwardBacktester(
        initial_train_days=30,
        test_days=7,
        model_type="random_forest",
    )
    summary = backtester.backtest(data, target_columns=DEFAULT_TARGET_COLUMNS)
    backtester.print_summary(summary)
    return summary
