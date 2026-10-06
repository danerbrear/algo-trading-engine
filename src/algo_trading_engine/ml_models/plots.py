#!/usr/bin/env python3
"""
Plotting utilities for the LSTM Options Trading Model
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from algo_trading_engine.plotting import HeatmapSpec, PlotConfig, build_plot_spec, show_plot
from algo_trading_engine.plotting.spec import BarSpec


class ModelPlotter:
    """Class for creating various plots for model evaluation and results visualization"""

    def __init__(self, symbol: str = "SPY", plot_config: PlotConfig | None = None):
        self.symbol = symbol
        self.plot_config = plot_config or PlotConfig()

    def plot_confusion_matrix(self, confusion_matrix, class_labels):
        spec = HeatmapSpec(
            matrix=np.asarray(confusion_matrix),
            row_labels=tuple(str(label) for label in class_labels),
            col_labels=tuple(str(label) for label in class_labels),
            name="confusion_matrix",
            title=f"Confusion Matrix of Option Trading Signals - {self.symbol}",
            x_label="Predicted Signal",
            y_label="True Signal",
            value_format="d",
        )
        show_plot(spec, config=self.plot_config)

    def plot_signal_distribution(self, test_actual, test_pred, class_labels):
        frame = pd.DataFrame(
            {
                "index": np.arange(len(test_actual)),
                "Actual Signal": test_actual,
                "Predicted Signal": test_pred,
            }
        )
        y_ticks = tuple((float(i), str(label)) for i, label in enumerate(class_labels))
        spec = build_plot_spec(
            frame,
            name="signal_distribution",
            x="index",
            title=f"Option Trading Signals: Predicted vs Actual - {self.symbol}",
            y_label="Signal",
            y_ticks=y_ticks,
        )
        show_plot(spec, config=self.plot_config)

    def plot_returns_comparison(self, predicted_returns, actual_returns):
        if predicted_returns is None or actual_returns is None:
            print("⚠️ Unable to generate returns comparison plot - insufficient data")
            return

        accumulated_predicted = np.cumsum(predicted_returns)
        accumulated_actual = np.cumsum(actual_returns)
        frame = pd.DataFrame(
            {
                "index": np.arange(len(predicted_returns)),
                f"Accumulated {self.symbol} Log Returns (×100)": accumulated_actual * 100,
                "Accumulated Strategy Returns": accumulated_predicted,
            }
        )
        spec = build_plot_spec(
            frame,
            name="returns_comparison",
            x="index",
            title=f"Accumulated Strategy Returns vs Accumulated {self.symbol} Log Returns Over Time",
            y_label="Accumulated Returns",
        )
        show_plot(spec, config=self.plot_config)

    def plot_training_history(self, history):
        loss_frame = pd.DataFrame({"epoch": np.arange(len(history.history["loss"])), "Training Loss": history.history["loss"]})
        if "val_loss" in history.history:
            loss_frame["Validation Loss"] = history.history["val_loss"]
        accuracy_frame = pd.DataFrame(
            {"epoch": np.arange(len(history.history["accuracy"])), "Training Accuracy": history.history["accuracy"]}
        )
        if "val_accuracy" in history.history:
            accuracy_frame["Validation Accuracy"] = history.history["val_accuracy"]

        loss_spec = build_plot_spec(
            loss_frame,
            name="training_loss",
            x="epoch",
            title=f"Model Loss - {self.symbol}",
            y_label="Loss",
        )
        accuracy_spec = build_plot_spec(
            accuracy_frame,
            name="training_accuracy",
            x="epoch",
            title=f"Model Accuracy - {self.symbol}",
            y_label="Accuracy",
        )
        show_plot([loss_spec, accuracy_spec], config=self.plot_config, ncols=2)

    def plot_feature_importance(self, feature_names, importance_scores):
        sorted_indices = np.argsort(importance_scores)[::-1]
        labels = tuple(feature_names[i] for i in sorted_indices)
        values = tuple(float(importance_scores[i]) for i in sorted_indices)
        spec = BarSpec(
            labels=labels,
            values=values,
            name="feature_importance",
            title=f"Feature Importance - {self.symbol}",
            x_label="Importance Score",
        )
        show_plot(spec, config=self.plot_config)

    def plot_market_states(self, market_states, dates):
        frame = pd.DataFrame({"timestamp": pd.to_datetime(dates), "Market State": market_states})
        unique_states = np.unique(market_states)
        y_ticks = tuple((float(state), f"State {state}") for state in unique_states)
        spec = build_plot_spec(
            frame,
            name="market_states",
            title=f"Market State Transitions - {self.symbol}",
            y_label="Market State",
            y_ticks=y_ticks,
        )
        show_plot(spec, config=self.plot_config)

    def plot_all_results(
        self,
        results,
        test_actual,
        test_pred,
        predicted_returns=None,
        actual_returns=None,
        history=None,
    ):
        print("📊 Generating plots...")
        self.plot_confusion_matrix(results["confusion_matrix"], results["class_labels"])
        self.plot_signal_distribution(test_actual, test_pred, results["class_labels"])
        if predicted_returns is not None and actual_returns is not None:
            self.plot_returns_comparison(predicted_returns, actual_returns)
        if history is not None:
            self.plot_training_history(history)
        print("✅ All plots generated successfully!")


def create_plotter(symbol: str = "SPY", plot_config: PlotConfig | None = None) -> ModelPlotter:
    """Factory function to create a ModelPlotter instance."""
    return ModelPlotter(symbol=symbol, plot_config=plot_config)
