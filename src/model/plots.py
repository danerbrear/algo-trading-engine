#!/usr/bin/env python3
"""
Plotting utilities for the LSTM Options Trading Model
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class ModelPlotter:
    """Class for creating various plots for model evaluation and results visualization"""
    
    def __init__(self, symbol='SPY'):
        """
        Initialize the plotter
        
        Args:
            symbol: Stock symbol for plot titles
        """
        self.symbol = symbol
        
        # Set style for better-looking plots
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_confusion_matrix(self, confusion_matrix, class_labels):
        """
        Plot confusion matrix heatmap
        
        Args:
            confusion_matrix: 2D array of confusion matrix values
            class_labels: List of class labels
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_matrix, 
            annot=True, 
            fmt='d',
            xticklabels=class_labels,
            yticklabels=class_labels
        )
        plt.title(f'Confusion Matrix of Option Trading Signals - {self.symbol}')
        plt.xlabel('Predicted Signal')
        plt.ylabel('True Signal')
        plt.yticks(range(len(class_labels)), class_labels)
        plt.xticks(range(len(class_labels)), class_labels, rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_signal_distribution(self, test_actual, test_pred, class_labels):
        """
        Plot signal distribution over time
        
        Args:
            test_actual: Actual signal values
            test_pred: Predicted signal values
            class_labels: List of class labels
        """
        plt.figure(figsize=(15, 6))
        
        # Create time points
        time_points = range(len(test_actual))
        
        # Plot actual vs predicted signals
        plt.plot(time_points, test_actual, label='Actual Signal', alpha=0.6)
        plt.plot(time_points, test_pred, label='Predicted Signal', alpha=0.6)
        
        plt.title(f'Option Trading Signals: Predicted vs Actual - {self.symbol}')
        plt.xlabel('Time')
        plt.ylabel('Signal')
        plt.yticks(range(len(class_labels)), class_labels)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_returns_comparison(self, predicted_returns, actual_returns):
        """
        Plot accumulated predicted returns vs actual log returns over time
        
        Args:
            predicted_returns: Array of predicted strategy returns
            actual_returns: Array of actual log returns
        """
        if predicted_returns is None or actual_returns is None:
            print("⚠️ Unable to generate returns comparison plot - insufficient data")
            return
        
        plt.figure(figsize=(15, 8))
        
        # Align time points with available data
        comparison_time_points = range(len(predicted_returns))
        
        # Calculate accumulated returns
        accumulated_predicted_returns = np.cumsum(predicted_returns)
        accumulated_actual_returns = np.cumsum(actual_returns)
        
        plt.plot(comparison_time_points, accumulated_actual_returns * 100, 
                label=f'Accumulated {self.symbol} Log Returns (×100)', alpha=0.8, linewidth=1.5, color='blue')
        plt.plot(comparison_time_points, accumulated_predicted_returns, 
                label='Accumulated Strategy Returns', alpha=0.8, linewidth=1.5, color='red')
        
        plt.title(f'Accumulated Strategy Returns vs Accumulated {self.symbol} Log Returns Over Time')
        plt.xlabel('Time')
        plt.ylabel('Accumulated Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, history):
        """
        Plot training history (loss and accuracy)
        
        Args:
            history: Keras training history object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title(f'Model Loss - {self.symbol}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title(f'Model Accuracy - {self.symbol}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, feature_names, importance_scores):
        """
        Plot feature importance (if available from model)
        
        Args:
            feature_names: List of feature names
            importance_scores: Array of importance scores
        """
        plt.figure(figsize=(12, 8))
        
        # Sort features by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_scores = importance_scores[sorted_indices]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(sorted_features))
        plt.barh(y_pos, sorted_scores)
        plt.yticks(y_pos, sorted_features)
        plt.xlabel('Importance Score')
        plt.title(f'Feature Importance - {self.symbol}')
        plt.gca().invert_yaxis()  # Invert y-axis to show most important at top
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()
    
    def plot_market_states(self, market_states, dates):
        """
        Plot market state transitions over time
        
        Args:
            market_states: Array of market state values
            dates: Array of corresponding dates
        """
        plt.figure(figsize=(15, 6))
        
        plt.plot(dates, market_states, 'o-', alpha=0.7, markersize=3)
        plt.title(f'Market State Transitions - {self.symbol}')
        plt.xlabel('Date')
        plt.ylabel('Market State')
        plt.grid(True, alpha=0.3)
        
        # Add state labels if available
        unique_states = np.unique(market_states)
        plt.yticks(unique_states, [f'State {s}' for s in unique_states])
        
        plt.tight_layout()
        plt.show()
    
    def plot_all_results(self, results, test_actual, test_pred, predicted_returns=None, actual_returns=None, history=None):
        """
        Plot all results in sequence
        
        Args:
            results: Dictionary containing evaluation results
            test_actual: Actual test values
            test_pred: Predicted test values
            predicted_returns: Optional predicted returns for comparison
            actual_returns: Optional actual returns for comparison
            history: Optional training history
        """
        print("📊 Generating plots...")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(results['confusion_matrix'], results['class_labels'])
        
        # Plot signal distribution
        self.plot_signal_distribution(test_actual, test_pred, results['class_labels'])
        
        # Plot returns comparison if data available
        if predicted_returns is not None and actual_returns is not None:
            self.plot_returns_comparison(predicted_returns, actual_returns)
        
        # Plot training history if available
        if history is not None:
            self.plot_training_history(history)
        
        print("✅ All plots generated successfully!")


def create_plotter(symbol='SPY'):
    """
    Factory function to create a ModelPlotter instance
    
    Args:
        symbol: Stock symbol for plot titles
        
    Returns:
        ModelPlotter instance
    """
    return ModelPlotter(symbol) 