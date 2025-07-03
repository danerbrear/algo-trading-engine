from ..common.data_retriever import DataRetriever
from .lstm_model import LSTMModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from .config import EPOCHS, BATCH_SIZE, SEQUENCE_LENGTH
import argparse
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class StockPredictor:
    def __init__(self, symbol='SPY', hmm_start_date='2010-01-01', lstm_start_date='2021-06-01', sequence_length=SEQUENCE_LENGTH, use_free_tier=False, quiet_mode=True):
        """Initialize StockPredictor with separate date ranges for HMM and LSTM
        
        Args:
            symbol: Stock symbol to analyze
            hmm_start_date: Start date for HMM training (market state classification)
            lstm_start_date: Start date for LSTM training (options signal prediction)  
            sequence_length: Length of sequences for LSTM
            use_free_tier: Whether to use free tier rate limiting (13 second timeout)
            quiet_mode: Whether to suppress detailed output for cleaner progress display
        """
        self.sequence_length = sequence_length
        self.data_retriever = DataRetriever(
            symbol=symbol, 
            hmm_start_date=hmm_start_date, 
            lstm_start_date=lstm_start_date,
            use_free_tier=use_free_tier,
            quiet_mode=quiet_mode
        )
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.analyzer = None

    def prepare_data(self):
        """Prepare the data for training"""
        # Fetch and prepare the data for LSTM
        self.data_retriever.prepare_data_for_lstm()

        # Initialize LSTM model with data preparation capabilities
        # Note: n_features will be determined dynamically in prepare_data()
        self.lstm_model = LSTMModel(
            sequence_length=self.sequence_length,
            n_features=9  # Updated to match new feature count
        )

        # Prepare LSTM-specific data
        self.X_train, self.y_train, self.X_test, self.y_test = \
            self.lstm_model.prepare_data(self.data_retriever, sequence_length=self.sequence_length)

    def train_model(self, epochs=EPOCHS, batch_size=BATCH_SIZE):
        """Train the LSTM model"""
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        # Train the model using the prepared LSTM model instance
        history = self.lstm_model.train(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size
        )
        return history

    def evaluate_model(self):
        """Evaluate the model and make predictions"""
        if self.lstm_model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Make predictions
        train_predictions = self.lstm_model.predict(self.X_train)
        test_predictions = self.lstm_model.predict(self.X_test)

        # Get prediction probabilities
        train_probs = self.lstm_model.predict_proba(self.X_train)
        test_probs = self.lstm_model.predict_proba(self.X_test)

        # Calculate metrics
        train_accuracy = np.mean(train_predictions == self.y_train)
        test_accuracy = np.mean(test_predictions == self.y_test)

        # Define class labels for the 3-class strategy system
        class_labels = ['Hold', 'Call Credit Spread', 'Put Credit Spread']

        # Get unique classes in the test set
        unique_classes = np.unique(np.concatenate([self.y_test, test_predictions]))
        used_labels = [class_labels[int(i)] for i in unique_classes]

        # Generate classification report
        test_report = classification_report(
            self.y_test, 
            test_predictions,
            target_names=used_labels
        )

        # Generate confusion matrix
        conf_matrix = confusion_matrix(self.y_test, test_predictions)

        return {
            'train_predictions': train_predictions,
            'test_predictions': test_predictions,
            'train_probs': train_probs,
            'test_probs': test_probs,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': test_report,
            'confusion_matrix': conf_matrix,
            'class_labels': used_labels
        }

    def get_strategy_returns_comparison(self):
        """Get predicted strategy returns vs actual SPY log returns for comparison plotting
        
        Returns:
            tuple: (predicted_returns, actual_spy_returns) both as numpy arrays
        """
        if self.lstm_model is None or self.X_test is None:
            return None, None

        try:
            # Get predicted strategy labels for test data
            test_predictions = self.lstm_model.predict(self.X_test)

            # Get the corresponding dates and strategy returns from the data retriever
            # We need to access the LSTM data that was used for testing
            lstm_data = self.data_retriever.lstm_data

            if lstm_data is None or len(lstm_data) == 0:
                return None, None

            # Calculate the starting index for test data in the original dataset
            # Test data starts after training data + sequence length
            total_samples = len(self.X_train) + len(self.X_test)
            train_size = len(self.X_train)
            sequence_length = self.sequence_length

            # The test data starts at: sequence_length + train_size
            test_start_idx = sequence_length + train_size
            test_end_idx = test_start_idx + len(test_predictions)

            # Ensure we don't go beyond available data
            test_end_idx = min(test_end_idx, len(lstm_data))
            actual_test_length = test_end_idx - test_start_idx

            if actual_test_length <= 0:
                return None, None

            # Trim predictions to match available data
            test_predictions = test_predictions[:actual_test_length]

            # Get actual SPY log returns for the test period
            actual_spy_returns = lstm_data['Log_Returns'].iloc[test_start_idx:test_end_idx].values

            # Calculate predicted strategy returns based on the predicted labels
            predicted_returns = np.zeros(len(test_predictions))

            for i, predicted_label in enumerate(test_predictions):
                predicted_label = int(predicted_label)
                data_idx = test_start_idx + i

                # Map predicted label to expected strategy return
                if predicted_label == 0:  # Hold
                    predicted_returns[i] = 0.0  # No return for hold
                elif predicted_label == 1:  # Call Credit Spread
                    if 'Future_Call_Credit_Return' in lstm_data.columns and data_idx < len(lstm_data):
                        predicted_returns[i] = lstm_data['Future_Call_Credit_Return'].iloc[data_idx]
                    else:
                        predicted_returns[i] = 0.08  # Default expected return
                elif predicted_label == 2:  # Put Credit Spread  
                    if 'Future_Put_Credit_Return' in lstm_data.columns and data_idx < len(lstm_data):
                        predicted_returns[i] = lstm_data['Future_Put_Credit_Return'].iloc[data_idx]
                    else:
                        predicted_returns[i] = 0.08  # Default expected return
                else:
                    predicted_returns[i] = 0.0
            
            # Handle NaN values by replacing with defaults
            predicted_returns = np.nan_to_num(predicted_returns, nan=0.0)
            actual_spy_returns = np.nan_to_num(actual_spy_returns, nan=0.0)

            return predicted_returns, actual_spy_returns

        except Exception as e:
            print(f"⚠️ Error calculating strategy returns comparison: {str(e)}")
            return None, None

    def plot_results(self, results):
        """Plot the results"""
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            results['confusion_matrix'], 
            annot=True, 
            fmt='d',
            xticklabels=results['class_labels'],
            yticklabels=results['class_labels']
        )
        plt.title('Confusion Matrix of Option Trading Signals')
        plt.xlabel('Predicted Signal')
        plt.ylabel('True Signal')
        plt.yticks(range(len(results['class_labels'])), results['class_labels'])
        plt.xticks(range(len(results['class_labels'])), results['class_labels'], rotation=45)
        plt.show()

        # Plot signal distribution over time
        plt.figure(figsize=(15, 6))
        
        # Get actual signals for test data
        test_actual = self.y_test
        test_pred = results['test_predictions']
        
        # Create time points
        time_points = range(len(test_actual))
        
        # Plot actual vs predicted signals
        plt.plot(time_points, test_actual, label='Actual Signal', alpha=0.6)
        plt.plot(time_points, test_pred, label='Predicted Signal', alpha=0.6)
        
        plt.title('Option Trading Signals: Predicted vs Actual')
        plt.xlabel('Time')
        plt.ylabel('Signal')
        plt.yticks(range(len(results['class_labels'])), results['class_labels'])
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot accumulated predicted returns vs actual SPY log returns over time
        plt.figure(figsize=(15, 8))
        
        # Get the predicted strategy returns and actual SPY log returns
        predicted_returns, actual_spy_returns = self.get_strategy_returns_comparison()
        
        if predicted_returns is not None and actual_spy_returns is not None:
            # Align time points with available data
            comparison_time_points = range(len(predicted_returns))
            
            # Calculate accumulated returns
            accumulated_predicted_returns = np.cumsum(predicted_returns)
            accumulated_spy_returns = np.cumsum(actual_spy_returns)
            
            plt.plot(comparison_time_points, accumulated_spy_returns * 100, 
                    label='Accumulated SPY Log Returns (×100)', alpha=0.8, linewidth=1.5, color='blue')
            plt.plot(comparison_time_points, accumulated_predicted_returns, 
                    label='Accumulated Strategy Returns', alpha=0.8, linewidth=1.5, color='red')
            
            plt.title('Accumulated Strategy Returns vs Accumulated SPY Log Returns Over Time')
            plt.xlabel('Time')
            plt.ylabel('Accumulated Returns')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            print("⚠️ Unable to generate returns comparison plot - insufficient data")

def save_model(model, hmm_model=None, mode='lstm_poc'):
    """Save both LSTM and HMM models to timestamped and latest folders"""
    # Only use the environment variable, default to a generic relative path if not set
    base_dir = os.environ.get('MODEL_SAVE_BASE_PATH', 'Trained_Models')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamp_dir = os.path.join(base_dir, mode, timestamp)
    latest_dir = os.path.join(base_dir, mode, 'latest')
    os.makedirs(timestamp_dir, exist_ok=True)
    os.makedirs(latest_dir, exist_ok=True)
    
    # Save LSTM model
    model_path = os.path.join(timestamp_dir, 'model.keras')
    latest_path = os.path.join(latest_dir, 'model.keras')
    model.save(model_path)
    model.save(latest_path)
    print(f"✅ LSTM model saved to {model_path} and {latest_path}")
    
    # Save HMM model if provided
    if hmm_model is not None:
        import pickle
        
        # Save HMM model and scaler
        hmm_data = {
            'hmm_model': hmm_model.hmm_model,
            'scaler': hmm_model.scaler,
            'n_states': hmm_model.n_states,
            'max_states': hmm_model.max_states
        }
        
        # Save to timestamped folder
        hmm_path = os.path.join(timestamp_dir, 'hmm_model.pkl')
        with open(hmm_path, 'wb') as f:
            pickle.dump(hmm_data, f)
        
        # Save to latest folder
        hmm_latest_path = os.path.join(latest_dir, 'hmm_model.pkl')
        with open(hmm_latest_path, 'wb') as f:
            pickle.dump(hmm_data, f)
        
        print(f"✅ HMM model saved to {hmm_path} and {hmm_latest_path}")
        
        # Save HMM model information
        info_content = f"""HMM Model Information
====================
Number of States: {hmm_model.n_states}
Max States: {hmm_model.max_states}
Model Type: GaussianHMM
Covariance Type: {hmm_model.hmm_model.covariance_type}
Transition Matrix Shape: {hmm_model.hmm_model.transmat_.shape}
Emission Parameters Shape: {hmm_model.hmm_model.means_.shape}
Scaler Type: StandardScaler
Features: Returns, Volatility, Price_to_SMA20, SMA20_to_SMA50, Volume_Ratio

State Characteristics:
"""
        
        # Add state characteristics if available
        if hasattr(hmm_model, 'hmm_model') and hmm_model.hmm_model is not None:
            try:
                # Get state characteristics (this would need data, so we'll add a note)
                info_content += """
Note: State characteristics require training data to calculate.
Use the get_state_description() method with your data to see state details.
"""
            except:
                pass
        
        # Save info to timestamped folder
        info_path = os.path.join(timestamp_dir, 'hmm_model_info.txt')
        with open(info_path, 'w') as f:
            f.write(info_content)
        
        # Save info to latest folder
        info_latest_path = os.path.join(latest_dir, 'hmm_model_info.txt')
        with open(info_latest_path, 'w') as f:
            f.write(info_content)
        
        print(f"✅ HMM model info saved to {info_path} and {info_latest_path}")

if __name__ == "__main__":
    # Example usage with separate date ranges
    print("🚀 Starting Options Trading LSTM Model with Separate HMM/LSTM Data")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description="Options Trading LSTM Model")
    parser.add_argument('-s', '--save', action='store_true', help='Save the trained model')
    parser.add_argument('--mode', type=str, default='lstm_poc', help='Mode label for model saving (e.g., lstm_poc)')
    parser.add_argument('-f', '--free', action='store_true', help='Use free tier rate limiting (13 second timeout between API requests)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress detailed output during processing for cleaner progress display (default)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed output during processing (opposite of --quiet)')
    args = parser.parse_args()
    
    # Determine quiet mode: default is True (quiet) unless --verbose is specified
    quiet_mode = not args.verbose  # If verbose is True, quiet_mode becomes False
    
    predictor = StockPredictor(
        symbol='SPY',
        hmm_start_date='2010-01-01',
        lstm_start_date='2021-06-01',
        use_free_tier=args.free,
        quiet_mode=quiet_mode
    )
    
    predictor.prepare_data()
    
    # Train and evaluate LSTM model
    print("\n🏋️ Training LSTM Model...")
    history = predictor.train_model()
    
    # Save the model if requested
    if args.save:
        save_model(predictor.lstm_model.model, hmm_model=predictor.data_retriever.state_classifier, mode=args.mode)
    
    print("\n📊 Evaluating Model Performance...")
    results = predictor.evaluate_model()
    predictor.plot_results(results)
    
    print("\nClassification Report:")
    print(results['classification_report'])
    print(f"\nTraining Accuracy: {results['train_accuracy']:.2f}")
    print(f"Testing Accuracy: {results['test_accuracy']:.2f}") 