from data_retriever import DataRetriever
from lstm_model import LSTMModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from config import EPOCHS, BATCH_SIZE, SEQUENCE_LENGTH

class StockPredictor:
    def __init__(self, symbol='SPY', start_date='2024-06-01', sequence_length=SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.data_retriever = DataRetriever(symbol=symbol, start_date=start_date)
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.analyzer = None
        
    def prepare_data(self):
        """Prepare the data for training"""
        # Fetch and prepare the data
        self.data_retriever.fetch_data()
        self.X_train, self.y_train, self.X_test, self.y_test = \
            self.data_retriever.prepare_data(sequence_length=self.sequence_length)
            
    def train_model(self, epochs=EPOCHS, batch_size=BATCH_SIZE):
        """Train the LSTM model"""
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
            
        # Initialize and train the model
        n_features = self.X_train.shape[2]
        
        self.model = LSTMModel(
            sequence_length=self.sequence_length,
            n_features=n_features
        )
        history = self.model.train(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size
        )
        return history
        
    def evaluate_model(self):
        """Evaluate the model and make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
            
        # Make predictions
        train_predictions = self.model.predict(self.X_train)
        test_predictions = self.model.predict(self.X_test)
        
        # Get prediction probabilities
        train_probs = self.model.predict_proba(self.X_train)
        test_probs = self.model.predict_proba(self.X_test)
        
        # Calculate metrics
        train_accuracy = np.mean(train_predictions == self.y_train)
        test_accuracy = np.mean(test_predictions == self.y_test)
        
        # Define class labels
        class_labels = ['No Position', 'Buy Call', 'Buy Put']
        
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
        plt.yticks(range(3), results['class_labels'])
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot signal probabilities over time
        plt.figure(figsize=(15, 8))
        test_probs = results['test_probs']
        
        for i in range(3):
            plt.plot(time_points, test_probs[:, i], 
                    label=results['class_labels'][i], alpha=0.7)
        
        plt.title('Signal Probability Distribution Over Time')
        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example usage
    predictor = StockPredictor()  # Use default start_date='2023-06-01'
    predictor.prepare_data()
    
    # Train and evaluate LSTM model
    history = predictor.train_model()
    results = predictor.evaluate_model()
    predictor.plot_results(results)
    
    print("\nClassification Report:")
    print(results['classification_report'])
    print(f"\nTraining Accuracy: {results['train_accuracy']:.2f}")
    print(f"Testing Accuracy: {results['test_accuracy']:.2f}") 