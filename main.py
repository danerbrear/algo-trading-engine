from data_retriever import DataRetriever
from lstm_model import LSTMModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from config import EPOCHS, BATCH_SIZE, SEQUENCE_LENGTH

class StockPredictor:
    def __init__(self, symbol='SPY', start_date='2010-01-01', sequence_length=SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.data_retriever = DataRetriever(symbol=symbol, start_date=start_date)
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
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
        n_states = self.data_retriever.n_states
        
        self.model = LSTMModel(
            sequence_length=self.sequence_length,
            n_features=n_features,
            n_states=n_states
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
        
        # Generate classification report
        state_descriptions = [
            self.data_retriever.get_state_description(i) 
            for i in range(self.data_retriever.n_states)
        ]
        
        test_report = classification_report(
            self.y_test, 
            test_predictions,
            target_names=state_descriptions
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
            'state_descriptions': state_descriptions
        }
        
    def plot_results(self, results):
        """Plot the results"""
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            results['confusion_matrix'], 
            annot=True, 
            fmt='d',
            xticklabels=[desc.split(':')[0] for desc in results['state_descriptions']],
            yticklabels=[desc.split(':')[0] for desc in results['state_descriptions']]
        )
        plt.title('Confusion Matrix of Market States')
        plt.xlabel('Predicted State')
        plt.ylabel('True State')
        plt.show()
        
        # Plot state distribution over time
        plt.figure(figsize=(15, 6))
        
        # Get actual states for test data
        test_actual = self.y_test
        test_pred = results['test_predictions']
        
        # Create time points
        time_points = range(len(test_actual))
        
        # Plot actual vs predicted states
        plt.plot(time_points, test_actual, label='Actual State', alpha=0.6)
        plt.plot(time_points, test_pred, label='Predicted State', alpha=0.6)
        
        plt.title('Market State Predictions vs Actual')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.yticks(
            range(self.data_retriever.n_states),
            [desc.split(':')[0] for desc in results['state_descriptions']]
        )
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot state probabilities over time
        plt.figure(figsize=(15, 8))
        test_probs = results['test_probs']
        
        for i in range(self.data_retriever.n_states):
            plt.plot(time_points, test_probs[:, i], 
                    label=results['state_descriptions'][i], alpha=0.7)
        
        plt.title('State Probability Distribution Over Time')
        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example usage
    predictor = StockPredictor(start_date='2015-01-01')
    predictor.prepare_data()
    history = predictor.train_model()
    results = predictor.evaluate_model()
    predictor.plot_results(results)
    
    print("\nMarket State Descriptions:")
    for desc in results['state_descriptions']:
        print(desc)
        
    print("\nClassification Report:")
    print(results['classification_report'])
    print(f"\nTraining Accuracy: {results['train_accuracy']:.2f}")
    print(f"Testing Accuracy: {results['test_accuracy']:.2f}") 