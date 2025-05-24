import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from keras.optimizers import Adam
from keras.metrics import SparseCategoricalAccuracy
from keras.callbacks import EarlyStopping
import numpy as np

# Enable GPU memory growth to prevent TF from taking all GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        print("GPU memory growth setting failed. Using default configuration.")

class LSTMModel:
    def __init__(self, sequence_length, n_features=5, n_states=None):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_states = n_states if n_states is not None else 3
        self.model = self._build_model()
        
    def _build_model(self):
        """Build an optimized LSTM model architecture"""
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=(self.sequence_length, self.n_features)))
        
        # First LSTM layer with batch normalization
        model.add(LSTM(units=128, return_sequences=True,
                      activation='relu',
                      unroll=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        # Second LSTM layer
        model.add(LSTM(units=64, return_sequences=False,
                      activation='relu',
                      unroll=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        # Dense layers
        model.add(Dense(units=32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        
        # Output layer
        model.add(Dense(units=self.n_states, activation='softmax'))
        
        # Use Adam optimizer with learning rate schedule
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True
        )
        optimizer = Adam(learning_rate=lr_schedule)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=[SparseCategoricalAccuracy()],
            jit_compile=True
        )
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=64, validation_split=0.1):
        """Train the LSTM model with early stopping"""
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            min_delta=0.001
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        return history
    
    def predict(self, X):
        """Make predictions using the trained model"""
        # Use a larger batch size for prediction
        pred_probs = self.model.predict(X, batch_size=128)
        return np.argmax(pred_probs, axis=1)
    
    def predict_proba(self, X):
        """Get probability distributions for each state"""
        return self.model.predict(X, batch_size=128)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model performance"""
        return self.model.evaluate(X_test, y_test, batch_size=128, verbose=0) 