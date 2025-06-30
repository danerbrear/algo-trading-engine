import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, LayerNormalization
from keras.optimizers import Adam
from keras.metrics import SparseCategoricalAccuracy
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
import numpy as np
from .config import (
    EPOCHS, BATCH_SIZE, VALIDATION_SPLIT, LSTM_UNITS, 
    DENSE_UNITS, DROPOUT_RATE, LEARNING_RATE,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA
)

# Enable GPU memory growth to prevent TF from taking all GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        print("GPU memory growth setting failed. Using default configuration.")

class LSTMModel:
    def __init__(self, sequence_length, n_features):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = 3  # 0: Hold, 1: Call Debit Spread, 2: Put Debit Spread
        self.model = self._build_model()
        
    def _build_model(self):
        """Build an enhanced LSTM model for options trading signals"""
        model = Sequential([
            # Input layer
            Input(shape=(self.sequence_length, self.n_features)),
            
            # First Bidirectional LSTM layer
            Bidirectional(LSTM(units=LSTM_UNITS[0], return_sequences=True,
                             kernel_regularizer=l2(0.01))),
            LayerNormalization(),
            Dropout(DROPOUT_RATE),
            
            # Second Bidirectional LSTM layer (final LSTM layer)
            Bidirectional(LSTM(units=LSTM_UNITS[1], return_sequences=False,
                             kernel_regularizer=l2(0.01))),
            LayerNormalization(),
            Dropout(DROPOUT_RATE),
            
            # Dense layers
            Dense(units=DENSE_UNITS[0], activation='relu',
                 kernel_regularizer=l2(0.01)),
            LayerNormalization(),
            Dropout(DROPOUT_RATE),
            
            # Output layer for 3-class classification
            Dense(units=self.n_classes, activation='softmax')
        ])
        
        # Use fixed learning rate with Adam optimizer
        optimizer = Adam(learning_rate=LEARNING_RATE)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            jit_compile=True
        )
        return model
    
    def train(self, X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT):
        """Train the LSTM model with early stopping and class weights"""
        # Early stopping with restoration of best weights
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            min_delta=EARLY_STOPPING_MIN_DELTA
        )
        
        # Calculate class weights to handle imbalance
        unique, counts = np.unique(y_train, return_counts=True)
        total = np.sum(counts)
        class_weights = {c: total / (len(counts) * count) for c, count in zip(unique, counts)}
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            class_weight=class_weights,
            verbose=1
        )
        return history
    
    def predict(self, X):
        """Make predictions using the trained model"""
        pred_probs = self.model.predict(X, batch_size=128)
        return np.argmax(pred_probs, axis=1)
    
    def predict_proba(self, X):
        """Get probability distributions for each class"""
        return self.model.predict(X, batch_size=128)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model performance"""
        return self.model.evaluate(X_test, y_test, batch_size=128, verbose=1) 