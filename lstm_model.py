import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional, LayerNormalization
from keras.optimizers import Adam
from keras.metrics import SparseCategoricalAccuracy
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
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
        """Build an enhanced LSTM model architecture"""
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=(self.sequence_length, self.n_features)))
        
        # First Bidirectional LSTM layer
        model.add(Bidirectional(LSTM(units=128, return_sequences=True,
                                   activation='tanh',
                                   recurrent_activation='sigmoid',
                                   recurrent_dropout=0.0,
                                   kernel_regularizer=l2(0.01),
                                   recurrent_regularizer=l2(0.01),
                                   unroll=True)))
        model.add(LayerNormalization())
        model.add(Dropout(0.4))
        
        # Second Bidirectional LSTM layer
        model.add(Bidirectional(LSTM(units=64, return_sequences=True,
                                   activation='tanh',
                                   recurrent_activation='sigmoid',
                                   recurrent_dropout=0.0,
                                   kernel_regularizer=l2(0.01),
                                   recurrent_regularizer=l2(0.01),
                                   unroll=True)))
        model.add(LayerNormalization())
        model.add(Dropout(0.4))
        
        # Third LSTM layer
        model.add(LSTM(units=32, return_sequences=False,
                      activation='tanh',
                      recurrent_activation='sigmoid',
                      recurrent_dropout=0.0,
                      kernel_regularizer=l2(0.01),
                      recurrent_regularizer=l2(0.01),
                      unroll=True))
        model.add(LayerNormalization())
        model.add(Dropout(0.3))
        
        # Dense layers with residual connections
        model.add(Dense(units=64, activation='selu',
                       kernel_regularizer=l2(0.01)))
        model.add(LayerNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(units=32, activation='selu',
                       kernel_regularizer=l2(0.01)))
        model.add(LayerNormalization())
        model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(units=self.n_states, activation='softmax'))
        
        # Use fixed learning rate with decay
        optimizer = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=[
                SparseCategoricalAccuracy(name='accuracy'),
            ],
            jit_compile=True
        )
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        """Train the LSTM model with enhanced callbacks and monitoring"""
        # Early stopping with restoration of best weights
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            min_delta=0.001
        )
        
        # Class weights to handle imbalance
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
        """Get probability distributions for each state"""
        return self.model.predict(X, batch_size=128)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model performance"""
        return self.model.evaluate(X_test, y_test, batch_size=128, verbose=1) 