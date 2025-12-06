import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, LayerNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
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
        self.n_classes = 3  # 0: Hold, 1: Call Credit Spread, 2: Put Credit Spread
        self.model = self._build_model()
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.original_features = None
        self.feature_recommendations = None
        
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
    
    def prepare_data(self, data_retriever, sequence_length=60):
        """Prepare data for LSTM model with enhanced features using separate date ranges
        
        Args:
            data_retriever: DataRetriever instance with data and options_handler
            sequence_length: Length of sequences for LSTM
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        print(f"\n🚀 Phase 6: Preparing LSTM features ({len(data_retriever.lstm_data)} samples)")
        
        # Prepare feature matrix for LSTM (including all features and market states)
        self.feature_columns = [
            'High_Low_Range',
            'SMA20_to_SMA50', 'SMA20_to_SMA50_Lag1', 'SMA20_to_SMA50_Lag5', 
            'SMA20_to_SMA50_MA5', 'SMA20_to_SMA50_MA10', 'SMA20_to_SMA50_Std5', 'SMA20_to_SMA50_Momentum',
            'RSI', 'RSI_Lag1', 'RSI_Lag5', 'RSI_MA5', 'RSI_MA10', 'RSI_Std5', 'RSI_Momentum', 
            'RSI_Overbought', 'RSI_Oversold',
            'MACD_Hist',
            'Volume_Ratio', 'OBV',
            'Put_Call_Ratio', 'Option_Volume_Ratio',
            'Market_State',
            'Days_Until_Next_CPI', 'Days_Since_Last_CPI',
            'Days_Until_Next_CC', 'Days_Since_Last_CC',
            'Days_Until_Next_FFR', 'Days_Since_Last_FFR'
        ]
        
        # Store original features for reference
        self.original_features = self.feature_columns.copy()
        
        # Update n_features to match the actual number of features
        self.n_features = len(self.feature_columns)
        
        # Rebuild the model with the correct number of features
        self.model = self._build_model()
        
        # Analyze feature correlations before scaling
        correlation_pairs = self.analyze_feature_correlations(data_retriever.lstm_data, self.feature_columns, threshold=0.8)
        
        # Generate detailed feature reduction recommendations
        self.feature_recommendations = self.generate_feature_reduction_recommendations(correlation_pairs, self.feature_columns)
        
        # Ask user if they want to apply feature reduction
        print(f"\n🤔 Would you like to use the recommended reduced feature set?")
        print(f"   Current features: {len(self.feature_columns)}")
        if self.feature_recommendations['features_to_remove']:
            print(f"   Recommended features: {len(self.feature_recommendations['recommended_features'])}")
            print(f"   Potential reduction: {self.feature_recommendations['reduction_percentage']:.1f}%")
            print(f"   Note: You can modify this by editing the feature_columns list in prepare_data()")
        
        # Scale features (using original feature set for now - user can modify based on recommendations)
        features = self.scaler.fit_transform(data_retriever.lstm_data[self.feature_columns])
        
        # Critical: Check for NaN or infinite values that cause training to fail
        print(f"🔍 Data validation:")
        print(f"   NaN values in features: {np.isnan(features).sum()}")
        print(f"   Infinite values in features: {np.isinf(features).sum()}")
        print(f"   NaN values in labels: {data_retriever.lstm_data['Option_Signal'].isna().sum()}")
        
        # Clean the data - remove rows with NaN or infinite values
        valid_mask = ~(np.isnan(features).any(axis=1) | np.isinf(features).any(axis=1))
        features = features[valid_mask]
        clean_lstm_data = data_retriever.lstm_data[valid_mask].copy()
        
        print(f"   📊 Cleaned data: {len(features)} samples (removed {(~valid_mask).sum()} invalid samples)")
        
        # Create sequences using numpy operations for better performance
        n_samples = len(features) - sequence_length
        
        # Pre-allocate arrays
        X = np.zeros((n_samples, sequence_length, len(self.feature_columns)))
        y = np.zeros(n_samples)
        
        # Fill arrays using vectorized operations
        for i in range(sequence_length, len(features)):
            X[i-sequence_length] = features[i-sequence_length:i]
            y[i-sequence_length] = clean_lstm_data['Option_Signal'].iloc[i]
        
        # Split into train and test sets (80-20 split)
        train_size = int(len(X) * 0.8)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        print(f"✅ Data preparation complete:")
        print(f"   🏋️ Training samples: {len(X_train)}")
        print(f"   🧪 Testing samples: {len(X_test)}")
        print(f"   📊 Features per sample: {len(self.feature_columns)}")
        
        return X_train, y_train, X_test, y_test
    
    def analyze_feature_correlations(self, data, feature_columns, threshold=0.8):
        """Analyze correlations between features to identify redundant features
        
        Args:
            data: DataFrame containing the features
            feature_columns: List of feature column names to analyze
            threshold: Correlation threshold above which features are considered highly correlated
            
        Returns:
            List of dictionaries containing highly correlated feature pairs
        """
        print(f"\n🔍 Analyzing feature correlations (threshold: {threshold})")
        
        # Calculate correlation matrix for the specified features
        feature_data = data[feature_columns].copy()
        correlation_matrix = feature_data.corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        
        for i, col1 in enumerate(correlation_matrix.columns):
            for j, col2 in enumerate(correlation_matrix.columns[i+1:], i+1):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > threshold:
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr
                    })
        
        # Display results
        if high_corr_pairs:
            print(f"⚠️  Found {len(high_corr_pairs)} highly correlated feature pairs:")
            print("   Feature 1              Feature 2              Correlation")
            print("   " + "="*60)
            
            for pair in sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True):
                print(f"   {pair['feature1']:<20} {pair['feature2']:<20} {pair['correlation']:>8.3f}")
                
            print("\n💡 Recommendations:")
            processed_features = set()
            for pair in high_corr_pairs:
                feat1, feat2 = pair['feature1'], pair['feature2']
                if feat1 not in processed_features and feat2 not in processed_features:
                    print(f"   • Consider removing one of: {feat1} or {feat2}")
                    processed_features.add(feat1)
                    processed_features.add(feat2)
        else:
            print(f"✅ No highly correlated features found (threshold: {threshold})")
        
        # Additional analysis: Feature statistics
        print(f"\n📊 Feature Statistics:")
        print("   Feature                Mean        Std       Min       Max     NaN%")
        print("   " + "="*70)
        
        for feature in feature_columns:
            if feature in feature_data.columns:
                series = feature_data[feature]
                nan_pct = (series.isna().sum() / len(series)) * 100
                print(f"   {feature:<20} {series.mean():>8.3f} {series.std():>8.3f} "
                      f"{series.min():>8.3f} {series.max():>8.3f} {nan_pct:>6.1f}%")
        
        return high_corr_pairs

    def generate_feature_reduction_recommendations(self, correlation_pairs, feature_columns):
        """Generate specific recommendations for feature reduction based on correlation analysis
        
        Args:
            correlation_pairs: List of highly correlated feature pairs
            feature_columns: List of all feature columns
            
        Returns:
            Dictionary with recommendations for feature reduction
        """
        print(f"\n🎯 Feature Reduction Recommendations:")
        
        if not correlation_pairs:
            print("✅ No highly correlated features detected. Current feature set appears optimal.")
            return {'recommended_features': feature_columns, 'features_to_remove': []}
        
        # Analyze specific redundant pairs and provide targeted recommendations
        features_to_remove = []
        recommendations = []
        
        for pair in correlation_pairs:
            feat1, feat2 = pair['feature1'], pair['feature2']
            corr = pair['correlation']
            
            # Specific recommendations based on feature types
            if feat1 == 'Volume_Ratio' and feat2 == 'OBV':
                features_to_remove.append('OBV')
                recommendations.append(f"• Remove 'OBV' - Volume_Ratio is more normalized and stable (r={corr:.3f})")
                
            elif feat1 == 'MACD' and feat2 == 'MACD_Hist':
                features_to_remove.append('MACD')
                recommendations.append(f"• Remove 'MACD' - MACD_Hist contains the same signal information (r={corr:.3f})")
                
            elif feat1 == 'ATM_Call_Return' and feat2 == 'ATM_Put_Return':
                features_to_remove.append('ATM_Put_Return')
                recommendations.append(f"• Remove 'ATM_Put_Return' - Highly correlated with ATM_Call_Return (r={corr:.3f})")
                
            elif 'Price_to_SMA20' in [feat1, feat2] and 'SMA20_to_SMA50' in [feat1, feat2]:
                features_to_remove.append('SMA20_to_SMA50')
                recommendations.append(f"• Consider removing 'SMA20_to_SMA50' - Price_to_SMA20 is more direct trend signal (r={corr:.3f})")
                
            else:
                # Generic recommendation for other pairs
                recommendations.append(f"• High correlation between '{feat1}' and '{feat2}' (r={corr:.3f}) - consider removing one")
        
        # Remove duplicates
        features_to_remove = list(set(features_to_remove))
        recommended_features = [f for f in feature_columns if f not in features_to_remove]
        
        # Display recommendations
        if recommendations:
            for rec in recommendations:
                print(f"   {rec}")
            
            print(f"\n📋 Proposed Feature Set ({len(recommended_features)} features):")
            for i, feature in enumerate(recommended_features, 1):
                print(f"   {i:2d}. {feature}")
            
            print(f"\n❌ Suggested Features to Remove ({len(features_to_remove)} features):")
            for feature in features_to_remove:
                print(f"   • {feature}")
                
            reduction_pct = (len(features_to_remove) / len(feature_columns)) * 100
            print(f"\n📊 Feature Reduction: {len(feature_columns)} → {len(recommended_features)} features ({reduction_pct:.1f}% reduction)")
        
        return {
            'recommended_features': recommended_features,
            'features_to_remove': features_to_remove,
            'reduction_percentage': (len(features_to_remove) / len(feature_columns)) * 100
        }

    def apply_feature_reduction(self, use_reduced_features=False):
        """Apply feature reduction based on correlation analysis recommendations
        
        Args:
            use_reduced_features: Whether to use the reduced feature set
            
        Returns:
            List of features to use for model training
        """
        if not hasattr(self, 'feature_recommendations'):
            print("⚠️ Feature recommendations not available. Run prepare_data() first.")
            return None
            
        if use_reduced_features and self.feature_recommendations['features_to_remove']:
            recommended_features = self.feature_recommendations['recommended_features']
            print(f"\n✅ Applying feature reduction: Using {len(recommended_features)} features")
            return recommended_features
        else:
            print(f"\n📋 Using original feature set: {len(self.original_features)} features")
            return self.original_features
    
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
    
    def _predict(self, X):
        """Make predictions using the trained model (private method)"""
        pred_probs = self.model.predict(X, batch_size=128)
        return np.argmax(pred_probs, axis=1)
    
    def predict(self, X):
        """Make predictions using the trained model (public wrapper)"""
        return self._predict(X)
    
    def predict_proba(self, X):
        """Get probability distributions for each class"""
        return self.model.predict(X, batch_size=128)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model performance"""
        return self.model.evaluate(X_test, y_test, batch_size=128, verbose=1)
    
    @staticmethod
    def calculate_option_signals(data, holding_period: int = 15, min_return_threshold: float = 0.08):
        """Calculate trading signals based on options strategies using real multi-strike data
        
        Strategy Classes:
        0: Hold - No position (when no strategy meets criteria)
        1: Call Credit Spread - Moderately bearish (sell ATM call + buy ATM+5 call)
        2: Put Credit Spread - Moderately bullish (sell ATM put + buy ATM-5 put) 
        
        Args:
            data: DataFrame with options pricing data
            holding_period: Number of days to hold the strategy (default: 15)
            min_return_threshold: Minimum return required to generate a signal (default: 8%)
            
        Returns:
            DataFrame with Option_Signal column added
        """        
        print(f"Generating strategy labels using multi-strike data with {holding_period}-day holding period and {min_return_threshold:.1%} minimum threshold")
        
        # Initialize the signal column
        data['Option_Signal'] = 0  # Default to Hold
        
        # Check if required columns exist, if not create them with NaN values
        required_columns = ['Call_ATM_Plus5_Price', 'Put_ATM_Minus5_Price', 'Call_Price', 'Put_Price']
        for col in required_columns:
            if col not in data.columns:
                data[col] = np.nan
                print(f"⚠️ Warning: {col} column not found, using NaN values")
        
        # Multi-strike returns for realistic strategy calculations
        data['Call_Plus5_Return'] = data['Call_ATM_Plus5_Price'].pct_change(fill_method=None)
        data['Put_Minus5_Return'] = data['Put_ATM_Minus5_Price'].pct_change(fill_method=None)
        
        # Stock movement for context
        stock_return = data['Close'].shift(-holding_period) / data['Close'] - 1
        data['Future_Stock_Return'] = stock_return
        
        # Strategy Return Calculations using actual multi-strike data
        
        # 1. Call Credit Spread: Sell ATM Call + Buy ATM+5 Call
        atm_call_future = data['Call_Price'].shift(-holding_period) / data['Call_Price'] - 1
        plus5_call_future = data['Call_ATM_Plus5_Price'].shift(-holding_period) / data['Call_ATM_Plus5_Price'] - 1
        
        # Credit spread P&L: Short leg profit - Long leg loss (reversed from debit spread)
        data['Future_Call_Credit_Return'] = -atm_call_future + plus5_call_future
        print("✅ Using REAL Call Credit Spread calculation (Sell ATM Call + Buy ATM+5 Call)")
        
        # 2. Put Credit Spread: Sell ATM Put + Buy ATM-5 Put
        atm_put_future = data['Put_Price'].shift(-holding_period) / data['Put_Price'] - 1
        minus5_put_future = data['Put_ATM_Minus5_Price'].shift(-holding_period) / data['Put_ATM_Minus5_Price'] - 1
        
        # Credit spread P&L: Short leg profit - Long leg loss (reversed from debit spread)
        data['Future_Put_Credit_Return'] = -atm_put_future + minus5_put_future
        print("✅ Using REAL Put Credit Spread calculation (Sell ATM Put + Buy ATM-5 Put)")
        
        # Strategy counters
        strategy_counts = [0, 0, 0]  # Hold, Call Credit, Put Credit
        strategy_names = ['Hold', 'Call Credit Spread', 'Put Credit Spread']
        
        # Generate labels based on best performing strategy with market context
        valid_strategies = 0
        for i in range(len(data) - holding_period):  # Exclude last holding_period rows
            
            # Get future returns for each strategy
            call_credit_return = data['Future_Call_Credit_Return'].iloc[i]
            put_credit_return = data['Future_Put_Credit_Return'].iloc[i]
            
            # Skip if we don't have complete option data
            if (pd.isna(call_credit_return) or pd.isna(put_credit_return)):
                continue
            
            valid_strategies += 1
            
            # Market regime factors for strategy selection
            recent_trend = data['SMA20_to_SMA50'].iloc[i] if not pd.isna(data['SMA20_to_SMA50'].iloc[i]) else 1.0
            
            # Find the best strategy that meets minimum threshold
            # Credit spreads work better in sideways/trending markets with less strict requirements
            strategy_returns = {
                1: call_credit_return if recent_trend < 0.998 else call_credit_return * 0.8,      # Favor in downtrends/neutral
                2: put_credit_return if recent_trend > 1.002 else put_credit_return * 0.8,       # Favor in uptrends/neutral
            }
            
            # Find best strategy above threshold
            best_strategy = 0  # Default to Hold
            best_return = 0
            
            for strategy_id, strategy_return in strategy_returns.items():
                if strategy_return > min_return_threshold and strategy_return > best_return:
                    best_strategy = strategy_id
                    best_return = strategy_return
            
            # Assign the best strategy
            data.iloc[i, data.columns.get_loc('Option_Signal')] = best_strategy
            strategy_counts[best_strategy] += 1
        
        # Display strategy distribution
        total_signals = sum(strategy_counts)
        
        if total_signals > 0:
            print(f"\nStrategy Distribution ({valid_strategies} valid samples):")
            for i, (name, count) in enumerate(zip(strategy_names, strategy_counts)):
                print(f"  {i}: {name:18} {count:4d} ({count/total_signals:.1%})")
            
            # Show average returns for each strategy when selected
            for strategy_id in range(1, 3):
                strategy_mask = data['Option_Signal'] == strategy_id
                if strategy_mask.sum() > 0:
                    if strategy_id == 1:
                        avg_return = data[strategy_mask]['Future_Call_Credit_Return'].mean()
                    else:  # strategy_id == 2
                        avg_return = data[strategy_mask]['Future_Put_Credit_Return'].mean()
                    
                    print(f"  Avg {strategy_names[strategy_id]} Return: {avg_return:.2%}")
        else:
            print("⚠️ No valid strategy signals generated - check multi-strike data availability")
        
        return data
    
    @staticmethod
    def calculate_option_features(data, options_handler, min_dte: int = 24, max_dte: int = 36) -> pd.DataFrame:
        """Calculate option-related features with multi-strike data for strategy modeling
        
        This method uses the options_handler to fetch options chain data and extracts
        relevant features for LSTM training.
        
        Args:
            data: pd.DataFrame with market data
            options_handler: OptionsHandler instance for fetching options data
            min_dte: Minimum days to expiration (default: 24)
            max_dte: Maximum days to expiration (default: 36)
        
        Returns:
            DataFrame with calculated option features
        """
        import pandas as pd
        from decimal import Decimal
        
        print(f"\n💰 Calculating option features for {len(data)} dates...")
        
        # Track successful and failed dates
        successful_dates = 0
        failed_dates = 0
        
        # Process each date in the data
        for current_date in data.index:
            date_key = current_date.strftime('%Y-%m-%d')
            
            try:
                # Get current price
                current_price = data.loc[current_date, 'Close']
                
                from ..common.options_dtos import StrikeRangeDTO, ExpirationRangeDTO, StrikePrice
                from ..common.models import Option, OptionType
                
                # Find ATM strike (closest to current price)
                atm_strike = round(current_price)
                
                # Define the specific strikes we need
                needed_strikes = {atm_strike, atm_strike + 5, atm_strike + 10, atm_strike - 5, atm_strike - 10}
                
                # Define strike range around current price (ATM ± 10)
                strike_range = StrikeRangeDTO(
                    min_strike=StrikePrice(Decimal(str(atm_strike - 10))),
                    max_strike=StrikePrice(Decimal(str(atm_strike + 10)))
                )
                
                # Define expiration range
                expiration_range = ExpirationRangeDTO(min_days=min_dte, max_days=max_dte)
                
                # Fetch only the contracts we need (more efficient than full chain)
                all_contracts = options_handler.get_contract_list_for_date(
                    date=current_date,
                    strike_range=strike_range,
                    expiration_range=expiration_range
                )
                
                if not all_contracts:
                    continue
                
                # Filter to only the strikes we actually need
                # Use strike_price.value to get the float value from StrikePrice object
                needed_contracts = [c for c in all_contracts if float(c.strike_price.value) in needed_strikes]
                
                if not needed_contracts:
                    continue
                
                # Fetch bars only for the contracts we need (much more efficient)
                # Organize by option type and strike
                calls_by_strike = {}
                puts_by_strike = {}
                
                for contract in needed_contracts:
                    bar = options_handler.get_option_bar(contract, current_date)
                    if bar:
                        # Convert contract and bar to Option object
                        option = Option.from_contract_and_bar(contract, bar)
                        strike_key = int(round(option.strike))
                        
                        if option.is_call:
                            calls_by_strike[strike_key] = option
                        else:
                            puts_by_strike[strike_key] = option
                
                # Extract ATM option features
                atm_call = calls_by_strike.get(atm_strike)
                atm_put = puts_by_strike.get(atm_strike)
                
                if atm_call and atm_put:
                    # Store ATM option features (Greeks will be None since not included in bar data)
                    data.loc[current_date, 'Call_IV'] = atm_call.implied_volatility or 0
                    data.loc[current_date, 'Put_IV'] = atm_put.implied_volatility or 0
                    data.loc[current_date, 'Call_Volume'] = atm_call.volume or 0
                    data.loc[current_date, 'Put_Volume'] = atm_put.volume or 0
                    data.loc[current_date, 'Call_OI'] = atm_call.open_interest or 0
                    data.loc[current_date, 'Put_OI'] = atm_put.open_interest or 0
                    data.loc[current_date, 'Call_Price'] = atm_call.last_price
                    data.loc[current_date, 'Put_Price'] = atm_put.last_price
                    data.loc[current_date, 'Call_Delta'] = atm_call.delta or 0
                    data.loc[current_date, 'Put_Delta'] = atm_put.delta or 0
                    data.loc[current_date, 'Call_Gamma'] = atm_call.gamma or 0
                    data.loc[current_date, 'Put_Gamma'] = atm_put.gamma or 0
                    data.loc[current_date, 'Call_Theta'] = atm_call.theta or 0
                    data.loc[current_date, 'Put_Theta'] = atm_put.theta or 0
                    data.loc[current_date, 'Call_Vega'] = atm_call.vega or 0
                    data.loc[current_date, 'Put_Vega'] = atm_put.vega or 0
                    data.loc[current_date, 'Option_Volume_Ratio'] = (atm_call.volume + atm_put.volume) / data.loc[current_date, 'Volume'] if data.loc[current_date, 'Volume'] > 0 else 0
                    data.loc[current_date, 'Put_Call_Ratio'] = atm_put.volume / atm_call.volume if atm_call.volume > 0 else 1.0
                    
                    # Multi-strike options for strategy calculations
                    call_plus5 = calls_by_strike.get(atm_strike + 5)
                    call_plus10 = calls_by_strike.get(atm_strike + 10)
                    put_minus5 = puts_by_strike.get(atm_strike - 5)
                    put_minus10 = puts_by_strike.get(atm_strike - 10)
                    
                    if call_plus5:
                        data.loc[current_date, 'Call_ATM_Plus5_Price'] = call_plus5.last_price
                    if call_plus10:
                        data.loc[current_date, 'Call_ATM_Plus10_Price'] = call_plus10.last_price
                    if put_minus5:
                        data.loc[current_date, 'Put_ATM_Minus5_Price'] = put_minus5.last_price
                    if put_minus10:
                        data.loc[current_date, 'Put_ATM_Minus10_Price'] = put_minus10.last_price
                    
                    # Only count as successful if we set all required option features
                    successful_dates += 1
                else:
                    # No ATM call/put pair found - count as failure
                    print(f"⚠️  No ATM call/put pair found for {date_key} (strike: {atm_strike})")
                    failed_dates += 1
                
            except Exception as e:
                print(f"⚠️  Error processing {date_key}: {e}")
                failed_dates += 1
                continue
        
        print(f"✅ Processed {len(data)} days of option chain data")
        print(f"   Successful: {successful_dates}, Failed: {failed_dates}")
        
        return data