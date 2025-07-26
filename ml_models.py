import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class MLModels:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        
    def prepare_data(self, features_df, target_column='Target', test_size=0.2):
        """
        Prepare data for machine learning models
        """
        try:
            if features_df is None or features_df.empty:
                return None, None, None, None
            
            # Separate features and target
            feature_columns = [col for col in features_df.columns if col not in ['Target', 'Target_Direction']]
            X = features_df[feature_columns].copy()
            y = features_df[target_column].copy()
            
            # Remove any remaining NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            if len(X) < 50:
                return None, None, None, None
            
            # Use time series split to maintain temporal order
            tscv = TimeSeriesSplit(n_splits=5, test_size=int(len(X) * test_size))
            splits = list(tscv.split(X))
            train_idx, test_idx = splits[-1]  # Use the last split
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            return None, None, None, None
    
    def train_xgboost(self, features_df):
        """
        Train XGBoost model for price prediction
        """
        try:
            X_train, X_test, y_train, y_test = self.prepare_data(features_df)
            
            if X_train is None:
                return None, {}
            
            # Configure XGBoost parameters
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            
            # Train model
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Directional accuracy
            y_direction_true = (y_test > features_df.iloc[:-5]['Close'].iloc[-len(y_test):].values).astype(int)
            y_direction_pred = (y_pred > features_df.iloc[:-5]['Close'].iloc[-len(y_pred):].values).astype(int)
            directional_accuracy = accuracy_score(y_direction_true, y_direction_pred)
            
            # Generate future prediction
            latest_features = features_df.iloc[-1:].drop(['Target', 'Target_Direction'], axis=1)
            latest_scaled = self.scaler.transform(latest_features)
            future_prediction = model.predict(latest_scaled)[0]
            
            self.models['xgboost'] = model
            
            metrics = {
                'rmse': rmse,
                'mae': mae,
                'accuracy': directional_accuracy
            }
            
            return future_prediction, metrics
            
        except Exception as e:
            print(f"Error training XGBoost: {str(e)}")
            return None, {}
    
    def train_random_forest(self, features_df):
        """
        Train Random Forest model for price prediction
        """
        try:
            X_train, X_test, y_train, y_test = self.prepare_data(features_df)
            
            if X_train is None:
                return None, {}
            
            # Configure Random Forest parameters
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Directional accuracy
            y_direction_true = (y_test > features_df.iloc[:-5]['Close'].iloc[-len(y_test):].values).astype(int)
            y_direction_pred = (y_pred > features_df.iloc[:-5]['Close'].iloc[-len(y_pred):].values).astype(int)
            directional_accuracy = accuracy_score(y_direction_true, y_direction_pred)
            
            # Generate future prediction
            latest_features = features_df.iloc[-1:].drop(['Target', 'Target_Direction'], axis=1)
            latest_scaled = self.scaler.transform(latest_features)
            future_prediction = model.predict(latest_scaled)[0]
            
            self.models['random_forest'] = model
            
            metrics = {
                'rmse': rmse,
                'mae': mae,
                'accuracy': directional_accuracy
            }
            
            return future_prediction, metrics
            
        except Exception as e:
            print(f"Error training Random Forest: {str(e)}")
            return None, {}
    
    def train_lstm(self, features_df, sequence_length=20):
        """
        Simple Linear Regression model as LSTM replacement for now
        """
        try:
            if features_df is None or len(features_df) < 50:
                return None, {}
            
            # Use linear regression as a simple alternative
            from sklearn.linear_model import LinearRegression
            
            X_train, X_test, y_train, y_test = self.prepare_data(features_df)
            
            if X_train is None:
                return None, {}
            
            # Train linear model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Directional accuracy
            y_direction_true = (y_test > features_df.iloc[:-5]['Close'].iloc[-len(y_test):].values).astype(int)
            y_direction_pred = (y_pred > features_df.iloc[:-5]['Close'].iloc[-len(y_pred):].values).astype(int)
            directional_accuracy = accuracy_score(y_direction_true, y_direction_pred)
            
            # Generate future prediction
            latest_features = features_df.iloc[-1:].drop(['Target', 'Target_Direction'], axis=1)
            latest_scaled = self.scaler.transform(latest_features)
            future_prediction = model.predict(latest_scaled)[0]
            
            self.models['lstm'] = model
            
            metrics = {
                'rmse': rmse,
                'mae': mae,
                'accuracy': directional_accuracy
            }
            
            return future_prediction, metrics
            
        except Exception as e:
            print(f"Error training Linear model: {str(e)}")
            return None, {}
    
    def get_feature_importance(self, model_name):
        """
        Get feature importance for tree-based models
        """
        try:
            if model_name not in self.models:
                return None
            
            model = self.models[model_name]
            
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            
            return None
            
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            return None
    
    def ensemble_prediction(self, predictions):
        """
        Combine predictions from multiple models
        """
        try:
            if not predictions:
                return None
            
            # Simple average ensemble
            ensemble_pred = np.mean(list(predictions.values()))
            
            # Weighted ensemble based on model reliability
            weights = {
                'XGBoost': 0.4,
                'Random Forest': 0.35,
                'LSTM': 0.25
            }
            
            weighted_pred = 0
            total_weight = 0
            
            for model_name, prediction in predictions.items():
                if model_name in weights:
                    weighted_pred += prediction * weights[model_name]
                    total_weight += weights[model_name]
            
            if total_weight > 0:
                weighted_pred /= total_weight
                return weighted_pred
            
            return ensemble_pred
            
        except Exception as e:
            print(f"Error creating ensemble prediction: {str(e)}")
            return None
