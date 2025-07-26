import pandas as pd
import numpy as np
from typing import Optional

class TechnicalIndicators:
    def __init__(self):
        pass
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for the given data using basic pandas operations
        """
        if data is None or data.empty:
            return data
        
        df = data.copy()
        
        try:
            # Moving Averages
            df['MA_5'] = df['Close'].rolling(window=5).mean()
            df['MA_10'] = df['Close'].rolling(window=10).mean()
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            df['MA_50'] = df['Close'].rolling(window=50).mean()
            df['MA_200'] = df['Close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # RSI (Relative Strength Index)
            df['RSI'] = self._calculate_rsi(df['Close'], 14)
            
            # MACD (Moving Average Convergence Divergence)
            macd_line = df['EMA_12'] - df['EMA_26']
            macd_signal = macd_line.ewm(span=9).mean()
            df['MACD'] = macd_line
            df['MACD_Signal'] = macd_signal
            df['MACD_Histogram'] = macd_line - macd_signal
            
            # Bollinger Bands
            bb_middle = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = bb_middle + (bb_std * 2)
            df['BB_Middle'] = bb_middle
            df['BB_Lower'] = bb_middle - (bb_std * 2)
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # Stochastic Oscillator
            stoch_k, stoch_d = self._calculate_stochastic(df, 14, 3)
            df['Stoch_K'] = stoch_k
            df['Stoch_D'] = stoch_d
            
            # Williams %R
            df['Williams_R'] = self._calculate_williams_r(df, 14)
            
            # Average True Range (ATR)
            df['ATR'] = self._calculate_atr(df, 14)
            
            # On Balance Volume (OBV)
            df['OBV'] = self._calculate_obv(df)
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Commodity Channel Index (CCI)
            df['CCI'] = self._calculate_cci(df, 20)
            
            # Money Flow Index (MFI)
            df['MFI'] = self._calculate_mfi(df, 14)
            
            # Awesome Oscillator
            df['AO'] = self._calculate_ao(df)
            
            # Aroon
            aroon_up, aroon_down = self._calculate_aroon(df, 14)
            df['Aroon_Up'] = aroon_up
            df['Aroon_Down'] = aroon_down
            df['Aroon_Oscillator'] = aroon_up - aroon_down
            
            # ADX (Average Directional Index)
            adx, di_plus, di_minus = self._calculate_adx(df, 14)
            df['ADX'] = adx
            df['DI_Plus'] = di_plus
            df['DI_Minus'] = di_minus
            
            # Price-based features
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_5d'] = df['Close'].pct_change(5)
            df['High_Low_Ratio'] = df['High'] / df['Low']
            df['Close_Open_Ratio'] = df['Close'] / df['Open']
            
            # Volatility indicators
            df['Volatility_5d'] = df['Price_Change'].rolling(5).std()
            df['Volatility_20d'] = df['Price_Change'].rolling(20).std()
            
            # Support and Resistance levels
            df['Resistance_20d'] = df['High'].rolling(20).max()
            df['Support_20d'] = df['Low'].rolling(20).min()
            df['Price_Position'] = (df['Close'] - df['Support_20d']) / (df['Resistance_20d'] - df['Support_20d'])
            
            # Trend indicators
            df['Trend_5d'] = np.where(df['Close'] > df['Close'].shift(5), 1, -1)
            df['Trend_20d'] = np.where(df['Close'] > df['Close'].shift(20), 1, -1)
            
            # Gap analysis
            df['Gap'] = df['Open'] - df['Close'].shift(1)
            df['Gap_Percent'] = df['Gap'] / df['Close'].shift(1)
            
            return df
            
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            return data
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI (Relative Strength Index)"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_stochastic(self, df, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        try:
            lowest_low = df['Low'].rolling(window=k_period).min()
            highest_high = df['High'].rolling(window=k_period).max()
            stoch_k = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
            stoch_d = stoch_k.rolling(window=d_period).mean()
            return stoch_k, stoch_d
        except:
            return pd.Series([50] * len(df), index=df.index), pd.Series([50] * len(df), index=df.index)
    
    def _calculate_williams_r(self, df, period=14):
        """Calculate Williams %R"""
        try:
            highest_high = df['High'].rolling(window=period).max()
            lowest_low = df['Low'].rolling(window=period).min()
            williams_r = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))
            return williams_r
        except:
            return pd.Series([-50] * len(df), index=df.index)
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        try:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            return atr
        except:
            return pd.Series([1] * len(df), index=df.index)
    
    def _calculate_obv(self, df):
        """Calculate On Balance Volume"""
        try:
            obv = pd.Series(index=df.index, dtype=float)
            obv.iloc[0] = df['Volume'].iloc[0]
            
            for i in range(1, len(df)):
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
                elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            return obv
        except:
            return pd.Series([0] * len(df), index=df.index)
    
    def _calculate_cci(self, df, period=20):
        """Calculate Commodity Channel Index"""
        try:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            sma = typical_price.rolling(window=period).mean()
            mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma) / (0.015 * mean_deviation)
            return cci
        except:
            return pd.Series([0] * len(df), index=df.index)
    
    def _calculate_mfi(self, df, period=14):
        """Calculate Money Flow Index"""
        try:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
            
            positive_mf = positive_flow.rolling(window=period).sum()
            negative_mf = negative_flow.rolling(window=period).sum()
            
            mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
            return mfi
        except:
            return pd.Series([50] * len(df), index=df.index)
    
    def _calculate_ao(self, df):
        """Calculate Awesome Oscillator"""
        try:
            median_price = (df['High'] + df['Low']) / 2
            ao = median_price.rolling(window=5).mean() - median_price.rolling(window=34).mean()
            return ao
        except:
            return pd.Series([0] * len(df), index=df.index)
    
    def _calculate_aroon(self, df, period=14):
        """Calculate Aroon indicators"""
        try:
            aroon_up = pd.Series(index=df.index, dtype=float)
            aroon_down = pd.Series(index=df.index, dtype=float)
            
            for i in range(period, len(df)):
                high_idx = df['High'].iloc[i-period:i+1].idxmax()
                low_idx = df['Low'].iloc[i-period:i+1].idxmin()
                
                periods_since_high = i - df.index.get_loc(high_idx)
                periods_since_low = i - df.index.get_loc(low_idx)
                
                aroon_up.iloc[i] = ((period - periods_since_high) / period) * 100
                aroon_down.iloc[i] = ((period - periods_since_low) / period) * 100
            
            return aroon_up, aroon_down
        except:
            return pd.Series([50] * len(df), index=df.index), pd.Series([50] * len(df), index=df.index)
    
    def _calculate_adx(self, df, period=14):
        """Calculate ADX and Directional Indicators"""
        try:
            # Calculate True Range
            tr = self._calculate_atr(df, 1)
            
            # Calculate Directional Movement
            plus_dm = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
                              np.maximum(df['High'] - df['High'].shift(1), 0), 0)
            minus_dm = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
                               np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)
            
            plus_dm = pd.Series(plus_dm, index=df.index)
            minus_dm = pd.Series(minus_dm, index=df.index)
            
            # Smooth the values
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
            
            # Calculate ADX
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            
            return adx, plus_di, minus_di
        except:
            return (pd.Series([25] * len(df), index=df.index), 
                   pd.Series([25] * len(df), index=df.index),
                   pd.Series([25] * len(df), index=df.index))
    
    def prepare_ml_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Prepare features for machine learning models
        """
        try:
            # Ensure all indicators are calculated
            df = self.calculate_all_indicators(data)
            
            # Select relevant features for ML
            feature_columns = [
                # Price features
                'Open', 'High', 'Low', 'Close', 'Volume',
                
                # Moving averages
                'MA_5', 'MA_10', 'MA_20', 'MA_50',
                'EMA_12', 'EMA_26',
                
                # Oscillators
                'RSI', 'Stoch_K', 'Stoch_D', 'Williams_R', 'CCI', 'MFI',
                
                # MACD
                'MACD', 'MACD_Signal', 'MACD_Histogram',
                
                # Bollinger Bands
                'BB_Width', 'BB_Position',
                
                # Volume indicators
                'OBV', 'Volume_Ratio',
                
                # Trend indicators
                'ATR', 'ADX', 'DI_Plus', 'DI_Minus',
                'Aroon_Up', 'Aroon_Down', 'Aroon_Oscillator',
                
                # Price ratios and changes
                'Price_Change', 'Price_Change_5d',
                'High_Low_Ratio', 'Close_Open_Ratio',
                
                # Volatility
                'Volatility_5d', 'Volatility_20d',
                
                # Position indicators
                'Price_Position',
                
                # Trend signals
                'Trend_5d', 'Trend_20d',
                
                # Gap analysis
                'Gap_Percent'
            ]
            
            # Select only existing columns
            available_columns = [col for col in feature_columns if col in df.columns]
            
            features_df = df[available_columns].copy()
            
            # Add target variable (next week's closing price)
            features_df['Target'] = df['Close'].shift(-5)  # 5 days ahead (1 week)
            features_df['Target_Direction'] = np.where(features_df['Target'] > df['Close'], 1, 0)
            
            # Remove rows with NaN values
            features_df = features_df.dropna()
            
            if len(features_df) < 50:
                print("Insufficient data for ML training after cleaning")
                return None
            
            return features_df
            
        except Exception as e:
            print(f"Error preparing ML features: {str(e)}")
            return None
    
    def get_signal_strength(self, data: pd.DataFrame) -> dict:
        """
        Calculate overall signal strength based on multiple indicators
        """
        try:
            if data.empty or len(data) < 2:
                return {'strength': 0, 'direction': 'neutral', 'signals': []}
            
            signals = []
            current_data = data.iloc[-1]
            
            # RSI signals
            rsi = current_data.get('RSI', 50)
            if rsi > 70:
                signals.append({'indicator': 'RSI', 'signal': 'sell', 'strength': min((rsi - 70) / 10, 1)})
            elif rsi < 30:
                signals.append({'indicator': 'RSI', 'signal': 'buy', 'strength': min((30 - rsi) / 10, 1)})
            
            # MACD signals
            macd = current_data.get('MACD', 0)
            macd_signal = current_data.get('MACD_Signal', 0)
            if macd > macd_signal:
                signals.append({'indicator': 'MACD', 'signal': 'buy', 'strength': min(abs(macd - macd_signal) / macd_signal, 1) if macd_signal != 0 else 0.5})
            else:
                signals.append({'indicator': 'MACD', 'signal': 'sell', 'strength': min(abs(macd - macd_signal) / abs(macd_signal), 1) if macd_signal != 0 else 0.5})
            
            # Moving average signals
            close = current_data.get('Close', 0)
            ma_20 = current_data.get('MA_20', close)
            ma_50 = current_data.get('MA_50', close)
            
            if close > ma_20 > ma_50:
                signals.append({'indicator': 'MA', 'signal': 'buy', 'strength': 0.7})
            elif close < ma_20 < ma_50:
                signals.append({'indicator': 'MA', 'signal': 'sell', 'strength': 0.7})
            
            # Bollinger Bands signals
            bb_position = current_data.get('BB_Position', 0.5)
            if bb_position > 1:
                signals.append({'indicator': 'BB', 'signal': 'sell', 'strength': min(bb_position - 1, 1)})
            elif bb_position < 0:
                signals.append({'indicator': 'BB', 'signal': 'buy', 'strength': min(abs(bb_position), 1)})
            
            # Calculate overall strength and direction
            buy_strength = sum([s['strength'] for s in signals if s['signal'] == 'buy'])
            sell_strength = sum([s['strength'] for s in signals if s['signal'] == 'sell'])
            
            total_strength = buy_strength + sell_strength
            
            if total_strength == 0:
                return {'strength': 0, 'direction': 'neutral', 'signals': signals}
            
            net_strength = (buy_strength - sell_strength) / total_strength
            
            if net_strength > 0.2:
                direction = 'buy'
            elif net_strength < -0.2:
                direction = 'sell'
            else:
                direction = 'neutral'
            
            return {
                'strength': abs(net_strength),
                'direction': direction,
                'signals': signals,
                'buy_strength': buy_strength,
                'sell_strength': sell_strength
            }
            
        except Exception as e:
            print(f"Error calculating signal strength: {str(e)}")
            return {'strength': 0, 'direction': 'neutral', 'signals': []}
