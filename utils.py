import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def format_currency(value, decimals=2):
    """
    Format currency values with proper formatting
    """
    try:
        if pd.isna(value) or value is None:
            return "N/A"
        
        if abs(value) >= 1e9:
            return f"${value/1e9:.{decimals}f}B"
        elif abs(value) >= 1e6:
            return f"${value/1e6:.{decimals}f}M"
        elif abs(value) >= 1e3:
            return f"${value/1e3:.{decimals}f}K"
        else:
            return f"${value:.{decimals}f}"
    except:
        return "N/A"

def format_percentage(value, decimals=2):
    """
    Format percentage values
    """
    try:
        if pd.isna(value) or value is None:
            return "N/A"
        return f"{value:.{decimals}f}%"
    except:
        return "N/A"

def get_color_for_change(value):
    """
    Get appropriate color for positive/negative changes
    """
    try:
        if pd.isna(value) or value is None:
            return "#212121"  # Neutral color
        
        if value > 0:
            return "#388E3C"  # Profit green
        elif value < 0:
            return "#D32F2F"  # Loss red
        else:
            return "#212121"  # Neutral
    except:
        return "#212121"

def calculate_returns(prices):
    """
    Calculate returns from price series
    """
    try:
        if len(prices) < 2:
            return pd.Series([])
        
        returns = prices.pct_change().dropna()
        return returns
    except:
        return pd.Series([])

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate Sharpe ratio
    """
    try:
        if returns.empty or returns.std() == 0:
            return 0
        
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        
        return excess_returns / volatility
    except:
        return 0

def calculate_max_drawdown(prices):
    """
    Calculate maximum drawdown
    """
    try:
        if len(prices) < 2:
            return 0
        
        peak = prices.cummax()
        drawdown = (prices - peak) / peak
        max_drawdown = drawdown.min()
        
        return abs(max_drawdown) * 100
    except:
        return 0

def calculate_volatility(returns, annualize=True):
    """
    Calculate volatility (standard deviation of returns)
    """
    try:
        if returns.empty:
            return 0
        
        vol = returns.std()
        
        if annualize:
            vol *= np.sqrt(252)  # Annualize assuming 252 trading days
        
        return vol * 100  # Convert to percentage
    except:
        return 0

def validate_data_quality(data):
    """
    Validate data quality and return issues found
    """
    issues = []
    
    try:
        if data is None or data.empty:
            issues.append("Data is empty")
            return issues
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        # Check for missing values
        if data.isnull().any().any():
            null_counts = data.isnull().sum()
            null_columns = null_counts[null_counts > 0].to_dict()
            issues.append(f"Missing values found: {null_columns}")
        
        # Check for negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns and (data[col] <= 0).any():
                issues.append(f"Non-positive values found in {col}")
        
        # Check for logical inconsistencies
        if all(col in data.columns for col in ['High', 'Low', 'Open', 'Close']):
            if (data['High'] < data['Low']).any():
                issues.append("High prices less than Low prices detected")
            
            if (data['High'] < data['Open']).any() or (data['High'] < data['Close']).any():
                issues.append("High prices less than Open/Close prices detected")
            
            if (data['Low'] > data['Open']).any() or (data['Low'] > data['Close']).any():
                issues.append("Low prices greater than Open/Close prices detected")
        
        # Check data length
        if len(data) < 50:
            issues.append(f"Insufficient data length: {len(data)} rows (minimum 50 recommended)")
        
        return issues
        
    except Exception as e:
        issues.append(f"Error validating data: {str(e)}")
        return issues

def clean_data(data):
    """
    Clean and prepare data for analysis
    """
    try:
        if data is None or data.empty:
            return data
        
        # Make a copy to avoid modifying original data
        cleaned_data = data.copy()
        
        # Remove rows with all NaN values
        cleaned_data = cleaned_data.dropna(how='all')
        
        # Forward fill missing values (carry forward last known value)
        cleaned_data = cleaned_data.fillna(method='ffill')
        
        # Remove any remaining NaN values
        cleaned_data = cleaned_data.dropna()
        
        # Ensure price columns are positive
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = cleaned_data[col].abs()
        
        # Ensure Volume is positive
        if 'Volume' in cleaned_data.columns:
            cleaned_data['Volume'] = cleaned_data['Volume'].abs()
        
        return cleaned_data
        
    except Exception as e:
        print(f"Error cleaning data: {str(e)}")
        return data

def get_market_session_info():
    """
    Get information about current market session
    """
    try:
        now = datetime.now()
        
        # US market hours (Eastern Time)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Check if it's a weekday
        is_weekday = now.weekday() < 5
        
        # Check if market is open
        is_market_hours = market_open <= now <= market_close
        
        is_market_open = is_weekday and is_market_hours
        
        if is_market_open:
            status = "Market Open"
            next_event = f"Market closes at {market_close.strftime('%H:%M')}"
        elif is_weekday and now < market_open:
            status = "Pre-Market"
            next_event = f"Market opens at {market_open.strftime('%H:%M')}"
        elif is_weekday and now > market_close:
            status = "After-Hours"
            next_event = "Market opens tomorrow at 09:30"
        else:
            status = "Market Closed (Weekend)"
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 1
            next_monday = now + timedelta(days=days_until_monday)
            next_event = f"Market opens {next_monday.strftime('%A')} at 09:30"
        
        return {
            'status': status,
            'is_open': is_market_open,
            'next_event': next_event,
            'current_time': now.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        return {
            'status': 'Unknown',
            'is_open': False,
            'next_event': 'Unable to determine',
            'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

def calculate_support_resistance(data, window=20):
    """
    Calculate support and resistance levels
    """
    try:
        if len(data) < window:
            return None, None
        
        # Calculate rolling highs and lows
        resistance = data['High'].rolling(window=window).max()
        support = data['Low'].rolling(window=window).min()
        
        return support.iloc[-1], resistance.iloc[-1]
        
    except Exception as e:
        print(f"Error calculating support/resistance: {str(e)}")
        return None, None

def get_trading_recommendation(signals_strength):
    """
    Generate trading recommendation based on signal strength
    """
    try:
        if signals_strength is None:
            return "HOLD", "Insufficient data for recommendation"
        
        strength = signals_strength.get('strength', 0)
        direction = signals_strength.get('direction', 'neutral')
        
        if direction == 'buy' and strength > 0.7:
            return "STRONG BUY", f"High conviction buy signal (strength: {strength:.2f})"
        elif direction == 'buy' and strength > 0.4:
            return "BUY", f"Moderate buy signal (strength: {strength:.2f})"
        elif direction == 'sell' and strength > 0.7:
            return "STRONG SELL", f"High conviction sell signal (strength: {strength:.2f})"
        elif direction == 'sell' and strength > 0.4:
            return "SELL", f"Moderate sell signal (strength: {strength:.2f})"
        else:
            return "HOLD", f"Neutral signals (strength: {strength:.2f})"
            
    except Exception as e:
        return "HOLD", f"Error generating recommendation: {str(e)}"
