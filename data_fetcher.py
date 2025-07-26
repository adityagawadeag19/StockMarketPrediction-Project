import yfinance as yf
import pandas as pd
import requests
import os
from datetime import datetime, timedelta
import streamlit as st

class DataFetcher:
    def __init__(self):
        self.alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
        self.base_url = "https://www.alphavantage.co/query"
    
    def get_stock_data(self, symbol, period="1y"):
        """
        Fetch stock data using yfinance as primary source
        Falls back to Alpha Vantage if needed
        """
        try:
            # Primary: Use yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                st.warning(f"No data found for {symbol} using yfinance, trying Alpha Vantage...")
                return self._get_alpha_vantage_data(symbol)
            
            # Clean and prepare data
            data = data.reset_index()
            data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            data = data.set_index('Date')
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            # Remove any rows with missing data
            data = data.dropna()
            
            if len(data) < 50:
                st.warning(f"Insufficient data for {symbol}. Only {len(data)} days available.")
                return None
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def _get_alpha_vantage_data(self, symbol):
        """
        Fetch data from Alpha Vantage API as backup
        """
        try:
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'apikey': self.alpha_vantage_api_key,
                'outputsize': 'full'
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            data = response.json()
            
            if 'Error Message' in data:
                st.error(f"Alpha Vantage Error: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                st.warning("Alpha Vantage API call frequency limit reached. Please try again later.")
                return None
            
            if 'Time Series (Daily)' not in data:
                st.error("Unexpected response format from Alpha Vantage")
                return None
            
            # Parse Alpha Vantage data
            time_series = data['Time Series (Daily)']
            
            df_data = []
            for date, values in time_series.items():
                df_data.append({
                    'Date': pd.to_datetime(date),
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Volume': int(values['6. volume'])
                })
            
            df = pd.DataFrame(df_data)
            df = df.set_index('Date').sort_index()
            
            # Limit to last year for consistency
            one_year_ago = datetime.now() - timedelta(days=365)
            df = df[df.index >= one_year_ago]
            
            return df
            
        except requests.exceptions.RequestException as e:
            st.error(f"Network error fetching from Alpha Vantage: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Error parsing Alpha Vantage data: {str(e)}")
            return None
    
    def get_options_data(self, symbol):
        """
        Fetch options data for the given symbol
        Note: yfinance provides limited options data
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get options expiration dates
            exp_dates = ticker.options
            
            if not exp_dates:
                return None
            
            # Get options chain for the nearest expiration
            options_chain = ticker.option_chain(exp_dates[0])
            
            return {
                'calls': options_chain.calls,
                'puts': options_chain.puts,
                'expiration': exp_dates[0]
            }
            
        except Exception as e:
            st.warning(f"Could not fetch options data for {symbol}: {str(e)}")
            return None
    
    def get_market_indicators(self):
        """
        Fetch broad market indicators for context
        """
        try:
            # Fetch major indices
            indices = {
                'SPY': yf.Ticker('SPY').history(period='5d'),
                'QQQ': yf.Ticker('QQQ').history(period='5d'),
                'VIX': yf.Ticker('^VIX').history(period='5d')
            }
            
            market_data = {}
            for symbol, data in indices.items():
                if not data.empty:
                    current = data['Close'].iloc[-1]
                    prev = data['Close'].iloc[-2]
                    change = ((current - prev) / prev) * 100
                    
                    market_data[symbol] = {
                        'current': current,
                        'change': change
                    }
            
            return market_data
            
        except Exception as e:
            st.warning(f"Could not fetch market indicators: {str(e)}")
            return {}
    
    def validate_symbol(self, symbol):
        """
        Validate if a stock symbol exists and has data
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check if we got valid info
            if 'symbol' in info and info['symbol']:
                return True
            
            return False
            
        except Exception:
            return False
    
    def get_stock_info(self, symbol):
        """
        Get additional stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', None),
                'dividend_yield': info.get('dividendYield', None),
                'beta': info.get('beta', None)
            }
            
        except Exception as e:
            st.warning(f"Could not fetch stock info for {symbol}: {str(e)}")
            return {'name': symbol}
