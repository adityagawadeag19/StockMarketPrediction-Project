import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Import our modules
from data_fetcher import DataFetcher
from utils import format_currency, format_percentage, get_color_for_change

# Page configuration
st.set_page_config(
    page_title="Options Trading Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1976D2 0%, #42A5F5 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1976D2;
    }
    
    .bullish-card {
        border-left-color: #388E3C !important;
        background: linear-gradient(90deg, #E8F5E8 0%, #F1F8E9 100%);
    }
    
    .bearish-card {
        border-left-color: #D32F2F !important;
        background: linear-gradient(90deg, #FFEBEE 0%, #FFCDD2 100%);
    }
    
    .neutral-card {
        border-left-color: #F57C00 !important;
        background: linear-gradient(90deg, #FFF3E0 0%, #FFE0B2 100%);
    }
    
    .strategy-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    
    .confidence-high {
        border-left: 5px solid #388E3C;
    }
    
    .confidence-medium {
        border-left: 5px solid #F57C00;
    }
    
    .confidence-low {
        border-left: 5px solid #D32F2F;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #F5F5F5 0%, #FAFAFA 100%);
    }
    
    .scalping-setup {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #2196F3;
    }
    
    .risk-warning {
        background: linear-gradient(90deg, #FFF3E0 0%, #FFCC02 20%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = DataFetcher()

def calculate_basic_indicators(data):
    """Calculate basic technical indicators without external libraries"""
    df = data.copy()
    
    # Moving Averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    bb_middle = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = bb_middle + (bb_std * 2)
    df['BB_Lower'] = bb_middle - (bb_std * 2)
    
    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    
    # ATR for volatility
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # Volume indicators
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Price momentum
    df['Price_Change_1d'] = df['Close'].pct_change()
    df['Price_Change_5d'] = df['Close'].pct_change(5)
    
    return df

def simple_prediction(data):
    """Simple price prediction using linear trend"""
    try:
        # Use last 20 days for trend calculation
        recent_data = data['Close'].tail(20)
        x = np.arange(len(recent_data))
        
        # Simple linear regression
        slope = np.polyfit(x, recent_data.values, 1)[0]
        current_price = data['Close'].iloc[-1]
        
        # Predict 5 days ahead (1 week)
        predicted_price = current_price + (slope * 5)
        
        return predicted_price
    except:
        return data['Close'].iloc[-1]  # Return current price if prediction fails

def generate_trading_signals(data):
    """Generate simple trading signals"""
    signals = []
    current_price = data['Close'].iloc[-1]
    rsi = data['RSI'].iloc[-1]
    ma_20 = data['MA_20'].iloc[-1]
    ma_50 = data['MA_50'].iloc[-1]
    
    # RSI signals
    if rsi < 30:
        signals.append("🟢 RSI Oversold - Buy Signal")
    elif rsi > 70:
        signals.append("🔴 RSI Overbought - Sell Signal")
    else:
        signals.append("🟡 RSI Neutral")
    
    # Moving average signals
    if current_price > ma_20 > ma_50:
        signals.append("🟢 Price above MAs - Uptrend")
    elif current_price < ma_20 < ma_50:
        signals.append("🔴 Price below MAs - Downtrend")
    else:
        signals.append("🟡 Mixed MA signals")
    
    return signals

def get_next_expiration_dates(current_date, num_days=30):
    """Get next expiration dates (Fridays) for options"""
    expiration_dates = []
    current = current_date
    
    # Find next Friday
    days_ahead = 4 - current.weekday()  # Friday is weekday 4
    if days_ahead <= 0:  # Target day already passed this week
        days_ahead += 7
    
    next_friday = current + timedelta(days=days_ahead)
    
    # Generate next few Fridays
    for i in range(6):  # Next 6 expiration dates
        exp_date = next_friday + timedelta(weeks=i)
        expiration_dates.append(exp_date)
    
    return expiration_dates

def generate_daily_options_strategies(data, symbol, strategy_pref):
    """Generate daily options strategies for next 30 days with proper expiration dates"""
    current_price = data['Close'].iloc[-1]
    current_date = datetime.now()
    expiration_dates = get_next_expiration_dates(current_date)
    
    daily_strategies = []
    
    # Generate strategies for next 30 days
    for day_offset in range(1, 31):
        target_date = current_date + timedelta(days=day_offset)
        
        # Skip weekends for trading
        if target_date.weekday() >= 5:
            continue
        
        # Find appropriate expiration date (next Friday after target date)
        exp_date = None
        for exp in expiration_dates:
            if exp >= target_date:
                exp_date = exp
                break
        
        if not exp_date:
            continue
        
        # Calculate days to expiration
        days_to_exp = (exp_date - target_date).days
        
        # Get recent market indicators
        rsi = data['RSI'].iloc[-1]
        atr = data['ATR'].iloc[-1]
        volume_ratio = data['Volume_Ratio'].iloc[-1]
        price_change_5d = data['Price_Change_5d'].iloc[-1] * 100
        
        # Generate volatility-based prediction
        volatility = data['Close'].pct_change().tail(20).std() * np.sqrt(252)
        daily_move = volatility / np.sqrt(252) * current_price
        
        # Determine market regime
        if rsi > 70:
            regime = "Overbought"
            bias = "Bearish"
        elif rsi < 30:
            regime = "Oversold"
            bias = "Bullish"
        elif price_change_5d > 3:
            regime = "Trending Up"
            bias = "Bullish"
        elif price_change_5d < -3:
            regime = "Trending Down"
            bias = "Bearish"
        else:
            regime = "Range-bound"
            bias = "Neutral"
        
        # Generate strategy based on days to expiration and market regime
        strategies = []
        
        if days_to_exp <= 7:  # Weekly options
            if bias == "Bullish":
                strike_call = round(current_price * 1.01, 2)
                strategies.append({
                    'date': target_date.strftime('%Y-%m-%d'),
                    'type': 'Buy Weekly Call',
                    'strike': strike_call,
                    'expiration': exp_date.strftime('%Y-%m-%d'),
                    'days_to_exp': days_to_exp,
                    'confidence': min(85, 60 + abs(price_change_5d)),
                    'rationale': f"{regime} - Expected upward move",
                    'max_risk': f"${strike_call * 0.05:.2f} per contract",
                    'target_profit': f"${strike_call * 0.15:.2f} per contract"
                })
            elif bias == "Bearish":
                strike_put = round(current_price * 0.99, 2)
                strategies.append({
                    'date': target_date.strftime('%Y-%m-%d'),
                    'type': 'Buy Weekly Put',
                    'strike': strike_put,
                    'expiration': exp_date.strftime('%Y-%m-%d'),
                    'days_to_exp': days_to_exp,
                    'confidence': min(85, 60 + abs(price_change_5d)),
                    'rationale': f"{regime} - Expected downward move",
                    'max_risk': f"${strike_put * 0.05:.2f} per contract",
                    'target_profit': f"${strike_put * 0.15:.2f} per contract"
                })
            else:
                # Iron Condor for neutral
                strategies.append({
                    'date': target_date.strftime('%Y-%m-%d'),
                    'type': 'Iron Condor',
                    'strike': f"{current_price * 0.98:.2f} / {current_price * 1.02:.2f}",
                    'expiration': exp_date.strftime('%Y-%m-%d'),
                    'days_to_exp': days_to_exp,
                    'confidence': 65,
                    'rationale': f"{regime} - Range-bound movement expected",
                    'max_risk': "$200 per spread",
                    'target_profit': "$100 per spread"
                })
        
        elif days_to_exp <= 30:  # Monthly options
            if bias == "Bullish":
                strike_call = round(current_price * 1.02, 2)
                strategies.append({
                    'date': target_date.strftime('%Y-%m-%d'),
                    'type': 'Buy Monthly Call',
                    'strike': strike_call,
                    'expiration': exp_date.strftime('%Y-%m-%d'),
                    'days_to_exp': days_to_exp,
                    'confidence': min(80, 55 + abs(price_change_5d)),
                    'rationale': f"{regime} - Medium-term bullish outlook",
                    'max_risk': f"${strike_call * 0.08:.2f} per contract",
                    'target_profit': f"${strike_call * 0.25:.2f} per contract"
                })
            elif bias == "Bearish":
                strike_put = round(current_price * 0.98, 2)
                strategies.append({
                    'date': target_date.strftime('%Y-%m-%d'),
                    'type': 'Buy Monthly Put',
                    'strike': strike_put,
                    'expiration': exp_date.strftime('%Y-%m-%d'),
                    'days_to_exp': days_to_exp,
                    'confidence': min(80, 55 + abs(price_change_5d)),
                    'rationale': f"{regime} - Medium-term bearish outlook",
                    'max_risk': f"${strike_put * 0.08:.2f} per contract",
                    'target_profit': f"${strike_put * 0.25:.2f} per contract"
                })
        
        if strategies:
            daily_strategies.extend(strategies)
    
    return daily_strategies

def generate_scalping_strategy(data, symbol):
    """Generate scalping strategy for next trading day"""
    current_price = data['Close'].iloc[-1]
    atr = data['ATR'].iloc[-1]
    volume_ratio = data['Volume_Ratio'].iloc[-1]
    rsi = data['RSI'].iloc[-1]
    
    # Calculate support and resistance levels
    recent_data = data.tail(20)
    resistance = recent_data['High'].max()
    support = recent_data['Low'].min()
    
    # Calculate pivot points
    prev_high = data['High'].iloc[-1]
    prev_low = data['Low'].iloc[-1]
    prev_close = data['Close'].iloc[-1]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    scalping_setups = []
    
    # Momentum scalp
    if volume_ratio > 1.5 and abs(data['Price_Change_1d'].iloc[-1]) > 0.02:
        scalping_setups.append({
            'type': 'Momentum Scalp',
            'entry_range': f"${current_price * 0.998:.2f} - ${current_price * 1.002:.2f}",
            'stop_loss': f"${current_price - (atr * 0.5):.2f}",
            'take_profit': f"${current_price + (atr * 1.0):.2f}",
            'time_frame': '1-5 minutes',
            'rationale': f"High volume ({volume_ratio:.1f}x avg) with momentum",
            'risk_reward': '1:2'
        })
    
    # Mean reversion scalp
    if rsi > 75:
        scalping_setups.append({
            'type': 'Mean Reversion Short',
            'entry_range': f"${current_price * 1.001:.2f} - ${current_price * 1.003:.2f}",
            'stop_loss': f"${current_price + (atr * 0.3):.2f}",
            'take_profit': f"${current_price - (atr * 0.6):.2f}",
            'time_frame': '5-15 minutes',
            'rationale': f"Overbought RSI ({rsi:.1f}) - expect pullback",
            'risk_reward': '1:2'
        })
    elif rsi < 25:
        scalping_setups.append({
            'type': 'Mean Reversion Long',
            'entry_range': f"${current_price * 0.997:.2f} - ${current_price * 0.999:.2f}",
            'stop_loss': f"${current_price - (atr * 0.3):.2f}",
            'take_profit': f"${current_price + (atr * 0.6):.2f}",
            'time_frame': '5-15 minutes',
            'rationale': f"Oversold RSI ({rsi:.1f}) - expect bounce",
            'risk_reward': '1:2'
        })
    
    # Breakout scalp
    if current_price > resistance * 0.999:
        scalping_setups.append({
            'type': 'Breakout Long',
            'entry_range': f"${resistance:.2f} - ${resistance * 1.002:.2f}",
            'stop_loss': f"${resistance * 0.998:.2f}",
            'take_profit': f"${resistance * 1.006:.2f}",
            'time_frame': '1-3 minutes',
            'rationale': f"Breaking above resistance ${resistance:.2f}",
            'risk_reward': '1:3'
        })
    elif current_price < support * 1.001:
        scalping_setups.append({
            'type': 'Breakdown Short',
            'entry_range': f"${support * 0.998:.2f} - ${support:.2f}",
            'stop_loss': f"${support * 1.002:.2f}",
            'take_profit': f"${support * 0.994:.2f}",
            'time_frame': '1-3 minutes',
            'rationale': f"Breaking below support ${support:.2f}",
            'risk_reward': '1:3'
        })
    
    # Pivot level scalp
    scalping_setups.append({
        'type': 'Pivot Level Trade',
        'key_levels': {
            'Pivot': f"${pivot:.2f}",
            'R1': f"${r1:.2f}",
            'S1': f"${s1:.2f}",
            'R2': f"${r2:.2f}",
            'S2': f"${s2:.2f}"
        },
        'strategy': 'Buy at support levels, Sell at resistance levels',
        'time_frame': '5-15 minutes',
        'rationale': 'Pivot point levels act as dynamic support/resistance'
    })
    
    return scalping_setups

def create_interactive_price_chart(data):
    """Create interactive price chart with technical indicators"""
    recent_data = data.tail(60).copy()
    
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.6, 0.25, 0.15],
        vertical_spacing=0.05,
        subplot_titles=('Price & Moving Averages', 'Volume', 'RSI'),
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=recent_data.index,
        open=recent_data['Open'],
        high=recent_data['High'],
        low=recent_data['Low'],
        close=recent_data['Close'],
        name="Price",
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444'
    ), row=1, col=1)
    
    # Moving averages
    fig.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data['MA_20'],
        mode='lines',
        name='MA 20',
        line=dict(color='#1976D2', width=1)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data['MA_50'],
        mode='lines',
        name='MA 50',
        line=dict(color='#FF9800', width=1)
    ), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data['BB_Upper'],
        mode='lines',
        name='BB Upper',
        line=dict(color='rgba(128,128,128,0.3)', width=1),
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data['BB_Lower'],
        mode='lines',
        name='BB Lower',
        line=dict(color='rgba(128,128,128,0.3)', width=1),
        fill='tonexty',
        fillcolor='rgba(128,128,128,0.1)',
        showlegend=False
    ), row=1, col=1)
    
    # Volume bars
    colors = ['#00ff88' if recent_data['Close'].iloc[i] > recent_data['Open'].iloc[i] 
              else '#ff4444' for i in range(len(recent_data))]
    
    fig.add_trace(go.Bar(
        x=recent_data.index,
        y=recent_data['Volume'],
        name='Volume',
        marker_color=colors,
        showlegend=False
    ), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data['RSI'],
        mode='lines',
        name='RSI',
        line=dict(color='#9C27B0', width=2)
    ), row=3, col=1)
    
    # RSI reference lines
    fig.add_shape(type="line", x0=recent_data.index[0], x1=recent_data.index[-1], 
                  y0=70, y1=70, line=dict(color="red", dash="dash"), row=3, col=1)
    fig.add_shape(type="line", x0=recent_data.index[0], x1=recent_data.index[-1], 
                  y0=30, y1=30, line=dict(color="green", dash="dash"), row=3, col=1)
    
    fig.update_layout(
        title="Interactive Price Analysis",
        height=700,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_white'
    )
    
    return fig

def create_volatility_gauge(data):
    """Create volatility gauge chart"""
    volatility = data['Close'].pct_change().tail(20).std() * np.sqrt(252) * 100
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = volatility,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Volatility %"},
        delta = {'reference': 20, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 60]},
            'bar': {'color': "#1976D2"},
            'steps': [
                {'range': [0, 15], 'color': "#E8F5E8"},
                {'range': [15, 30], 'color': "#FFF3E0"},
                {'range': [30, 60], 'color': "#FFEBEE"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 40}}))
    
    fig.update_layout(height=300)
    return fig

def create_confidence_chart(strategies):
    """Create confidence level chart for strategies"""
    if not strategies or len(strategies) == 0:
        return None
    
    try:
        # Get up to 10 strategies for display
        display_strategies = strategies[:10]
        strategy_types = []
        confidences = []
        
        for s in display_strategies:
            if 'type' in s and 'confidence' in s:
                # Truncate long strategy names for better display
                strategy_name = s['type']
                if len(strategy_name) > 15:
                    strategy_name = strategy_name[:12] + "..."
                strategy_types.append(strategy_name)
                confidences.append(float(s['confidence']))
        
        if not strategy_types or not confidences:
            return None
        
        # Color code based on confidence levels
        colors = []
        for c in confidences:
            if c > 70:
                colors.append('#388E3C')  # Green for high confidence
            elif c > 50:
                colors.append('#F57C00')  # Orange for medium confidence
            else:
                colors.append('#D32F2F')  # Red for low confidence
        
        fig = go.Figure(data=[
            go.Bar(
                x=strategy_types, 
                y=confidences, 
                marker_color=colors,
                text=[f"{c:.0f}%" for c in confidences],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Strategy Confidence Levels",
            xaxis_title="Strategy Type",
            yaxis_title="Confidence %",
            height=400,
            template='plotly_white',
            showlegend=False,
            yaxis=dict(range=[0, 100])
        )
        
        # Add reference lines for confidence levels
        fig.add_hline(y=70, line_dash="dash", line_color="#388E3C", opacity=0.3, annotation_text="High Confidence")
        fig.add_hline(y=50, line_dash="dash", line_color="#F57C00", opacity=0.3, annotation_text="Medium Confidence")
        
        return fig
        
    except Exception as e:
        print(f"Error creating confidence chart: {e}")
        return None

def main():
    # Main header with gradient background
    st.markdown("""
    <div class="main-header">
        <h1>📈 Options Trading Prediction System</h1>
        <p>AI-powered weekly options and futures trading signals with interactive analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for user inputs with enhanced styling
    with st.sidebar:
        st.markdown("### 🎯 Trading Configuration")
        
        # Popular symbols quick select
        st.markdown("**Quick Select:**")
        popular_symbols = st.selectbox(
            "Popular Stocks",
            ["Custom", "AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "NFLX", "AMD", "SPY"],
            help="Choose from popular trading symbols"
        )
        
        # Stock symbol input
        if popular_symbols == "Custom":
            symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter stock ticker (e.g., AAPL, MSFT)")
        else:
            symbol = popular_symbols
            st.text_input("Selected Symbol", value=symbol, disabled=True)
        
        st.markdown("---")
        
        # Time period selection with icons
        period = st.selectbox(
            "📊 Historical Period",
            ["6mo", "1y", "2y", "3y", "5y"],
            index=1,
            help="Historical data period for analysis"
        )
        
        # Strategy preference with descriptions
        strategy_pref = st.selectbox(
            "🎯 Strategy Preference",
            ["Conservative", "Moderate", "Aggressive"],
            index=1,
            help="Risk tolerance for options strategies"
        )
        
        # Advanced options in expander
        with st.expander("⚙️ Advanced Options"):
            show_detailed_analysis = st.checkbox("Show Detailed Technical Analysis", value=True)
            auto_refresh = st.checkbox("Auto-refresh Data", value=False)
            if auto_refresh:
                refresh_interval = st.slider("Refresh Interval (minutes)", 1, 60, 5)
        
        st.markdown("---")
        
        # Analysis button with custom styling
        analyze_button = st.button("🔍 Analyze & Predict", type="primary", use_container_width=True)
        
        if analyze_button:
            st.balloons()  # Fun animation when analyzing
        
        # Real-time market status
        st.markdown("### 📊 Market Status")
        market_time = datetime.now()
        if market_time.weekday() < 5 and 9 <= market_time.hour <= 16:
            st.success("🟢 Market Open")
        else:
            st.error("🔴 Market Closed")
        
        st.markdown(f"**Last Updated:** {market_time.strftime('%H:%M:%S')}")
        
        # Risk disclaimer in sidebar
        st.markdown("""
        <div class="risk-warning">
            <strong>⚠️ Risk Warning</strong><br>
            Trading involves substantial risk. This is for educational purposes only.
        </div>
        """, unsafe_allow_html=True)
    
    if analyze_button and symbol:
        with st.spinner(f"Fetching data and analyzing {symbol}..."):
            try:
                # Fetch stock data
                stock_data = st.session_state.data_fetcher.get_stock_data(symbol, period)
                
                if stock_data is None or stock_data.empty:
                    st.error(f"Could not fetch data for {symbol}. Please check the symbol and try again.")
                    return
                
                # Calculate technical indicators
                stock_data = calculate_basic_indicators(stock_data)
                
                # Display current stock info with enhanced styling
                current_price = stock_data['Close'].iloc[-1]
                prev_price = stock_data['Close'].iloc[-2]
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100
                
                # Determine market sentiment for styling
                sentiment_class = "bullish-card" if change_pct > 0 else "bearish-card" if change_pct < 0 else "neutral-card"
                
                # Create metrics with enhanced cards
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card {sentiment_class}">
                        <h3>{symbol} Current Price</h3>
                        <h2>{format_currency(current_price)}</h2>
                        <p style="color: {'#388E3C' if change_pct > 0 else '#D32F2F' if change_pct < 0 else '#757575'};">
                            {change:+.2f} ({change_pct:+.2f}%)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    volume = stock_data['Volume'].iloc[-1]
                    volume_change = ((volume / stock_data['Volume_MA'].iloc[-1]) - 1) * 100
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Volume</h3>
                        <h2>{volume:,.0f}</h2>
                        <p style="color: {'#388E3C' if volume_change > 0 else '#D32F2F'};">
                            {volume_change:+.1f}% vs avg
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    volatility = stock_data['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
                    vol_color = "#D32F2F" if volatility > 30 else "#388E3C" if volatility < 15 else "#F57C00"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>20-Day Volatility</h3>
                        <h2 style="color: {vol_color};">{volatility:.1f}%</h2>
                        <p>{'High' if volatility > 30 else 'Low' if volatility < 15 else 'Moderate'} Risk</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    rsi = stock_data['RSI'].iloc[-1]
                    rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                    rsi_color = "#D32F2F" if rsi > 70 else "#388E3C" if rsi < 30 else "#F57C00"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>RSI (14)</h3>
                        <h2 style="color: {rsi_color};">{rsi:.1f}</h2>
                        <p>{rsi_status}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Create tabs for different sections
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Technical Analysis", "🤖 Predictions", "📋 Daily Options", "⚡ Scalping", "📊 Summary"])
                
                with tab1:
                    st.markdown("### 📊 Interactive Technical Analysis")
                    
                    # Create interactive price chart
                    try:
                        price_chart = create_interactive_price_chart(stock_data)
                        st.plotly_chart(price_chart, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not create interactive chart: {str(e)}")
                        st.info("Falling back to basic analysis...")
                    
                    # Two column layout for additional analysis
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Technical signals with enhanced styling
                        st.markdown("### 📋 Trading Signals")
                        signals = generate_trading_signals(stock_data)
                        
                        for signal in signals:
                            if "🟢" in signal:
                                st.success(signal)
                            elif "🔴" in signal:
                                st.error(signal)
                            else:
                                st.warning(signal)
                    
                    with col2:
                        # Volatility gauge
                        try:
                            volatility_gauge = create_volatility_gauge(stock_data)
                            st.plotly_chart(volatility_gauge, use_container_width=True)
                        except Exception as e:
                            st.info("Volatility analysis unavailable")
                        
                        # Key levels
                        st.markdown("### 🎯 Key Levels")
                        recent_data = stock_data.tail(20)
                        resistance = recent_data['High'].max()
                        support = recent_data['Low'].min()
                        
                        st.markdown(f"""
                        <div class="scalping-setup">
                            <strong>Resistance:</strong> ${resistance:.2f}<br>
                            <strong>Support:</strong> ${support:.2f}<br>
                            <strong>Range:</strong> {((resistance - support) / current_price * 100):.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                
                with tab2:
                    st.subheader("🤖 Price Predictions")
                    
                    # Generate prediction
                    predicted_price = simple_prediction(stock_data)
                    change = predicted_price - current_price
                    change_pct = (change / current_price) * 100
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            label="Next Week Prediction",
                            value=format_currency(predicted_price),
                            delta=f"{change:+.2f} ({change_pct:+.2f}%)"
                        )
                    
                    with col2:
                        confidence = max(50, min(85, 70 - abs(change_pct)))
                        st.metric(
                            label="Confidence Level",
                            value=f"{confidence:.0f}%"
                        )
                    
                    if change_pct > 2:
                        st.success(f"🟢 Bullish Signal: {change_pct:.1f}% upward move expected")
                    elif change_pct < -2:
                        st.error(f"🔴 Bearish Signal: {change_pct:.1f}% downward move expected")
                    else:
                        st.warning(f"🟡 Neutral Signal: {change_pct:.1f}% movement expected")
                
                with tab3:
                    st.markdown("### 📋 Daily Options Strategies (Next 30 Days)")
                    
                    daily_strategies = generate_daily_options_strategies(stock_data, symbol, strategy_pref)
                    
                    if daily_strategies:
                        # Strategy filter
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            strategy_filter = st.multiselect(
                                "Filter by Strategy Type",
                                options=list(set([s['type'] for s in daily_strategies])),
                                default=list(set([s['type'] for s in daily_strategies]))
                            )
                        with col2:
                            confidence_filter = st.slider("Minimum Confidence %", 0, 100, 50)
                        
                        # Filter strategies
                        filtered_strategies = [s for s in daily_strategies 
                                             if s['type'] in strategy_filter and s['confidence'] >= confidence_filter]
                        
                        # Confidence chart
                        if len(filtered_strategies) > 0:
                            try:
                                confidence_chart = create_confidence_chart(filtered_strategies)
                                if confidence_chart:
                                    st.plotly_chart(confidence_chart, use_container_width=True)
                                else:
                                    # Fallback to simple confidence display
                                    st.markdown("### Strategy Confidence Overview")
                                    high_conf = len([s for s in filtered_strategies if s.get('confidence', 0) > 70])
                                    med_conf = len([s for s in filtered_strategies if 50 < s.get('confidence', 0) <= 70])
                                    low_conf = len([s for s in filtered_strategies if s.get('confidence', 0) <= 50])
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("High Confidence", high_conf, delta="70%+")
                                    with col2:
                                        st.metric("Medium Confidence", med_conf, delta="50-70%")
                                    with col3:
                                        st.metric("Low Confidence", low_conf, delta="<50%")
                            except Exception as e:
                                st.error(f"Chart error: {str(e)}")
                                # Show simple confidence metrics as fallback
                                high_conf = len([s for s in filtered_strategies if s.get('confidence', 0) > 70])
                                med_conf = len([s for s in filtered_strategies if 50 < s.get('confidence', 0) <= 70])
                                low_conf = len([s for s in filtered_strategies if s.get('confidence', 0) <= 50])
                                
                                st.markdown("### Strategy Confidence Summary")
                                st.write(f"• **High Confidence (70%+):** {high_conf} strategies")
                                st.write(f"• **Medium Confidence (50-70%):** {med_conf} strategies") 
                                st.write(f"• **Low Confidence (<50%):** {low_conf} strategies")
                        else:
                            st.warning("No strategies match your filter criteria. Adjust the filters to see more strategies.")
                        
                        # Group strategies by week
                        strategies_by_week = {}
                        for strategy in filtered_strategies:
                            week_start = datetime.strptime(strategy['date'], '%Y-%m-%d')
                            week_key = week_start.strftime('Week of %Y-%m-%d')
                            
                            if week_key not in strategies_by_week:
                                strategies_by_week[week_key] = []
                            strategies_by_week[week_key].append(strategy)
                        
                        # Display strategies by week with enhanced cards
                        for week, week_strategies in list(strategies_by_week.items())[:4]:
                            with st.expander(f"📅 {week} ({len(week_strategies)} strategies)", expanded=(week == list(strategies_by_week.keys())[0])):
                                for strategy in week_strategies:
                                    confidence_class = "confidence-high" if strategy['confidence'] > 70 else "confidence-medium" if strategy['confidence'] > 50 else "confidence-low"
                                    
                                    st.markdown(f"""
                                    <div class="strategy-card {confidence_class}">
                                        <div style="display: flex; justify-content: space-between; align-items: center;">
                                            <h4>{strategy['type']}</h4>
                                            <span style="background: {'#388E3C' if strategy['confidence'] > 70 else '#F57C00' if strategy['confidence'] > 50 else '#D32F2F'}; 
                                                         color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px;">
                                                {strategy['confidence']:.0f}% confidence
                                            </span>
                                        </div>
                                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin: 1rem 0;">
                                            <div>
                                                <strong>Date:</strong> {strategy['date']}<br>
                                                <strong>Strike:</strong> ${strategy['strike']}<br>
                                                <strong>Expiration:</strong> {strategy['expiration']}
                                            </div>
                                            <div>
                                                <strong>Days to Exp:</strong> {strategy['days_to_exp']}<br>
                                                <strong>Max Risk:</strong> {strategy['max_risk']}<br>
                                                <strong>Target Profit:</strong> {strategy['target_profit']}
                                            </div>
                                            <div>
                                                <strong>Rationale:</strong><br>
                                                <em>{strategy['rationale']}</em>
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        st.info(f"Showing {len(filtered_strategies)} strategies out of {len(daily_strategies)} total")
                    else:
                        st.info("No options strategies generated for the selected parameters.")
                
                with tab4:
                    st.markdown("### ⚡ Scalping Strategies (Next Trading Day)")
                    
                    scalping_setups = generate_scalping_strategy(stock_data, symbol)
                    
                    if scalping_setups:
                        # Scalping setup selector
                        setup_types = [setup['type'] for setup in scalping_setups]
                        selected_setup = st.selectbox("Choose Scalping Setup", setup_types)
                        
                        # Find selected setup
                        current_setup = next(setup for setup in scalping_setups if setup['type'] == selected_setup)
                        
                        if current_setup['type'] == 'Pivot Level Trade':
                            # Pivot levels display
                            st.markdown("#### 🎯 Key Pivot Levels")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                for level, price in list(current_setup['key_levels'].items())[:3]:
                                    color = "#D32F2F" if "R" in level else "#388E3C" if "S" in level else "#1976D2"
                                    st.markdown(f"""
                                    <div style="background: {color}; color: white; padding: 0.5rem; margin: 0.2rem 0; border-radius: 5px; text-align: center;">
                                        <strong>{level}:</strong> {price}
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            with col2:
                                for level, price in list(current_setup['key_levels'].items())[3:]:
                                    color = "#D32F2F" if "R" in level else "#388E3C" if "S" in level else "#1976D2"
                                    st.markdown(f"""
                                    <div style="background: {color}; color: white; padding: 0.5rem; margin: 0.2rem 0; border-radius: 5px; text-align: center;">
                                        <strong>{level}:</strong> {price}
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="scalping-setup">
                                <strong>Strategy:</strong> {current_setup['strategy']}<br>
                                <strong>Time Frame:</strong> {current_setup['time_frame']}<br>
                                <strong>Rationale:</strong> {current_setup['rationale']}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        else:
                            # Regular scalping setup
                            setup_color = "#388E3C" if 'Momentum' in current_setup['type'] else "#F57C00" if 'Breakout' in current_setup['type'] else "#2196F3"
                            
                            st.markdown(f"""
                            <div class="strategy-card" style="border-left: 5px solid {setup_color};">
                                <h4>{current_setup['type']}</h4>
                                
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin: 1rem 0;">
                                    <div>
                                        <strong>📍 Entry Range:</strong><br>
                                        <span style="color: {setup_color}; font-size: 1.1em;">{current_setup['entry_range']}</span><br><br>
                                        
                                        <strong>🛑 Stop Loss:</strong><br>
                                        <span style="color: #D32F2F; font-size: 1.1em;">{current_setup['stop_loss']}</span><br><br>
                                        
                                        <strong>🎯 Take Profit:</strong><br>
                                        <span style="color: #388E3C; font-size: 1.1em;">{current_setup['take_profit']}</span>
                                    </div>
                                    
                                    <div>
                                        <strong>⏱️ Time Frame:</strong><br>
                                        {current_setup['time_frame']}<br><br>
                                        
                                        <strong>📊 Risk:Reward:</strong><br>
                                        <span style="color: #1976D2; font-weight: bold;">{current_setup['risk_reward']}</span><br><br>
                                        
                                        <strong>💡 Rationale:</strong><br>
                                        <em>{current_setup['rationale']}</em>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Setup quality indicator
                            if 'Momentum' in current_setup['type']:
                                st.success("🟢 High probability momentum setup - Strong directional move expected")
                            elif 'Breakout' in current_setup['type']:
                                st.warning("🟡 Breakout setup - Confirm with volume and follow-through")
                            else:
                                st.info("🔵 Mean reversion setup - Counter-trend play with tight stops")
                        
                        # Trading tips
                        st.markdown("#### 💡 Scalping Tips")
                        tips = [
                            "Use 1-minute or 5-minute timeframes for entries",
                            "Always set stop losses before entering trades",
                            "Monitor volume for confirmation of breakouts",
                            "Take profits quickly - scalping is about small, frequent gains",
                            "Avoid trading during low volume periods",
                            "Use proper position sizing (1-2% risk per trade)"
                        ]
                        
                        for tip in tips:
                            st.markdown(f"• {tip}")
                    
                    else:
                        st.info("No scalping setups identified for current market conditions.")
                
                with tab5:
                    st.subheader("📊 Trading Summary")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Market Assessment:**")
                        rsi = stock_data['RSI'].iloc[-1]
                        if rsi > 70:
                            st.error("🔴 Overbought conditions")
                        elif rsi < 30:
                            st.success("🟢 Oversold conditions")
                        else:
                            st.warning("🟡 Neutral conditions")
                        
                        volatility = stock_data['Close'].pct_change().tail(20).std() * np.sqrt(252) * 100
                        st.write(f"**Volatility:** {volatility:.1f}%")
                        
                        if volatility > 30:
                            st.error("High volatility - increase position sizing caution")
                        elif volatility < 15:
                            st.success("Low volatility - good for income strategies")
                        else:
                            st.info("Moderate volatility - balanced approach")
                    
                    with col2:
                        st.write("**Recommended Actions:**")
                        
                        # Count strategy types
                        weekly_strategies = len([s for s in daily_strategies if s.get('days_to_exp', 0) <= 7])
                        monthly_strategies = len([s for s in daily_strategies if s.get('days_to_exp', 0) > 7])
                        scalping_opportunities = len(scalping_setups) - 1  # Exclude pivot levels
                        
                        st.write(f"📈 **Weekly Options:** {weekly_strategies} opportunities")
                        st.write(f"📊 **Monthly Options:** {monthly_strategies} opportunities")
                        st.write(f"⚡ **Scalping Setups:** {scalping_opportunities} identified")
                        
                        if predicted_price > current_price * 1.02:
                            st.success("🟢 **Overall Bias:** Bullish")
                        elif predicted_price < current_price * 0.98:
                            st.error("🔴 **Overall Bias:** Bearish")
                        else:
                            st.warning("🟡 **Overall Bias:** Neutral")
                    
                    st.markdown("""
                    ---
                    **⚠️ Risk Disclaimer:** This analysis is for educational purposes only. 
                    Always conduct your own research and consider your risk tolerance before trading.
                    """)

                
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.info("Please try again with a different symbol.")
    
    elif not symbol:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Options Trading Prediction System
        
        This system provides:
        - **Real-time stock data** from free APIs
        - **Technical analysis** with RSI, MACD, and moving averages
        - **Price predictions** for the next week
        - **Options trading strategies** based on market analysis
        
        **How to use:**
        1. Enter a stock symbol in the sidebar (e.g., AAPL, MSFT, GOOGL)
        2. Select your preferred time period and risk tolerance
        3. Click "Analyze & Predict" to get trading signals
        
        **Note:** This is for educational purposes only. Always do your own research before trading.
        """)

if __name__ == "__main__":
    main()