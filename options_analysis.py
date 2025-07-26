import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import math

class OptionsAnalysis:
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% risk-free rate assumption
        
    def black_scholes_call(self, S, K, T, r, sigma):
        """
        Calculate Black-Scholes call option price
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        """
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            return call_price
        except:
            return 0
    
    def black_scholes_put(self, S, K, T, r, sigma):
        """
        Calculate Black-Scholes put option price
        """
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            return put_price
        except:
            return 0
    
    def calculate_greeks(self, S, K, T, r, sigma, option_type='call'):
        """
        Calculate option Greeks
        """
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Delta
            if option_type == 'call':
                delta = norm.cdf(d1)
            else:
                delta = norm.cdf(d1) - 1
            
            # Gamma
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            # Theta (per day)
            if option_type == 'call':
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                        - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            else:
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                        + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            
            # Vega (per 1% change in volatility)
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega
            }
        except:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    def calculate_implied_volatility(self, data, window=20):
        """
        Calculate historical volatility as proxy for implied volatility
        """
        try:
            returns = data['Close'].pct_change().dropna()
            if len(returns) < window:
                return 0.20  # Default 20% volatility
            
            volatility = returns.rolling(window).std().iloc[-1] * np.sqrt(252)
            return max(0.05, min(2.0, volatility))  # Cap between 5% and 200%
        except:
            return 0.20
    
    def generate_strike_prices(self, current_price, num_strikes=5):
        """
        Generate reasonable strike prices around current price
        """
        strikes = []
        
        # Calculate strike spacing based on price level
        if current_price < 50:
            spacing = 2.5
        elif current_price < 100:
            spacing = 5
        elif current_price < 200:
            spacing = 10
        else:
            spacing = 20
        
        # Generate strikes around current price
        for i in range(-num_strikes//2, num_strikes//2 + 1):
            strike = current_price + (i * spacing)
            if strike > 0:
                strikes.append(round(strike, 2))
        
        return sorted(strikes)
    
    def analyze_options_opportunity(self, data, strategy_pref='Moderate'):
        """
        Analyze current market conditions for options opportunities
        """
        try:
            current_price = data['Close'].iloc[-1]
            volatility = self.calculate_implied_volatility(data)
            
            # Calculate price momentum
            price_change_5d = (current_price / data['Close'].iloc[-6] - 1) * 100
            price_change_20d = (current_price / data['Close'].iloc[-21] - 1) * 100
            
            # Calculate volatility trend
            vol_5d = data['Close'].pct_change().tail(5).std() * np.sqrt(252)
            vol_20d = data['Close'].pct_change().tail(20).std() * np.sqrt(252)
            vol_trend = (vol_5d / vol_20d - 1) * 100
            
            # RSI and other indicators
            rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else 50
            
            opportunities = {
                'high_volatility': volatility > 0.30,
                'low_volatility': volatility < 0.15,
                'strong_uptrend': price_change_5d > 5 and rsi < 70,
                'strong_downtrend': price_change_5d < -5 and rsi > 30,
                'oversold': rsi < 30,
                'overbought': rsi > 70,
                'vol_expansion': vol_trend > 20,
                'vol_contraction': vol_trend < -20
            }
            
            return {
                'current_price': current_price,
                'volatility': volatility,
                'price_change_5d': price_change_5d,
                'price_change_20d': price_change_20d,
                'vol_trend': vol_trend,
                'rsi': rsi,
                'opportunities': opportunities
            }
            
        except Exception as e:
            print(f"Error analyzing options opportunity: {str(e)}")
            return None
    
    def generate_options_signals(self, data, symbol, strategy_pref='Moderate'):
        """
        Generate options trading signals based on market analysis
        """
        try:
            analysis = self.analyze_options_opportunity(data, strategy_pref)
            if not analysis:
                return []
            
            current_price = analysis['current_price']
            volatility = analysis['volatility']
            opportunities = analysis['opportunities']
            
            signals = []
            
            # Time to expiration (1 week for weekly options)
            T = 7 / 365
            
            # Generate strike prices
            strikes = self.generate_strike_prices(current_price)
            
            # Strategy 1: Directional plays based on momentum
            if opportunities['strong_uptrend'] and not opportunities['overbought']:
                # Buy calls
                strike = min([s for s in strikes if s >= current_price])
                greeks = self.calculate_greeks(current_price, strike, T, self.risk_free_rate, volatility, 'call')
                premium = self.black_scholes_call(current_price, strike, T, self.risk_free_rate, volatility)
                
                signals.append({
                    'strategy': 'Long Call',
                    'type': 'call',
                    'strike_price': strike,
                    'expiration': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
                    'confidence': min(95, 60 + abs(analysis['price_change_5d'])),
                    'expected_return': max(20, abs(analysis['price_change_5d']) * 3),
                    'max_risk': 100,  # Premium paid
                    'implied_volatility': volatility * 100,
                    'delta': greeks['delta'],
                    'gamma': greeks['gamma'],
                    'theta': greeks['theta'],
                    'premium': premium,
                    'rationale': f"Strong upward momentum ({analysis['price_change_5d']:.1f}% in 5 days) with RSI at {analysis['rsi']:.1f}"
                })
            
            elif opportunities['strong_downtrend'] and not opportunities['oversold']:
                # Buy puts
                strike = max([s for s in strikes if s <= current_price])
                greeks = self.calculate_greeks(current_price, strike, T, self.risk_free_rate, volatility, 'put')
                premium = self.black_scholes_put(current_price, strike, T, self.risk_free_rate, volatility)
                
                signals.append({
                    'strategy': 'Long Put',
                    'type': 'put',
                    'strike_price': strike,
                    'expiration': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
                    'confidence': min(95, 60 + abs(analysis['price_change_5d'])),
                    'expected_return': max(20, abs(analysis['price_change_5d']) * 3),
                    'max_risk': 100,
                    'implied_volatility': volatility * 100,
                    'delta': greeks['delta'],
                    'gamma': greeks['gamma'],
                    'theta': greeks['theta'],
                    'premium': premium,
                    'rationale': f"Strong downward momentum ({analysis['price_change_5d']:.1f}% in 5 days) with RSI at {analysis['rsi']:.1f}"
                })
            
            # Strategy 2: Volatility plays
            if opportunities['low_volatility'] and opportunities['vol_expansion']:
                # Long straddle - expecting big move
                atm_strike = min(strikes, key=lambda x: abs(x - current_price))
                
                call_greeks = self.calculate_greeks(current_price, atm_strike, T, self.risk_free_rate, volatility, 'call')
                put_greeks = self.calculate_greeks(current_price, atm_strike, T, self.risk_free_rate, volatility, 'put')
                
                call_premium = self.black_scholes_call(current_price, atm_strike, T, self.risk_free_rate, volatility)
                put_premium = self.black_scholes_put(current_price, atm_strike, T, self.risk_free_rate, volatility)
                
                signals.append({
                    'strategy': 'Long Straddle',
                    'type': 'straddle',
                    'strike_price': atm_strike,
                    'expiration': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
                    'confidence': 70,
                    'expected_return': 40,
                    'max_risk': 100,
                    'implied_volatility': volatility * 100,
                    'delta': call_greeks['delta'] + put_greeks['delta'],
                    'gamma': call_greeks['gamma'] + put_greeks['gamma'],
                    'theta': call_greeks['theta'] + put_greeks['theta'],
                    'premium': call_premium + put_premium,
                    'rationale': f"Low volatility ({volatility*100:.1f}%) with expanding volatility trend ({analysis['vol_trend']:.1f}%)"
                })
            
            # Strategy 3: Mean reversion plays
            if opportunities['oversold'] and strategy_pref in ['Moderate', 'Aggressive']:
                # Buy calls on oversold bounce
                otm_strike = min([s for s in strikes if s > current_price * 1.02])
                greeks = self.calculate_greeks(current_price, otm_strike, T, self.risk_free_rate, volatility, 'call')
                premium = self.black_scholes_call(current_price, otm_strike, T, self.risk_free_rate, volatility)
                
                signals.append({
                    'strategy': 'Oversold Bounce Call',
                    'type': 'call',
                    'strike_price': otm_strike,
                    'expiration': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
                    'confidence': 65,
                    'expected_return': 30,
                    'max_risk': 100,
                    'implied_volatility': volatility * 100,
                    'delta': greeks['delta'],
                    'gamma': greeks['gamma'],
                    'theta': greeks['theta'],
                    'premium': premium,
                    'rationale': f"Oversold condition (RSI: {analysis['rsi']:.1f}) suggesting potential bounce"
                })
            
            elif opportunities['overbought'] and strategy_pref in ['Moderate', 'Aggressive']:
                # Buy puts on overbought pullback
                otm_strike = max([s for s in strikes if s < current_price * 0.98])
                greeks = self.calculate_greeks(current_price, otm_strike, T, self.risk_free_rate, volatility, 'put')
                premium = self.black_scholes_put(current_price, otm_strike, T, self.risk_free_rate, volatility)
                
                signals.append({
                    'strategy': 'Overbought Pullback Put',
                    'type': 'put',
                    'strike_price': otm_strike,
                    'expiration': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
                    'confidence': 65,
                    'expected_return': 30,
                    'max_risk': 100,
                    'implied_volatility': volatility * 100,
                    'delta': greeks['delta'],
                    'gamma': greeks['gamma'],
                    'theta': greeks['theta'],
                    'premium': premium,
                    'rationale': f"Overbought condition (RSI: {analysis['rsi']:.1f}) suggesting potential pullback"
                })
            
            # Sort signals by confidence
            signals.sort(key=lambda x: x['confidence'], reverse=True)
            
            return signals[:3]  # Return top 3 signals
            
        except Exception as e:
            print(f"Error generating options signals: {str(e)}")
            return []
    
    def calculate_option_payoff(self, price_range, strike_price, option_type, premium):
        """
        Calculate option payoff for profit/loss diagram
        """
        try:
            payoffs = []
            
            for price in price_range:
                if option_type == 'call':
                    intrinsic_value = max(0, price - strike_price)
                    payoff = intrinsic_value - premium
                elif option_type == 'put':
                    intrinsic_value = max(0, strike_price - price)
                    payoff = intrinsic_value - premium
                elif option_type == 'straddle':
                    call_value = max(0, price - strike_price)
                    put_value = max(0, strike_price - price)
                    payoff = call_value + put_value - premium
                else:
                    payoff = 0
                
                payoffs.append(payoff)
            
            return payoffs
            
        except Exception as e:
            print(f"Error calculating option payoff: {str(e)}")
            return [0] * len(price_range)
    
    def calculate_break_even_points(self, strike_price, premium, option_type):
        """
        Calculate break-even points for options strategies
        """
        try:
            if option_type == 'call':
                return [strike_price + premium]
            elif option_type == 'put':
                return [strike_price - premium]
            elif option_type == 'straddle':
                return [strike_price - premium, strike_price + premium]
            else:
                return []
        except:
            return []
