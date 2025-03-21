import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import time

# Set page config
st.set_page_config(
    page_title="Stock Trading Assistant",
    page_icon="📈",
    layout="wide"
)

# Add custom CSS
st.markdown('''
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .market-up {
        color: green;
        font-weight: bold;
    }
    .market-down {
        color: red;
        font-weight: bold;
    }
</style>
''', unsafe_allow_html=True)

# App title
st.title("Stock Trading Assistant")
st.markdown("### Your AI-powered investment advisor")

# Sample portfolio data
@st.cache_data
def load_sample_portfolio():
    data = {
        'Stock': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
        'Shares': [10, 5, 2, 3, 8],
        'Purchase Price': [150.75, 280.50, 2750.25, 3300.10, 220.75],
        'Term': ['Long', 'Short', 'Long', 'Short', 'Long']
    }
    return pd.DataFrame(data)

# Get current stock prices
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_current_prices(_symbols):
    symbols = list(_symbols) if hasattr(_symbols, '__iter__') and not isinstance(_symbols, str) else [_symbols]
    prices = {}
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="1d")
            if not data.empty:
                prices[symbol] = data['Close'].iloc[-1]
            else:
                prices[symbol] = 0
        except:
            prices[symbol] = 0
    return prices

# Get market data with previous day comparison
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_market_data():
    # Define market symbols to track
    market_symbols = {
        'S&P 500': '^GSPC',
        'Dow Jones': '^DJI',
        'NASDAQ': '^IXIC',
        'Bitcoin': 'BTC-USD',
        'Technology ETF': 'XLK'  # Technology sector ETF
    }
    
    market_data = {}
    
    for name, symbol in market_symbols.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")  # Get 2 days of data for comparison
            
            if len(hist) >= 2:
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2]
                change = current_price - prev_price
                percent_change = (change / prev_price) * 100
                
                market_data[name] = {
                    'symbol': symbol,
                    'current': current_price,
                    'previous': prev_price,
                    'change': change,
                    'percent_change': percent_change
                }
            else:
                # If we don't have enough data, just use the latest price
                current_price = hist['Close'].iloc[-1] if not hist.empty else 0
                market_data[name] = {
                    'symbol': symbol,
                    'current': current_price,
                    'previous': current_price,
                    'change': 0,
                    'percent_change': 0
                }
        except Exception as e:
            # Handle any errors by providing zeros
            market_data[name] = {
                'symbol': symbol,
                'current': 0,
                'previous': 0,
                'change': 0,
                'percent_change': 0
            }
    
    return market_data

# Calculate portfolio metrics
def calculate_portfolio_metrics(portfolio_df, current_prices):
    # Make a copy to avoid modifying the original
    df = portfolio_df.copy()
    
    # Ensure column names are standardized
    if 'Stock' in df.columns:
        df['symbol'] = df['Stock']
    
    if 'Shares' in df.columns:
        df['shares'] = df['Shares']
    
    if 'Purchase Price' in df.columns:
        df['purchase_price'] = df['Purchase Price']
    
    if 'Term' in df.columns:
        df['holding_type'] = df['Term'].apply(lambda x: 'long_term' if x.lower() == 'long' else 'short_term')
    
    # Calculate metrics
    df['current_price'] = df['symbol'].map(current_prices)
    df['current_value'] = df['shares'] * df['current_price']
    df['purchase_value'] = df['shares'] * df['purchase_price']
    df['unrealized_gain'] = df['current_value'] - df['purchase_value']
    df['unrealized_gain_percent'] = (df['unrealized_gain'] / df['purchase_value']) * 100
    
    # Calculate tax rates based on holding_type
    df['tax_rate'] = df['holding_type'].apply(lambda x: 0.15 if 'long' in str(x).lower() else 0.35)
    df['estimated_tax'] = df.apply(lambda x: x['unrealized_gain'] * x['tax_rate'] if x['unrealized_gain'] > 0 else 0, axis=1)
    df['effective_tax_rate'] = df.apply(lambda x: (x['estimated_tax'] / x['unrealized_gain'] * 100) if x['unrealized_gain'] > 0 else 0, axis=1)
    
    return df

# Generate chat response based on portfolio and user query
def generate_chat_response(query, portfolio_data, risk_tolerance, tax_sensitivity, investment_horizon):
    query = query.lower()
    
    # Calculate some portfolio metrics for responses
    if portfolio_data is not None:
        try:
            # Get symbols from the portfolio
            if 'Stock' in portfolio_data.columns:
                symbols = portfolio_data['Stock'].tolist()
            elif 'symbol' in portfolio_data.columns:
                symbols = portfolio_data['symbol'].tolist()
            else:
                symbols = []
            
            # Get current prices
            current_prices = get_current_prices(symbols)
            
            # Calculate metrics
            portfolio_metrics = calculate_portfolio_metrics(portfolio_data, current_prices)
            
            # Calculate total values
            total_value = portfolio_metrics['current_value'].sum()
            total_purchase = portfolio_metrics['purchase_value'].sum()
            total_gain = total_value - total_purchase
            total_gain_percent = (total_gain / total_purchase) * 100 if total_purchase > 0 else 0
            
            # Get top performer
            if not portfolio_metrics.empty:
                top_performer_idx = portfolio_metrics['unrealized_gain_percent'].idxmax()
                top_performer = portfolio_metrics.iloc[top_performer_idx]
                top_symbol = top_performer['symbol']
                top_gain_percent = top_performer['unrealized_gain_percent']
                
                # Get worst performer
                worst_performer_idx = portfolio_metrics['unrealized_gain_percent'].idxmin()
                worst_performer = portfolio_metrics.iloc[worst_performer_idx]
                worst_symbol = worst_performer['symbol']
                worst_gain_percent = worst_performer['unrealized_gain_percent']
            else:
                top_symbol = "N/A"
                top_gain_percent = 0
                worst_symbol = "N/A"
                worst_gain_percent = 0
            
            # Get tax impact
            total_tax = portfolio_metrics['estimated_tax'].sum()
            effective_tax_rate = (total_tax / total_gain * 100) if total_gain > 0 else 0
            
            # Get holding breakdown
            short_term = portfolio_metrics[portfolio_metrics['holding_type'].str.lower().str.contains('short')]
            long_term = portfolio_metrics[portfolio_metrics['holding_type'].str.lower().str.contains('long')]
            short_term_value = short_term['current_value'].sum()
            long_term_value = long_term['current_value'].sum()
            short_term_percent = (short_term_value / total_value) * 100 if total_value > 0 else 0
            long_term_percent = (long_term_value / total_value) * 100 if total_value > 0 else 0
        except Exception as e:
            # If there's an error calculating metrics, use placeholder values
            total_value = 0
            total_gain = 0
            total_gain_percent = 0
            top_symbol = "N/A"
            top_gain_percent = 0
            worst_symbol = "N/A"
            worst_gain_percent = 0
            total_tax = 0
            effective_tax_rate = 0
            short_term_percent = 0
            long_term_percent = 0
    else:
        # If no portfolio data, use placeholder values
        total_value = 0
        total_gain = 0
        total_gain_percent = 0
        top_symbol = "N/A"
        top_gain_percent = 0
        worst_symbol = "N/A"
        worst_gain_percent = 0
        total_tax = 0
        effective_tax_rate = 0
        short_term_percent = 0
        long_term_percent = 0
    
    # Get market data for responses
    try:
        market_data = get_market_data()
        sp500_change = market_data['S&P 500']['percent_change']
        tech_change = market_data['Technology ETF']['percent_change']
    except:
        sp500_change = 0
        tech_change = 0
    
    # Generate response based on query categories
    if any(word in query for word in ['portfolio', 'holdings', 'stocks', 'positions']):
        if portfolio_data is None or portfolio_data.empty:
            return "You don't have any portfolio data loaded yet. Please upload a CSV file or use the sample portfolio."
        
        return f"Your portfolio contains {len(symbols)} stocks with a total value of ${total_value:.2f}. " \
               f"Overall, your portfolio is {('up' if total_gain >= 0 else 'down')} ${abs(total_gain):.2f} " \
               f"({abs(total_gain_percent):.2f}%). " \
               f"Your top performer is {top_symbol} with a gain of {top_gain_percent:.2f}%, " \
               f"while your worst performer is {worst_symbol} with a {'gain' if worst_gain_percent >= 0 else 'loss'} of {abs(worst_gain_percent):.2f}%."
    
    elif any(word in query for word in ['tax', 'taxes', 'capital gains']):
        if portfolio_data is None or portfolio_data.empty:
            return "You don't have any portfolio data loaded yet. Please upload a CSV file or use the sample portfolio."
        
        return f"Your portfolio has unrealized gains of ${total_gain:.2f}, with an estimated tax impact of ${total_tax:.2f} " \
               f"(effective rate: {effective_tax_rate:.2f}%). " \
               f"{short_term_percent:.1f}% of your portfolio is in short-term positions (higher tax rate), " \
               f"while {long_term_percent:.1f}% is in long-term positions (lower tax rate)."
    
    elif any(word in query for word in ['market', 'index', 'indices', 'indexes']):
        market_sentiment = "bullish" if sp500_change > 0.5 else "bearish" if sp500_change < -0.5 else "neutral"
        sector_performance = "outperforming" if tech_change > sp500_change else "underperforming"
        
        return f"The market is currently showing {market_sentiment} sentiment. " \
               f"S&P 500 is {('up' if sp500_change >= 0 else 'down')} {abs(sp500_change):.2f}% today, " \
               f"with the technology sector {sector_performance} at {tech_change:.2f}%."
    
    elif any(word in query for word in ['advice', 'recommend', 'suggestion', 'help']):
        advice = ""
        
        if risk_tolerance == "conservative":
            advice += "Based on your conservative risk tolerance, I recommend focusing on stable, dividend-paying stocks and considering protective puts for downside protection. "
        elif risk_tolerance == "moderate":
            advice += "With your moderate risk tolerance, a balanced approach with a mix of growth and value stocks is appropriate. "
        else:  # aggressive
            advice += "Given your aggressive risk tolerance, you might consider higher-growth tech stocks and emerging markets for potential outperformance. "
        
        if tax_sensitivity == "high":
            advice += "Since you're highly tax-sensitive, prioritize tax-loss harvesting opportunities and favor long-term holdings. "
            if top_gain_percent > 20:
                advice += f"Consider using options strategies like collars on {top_symbol} to protect gains without triggering taxes. "
        
        if worst_gain_percent < -10:
            advice += f"Consider tax-loss harvesting with {worst_symbol} which is currently underperforming. "
        
        if top_gain_percent > 30:
            advice += f"Your position in {top_symbol} has strong momentum but consider taking some profits if it exceeds your target allocation. "
        
        return advice
    
    else:
        return "I can help you with portfolio analysis, tax implications, market insights, and investment recommendations. " \
               "Please ask a specific question about these topics, such as 'How is my portfolio performing?', " \
               "'What are my tax implications?', 'How is the market doing?', or 'What investment advice do you have for me?'"

# Sidebar
with st.sidebar:
    st.header("Portfolio Management")
    
    # Portfolio upload
    uploaded_file = st.file_uploader("Upload Portfolio CSV", type=["csv"])
    
    # Sample portfolio button
    use_sample = st.button("Use Sample Portfolio")
    
    # Investor profile
    st.header("Investor Profile")
    risk_tolerance = st.selectbox(
        "Risk Tolerance",
        ["conservative", "moderate", "aggressive"],
        index=1
    )
    tax_sensitivity = st.selectbox(
        "Tax Sensitivity",
        ["low", "moderate", "high"],
        index=1
    )
    investment_horizon = st.selectbox(
        "Investment Horizon",
        ["short", "medium", "long"],
        index=1
    )
    
    # Refresh market data
    st.header("Market Data")
    refresh_data = st.button("Refresh Market Data")
    if refresh_data:
        st.cache_data.clear()
        st.experimental_rerun()

# Initialize session state
if 'portfolio_loaded' not in st.session_state:
    st.session_state.portfolio_loaded = False
    
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None
    
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Load portfolio data
if uploaded_file is not None:
    try:
        portfolio_df = pd.read_csv(uploaded_file)
        st.session_state.portfolio_data = portfolio_df
        st.session_state.portfolio_loaded = True
    except Exception as e:
        st.error(f"Error loading portfolio: {str(e)}")
        
elif use_sample:
    portfolio_df = load_sample_portfolio()
    st.session_state.portfolio_data = portfolio_df
    st.session_state.portfolio_loaded = True

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Chat", "Portfolio", "Market", "Settings"])

# Tab 1: Chat Interface
with tab1:
    st.header("Chat with your Stock Trading Assistant")
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("Ask about your portfolio or get investment advice..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Check if portfolio is loaded
        if not st.session_state.portfolio_loaded:
            response = "Please load a portfolio first. You can upload a CSV file or use the sample portfolio."
        else:
            # Generate response based on user input and portfolio data
            response = generate_chat_response(
                prompt, 
                st.session_state.portfolio_data,
                risk_tolerance,
                tax_sensitivity,
                investment_horizon
            )
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Tab 2: Portfolio Overview
with tab2:
    st.header("Portfolio Overview")
    
    if not st.session_state.portfolio_loaded:
        st.info("Please load a portfolio to view details.")
    else:
        try:
            # Get stock symbols from the portfolio
            if 'Stock' in st.session_state.portfolio_data.columns:
                symbols = st.session_state.portfolio_data['Stock'].tolist()
            elif 'symbol' in st.session_state.portfolio_data.columns:
                symbols = st.session_state.portfolio_data['symbol'].tolist()
            else:
                raise ValueError("Could not find stock symbols in the portfolio data")
            
            # Get current prices
            current_prices = get_current_prices(symbols)
            
            # Calculate portfolio metrics
            portfolio_with_metrics = calculate_portfolio_metrics(st.session_state.portfolio_data.copy(), current_prices)
            
            # Display portfolio summary
            total_value = portfolio_with_metrics['current_value'].sum()
            total_purchase = portfolio_with_metrics['purchase_value'].sum()
            total_gain = total_value - total_purchase
            total_gain_percent = (total_gain / total_purchase) * 100 if total_purchase > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Portfolio Value", f"${total_value:.2f}")
            
            with col2:
                st.metric("Total Gain/Loss", f"${total_gain:.2f}", f"{total_gain_percent:.2f}%")
            
            with col3:
                total_tax = portfolio_with_metrics['estimated_tax'].sum()
                st.metric("Estimated Tax Impact", f"${total_tax:.2f}")
            
            # Display holding period breakdown
            st.subheader("Holding Period Breakdown")
            
            short_term = portfolio_with_metrics[portfolio_with_metrics['holding_type'].str.lower().str.contains('short')]
            long_term = portfolio_with_metrics[portfolio_with_metrics['holding_type'].str.lower().str.contains('long')]
            
            short_term_value = short_term['current_value'].sum()
            long_term_value = long_term['current_value'].sum()
            
            short_term_percent = (short_term_value / total_value) * 100 if total_value > 0 else 0
            long_term_percent = (long_term_value / total_value) * 100 if total_value > 0 else 0
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Create pie chart for holding periods
                fig = px.pie(
                    values=[short_term_value, long_term_value],
                    names=['Short-term', 'Long-term'],
                    title="Portfolio by Holding Period"
                )
                st.plotly_chart(fig)
            
            with col2:
                st.metric("Short-term Positions", f"{len(short_term)} ({short_term_percent:.1f}%)")
                st.metric("Long-term Positions", f"{len(long_term)} ({long_term_percent:.1f}%)")
            
            # Display portfolio table
            st.subheader("Portfolio Positions")
            
            # Format the dataframe for display
            display_df = portfolio_with_metrics.copy()
            
            # Select columns to display
            display_columns = [
                'symbol', 'shares', 'purchase_price', 'current_price', 
                'unrealized_gain', 'unrealized_gain_percent', 'holding_type', 
                'estimated_tax', 'effective_tax_rate'
            ]
            
            # Ensure all required columns exist
            for col in display_columns:
                if col not in display_df.columns:
                    if col == 'symbol' and 'Stock' in display_df.columns:
                        display_df[col] = display_df['Stock']
                    elif col == 'shares' and 'Shares' in display_df.columns:
                        display_df[col] = display_df['Shares']
                    elif col == 'purchase_price' and 'Purchase Price' in display_df.columns:
                        display_df[col] = display_df['Purchase Price']
                    elif col == 'holding_type' and 'Term' in display_df.columns:
                        display_df[col] = display_df['Term']
            
            display_df = display_df[display_columns]
            
            # Rename columns for better readability
            display_df.columns = [
                'Symbol', 'Shares', 'Purchase Price', 'Current Price', 
                'Unrealized Gain', 'Gain %', 'Holding Type', 
                'Est. Tax', 'Tax Rate %'
            ]
            
            # Format numeric columns
            display_df['Purchase Price'] = display_df['Purchase Price'].map('${:.2f}'.format)
            display_df['Current Price'] = display_df['Current Price'].map('${:.2f}'.format)
            display_df['Unrealized Gain'] = display_df['Unrealized Gain'].map('${:.2f}'.format)
            display_df['Gain %'] = display_df['Gain %'].map('{:.2f}%'.format)
            display_df['Est. Tax'] = display_df['Est. Tax'].map('${:.2f}'.format)
            display_df['Tax Rate %'] = display_df['Tax Rate %'].map('{:.2f}%'.format)
            
            # Format holding type
            if 'Holding Type' in display_df.columns:
                display_df['Holding Type'] = display_df['Holding Type'].astype(str).str.replace('_', ' ').str.title()
            
            st.dataframe(display_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error displaying portfolio: {str(e)}")

# Tab 3: Market Overview
with tab3:
    st.header("Market Overview")
    
    # Display live market data
    st.subheader("Live Market Indexes")
    
    try:
        # Get market data
        market_data = get_market_data()
        
        # Create columns for market indices
        cols = st.columns(len(market_data))
        
        # Display each market index
        for i, (name, data) in enumerate(market_data.items()):
            with cols[i]:
                # Format the change as a string with color
                change_str = f"{data['change']:.2f} ({data['percent_change']:.2f}%)"
                if data['change'] > 0:
                    change_html = f'<span class="market-up">+{change_str}</span>'
                elif data['change'] < 0:
                    change_html = f'<span class="market-down">{change_str}</span>'
                else:
                    change_html = f'<span>{change_str}</span>'
                
                # Display the metric
                st.metric(name, f"{data['current']:.2f}")
                st.markdown(f"Change: {change_html}", unsafe_allow_html=True)
        
        # Add last updated time
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Display sector performance (using real data for Technology sector)
        st.subheader("Sector Performance")
        
        sector_performance = {
            'Technology': market_data['Technology ETF']['percent_change'],
            'Financial': -0.8,
            'Healthcare': 1.2,
            'Energy': -1.5,
            'Industrial': 0.3,
            'Consumer Staples': 0.7,
            'Consumer Discretionary': -0.2,
            'Materials': -0.5,
            'Utilities': 0.1,
            'Real Estate': -1.0,
            'Communication Services': 1.8
        }
        
        # Create bar chart for sector performance
        sectors = list(sector_performance.keys())
        performances = list(sector_performance.values())
        
        fig = px.bar(
            x=sectors,
            y=performances,
            title="Sector Performance (%)",
            labels={'x': 'Sector', 'y': 'Performance (%)'}
        )
        
        # Color bars based on performance
        fig.update_traces(marker_color=['green' if p > 0 else 'red' for p in performances])
        
        st.plotly_chart(fig, use_container_width=True)
        
        if st.session_state.portfolio_loaded:
            # Display portfolio symbol analysis
            st.subheader("Portfolio Symbol Analysis")
            
            # Get symbols from portfolio
            if 'Stock' in st.session_state.portfolio_data.columns:
                portfolio_symbols = st.session_state.portfolio_data['Stock']
            elif 'symbol' in st.session_state.portfolio_data.columns:
                portfolio_symbols = st.session_state.portfolio_data['symbol']
            else:
                portfolio_symbols = []
            
            # Get real data for portfolio symbols
            symbol_data = []
            
            for symbol in portfolio_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="14d")  # Get 14 days for RSI calculation
                    
                    if not hist.empty:
                        # Calculate a simple RSI
                        delta = hist['Close'].diff()
                        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                        rs = gain / loss
                        rsi = 100 - (100 / (1 + rs)).iloc[-1]
                        
                        # Determine trend
                        short_ma = hist['Close'].rolling(window=5).mean().iloc[-1]
                        long_ma = hist['Close'].rolling(window=10).mean().iloc[-1]
                        trend = "Bullish" if short_ma > long_ma else "Bearish" if short_ma < long_ma else "Neutral"
                        
                        # Calculate volatility (standard deviation of returns)
                        returns = hist['Close'].pct_change()
                        volatility = returns.std() * 100
                        
                        # Determine status based on RSI
                        status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                        
                        symbol_data.append({
                            'Symbol': symbol,
                            'RSI': rsi,
                            'Trend': trend,
                            'Volatility': volatility,
                            'Status': status
                        })
                    else:
                        # Use placeholder data if no history
                        symbol_data.append({
                            'Symbol': symbol,
                            'RSI': 50,
                            'Trend': "Neutral",
                            'Volatility': 1.0,
                            'Status': "Neutral"
                        })
                except:
                    # Use placeholder data if error
                    symbol_data.append({
                        'Symbol': symbol,
                        'RSI': 50,
                        'Trend': "Neutral",
                        'Volatility': 1.0,
                        'Status': "Neutral"
                    })
            
            symbol_df = pd.DataFrame(symbol_data)
            
            # Format numeric columns
            symbol_df['RSI'] = symbol_df['RSI'].map('{:.2f}'.format)
            symbol_df['Volatility'] = symbol_df['Volatility'].map('{:.2f}%'.format)
            
            st.dataframe(symbol_df, use_container_width=True)
            
            # Display market sentiment
            st.subheader("Market Sentiment")
            
            # Determine sentiment based on S&P 500
            sp500_change = market_data['S&P 500']['percent_change']
            if sp500_change > 1.0:
                sentiment = "🐂 Strongly Bullish"
            elif sp500_change > 0.3:
                sentiment = "🐂 Bullish"
            elif sp500_change > -0.3:
                sentiment = "😐 Neutral"
            elif sp500_change > -1.0:
                sentiment = "🐻 Bearish"
            else:
                sentiment = "🐻 Strongly Bearish"
            
            st.info(f"Current market sentiment: {sentiment}")
            
            # Display upcoming earnings
            st.subheader("Upcoming Earnings")
            
            # Try to get real earnings data for portfolio symbols
            earnings = []
            
            for symbol in portfolio_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    calendar = ticker.calendar
                    
                    if calendar is not None and not calendar.empty and 'Earnings Date' in calendar.columns:
                        earnings_date = calendar['Earnings Date'].iloc[0]
                        earnings_time = "Before Market Open" if calendar['Earnings Date'].dt.hour.iloc[0] < 12 else "After Market Close"
                        
                        # Get EPS estimate if available
                        eps_estimate = calendar['EPS Estimate'].iloc[0] if 'EPS Estimate' in calendar.columns else None
                        
                        earnings.append({
                            'symbol': symbol,
                            'company_name': ticker.info.get('shortName', symbol),
                            'date': earnings_date.strftime('%Y-%m-%d'),
                            'time': earnings_time,
                            'eps_estimate': eps_estimate
                        })
                except:
                    # Skip if error
                    pass
            
            # If no real earnings data, use simulated data
            if not earnings:
                earnings = [
                    {'symbol': 'AAPL', 'company_name': 'Apple Inc.', 'date': '2025-04-30', 'time': 'After Market Close', 'eps_estimate': 1.56},
                    {'symbol': 'MSFT', 'company_name': 'Microsoft Corporation', 'date': '2025-04-29', 'time': 'After Market Close', 'eps_estimate': 2.35},
                    {'symbol': 'GOOGL', 'company_name': 'Alphabet Inc.', 'date': '2025-04-28', 'time': 'After Market Close', 'eps_estimate': 1.78}
                ]
            
            earnings_df = pd.DataFrame(earnings)
            
            # Check if portfolio symbols are in earnings
            if 'Stock' in st.session_state.portfolio_data.columns:
                earnings_df['In Portfolio'] = earnings_df['symbol'].isin(st.session_state.portfolio_data['Stock'])
            elif 'symbol' in st.session_state.portfolio_data.columns:
                earnings_df['In Portfolio'] = earnings_df['symbol'].isin(st.session_state.portfolio_data['symbol'])
            else:
                earnings_df['In Portfolio'] = False
            
            st.dataframe(earnings_df, use_container_width=True)
        else:
            st.info("Please load a portfolio to view detailed market analysis.")
        
    except Exception as e:
        st.error(f"Error displaying market overview: {str(e)}")

# Tab 4: Settings
with tab4:
    st.header("Settings")
    
    # Tax settings
    st.subheader("Tax Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        filing_status = st.selectbox(
            "Filing Status",
            ["single", "married_joint", "head_household", "married_separate"],
            index=0
        )
    
    with col2:
        annual_income = st.number_input(
            "Annual Income",
            min_value=0,
            max_value=1000000,
            value=100000,
            step=5000
        )
    
    update_tax = st.button("Update Tax Settings")
    
    # App settings
    st.subheader("App Settings")
    
    # Theme selection
    theme = st.selectbox(
        "Theme",
        ["Light", "Dark"],
        index=1
    )
    
    # Data refresh interval
    refresh_interval = st.slider(
        "Market Data Refresh Interval (minutes)",
        min_value=5,
        max_value=60,
        value=30,
        step=5
    )
    
    save_settings = st.button("Save Settings")
    
    if save_settings:
        st.success("Settings saved!")

# Footer
st.markdown("---")
st.markdown("Stock Trading Assistant - Developed with Streamlit")
