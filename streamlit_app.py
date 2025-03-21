import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf

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
</style>
''', unsafe_allow_html=True)

# App title
st.title("Stock Trading Assistant")
st.markdown("### Your AI-powered investment advisor")

# Sample portfolio data
@st.cache_data
def load_sample_portfolio():
    data = {
        'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
        'shares': [10, 5, 2, 3, 8],
        'purchase_price': [150.75, 280.50, 2750.25, 3300.10, 220.75],
        'purchase_date': ['2023-01-15', '2023-03-20', '2022-11-05', '2023-05-10', '2022-08-22']
    }
    return pd.DataFrame(data)

# Get current stock prices
@st.cache_data(ttl=3600)
def get_current_prices(tickers):
    prices = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d")
            if not data.empty:
                prices[ticker] = data['Close'].iloc[-1]
            else:
                prices[ticker] = 0
        except:
            prices[ticker] = 0
    return prices

# Calculate portfolio metrics
def calculate_portfolio_metrics(portfolio_df, current_prices):
    portfolio_df['current_price'] = portfolio_df['ticker'].map(current_prices)
    portfolio_df['current_value'] = portfolio_df['shares'] * portfolio_df['current_price']
    portfolio_df['purchase_value'] = portfolio_df['shares'] * portfolio_df['purchase_price']
    portfolio_df['unrealized_gain'] = portfolio_df['current_value'] - portfolio_df['purchase_value']
    portfolio_df['unrealized_gain_percent'] = (portfolio_df['unrealized_gain'] / portfolio_df['purchase_value']) * 100
    
    # Calculate holding period
    portfolio_df['purchase_date'] = pd.to_datetime(portfolio_df['purchase_date'])
    portfolio_df['holding_period_days'] = (datetime.now() - portfolio_df['purchase_date']).dt.days
    portfolio_df['holding_type'] = portfolio_df['holding_period_days'].apply(lambda x: 'long_term' if x >= 365 else 'short_term')
    
    # Calculate tax rates (simplified)
    portfolio_df['tax_rate'] = portfolio_df['holding_type'].apply(lambda x: 0.15 if x == 'long_term' else 0.35)
    portfolio_df['estimated_tax'] = portfolio_df.apply(lambda x: x['unrealized_gain'] * x['tax_rate'] if x['unrealized_gain'] > 0 else 0, axis=1)
    portfolio_df['effective_tax_rate'] = portfolio_df.apply(lambda x: (x['estimated_tax'] / x['unrealized_gain'] * 100) if x['unrealized_gain'] > 0 else 0, axis=1)
    
    return portfolio_df

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
            # Generate response based on user input
            if "portfolio" in prompt.lower() or "holdings" in prompt.lower():
                response = "Your portfolio contains 5 stocks with a total value of approximately $5,200. The top performer is NVDA with a gain of 45%."
            elif "tax" in prompt.lower():
                response = "Your portfolio has unrealized gains of approximately $1,800, with an estimated tax impact of $450 (effective rate: 25%)."
            elif "market" in prompt.lower():
                response = "The market is currently showing moderate volatility. S&P 500 is up 0.5% today, with technology and healthcare sectors outperforming."
            elif "advice" in prompt.lower() or "recommend" in prompt.lower():
                response = "Based on your moderate risk tolerance and tax sensitivity, I recommend holding your NVDA position which has strong momentum. Consider tax-loss harvesting with AMZN which is currently underperforming."
            else:
                response = "I can help you with portfolio analysis, tax implications, market insights, and investment recommendations. Please ask a specific question about these topics."
        
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
            # Get current prices
            current_prices = get_current_prices(st.session_state.portfolio_data['ticker'].tolist())
            
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
            
            short_term = portfolio_with_metrics[portfolio_with_metrics['holding_type'] == 'short_term']
            long_term = portfolio_with_metrics[portfolio_with_metrics['holding_type'] == 'long_term']
            
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
            display_df = display_df[[
                'ticker', 'shares', 'purchase_price', 'current_price', 
                'unrealized_gain', 'unrealized_gain_percent', 'holding_type', 
                'estimated_tax', 'effective_tax_rate'
            ]]
            
            # Rename columns for better readability
            display_df.columns = [
                'Ticker', 'Shares', 'Purchase Price', 'Current Price', 
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
            display_df['Holding Type'] = display_df['Holding Type'].str.replace('_', ' ').str.title()
            
            st.dataframe(display_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error displaying portfolio: {str(e)}")

# Tab 3: Market Overview
with tab3:
    st.header("Market Overview")
    
    if not st.session_state.portfolio_loaded:
        st.info("Please load a portfolio to view market analysis.")
    else:
        try:
            # Get market indices
            indices = {
                'S&P 500': '^GSPC',
                'Dow Jones': '^DJI',
                'NASDAQ': '^IXIC',
                'VIX': '^VIX'
            }
            
            index_values = get_current_prices(indices.values())
            index_display = {name: index_values.get(symbol, 0) for name, symbol in indices.items()}
            
            # Display market indices
            st.subheader("Market Indices")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("S&P 500", f"{index_display.get('S&P 500', 0):.2f}")
            
            with col2:
                st.metric("Dow Jones", f"{index_display.get('Dow Jones', 0):.2f}")
            
            with col3:
                st.metric("NASDAQ", f"{index_display.get('NASDAQ', 0):.2f}")
            
            with col4:
                st.metric("VIX", f"{index_display.get('VIX', 0):.2f}")
            
            # Display sector performance (simulated)
            st.subheader("Sector Performance")
            
            sector_performance = {
                'Technology': 2.5,
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
            
            # Display portfolio ticker analysis
            st.subheader("Portfolio Ticker Analysis")
            
            # Simulated technical indicators
            ticker_indicators = {
                ticker: {
                    'RSI': np.random.uniform(30, 70),
                    'Trend': np.random.choice(['Bullish', 'Bearish', 'Neutral']),
                    'Volatility': np.random.uniform(0.5, 2.5),
                    'Status': np.random.choice(['Neutral', 'Overbought', 'Oversold'])
                } for ticker in st.session_state.portfolio_data['ticker']
            }
            
            # Create a dataframe for ticker indicators
            ticker_data = []
            
            for ticker, indicators in ticker_indicators.items():
                ticker_data.append({
                    'Ticker': ticker,
                    'RSI': indicators['RSI'],
                    'Trend': indicators['Trend'],
                    'Volatility': indicators['Volatility'],
                    'Status': indicators['Status']
                })
            
            ticker_df = pd.DataFrame(ticker_data)
            
            # Format numeric columns
            ticker_df['RSI'] = ticker_df['RSI'].map('{:.2f}'.format)
            ticker_df['Volatility'] = ticker_df['Volatility'].map('{:.2f}%'.format)
            
            st.dataframe(ticker_df, use_container_width=True)
            
            # Display market sentiment
            st.subheader("Market Sentiment")
            sentiment = np.random.choice(['🐂 Bullish', '🐻 Bearish', '😐 Neutral', '🔥 Overbought', '❄️ Oversold'])
            
            st.info(f"Current market sentiment: {sentiment}")
            
            # Display upcoming earnings
            st.subheader("Upcoming Earnings")
            
            # Simulated earnings data
            earnings = [
                {'ticker': 'AAPL', 'compa<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>
