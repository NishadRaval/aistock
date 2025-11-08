import streamlit as st
import yfinance as yf
import finnhub  # Finnhub import
import pandas as pd
import numpy as np
import plotly.express as px
from xgboost import XGBClassifier  # <-- NEW: Upgraded Model
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta

# --- Configuration ---
st.set_page_config(page_title="AI Stock & Crypto Analyzer", page_icon="🚀", layout="wide")

# --- API Key ---
FINNHUB_API_KEY = "d47j8s9r01qtk51qd12gd47j8s9r01qtk51qd130"  # <-- PASTE YOUR KEY HERE
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# --- STOCK/CRYPTO HELPER FUNCTIONS ---

def calculate_rsi(data, window=14):
    """Calculates the Relative Strength Index (RSI)."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data(ttl="1h") 
def get_fundamental_score(info):
    """
    Calculates a comprehensive 'Long-Term Quality Score' (out of 7) 
    and returns a structured list of reasons.
    """
    score = 0
    reasons = [] 
    
    def get_metric(key, default=None):
        val = info.get(key, default)
        return val if val is not None else default

    try:
        # 1. Value: P/E Ratio (trailingPE)
        pe_ratio = get_metric('trailingPE', float('inf'))
        if 0 < pe_ratio < 20:
            score += 1
            reasons.append({"metric": "P/E Ratio", "status": "success", "value": f"{pe_ratio:.2f}",
                            "text": "The stock's price is low relative to its earnings (P/E < 20), suggesting it's reasonably priced."})
        elif pe_ratio <= 0:
            reasons.append({"metric": "P/E Ratio", "status": "error", "value": f"{pe_ratio:.2f}",
                            "text": "The company is not currently profitable, which is a significant risk flag."})
        else:
            reasons.append({"metric": "P/E Ratio", "status": "warning", "value": f"{pe_ratio:.2f}",
                            "text": "The stock's price is high relative to its earnings (P/E > 20), suggesting it may be overvalued."})

        # 2. Value: P/E-to-Growth (pegRatio)
        peg_ratio = get_metric('pegRatio', float('inf'))
        if 0 < peg_ratio < 1:
            score += 1
            reasons.append({"metric": "PEG Ratio", "status": "success", "value": f"{peg_ratio:.2f}",
                            "text": "A PEG ratio under 1 suggests the stock's price is low relative to its future earnings growth."})
        elif 1 < peg_ratio < 2:
            reasons.append({"metric": "PEG Ratio", "status": "warning", "value": f"{peg_ratio:.2f}",
                            "text": "The stock is fairly valued given its growth forecast. Not a bargain, but not expensive."})
        else:
            reasons.append({"metric": "PEG Ratio", "status": "error", "value": f"{peg_ratio:.2f}",
                            "text": "The stock appears expensive relative to its future growth, suggesting it may be overvalued."})

        # 3. Value: Price-to-Book (priceToBook)
        pb_ratio = get_metric('priceToBook', float('inf'))
        if 0 < pb_ratio < 1.5:
            score += 1
            reasons.append({"metric": "Price-to-Book Ratio", "status": "success", "value": f"{pb_ratio:.2f}",
                            "text": "The stock's price is low relative to the company's net assets, which can indicate it's undervalued."})
        else:
            reasons.append({"metric": "Price-to-Book Ratio", "status": "warning", "value": f"{pb_ratio:.2f}",
                            "text": "The stock price is high compared to its book value. This is common for tech companies but can be a sign of being overvalued."})

        # 4. Profitability: Profit Margin (profitMargins)
        profit_margin = get_metric('profitMargins', 0) * 100
        if profit_margin > 15:
            score += 1
            reasons.append({"metric": "Profit Margin", "status": "success", "value": f"{profit_margin:.2f}%",
                            "text": "The company keeps a large percentage of its revenue as profit, a very strong sign of efficiency."})
        elif profit_margin > 0:
            reasons.append({"metric": "Profit Margin", "status": "warning", "value": f"{profit_margin:.2f}%",
                            "text": "The company is profitable, but has thin margins, making it vulnerable to competition."})
        else:
            reasons.append({"metric": "Profit Margin", "status": "error", "value": f"{profit_margin:.2f}%",
                            "text": "The company is currently losing money, which is a major red flag."})

        # 5. Profitability: Return on Equity (returnOnEquity)
        roe = get_metric('returnOnEquity', 0) * 100
        if roe > 15:
            score += 1
            reasons.append({"metric": "Return on Equity (ROE)", "status": "success", "value": f"{roe:.2f}%",
                            "text": "The company generates high profits from its shareholders' money, indicating strong management."})
        else:
            reasons.append({"metric": "Return on Equity (ROE)", "status": "warning", "value": f"{roe:.2f}%",
                            "text": "The company is not generating strong returns for its shareholders."})

        # 6. Health: Debt-to-Equity (debtToEquity)
        debt_to_equity = get_metric('debtToEquity') 
        if debt_to_equity is None or debt_to_equity == 0:
            score += 1
            reasons.append({"metric": "Debt-to-Equity", "status": "success", "value": "No Debt",
                            "text": "The company has little to no debt, giving it a very strong and safe financial position."})
        elif debt_to_equity < 100:
            score += 1
            reasons.append({"metric": "Debt-to-Equity", "status": "success", "value": f"{debt_to_equity:.2f}",
                            "text": "The company's debt is well-managed and at a healthy level relative to its equity."})
        else:
            reasons.append({"metric": "Debt-to-Equity", "status": "error", "value": f"{debt_to_equity:.2f}",
                            "text": "The company uses a large amount of debt, which increases its risk."})

        # 7. Health: Current Ratio (currentRatio)
        current_ratio = get_metric('currentRatio', 0)
        if current_ratio > 1.5:
            score += 1
            reasons.append({"metric": "Current Ratio", "status": "success", "value": f"{current_ratio:.2f}",
                            "text": "The company has enough short-term assets to cover its short-term debts."})
        else:
            reasons.append({"metric": "Current Ratio", "status": "error", "value": f"{current_ratio:.2f}",
                            "text": "The company may struggle to pay its short-term bills, a significant risk factor."})

    except Exception as e:
        st.error(f"Error during scoring: {e}")

    return score, reasons, 7

@st.cache_data
def generate_long_term_summary(score, total_possible):
    """
    Generates a direct "Buy/Sell/Hold" recommendation and a detailed summary
    based on the fundamental score.
    """
    score_percentage = score / total_possible
    
    if score_percentage >= 0.8:
        rating = "Long-Term Buy 🟢"
        summary = (
            "**AI Analysis:** Based on **excellent fundamental health**, this company appears to be a strong candidate for a long-term (6+ months) 'Buy' position. "
            "It shows signs of profitability, efficiency, and financial stability."
        )
    elif score_percentage >= 0.4:
        rating = "Long-Term Hold 🟡"
        summary = (
            "**AI Analysis:** Based on **mixed fundamentals**, the AI suggests a 'Hold' position. "
            "The company shows a mix of strengths and weaknesses (e.g., good value but high debt, or good profitability but overvalued). "
            "Caution and further research are advised."
        )
    else: 
        rating = "Long-Term Sell 🔴"
        summary = (
            "**AI Analysis:** Based on **multiple fundamental red flags** (e.g., unprofitability, high debt, or extreme valuation), "
            "the AI suggests a 'Sell' or 'Avoid' position for long-term investors. "
            "The risk of a poor long-term outcome appears high."
        )
        
    return rating, summary

@st.cache_data(ttl="1h")
def fetch_stock_data(symbol):
    """Fetches historical data and company info from yfinance."""
    ticker = yf.Ticker(symbol)
    # --- UPGRADE 1: Get 10 years of data ---
    hist_df = ticker.history(period="10y")
    info = ticker.info
    if hist_df.empty: 
        return None, None
    return hist_df, info

@st.cache_data(ttl="1h")
def fetch_finnhub_news(symbol):
    """Fetches recent company news from Finnhub."""
    try:
        if FINNHUB_API_KEY == "YOUR_FINNHUB_API_KEY_HERE":
            return [{"headline": "Please add your Finnhub API key to the code to see news.", "source": "App", "url": "https://finnhub.io/"}]
        
        # Get today and yesterday's date
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Make the API call
        news = finnhub_client.company_news(symbol, _from=yesterday, to=today)
        return news
    except Exception as e:
        # Fails silently for the user
        print(f"Silent Error fetching news: {e}") 
        return [] # Just return an empty list.

def calculate_technical_indicators(df):
    """Calculates MAs, RSI, and Volume."""
    df = df.copy()
    
    # Standard MAs
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI (uses our existing helper)
    df['RSI_14'] = calculate_rsi(df['Close'], window=14)
    
    # 'volume' is from yfinance, 'Volume' with a capital V
    df['volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    
    df_processed = df.dropna()
    return df_processed, df.iloc[[-1]] # Return processed data AND the last row

# --- UPGRADED: create_model function with XGBoost ---
@st.cache_data(ttl="1.h")
def create_model(df):
    """
    Creates, trains, and returns a short-term prediction model
    using XGBoost on 10 years of data.
    """
    n_days = 5
    threshold = 0.02
    df['Future_Price'] = df['Close'].shift(-n_days)
    df['Future_Return'] = (df['Future_Price'] - df['Close']) / df['Close']
    
    # Remap signals for ML (0=Sell, 1=Hold, 2=Buy)
    df['Signal'] = np.where(df['Future_Return'] > threshold, 2,  # Buy
                   np.where(df['Future_Return'] < -threshold, 0, 1)) # Sell, else Hold
    
    # Back to the original 4 features
    features = [
        'MA_20', 
        'MA_50', 
        'RSI_14', 
        'volume'
    ]
    
    df = df.dropna()
    
    available_features = [f for f in features if f in df.columns]
    
    X = df[available_features]
    y = df['Signal']
    
    if len(X) < 200: # Need at least ~200 data points for 10y
        return None, "Not enough data to train model."
        
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    if len(X_test) == 0: 
        return None, "Not enough test data."
        
    # --- UPGRADE 2: Use XGBClassifier ---
    model = XGBClassifier(
        n_estimators=100, 
        random_state=42,
        use_label_encoder=False,  # To avoid a warning
        eval_metric='mlogloss'    # Metric for multi-class classification
    )
    model.fit(X_train, y_train)
    
    # Train final model on ALL data
    final_model = XGBClassifier(
        n_estimators=100, 
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    final_model.fit(X, y)
    
    return final_model, available_features

# --- ============================= ---
# ---      STREAMLIT APP LAYOUT     ---
# --- ============================= ---

st.title("🚀 AI Stock & Crypto Analyzer")
st.write("Analyzes Indian (NSE/BSE), US, and Crypto tickers.")

st.header("Stock / Crypto Analyzer")
stock_symbol = st.text_input(
    "Enter a symbol (e.g., AAPL, INFY.NS, RELIANCE.NS, BTC-USD)", 
    "RELIANCE.NS"
).upper()

if st.button("Analyze & Predict"):
    # --- UPGRADE: Updated spinner text ---
    with st.spinner(f"Analyzing {stock_symbol} (using 10 years of data)... This may take a moment."):
        try:
            hist_df, info = fetch_stock_data(stock_symbol)
            news = fetch_finnhub_news(stock_symbol) 
            
            if hist_df is None:
                st.error(f"Could not fetch data for {stock_symbol}. Is the ticker correct?")
            else:
                # Run all AI/data processing
                tech_df, last_row = calculate_technical_indicators(hist_df)
                model, model_features = create_model(tech_df.copy())
                
                prediction_text = "N/A"
                if model:
                    features_to_predict = last_row[model_features]
                    if features_to_predict.isnull().values.any():
                        prediction_text = "Wait (Indicators calculating)"
                    else:
                        prediction = model.predict(features_to_predict)[0]
                        
                        if prediction == 2: 
                            prediction_text = "Buy 📈"
                        elif prediction == 0: 
                            prediction_text = "Sell 📉"
                        else: 
                            prediction_text = "Hold 😐"
                else: 
                    st.warning(f"Could not train AI model: {model_features}")

                score, reasons, total_possible = get_fundamental_score(info)
                rating, summary = generate_long_term_summary(score, total_possible)
                pe_ratio = info.get('trailingPE', 'N/A')
                pe_text = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A"

                #
                # --- PROFESSIONAL "CARD" LAYOUT ---
                #
                st.header(f"Analysis for {info.get('longName', stock_symbol)}")
                
                # --- Row 1: Key Metrics (HONEST VERSION) ---
                with st.container(border=True):
                    st.subheader("Key Metrics & Predictions")
                    cols = st.columns(3) # 3 columns
                    cols[0].metric("AI Prediction (5-Day)", prediction_text, help="A short-term technical guess based on 10 years of data.")
                    cols[1].metric("Long-Term Score", f"{score} / {total_possible}", help="A fundamental score for Value, Profitability, and Health.")
                    cols[2].metric("P/E Ratio", pe_text, help="Price-to-Earnings Ratio (Value)")

                # --- Row 2: Long-Term Recommendation ---
                with st.container(border=True):
                    st.subheader("Long-Term AI Recommendation (3-6+ Months)")
                    st.warning("⚠️ **Disclaimer:** This is an AI-generated analysis based *only* on fundamental data. It is **not** financial advice. This is an educational tool for research.")
                    
                    col1_summ, col2_summ = st.columns([1, 2])
                    with col1_summ:
                        st.metric("AI Recommendation", rating)
                    with col2_summ:
                        st.markdown(summary)

                # --- Row 3: Chart & News ---
                with st.container(border=True):
                    col1_chart, col2_news = st.columns([2, 1])
                    with col1_chart:
                        # --- UPGRADE: Updated chart title ---
                        st.subheader("Price Chart (Last 10 Years)")
                        fig = px.line(hist_df, y='Close', title=f'{stock_symbol} Closing Price')
                        st.plotly_chart(fig, use_container_width=True)

                    with col2_news:
                        st.subheader("Recent News")
                        if news:
                            for item in news[:5]: # Show top 5 news items
                                title = item.get('headline', 'No Title Available')
                                source = item.get('source', 'No Source')
                                link = item.get('url', None)
                                
                                st.markdown(f"**{title}**")
                                if link and source:
                                    st.markdown(f"_{source}_ | [Read More]({link})")
                                elif source:
                                    st.markdown(f"_{source}_")
                                st.divider()
                        else:
                            st.write("No recent news found for this ticker.")
                
                # --- Row 4: Detailed Breakdowns ---
                with st.container(border=True):
                    st.subheader("Detailed Analysis")
                    col1_detail, col2_detail = st.columns(2)

                    with col1_detail:
                        st.subheader("Detailed Fundamental Breakdown")
                        for item in reasons:
                            val = item.get('value', 'N/A')
                            if item['status'] == 'success':
                                st.success(f"**{item['metric']} ({val}):** {item['text']}")
                            elif item['status'] == 'warning':
                                st.warning(f"**{item['metric']} ({val}):** {item['text']}")
                            else:
                                st.error(f"**{item['metric']} ({val}):** {item['text']}")
                    
                    with col2_detail:
                        st.subheader("Technical Indicators (Recent)")
                        st.dataframe(tech_df.tail(), use_container_width=True)

                        st.subheader("Company Overview")
                        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                        st.write(f"**Website:** {info.get('website', 'N/A')}")
                        summary_text = info.get('longBusinessSummary', 'No company description available.')
                        st.markdown(summary_text[:400] + "...")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

            # --- ============================= ---
# ---           FOOTER              ---
# --- ============================= ---

st.divider()

st.caption("""
© 2025 Nishad Raval. All Rights Reserved.  
This is an educational portfolio project. The information and AI predictions provided are for informational purposes only and **do not constitute financial advice**.  
This application does not collect, store, or share any personal user data.
""")