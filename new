# pip install streamlit
# streamlit run your_file.py
# pip install streamlit yfinance pandas matplotlib pmdarima

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima

# site title
st.set_page_config(page_title="S&P 500 Stock Finder", layout="wide")
# list of all s&p tickers
SP500 = [
    "MMM", "AOS", "ABT", "ABBV", "ACN", "ATVI", "ADM", "ADBE", "AAP", "AES",
    "AFL", "A", "APD", "AKAM", "ALK", "ALB", "ARE", "ALGN", "ALLE", "LNT",
    "ALL", "GOOGL", "GOOG", "MO", "AMZN", "AMCR", "AEE", "AAL", "AEP", "AXP",
    "AIG", "AMT", "AWK", "AMP", "ABC", "AME", "AMGN", "APH", "ADI", "ANSS",
    "ANTM", "AON", "AOS", "APA", "AIV", "AAPL", "AMAT", "APTV", "ADM", "ARNC",
    "ANET", "AJG", "AIZ", "ATO", "T", "ADSK", "ADP", "AZO", "AVB", "AVY",
    "BKR", "BLL", "BAC", "BK", "BAX", "BDX", "BRK.B", "BBY", "BIO", "TECH",
    "BIIB", "BLK", "HRB", "BA", "BKNG", "BWA", "BXP", "BSX", "BMY", "AVGO",
    "BR", "BF.B", "CHRW", "COG", "CDNS", "CPB", "COF", "CAH", "KMX", "CCL",
    "CARR", "CAT", "CBOE", "CBRE", "CDW", "CE", "CNC", "CNP", "CTL", "CERN",
    "CF", "SCHW", "CHTR", "CVX", "CMG", "CB", "CHD", "CI", "CINF", "CTAS",
    "CLX", "CME", "CMS", "KO", "CTSH", "CL", "CMCSA", "CMA", "CAG", "CXO",
    "CTRA", "COP", "ED", "STZ", "CPRT", "GLW", "COST", "CCI", "CSX", "CME",
    "CVS", "DHI", "DHR", "DRI", "DHR", "DVA", "DE", "DAL", "XRAY", "DVN",
    "DXC", "FANG", "DLR", "DFS", "DISCA", "DISCK", "DISH", "DG", "DLTR", "D",
    "DOV", "DOW", "DTE", "DUK", "DRE", "DD", "DXC", "EMN", "ETN", "EBAY",
    "ECL", "EIX", "EW", "EA", "EMR", "ETR", "EOG", "EFX", "EQIX", "EQR",
    "ESS", "EL", "EVRG", "ES", "EXC", "EXPE", "EXPD", "EXPW", "EXR", "XOM",
    "FFIV", "FBHS", "FAST", "FRT", "FDX", "FIS", "FITB", "FRC", "FE", "FISV",
    "FLT", "FMC", "F", "FTV", "FB", "FOXA", "FOX", "BEN", "FCX", "GPS",
    "GRMN", "IT", "GD", "GE", "GIS", "GM", "GPC", "GILD", "GL", "GPN",
    "GS", "GT", "GWW", "HAL", "HBI", "HOG", "HIG", "HAS", "HCA", "PEAK",
    "HP", "HSIC", "HSY", "HES", "HPE", "HLT", "HOLX", "HD", "HON", "HRL",
    "HST", "HPQ", "HUM", "HBAN", "HII", "IEX", "IHS", "ILMN", "INCY", "IDXX",
    "IFF", "IP", "IPG", "INTC", "ICE", "IBM", "IFF", "INTU", "ISRG", "IVZ",
    "IPGP", "JKHY", "J", "JBHT", "SJM", "JNJ", "JCI", "JPM", "JNPR", "KSU",
    "K", "KEY", "KMB", "KIM", "KMI", "KSS", "KR", "KHC", "KIM", "KMI",
    "KO", "CL", "KR", "LHX", "LH", "LRCX", "LW", "LVS", "LEG", "LDOS",
    "LEN", "LLY", "LNC", "LIN", "LYV", "LKQ", "LMT", "L", "LOW", "LYB",
    "MTB", "MRO", "MPC", "MAR", "MMC", "MLR", "MHK", "TAP", "MCD", "MCK",
    "MDT", "MRK", "MET", "MGM", "KORS", "MCHP", "MU", "MSFT", "MA", "MTD",
    "MHK", "MCO", "MS", "MOS", "MSI", "MSCI", "MYL", "NDAQ", "NOV", "NAVI",
    "NUE", "NVR", "NEE", "NLSN", "NKE", "NI", "NBL", "NSC", "NTRS", "NOC",
    "NCLH", "NRG", "NUE", "NVDA", "NVR", "ORLY", "OI", "OKE", "ORCL", "OGN",
    "OIH", "OXY", "PCAR", "PKG", "PH", "PAYX", "PYPL", "PNR", "PEP", "PKI",
    "PFE", "PM", "PSX", "PNW", "PXD", "PNC", "PPG", "PPL", "PFG", "PG",
    "PLD", "PRU", "PEG", "PSA", "PHM", "PVH", "QRVO", "QCOM", "PWR", "DGX",
    "RMD", "RJF", "RTN", "O", "REG", "REGN", "RF", "RSG", "RMD", "RHI",
    "ROK", "ROL", "ROP", "ROST", "RCL", "SPGI", "CRM", "SBAC", "SLB", "STX",
    "STT", "SYK", "SIVB", "SNPS", "SBUX", "STZ", "SO", "LUV", "SWK", "SLG",
    "SNA", "SRE", "SPG", "SWKS", "SYY", "TMUS", "TROW", "TTWO", "TPR", "TRV",
    "TRMB", "TGT", "TEL", "TDY", "TFX", "TER", "TSLA", "TXN", "TXT", "TMO",
    "TJX", "TSCO", "TT", "TMUS", "UDR", "ULTA", "USB", "UAA", "UNP", "UAL",
    "UNH", "UPS", "URI", "UTX", "VLO", "VTR", "VZ", "VRTX", "VIAC", "VFC",
    "VTRS", "VRSK", "VRSN", "VMC", "VFC", "VFC", "VNO", "VMC", "VZ", "WAB",
    "WMT", "WBA", "WM", "WAT", "WEC", "WELL", "WDC", "WU", "WRB", "WMB",
    "WHR", "WLTW", "WY", "WMB", "WEC", "WELL", "WST", "XEL", "XLNX", "XYL",
    "YUM", "ZBH", "ZION", "ZTS"
]


# Checks user input of stock ticker
def validate_ticker(ticker):
    ticker = ticker.upper()
    if ticker in SP500:
        return True, ""
    else:
        return False, "Ticker not in S&P 500 or invalid."


# Prevents errors in case some values (like beta or PE rate) are missing from yfinance
def safe(val, default="N/A"):
    if val is not None:
        return val
    else:
        return default


# Load stock data for given ticker and timeframe
@st.cache_data
def load_stock_data(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        data_intervals = {
            "1d": "1m",
            "1mo": "1d",
            "3mo": "1d",
            "6mo": "1d",
            "1y": "1d",
            "5y": "1wk",
            "max": "1mo"
        }
        interval = data_intervals.get(period, "1d")
        # hist = previous stock data
        hist = stock.history(period=period, interval=interval)
        if hist.empty:
            raise ValueError("No data on stock")
        info = stock.info
        return hist, info
    except Exception as e:
        st.warning(f"Failed to load {ticker} data. Error: {e}")
        return pd.DataFrame(), {}


# get data from the past year for arima forecasting model
@st.cache_data
def load_arima_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y", interval="1d")
        if hist.empty:
            raise ValueError("No data on stock")
        # converting data to dates (ex: 2025-11-11)
        hist.index = pd.to_datetime(hist.index)
        return hist["Close"]
    except Exception as e:
        st.warning(f"Failed to load {ticker} data for ARIMA. Error: {e}")
        return pd.Series(dtype=float)


# ARIMA forecast
def auto_arima_with_returns(series, steps=5):
    try:
        returns = series.pct_change().dropna()
        # auto arima automatically takes the model with the lowest AIC (best fit model)
        model = auto_arima(returns, seasonal=False, error_action="ignore", suppress_warnings=True)
        forecast_returns = model.predict(n_periods=steps)
        last_price = series.iloc[-1]
        forecast_prices = [last_price * (1 + r) for r in forecast_returns]
        # gets details about auto arima model
        details = {
            "forecast": forecast_prices,
            "order": model.order,
            "aic": model.aic(),
            "bic": model.bic(),
            "residuals_std": model.resid().std()
        }
        return details
    except:
        return {"forecast": [], "order": None, "aic": None, "bic": None, "residuals_std": None}


def generate_recommendation(hist, arima_details, info):
    if hist.empty:
        return {
            "recommendation": "No data",
            "total points": 0,
            "points_breakdown": {}
        }

    current = hist["Close"].iloc[-1]
    low_52 = hist["Close"].min()
    high_52 = hist["Close"].max()

    # ARIMA points
    forecast = arima_details.get("forecast", [])
    days_higher = sum(f > current for f in forecast)

    if days_higher == 1:
        arima_pts = 0
    elif 2 <= days_higher <= 3:
        arima_pts = 2
    elif days_higher == 4:
        arima_pts = 3
    elif days_higher >= 5:
        arima_pts = 4
    else:
        arima_pts = 0

    # calculates if price level is considered on the high or lower end
    lower_quartile = low_52 + (high_52 - low_52) * 0.25
    upper_quartile = low_52 + (high_52 - low_52) * 0.75

    if current <= lower_quartile:
        price_pts = 3
    elif current >= upper_quartile:
        price_pts = 0
    else:
        price_pts = 1

    # PE ratio points to calculate if PE is greater than average
    pe = info.get("trailingPE", None)
    pe_avg = info.get("forwardPE", None)

    if pe is not None and pe_avg is not None:
        if pe < pe_avg:
            pe_pts = 0
        elif pe > pe_avg:
            pe_pts = 3
        else:
            pe_pts = 1
    else:
        pe_pts = 1

    # point system for recommending whether to buy sell or hold
    total_points = arima_pts + price_pts + pe_pts

    if total_points >= 7:
        recommendation = "Buy stock: high potential for growth"
    elif 4 <= total_points < 7:
        recommendation = "Hold stock: no big change in stock prices️"
    else:
        recommendation = "Sell stock: may decrease in the future"
    # prints out all the results
    return {
        "recommendation": recommendation,
        "total points": total_points,
        "points_breakdown": {
            "arima": arima_pts,
            "stock price": price_pts,
            "PE ratio": pe_pts
        }
    }


def plot_price_history(hist, period): #plots price history on forecast graph for 3 days prior
    fig, ax = plt.subplots(figsize=(10, 5))
    x_values = hist.index
    ax.plot(x_values, hist["Close"])

    if period == "1d":
        title = "Intraday Closing Prices (1 Day)"
    else:
        title = f"Closing Prices ({period})"

    # graph settings
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    return fig


# graph settings
def plot_forecast_zoomed(arima_series, forecast, days_before=3): #plots forecast
    fig, ax = plt.subplots(figsize=(10, 5))
    if arima_series.empty or len(forecast) == 0:
        return fig

    arima_series.index = pd.to_datetime(arima_series.index)
    arima_series = arima_series.asfreq("B").ffill()
    hist_zoom = arima_series.iloc[-days_before:]

    future_index = pd.date_range(
        start=arima_series.index[-1] + pd.Timedelta(days=1),
        periods=len(forecast),
        freq="B"
    )

    ax.plot(hist_zoom.index, hist_zoom.values, label="Historical (Daily Close)", linewidth=2)
    ax.plot(future_index, forecast, linestyle="--", marker="o", label="Predicted 5-Day Closing (ARIMA)")
    ax.set_title("5-day Prediction of closing prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    return fig


# Website layout
st.title("S&P 500 Stock Finder")
tabs = st.tabs(["Stock Overview", "Predictions", "Information Overview"])
# tabs of different sections of the website
with tabs[0]:  # tab that asks for ticker input and shows data of stock
    raw_input = st.text_input("Enter S&P 500 ticker (e.g., AAPL)")
    ticker = raw_input.upper().strip()
    valid, error = validate_ticker(ticker)
    if ticker and not valid:
        st.error(error)

    if valid:
        timeframe = st.selectbox("Select timeframe", ["1d", "1mo", "3mo", "6mo", "1y", "5y", "max"], index=0)
        hist, info = load_stock_data(ticker, period=timeframe)

        st.subheader(f"{safe(info.get('shortName'))} ({ticker})")
        st.subheader("Price Chart")
        st.pyplot(plot_price_history(hist, timeframe))

        # Key info metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Previous Close", safe(info.get("previousClose")))
        c1.metric("Open", safe(info.get("open")))
        c1.metric("Volume", safe(info.get("volume")))
        c2.metric("52W Low", safe(info.get("fiftyTwoWeekLow")))
        c2.metric("52W High", safe(info.get("fiftyTwoWeekHigh")))
        c2.metric("Avg Volume", safe(info.get("averageVolume")))
        c3.metric("Beta", safe(info.get("beta")))
        c3.metric("PE Ratio", safe(info.get("trailingPE")))
        c3.metric("EPS", safe(info.get("trailingEps")))

with tabs[1]:  # tab that shows forecasts
    if valid:
        arima_series = load_arima_data(ticker)
        arima_details = auto_arima_with_returns(arima_series)
        rec = generate_recommendation(hist, arima_details, info)
        st.subheader("5 Day Prediction of Closing Prices")
        st.pyplot(
            plot_forecast_zoomed(
                arima_series,
                arima_details.get("forecast", []),
                days_before=3
            )
        )
        # printing out previous results
        st.subheader("Recommendations")
        st.write(f"Recommendation: {rec['recommendation']}")
        st.write("Points breakdown:", rec["points_breakdown"])

        st.subheader("See more")
        st.write("ARIMA order:", arima_details.get("order"))
        st.write("AIC:", arima_details.get("aic"))
        st.write("BIC:", arima_details.get("bic"))
        st.write("Residual std:", arima_details.get("residuals_std"))

with tabs[2]:  # descriptions of vocab on the site
    st.subheader("Information Overview")

    st.markdown("""
**Vocabulary**


**Volume:** Total number of shares traded over a certain time frame. 
- High volume = strong interest, making it easier to buy and sell without impacting stock prices 
- Low volume = increased risk of market manipulation, making it more difficult to sell. 


**Average Volume:** The average number of shares traded per day over a time period. 


**Beta:** stock volatility in comparison to the overall market. 
- A beta of 1 = the stock moves in line with the market. 
- A beta greater than 1 =  higher volatility. 


**P/E Ratio (Price to Earnings Ratio):** Compares the current stock price to earnings per share.   


**EPS (Earnings Per Share):** Indicates the amount of profit a company generates for each share of stock. 
- Calculated by subtracting preferred dividends from the company’s net income. 
- Higher EPS suggests a more profitable company and potential for higher dividends. 


**Dividend:** A portion of a company’s profits distributed to shareholders 
- Higher dividends suggests financial stability and confidence in future cash flow. 


**Forecast:** The forecast was created using an ARIMA model
- AIC and BIC shows the fit of the model - the lower the AIC and BIC value the better fit the model is


**Recommendations:** The recommendations are based on the stock price, forecast, and PE ratio

**DISCLAIMER:** recommendations may not always be accurate
- Stock prices: it is better to buy low sell high
- Forecast: the forecast shows what the arima model suggests will happen to the stock
- PE ratio: higher PE ratio = buy ; lower PE ratio = sell
""")
