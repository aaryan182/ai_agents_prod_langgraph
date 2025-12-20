import yfinance as yf

def fetch_stock_data(ticker: str, period: str):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    if df.empty:
        raise ValueError("No market data found")
    
    return df