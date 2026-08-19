# Buyback June 15, 2025 for tax issue (wash sale rule: 30 days)
# select Base kernel python 3.9.7

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

def get_stock_info(ticker):
    """Get stock information including first available date."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return stock, info
    except Exception as e:
        print(f"❌ Error getting info for {ticker}: {e}")
        return None, None

def download_stock_data(ticker, start_date='max', end_date=None):
    """
    Download stock data from Yahoo Finance.
    
    Parameters:
    - ticker: Stock symbol
    - start_date: 'max' for all available data, or specific date 'YYYY-MM-DD'
    - end_date: End date, defaults to today
    """
    try:
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Method 1: Use 'max' period to get all available data
        if start_date == 'max' or start_date == 'all':
            print(f"📊 Downloading all available data for {ticker}...")
            stock_data = yf.download(ticker, period='max', progress=False)
        else:
            print(f"📊 Downloading {ticker} from {start_date} to {end_date}...")
            stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Handle multi-index columns (newer yfinance versions)
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.droplevel(1)
        
        # Check if data is empty
        if stock_data.empty:
            raise ValueError(f"No data found for {ticker}")
        
        # Ensure index is datetime
        stock_data.index = pd.to_datetime(stock_data.index)
        
        # Remove any future dates (data corruption check)
        today = pd.Timestamp.now().normalize()
        stock_data = stock_data[stock_data.index <= today]
        
        # Remove rows with NaN in Close price
        stock_data = stock_data.dropna(subset=['Close'])
        
        # Get first and last available dates
        first_date = stock_data.index[0].strftime('%Y-%m-%d')
        last_date = stock_data.index[-1].strftime('%Y-%m-%d')
        
        print(f"✅ {ticker}: Downloaded {len(stock_data)} rows")
        print(f"   📅 First available date: {first_date}")
        print(f"   📅 Last available date:  {last_date}")
        
        return stock_data
    
    except Exception as e:
        print(f"❌ Error downloading {ticker}: {e}")
        return pd.DataFrame()

def download_stock_data_from_date(ticker, years_back=None, start_date=None, end_date=None):
    """
    Download stock data with flexible date options.
    
    Parameters:
    - ticker: Stock symbol
    - years_back: Number of years back from today (e.g., 5 for 5 years)
    - start_date: Specific start date 'YYYY-MM-DD' (overrides years_back)
    - end_date: End date, defaults to today
    
    Examples:
        download_stock_data_from_date('AAPL', years_back=5)  # Last 5 years
        download_stock_data_from_date('AAPL', start_date='2020-01-01')  # From specific date
        download_stock_data_from_date('AAPL')  # All available data
    """
    try:
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Determine start date
        if start_date:
            # Use specific start date
            pass
        elif years_back:
            # Calculate start date based on years back
            start_dt = datetime.now() - timedelta(days=years_back * 365)
            start_date = start_dt.strftime('%Y-%m-%d')
        else:
            # Get all available data
            start_date = 'max'
        
        return download_stock_data(ticker, start_date, end_date)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return pd.DataFrame()

def get_first_available_date(ticker):
    """Get the first available trading date for a stock."""
    try:
        # Download minimal data with max period
        stock_data = yf.download(ticker, period='max', progress=False)
        
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.droplevel(1)
        
        if not stock_data.empty:
            first_date = stock_data.index[0]
            return first_date.strftime('%Y-%m-%d')
        return None
    except:
        return None

def get_value(series_or_value):
    """Safely extract scalar value from pandas Series, numpy array, or scalar."""
    if series_or_value is None:
        return np.nan
    
    if isinstance(series_or_value, pd.Series):
        if len(series_or_value) > 0:
            val = series_or_value.iloc[0]
            return float(val) if not pd.isna(val) else np.nan
        return np.nan
    
    if isinstance(series_or_value, np.ndarray):
        if series_or_value.size > 0:
            val = series_or_value.flat[0]
            return float(val) if not pd.isna(val) else np.nan
        return np.nan
    
    if hasattr(series_or_value, 'item'):
        try:
            return float(series_or_value.item())
        except:
            pass
    
    try:
        return float(series_or_value)
    except:
        return np.nan

def save_to_csv(stock_data, ticker, output_dir='stock_data'):
    """Save stock data to CSV file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    start_date = stock_data.index[0].strftime('%Y%m%d')
    end_date = stock_data.index[-1].strftime('%Y%m%d')
    filename = f"{ticker}_{start_date}_{end_date}.csv"
    filepath = os.path.join(output_dir, filename)
    
    stock_data.to_csv(filepath)
    print(f"   💾 Saved to: {filepath}")
    
    return filepath

def plot_price_and_volume(stock_data, ticker):
    """Plot only price and volume."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, 
                             gridspec_kw={'height_ratios': [3, 1]})
    
    dates = stock_data.index
    close_prices = stock_data['Close'].values.flatten()
    high_prices = stock_data['High'].values.flatten()
    low_prices = stock_data['Low'].values.flatten()
    volume = stock_data['Volume'].values.flatten()
    
    # Plot 1: Price
    axes[0].plot(dates, close_prices, label='Close', linewidth=1.5, color='blue')
    axes[0].fill_between(dates, low_prices, high_prices, alpha=0.2, color='blue', label='High-Low Range')
    
    latest_price = close_prices[-1]
    max_price = np.nanmax(close_prices)
    min_price = np.nanmin(close_prices)
    max_idx = np.nanargmax(close_prices)
    min_idx = np.nanargmin(close_prices)
    
    axes[0].axhline(y=latest_price, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
    axes[0].annotate(f'Current: ${latest_price:.2f}', 
                     xy=(dates[-1], latest_price), 
                     xytext=(10, 0), textcoords='offset points',
                     fontsize=10, color='green', fontweight='bold')
    
    axes[0].scatter(dates[max_idx], max_price, color='red', s=100, zorder=5, marker='v')
    axes[0].annotate(f'High: ${max_price:.2f}', 
                     xy=(dates[max_idx], max_price), 
                     xytext=(0, 10), textcoords='offset points',
                     fontsize=9, color='red', ha='center')
    
    axes[0].scatter(dates[min_idx], min_price, color='green', s=100, zorder=5, marker='^')
    axes[0].annotate(f'Low: ${min_price:.2f}', 
                     xy=(dates[min_idx], min_price), 
                     xytext=(0, -15), textcoords='offset points',
                     fontsize=9, color='green', ha='center')
    
    axes[0].set_title(f'{ticker} Stock Price', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price ($)', fontsize=12)
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Volume
    colors = ['green' if close_prices[i] >= close_prices[i-1] else 'red' 
              for i in range(1, len(close_prices))]
    colors.insert(0, 'gray')
    
    axes[1].bar(dates, volume, color=colors, alpha=0.7)
    avg_volume = np.nanmean(volume)
    axes[1].axhline(y=avg_volume, color='blue', linestyle='--', linewidth=1, 
                    label=f'Avg Volume: {avg_volume/1e6:.2f}M')
    
    axes[1].set_title('Volume', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Volume', fontsize=12)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    plt.tight_layout()
    filename = f'stock_data/plot_{ticker}_price_and_volume.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    #plt.show()

def print_price_summary(stock_data, ticker):
    """Print price and volume summary."""
    latest = stock_data.iloc[-1]
    first = stock_data.iloc[0]
    
    close_price = get_value(latest['Close'])
    open_price = get_value(latest['Open'])
    high_price = get_value(latest['High'])
    low_price = get_value(latest['Low'])
    volume = get_value(latest['Volume'])
    first_close = get_value(first['Close'])
    
    period_high = get_value(stock_data['High'].max())
    period_low = get_value(stock_data['Low'].min())
    avg_volume = get_value(stock_data['Volume'].mean())
    
    if not np.isnan(close_price) and not np.isnan(first_close) and first_close != 0:
        period_return = ((close_price - first_close) / first_close) * 100
    else:
        period_return = np.nan
    
    print(f"\n{'='*50}")
    print(f"  {ticker} PRICE & VOLUME SUMMARY")
    print(f"  Period: {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Total trading days: {len(stock_data)}")
    print(f"{'='*50}")
    print(f"  Latest Date:     {stock_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Open:            ${open_price:.2f}")
    print(f"  High:            ${high_price:.2f}")
    print(f"  Low:             ${low_price:.2f}")
    print(f"  Close:           ${close_price:.2f}")
    print(f"  Volume:          {volume/1e6:.2f}M")
    print(f"{'='*50}")
    print(f"  Period High:     ${period_high:.2f}")
    print(f"  Period Low:      ${period_low:.2f}")
    print(f"  Period Return:   {period_return:+.2f}%")
    print(f"  Avg Volume:      {avg_volume/1e6:.2f}M")
    print(f"{'='*50}\n")

def analyze_multiple_stocks(tickers, start_date='max', end_date=None, output_dir='stock_data'):
    """Analyze multiple stocks and save all data to CSV."""
    results = []
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for ticker in tickers:
        try:
            print(f"\n{'─'*40}")
            data = download_stock_data(ticker, start_date, end_date)
            
            if data.empty:
                print(f"   ⚠️ No data for {ticker}, skipping...")
                continue
            
            save_to_csv(data, ticker, output_dir)
            
            latest = data.iloc[-1]
            first = data.iloc[0]
            
            close_price = get_value(latest['Close'])
            first_close = get_value(first['Close'])
            volume = get_value(latest['Volume'])
            avg_volume = get_value(data['Volume'].mean())
            
            if not np.isnan(close_price) and not np.isnan(first_close) and first_close != 0:
                period_return = ((close_price - first_close) / first_close) * 100
            else:
                period_return = np.nan
            
            results.append({
                'Ticker': ticker,
                'First_Date': data.index[0].strftime('%Y-%m-%d'),
                'Last_Date': data.index[-1].strftime('%Y-%m-%d'),
                'Days': len(data),
                'Price': close_price,
                'Return%': period_return,
                'Volume': volume,
                'Avg_Volume': avg_volume
            })
            
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
    
    if not results:
        print("⚠️ No data collected for any stocks")
        return pd.DataFrame(), pd.DataFrame()
    
    summary_df = pd.DataFrame(results)
    
    # Format for display
    display_df = summary_df.copy()
    display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}" if not np.isnan(x) else "N/A")
    display_df['Return%'] = display_df['Return%'].apply(lambda x: f"{x:+.1f}%" if not np.isnan(x) else "N/A")
    display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{x/1e6:.2f}M" if not np.isnan(x) else "N/A")
    display_df['Avg_Volume'] = display_df['Avg_Volume'].apply(lambda x: f"{x/1e6:.2f}M" if not np.isnan(x) else "N/A")
    
    summary_path = os.path.join(output_dir, 'summary_all_stocks.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ Summary saved to: {summary_path}")
    
    return display_df, summary_df

# Main execution
if __name__ == "__main__":
    
    # Configuration
    ticker = 'NVDA'
    output_dir = 'stock_data'
    
    print(f"\n🚀 Starting Stock Analysis")
    print(f"   Today's date: {datetime.now().strftime('%Y-%m-%d')}")
    
    # ============================================================
    # OPTION 1: Get ALL available historical data (use period='max')
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  OPTION 1: ALL AVAILABLE DATA")
    print(f"{'='*60}")
    
    stock_data = download_stock_data(ticker, start_date='max')
    
    # ============================================================
    # OPTION 2: Get data for last N years
    # ============================================================
    # stock_data = download_stock_data_from_date(ticker, years_back=5)  # Last 5 years
    
    # ============================================================
    # OPTION 3: Get data from specific date
    # ============================================================
    # stock_data = download_stock_data_from_date(ticker, start_date='2020-01-01')
    
    # ============================================================
    # OPTION 4: Check first available date first, then download
    # ============================================================
    # first_date = get_first_available_date(ticker)
    # print(f"First available date for {ticker}: {first_date}")
    # stock_data = download_stock_data(ticker, start_date=first_date)
    
    if not stock_data.empty:
        save_to_csv(stock_data, ticker, output_dir)
        print_price_summary(stock_data, ticker)
        plot_price_and_volume(stock_data, ticker)
    else:
        print(f"❌ Failed to download data for {ticker}")
    
    # Analyze multiple stocks with all available data
    print(f"\n{'='*60}")
    print(f"  MULTIPLE STOCKS - ALL AVAILABLE DATA")
    print(f"{'='*60}")
    
    stocks = ['META', 'GOOGL','AMZN', 'MSFT']
    display_df, raw_df = analyze_multiple_stocks(stocks, start_date='max', output_dir=output_dir)
    
    if not display_df.empty:
        print(f"\n📈 Stock Summary (All Available Data):")
        print(display_df.to_string(index=False))
    
    print(f"\n✅ All data saved to '{output_dir}/' folder")



#output_dir = 'stock_data'
#stocks = ['META', 'GOOGL','AMZN', 'MSFT']
#display_df, raw_df = analyze_multiple_stocks(stocks, start_date='max', output_dir=output_dir)