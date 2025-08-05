import os
import requests
import pandas as pd
import time
from datetime import datetime
from dotenv import load_dotenv
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

API_KEY = os.getenv('ALPHA_KEY')
BASE_URL = 'https://www.alphavantage.co/query'
OUTPUT_DIR = 'downloaded'
INSTRUMENTS_FILE = 'instruments.txt'
CUTOFF_DATE = '2023-01-01'
PROGRESS_FILE = 'fetch_progress.json'

def load_instruments():
    """Load instruments from instruments.txt file"""
    instruments_path = os.path.join(os.path.dirname(__file__), INSTRUMENTS_FILE)
    with open(instruments_path, 'r') as f:
        instruments = [line.strip() for line in f if line.strip()]
    return instruments

def check_existing_download(symbol):
    """Check if we already have data for this symbol"""
    filename = os.path.join(OUTPUT_DIR, f"{symbol.replace('/', '_')}.csv")
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename, index_col=0, parse_dates=True)
            if len(df) > 0:
                print(f"Data for {symbol} already exists with {len(df)} rows")
                return True
        except:
            print(f"Error reading existing file for {symbol}, will re-download")
    return False

def fetch_daily_data(symbol, outputsize='full', retry_count=3):
    """Fetch daily time series data for a given symbol with retry logic"""
    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': symbol,
        'apikey': API_KEY,
        'outputsize': outputsize,
        'datatype': 'json'
    }
    
    for attempt in range(retry_count):
        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            data = response.json()
            
            if 'Error Message' in data:
                print(f"Error fetching {symbol}: {data['Error Message']}")
                return None
            elif 'Note' in data:
                print(f"API call limit reached: {data['Note']}")
                # This is a rate limit issue, return special marker
                return 'RATE_LIMIT'
            elif 'Time Series (Daily)' not in data:
                print(f"No data found for {symbol}")
                return None
            
            time_series = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns - adjusted data has more columns
            # We'll use adjusted close instead of regular close
            df.columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend_amount', 'split_coefficient']
            
            # Use adjusted close as our close price
            df['close'] = df['adjusted_close']
            
            # Keep only the OHLCV columns we need
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Filter data to exclude entries after 2023-01-01
            df = df[df.index < CUTOFF_DATE]
            
            if len(df) == 0:
                print(f"No data before {CUTOFF_DATE} for {symbol}")
                return None
            
            print(f"Fetched {len(df)} rows for {symbol} (up to {df.index.max().date()})")
            
            return df
            
        except requests.exceptions.Timeout:
            print(f"Timeout fetching {symbol} (attempt {attempt + 1}/{retry_count})")
            if attempt < retry_count - 1:
                time.sleep(5)
        except Exception as e:
            print(f"Exception fetching {symbol}: {str(e)} (attempt {attempt + 1}/{retry_count})")
            if attempt < retry_count - 1:
                time.sleep(5)
    
    return None

def save_data(df, symbol):
    """Save dataframe to CSV"""
    if df is None or (isinstance(df, str) and df == 'RATE_LIMIT'):
        return False
    
    if isinstance(df, pd.DataFrame):
        filename = os.path.join(OUTPUT_DIR, f"{symbol.replace('/', '_')}.csv")
        df.to_csv(filename)
        print(f"Saved {symbol} to {filename}")
        return True
    return False

def load_progress():
    """Load progress from file"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'last_processed_index': -1, 'successful': 0, 'failed': 0, 'skipped': 0}

def save_progress(progress):
    """Save progress to file"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load instruments
    try:
        instruments = load_instruments()
        print(f"Loaded {len(instruments)} instruments from {INSTRUMENTS_FILE}")
    except FileNotFoundError:
        print(f"Error: {INSTRUMENTS_FILE} not found!")
        return
    
    # Load previous progress
    progress = load_progress()
    start_index = progress['last_processed_index'] + 1
    
    if start_index > 0:
        print(f"Resuming from instrument #{start_index + 1} ({instruments[start_index]})")
        print(f"Previous progress - Successful: {progress['successful']}, Failed: {progress['failed']}, Skipped: {progress['skipped']}")
    
    print(f"API Key: {API_KEY[:8]}...")
    print(f"Data cutoff date: {CUTOFF_DATE}")
    print(f"Rate limit: 70 requests per minute")
    
    successful = progress['successful']
    failed = progress['failed']
    skipped = progress['skipped']
    api_calls = 0
    
    try:
        for i in range(start_index, len(instruments)):
            symbol = instruments[i]
            print(f"\n[{i+1}/{len(instruments)}] Processing {symbol}...")
            
            # Check if already downloaded
            if check_existing_download(symbol):
                skipped += 1
                progress.update({
                    'last_processed_index': i,
                    'successful': successful,
                    'failed': failed,
                    'skipped': skipped
                })
                save_progress(progress)
                continue
            
            # Fetch new data
            df = fetch_daily_data(symbol)
            api_calls += 1
            
            # Handle rate limit response
            if isinstance(df, str) and df == 'RATE_LIMIT':
                print("Rate limit hit, waiting 60 seconds before continuing...")
                time.sleep(60)
                # Retry the same symbol
                df = fetch_daily_data(symbol)
                api_calls += 1
            
            if save_data(df, symbol):
                successful += 1
            else:
                failed += 1
            
            # Save progress after each instrument
            progress.update({
                'last_processed_index': i,
                'successful': successful,
                'failed': failed,
                'skipped': skipped
            })
            save_progress(progress)
            
            # Rate limiting: 70 requests per minute
            # Add delay after every 65 requests to stay safely under limit
            if api_calls % 65 == 0 and api_calls > 0:
                print(f"Reached {api_calls} API calls, waiting 60 seconds to respect rate limit...")
                time.sleep(60)
            else:
                # Small delay between requests to be respectful
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n\nFetch interrupted by user. Progress has been saved.")
        print("Run the script again to resume from where you left off.")
        return
    except Exception as e:
        print(f"\n\nError occurred: {str(e)}")
        print("Progress has been saved. Run the script again to resume.")
        return
    
    # Clear progress file on successful completion
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
    
    print(f"\n{'='*50}")
    print(f"Data fetch complete!")
    print(f"Successful downloads: {successful}")
    print(f"Failed downloads: {failed}")
    print(f"Skipped (already exist): {skipped}")
    print(f"Total instruments: {len(instruments)}")
    print(f"Total API calls made: {api_calls}")
    print(f"Data saved in: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()