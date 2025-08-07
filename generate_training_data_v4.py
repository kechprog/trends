import numpy as np
import pandas as pd
import os
import sys
import argparse
from glob import glob
from datetime import datetime
import json
from scipy import stats
import multiprocessing as mp
from functools import partial

# Define S&P 500 tracking and highly correlated instruments
SP500_CORRELATED = {
    'SPY', 'VOO', 'IVV', 'SPX',  # Direct S&P 500 ETFs
    'QQQ', 'DIA', 'IWM',  # Major index ETFs
    'VTI', 'VTV', 'VUG',  # Broad market ETFs
    'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB',  # Sector SPDRs
    'VIG', 'VYM', 'DVY',  # Dividend ETFs
    'IWF', 'IWD', 'IWO', 'IWN',  # Russell ETFs
    'MDY', 'IJH', 'IJR',  # Mid/Small cap S&P ETFs
    'RSP', 'SPLG',  # Equal weight S&P
    'SSO', 'UPRO', 'SH', 'SDS', 'SPXU',  # Leveraged/Inverse S&P
    'VXX', 'UVXY', 'SVXY',  # VIX ETFs (inverse correlation)
    'AGG', 'BND', 'TLT', 'IEF', 'LQD', 'HYG',  # Bond ETFs (often correlated during risk-off)
    'GLD', 'SLV', 'GDX',  # Precious metals (sometimes correlated)
    'EEM', 'EFA', 'VWO', 'IEMG',  # International ETFs
}

def detect_regimes_adaptive(df, min_window=5, max_window=50, r2_threshold=0.86, 
                          volatility_factor=2.6, min_trend_strength=1.4):
    """
    Detect market regimes using adaptive windows with volatility-adjusted thresholds.
    """
    df = df.copy()
    df['log_price'] = np.log(df['close'])
    df['returns'] = df['log_price'].diff()
    
    segments = []
    i = 0
    
    while i < len(df):
        # Handle end of data
        if i >= len(df) - min_window:
            remaining = len(df) - i
            if remaining > 0:
                # Simple classification for remaining data
                total_return = (df['close'].iloc[-1] / df['close'].iloc[i] - 1) * 100
                
                if abs(total_return) < min_trend_strength:
                    regime = 'flat'
                elif total_return > 0:
                    regime = 'bull'
                else:
                    regime = 'bear'
                
                segments.append({
                    'start': i,
                    'end': len(df) - 1,
                    'start_date': df.index[i],
                    'end_date': df.index[-1],
                    'regime': regime,
                    'total_return': total_return,
                    'duration': remaining
                })
            break
        
        # Find the best window by looking for trend consistency
        best_window = min_window
        
        for window in range(min_window, min(max_window + 1, len(df) - i + 1)):
            # Get window data
            window_prices = df['close'].iloc[i:i+window].values
            window_log_prices = df['log_price'].iloc[i:i+window].values
            
            # Fit regression
            x = np.arange(window)
            slope, intercept, r_value, _, _ = stats.linregress(x, window_log_prices)
            r2 = r_value**2
            
            # Calculate trend line values
            trend_line = slope * x + intercept
            
            # Calculate deviations from trend
            deviations = window_log_prices - trend_line
            volatility = np.std(deviations)
            
            # Check if price has broken the trend channel significantly
            if window > min_window:
                # Current deviation from trend
                current_deviation = abs(deviations[-1])
                
                # Trend break conditions
                trend_broken = (
                    current_deviation > volatility * volatility_factor or
                    r2 < r2_threshold * 0.8
                )
                
                if trend_broken:
                    break
            
            best_window = window
        
        # Calculate regime for the segment
        segment_return = (df['close'].iloc[i + best_window - 1] / df['close'].iloc[i] - 1) * 100
        
        if abs(segment_return) < min_trend_strength:
            regime = 'flat'
        elif segment_return > 0:
            regime = 'bull'
        else:
            regime = 'bear'
        
        # Store segment
        segments.append({
            'start': i,
            'end': i + best_window - 1,
            'start_date': df.index[i],
            'end_date': df.index[i + best_window - 1],
            'regime': regime,
            'total_return': segment_return,
            'duration': best_window
        })
        
        # Move to next segment
        i += best_window
    
    # Apply segments to dataframe
    df['regime'] = 'flat'
    df['segment_id'] = -1
    
    for idx, seg in enumerate(segments):
        df.iloc[seg['start']:seg['end']+1, df.columns.get_loc('regime')] = seg['regime']
        df.iloc[seg['start']:seg['end']+1, df.columns.get_loc('segment_id')] = idx
    
    return df, segments


def find_segment_at_index(segments, idx):
    """Find which segment contains the given index"""
    for seg in segments:
        if seg['start'] <= idx <= seg['end']:
            return seg
    return None


def calculate_rsi(prices, period=14):
    """Calculate RSI for a price series"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(prices, period=20, num_std=2):
    """Calculate Bollinger Bands width and distance"""
    middle_band = prices.rolling(window=period).mean()
    std_dev = prices.rolling(window=period).std()
    upper_band = middle_band + (num_std * std_dev)
    lower_band = middle_band - (num_std * std_dev)
    
    # BB width (normalized volatility)
    bb_width = (upper_band - lower_band) / middle_band
    
    # BB distance (where price is relative to bands)
    # This can go beyond -0.5 to 0.5 when price breaks the bands
    bb_distance = (prices - middle_band) / (upper_band - lower_band)
    
    return bb_width, bb_distance


def generate_training_samples(df, segments, lookback_window, pre_lookback=20):
    """
    Generate training samples from data
    For each valid position i:
    - Input: data from [i, i+lookback_window-1]
    - Output: regime at i+lookback_window and duration from that point to end of regime
    
    pre_lookback: additional historical data needed for indicator calculation
    """
    X = []  # Input features
    y_class = []  # Regime class
    y_duration = []  # Duration metric
    sample_info = []  # Metadata for debugging
    
    # Map regime names to class indices
    regime_map = {'bull': 0, 'flat': 1, 'bear': 2}
    
    skipped_nan = 0
    skipped_no_segment = 0
    
    # Start from pre_lookback to ensure we have enough data for indicators
    # We need pre_lookback + lookback_window + 1 total points
    for i in range(pre_lookback, len(df) - lookback_window - 1):
        # Get extended data including pre_lookback period for indicators
        extended_data = df.iloc[i-pre_lookback:i+lookback_window]
        
        # Get lookback window [i, i+lookback_window-1]
        lookback_data = df.iloc[i:i+lookback_window]
        
        # The point we're predicting is at index i+lookback_window
        predict_idx = i + lookback_window
        
        # Find which segment contains the prediction point
        target_segment = find_segment_at_index(segments, predict_idx)
        if target_segment is None:
            skipped_no_segment += 1
            continue
        
        # Calculate technical indicators on extended data
        rsi_full = calculate_rsi(extended_data['close'], period=14)
        bb_width_full, bb_distance_full = calculate_bollinger_bands(extended_data['close'], period=20)
        
        # Extract only the lookback window portion (skip pre_lookback values)
        rsi_window = rsi_full.iloc[pre_lookback:].values
        bb_distance_window = bb_distance_full.iloc[pre_lookback:].values
        
        # Normalize features using only lookback window data
        # Price features: normalized log returns
        features = []
        for col in ['open', 'high', 'low', 'close']:
            returns = lookback_data[col].pct_change()
            log_rets = np.log(1 + returns)
            # Drop the first NaN value from pct_change
            log_rets = log_rets.iloc[1:]
            # Use lookback window for normalization
            mean = log_rets.mean()
            std = log_rets.std()
            if std > 0:
                normalized = (log_rets - mean) / std
            else:
                normalized = log_rets - mean
            features.append(normalized.values)
        
        # Volume feature: z-score normalized log volume
        log_volume = np.log(lookback_data['volume'])
        # Drop the first value to match the length of price returns
        log_volume = log_volume.iloc[1:]
        volume_mean = log_volume.mean()
        volume_std = log_volume.std()
        if volume_std > 0:
            volume_normalized = (log_volume - volume_mean) / volume_std
        else:
            volume_normalized = log_volume - volume_mean
        features.append(volume_normalized.values)
        
        # Process RSI (convert from [0,100] to [-1,1])
        rsi_normalized = (rsi_window[1:] - 50) / 50  # Skip first NaN to match other features
        features.append(rsi_normalized)
        
        # Process BB distance (already normalized, just need to match length)
        bb_distance_normalized = bb_distance_window[1:]  # Skip first to match other features
        # Additional normalization based on lookback window stats
        bb_mean = np.nanmean(bb_distance_normalized)
        bb_std = np.nanstd(bb_distance_normalized)
        if bb_std > 0:
            bb_distance_normalized = (bb_distance_normalized - bb_mean) / bb_std
        else:
            bb_distance_normalized = bb_distance_normalized - bb_mean
        features.append(bb_distance_normalized)
        
        # Stack features: shape (lookback_window-1, 7)
        X_sample = np.stack(features, axis=1)
        
        # Skip if we have NaN values (from first return calculation)
        if np.any(np.isnan(X_sample)):
            skipped_nan += 1
            continue
        
        # Calculate duration metric
        # Duration is from predict_idx to end of segment
        segment_start_idx = max(predict_idx, target_segment['start'])
        segment_end_idx = target_segment['end']
        
        # Get volumes from predict_idx to end of segment
        duration_volumes = df['volume'].iloc[segment_start_idx:segment_end_idx+1].values
        
        # Calculate z-score using lookback statistics
        log_duration_volumes = np.log(duration_volumes)
        duration_zscores = (log_duration_volumes - volume_mean) / volume_std if volume_std > 0 else log_duration_volumes - volume_mean
        duration_metric = duration_zscores.sum()
        
        # Add sample
        X.append(X_sample)
        y_class.append(regime_map[target_segment['regime']])
        y_duration.append(duration_metric)
        
        sample_info.append({
            'lookback_start': df.index[i].strftime('%Y-%m-%d'),
            'lookback_end': df.index[i+lookback_window-1].strftime('%Y-%m-%d'),
            'predict_date': df.index[predict_idx].strftime('%Y-%m-%d'),
            'segment_end_date': df.index[segment_end_idx].strftime('%Y-%m-%d'),
            'regime': target_segment['regime'],
            'duration_days': segment_end_idx - segment_start_idx + 1,
            'duration_metric': float(duration_metric),
            'total_return': target_segment['total_return']
        })
    
    return np.array(X), np.array(y_class), np.array(y_duration), sample_info


def process_instrument(csv_file, lookback_window, output_dir, pre_lookback=20):
    """Process a single instrument and save to individual file"""
    symbol = os.path.basename(csv_file).replace('.csv', '')
    
    try:
        # Load data
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        df = df.sort_index()
        
        # Skip if not enough data (need pre_lookback + lookback_window + buffer)
        min_required = pre_lookback + lookback_window + 10
        if len(df) < min_required:
            return {
                'symbol': symbol,
                'status': 'skipped',
                'reason': f'insufficient data ({len(df)} rows, need {min_required})',
                'samples': 0
            }
        
        # Detect regimes
        df_regime, segments = detect_regimes_adaptive(
            df, 
            min_window=5, 
            max_window=50, 
            r2_threshold=0.86, 
            volatility_factor=2.6, 
            min_trend_strength=1.4
        )
        
        # Generate training samples with pre_lookback
        X, y_class, y_duration, info = generate_training_samples(df, segments, lookback_window, pre_lookback)
        
        # Add symbol to info
        for item in info:
            item['symbol'] = symbol
        
        if len(X) == 0:
            return {
                'symbol': symbol,
                'status': 'no_samples',
                'samples': 0
            }
        
        # Determine if this is S&P 500 correlated
        is_sp500 = symbol.upper() in SP500_CORRELATED
        subset = 'val' if is_sp500 else 'train'
        
        # Create subdirectory for the subset
        subset_dir = os.path.join(output_dir, subset, f'n{lookback_window}')
        os.makedirs(subset_dir, exist_ok=True)
        
        # Save to individual file
        output_file = os.path.join(subset_dir, f'{symbol}.npz')
        np.savez_compressed(
            output_file,
            X=X,
            y_class=y_class,
            y_duration=y_duration
        )
        
        # Save info
        info_file = os.path.join(subset_dir, f'{symbol}_info.json')
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        return {
            'symbol': symbol,
            'status': 'success',
            'file': output_file,
            'samples': len(X),
            'is_sp500': is_sp500,
            'subset': subset,
            'regimes': {
                'bull': int(np.sum(y_class==0)),
                'flat': int(np.sum(y_class==1)),
                'bear': int(np.sum(y_class==2))
            }
        }
        
    except Exception as e:
        return {
            'symbol': symbol,
            'status': 'error',
            'error': str(e),
            'samples': 0
        }


def process_all_instruments_parallel(data_dir, lookback_window, output_dir, n_jobs=None):
    """Process all instruments in parallel and save individually"""
    if n_jobs is None:
        n_jobs = mp.cpu_count() - 1
    
    csv_files = sorted(glob(os.path.join(data_dir, '*.csv')))
    print(f"Found {len(csv_files)} CSV files")
    print(f"Using {n_jobs} parallel workers")
    
    # Process files in parallel
    process_func = partial(process_instrument, 
                          lookback_window=lookback_window, 
                          output_dir=output_dir)
    
    with mp.Pool(n_jobs) as pool:
        results = pool.map(process_func, csv_files)
    
    # Summarize results
    train_symbols = []
    val_symbols = []
    total_train_samples = 0
    total_val_samples = 0
    
    print("\nProcessing results:")
    for result in results:
        if result['status'] == 'success':
            if result['subset'] == 'val':
                val_symbols.append(result['symbol'])
                total_val_samples += result['samples']
            else:
                train_symbols.append(result['symbol'])
                total_train_samples += result['samples']
            print(f"  {result['symbol']}: {result['samples']} samples ({result['subset']})")
        elif result['status'] == 'skipped':
            print(f"  {result['symbol']}: Skipped - {result['reason']}")
        elif result['status'] == 'error':
            print(f"  {result['symbol']}: Error - {result['error']}")
    
    # Save manifest files
    train_manifest = {
        'lookback_window': lookback_window,
        'symbols': train_symbols,
        'n_instruments': len(train_symbols),
        'total_samples': total_train_samples
    }
    
    val_manifest = {
        'lookback_window': lookback_window,
        'symbols': val_symbols,
        'n_instruments': len(val_symbols),
        'total_samples': total_val_samples
    }
    
    # Save manifests
    train_dir = os.path.join(output_dir, 'train', f'n{lookback_window}')
    val_dir = os.path.join(output_dir, 'val', f'n{lookback_window}')
    
    if train_symbols:
        with open(os.path.join(train_dir, 'manifest.json'), 'w') as f:
            json.dump(train_manifest, f, indent=2)
    
    if val_symbols:
        with open(os.path.join(val_dir, 'manifest.json'), 'w') as f:
            json.dump(val_manifest, f, indent=2)
    
    print(f"\nInstrument breakdown:")
    print(f"  Training: {len(train_symbols)} instruments, {total_train_samples} samples")
    print(f"  Validation: {len(val_symbols)} instruments, {total_val_samples} samples")
    
    return total_train_samples, total_val_samples


def main():
    parser = argparse.ArgumentParser(description='Generate training data for market regime prediction (per-instrument files)')
    parser.add_argument('n', type=int, help='Lookback window size')
    parser.add_argument('--data-dir', default='data/downloaded', help='Directory containing CSV files')
    parser.add_argument('--output-dir', default='data/training', help='Output directory for training data')
    parser.add_argument('--n-jobs', type=int, default=None, help='Number of parallel jobs (default: CPU count - 1)')
    
    args = parser.parse_args()
    
    print(f"Generating training data with lookback window n={args.n}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all instruments in parallel
    train_samples, val_samples = process_all_instruments_parallel(
        args.data_dir, 
        args.n, 
        args.output_dir,
        args.n_jobs
    )
    
    if train_samples == 0 and val_samples == 0:
        print("No training data generated!")
    else:
        print(f"\nData generation complete!")
        print(f"Files saved in:")
        print(f"  Training: {args.output_dir}/train/n{args.n}/")
        print(f"  Validation: {args.output_dir}/val/n{args.n}/")


if __name__ == "__main__":
    main()