import numpy as np
import pandas as pd
import os
import sys
import argparse
from glob import glob
from datetime import datetime
import json
from scipy import stats

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


def generate_training_samples(df, segments, lookback_window):
    """
    Generate training samples from data
    For each valid position i:
    - Input: data from [i, i+lookback_window-1]
    - Output: regime at i+lookback_window and duration from that point to end of regime
    """
    X = []  # Input features
    y_class = []  # Regime class
    y_duration = []  # Duration metric
    sample_info = []  # Metadata for debugging
    
    # Map regime names to class indices
    regime_map = {'bull': 0, 'flat': 1, 'bear': 2}
    
    # We can generate samples from index 0 to len(df) - lookback_window - 1
    # Because we need lookback_window points for features and at least 1 point to predict
    for i in range(len(df) - lookback_window - 1):
        # Get lookback window [i, i+lookback_window-1]
        lookback_data = df.iloc[i:i+lookback_window]
        
        # The point we're predicting is at index i+lookback_window
        predict_idx = i + lookback_window
        
        # Find which segment contains the prediction point
        target_segment = find_segment_at_index(segments, predict_idx)
        if target_segment is None:
            continue
        
        # Normalize features using only lookback window data
        # Price features: normalized log returns
        features = []
        for col in ['open', 'high', 'low', 'close']:
            returns = lookback_data[col].pct_change()
            log_rets = np.log(1 + returns)
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
        volume_mean = log_volume.mean()
        volume_std = log_volume.std()
        if volume_std > 0:
            volume_normalized = (log_volume - volume_mean) / volume_std
        else:
            volume_normalized = log_volume - volume_mean
        features.append(volume_normalized.values)
        
        # Stack features: shape (lookback_window, 5)
        X_sample = np.stack(features, axis=1)
        
        # Skip if we have NaN values (from first return calculation)
        if np.any(np.isnan(X_sample)):
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


def process_instrument(csv_file, lookback_window):
    """Process a single instrument and generate training data"""
    symbol = os.path.basename(csv_file).replace('.csv', '')
    
    # Load data
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    df = df.sort_index()
    
    # Skip if not enough data
    if len(df) < lookback_window + 10:
        print(f"  Skipping {symbol}: insufficient data ({len(df)} rows)")
        return None, None, None, []
    
    # Detect regimes
    df_regime, segments = detect_regimes_adaptive(
        df, 
        min_window=5, 
        max_window=50, 
        r2_threshold=0.86, 
        volatility_factor=2.6, 
        min_trend_strength=1.4
    )
    
    # Generate training samples
    X, y_class, y_duration, info = generate_training_samples(df, segments, lookback_window)
    
    # Add symbol to info
    for item in info:
        item['symbol'] = symbol
    
    print(f"  Generated {len(X)} samples")
    if len(X) > 0:
        print(f"  Regimes: Bull={np.sum(y_class==0)}, Flat={np.sum(y_class==1)}, Bear={np.sum(y_class==2)}")
    
    return X, y_class, y_duration, info


def process_all_instruments(data_dir, lookback_window):
    """Process all instruments and generate training data"""
    all_X = []
    all_y_class = []
    all_y_duration = []
    all_info = []
    
    csv_files = sorted(glob(os.path.join(data_dir, '*.csv')))
    print(f"Found {len(csv_files)} CSV files")
    
    for i, csv_file in enumerate(csv_files):
        symbol = os.path.basename(csv_file).replace('.csv', '')
        print(f"\n[{i+1}/{len(csv_files)}] Processing {symbol}...")
        
        try:
            X, y_class, y_duration, info = process_instrument(csv_file, lookback_window)
            
            if X is not None and len(X) > 0:
                all_X.append(X)
                all_y_class.append(y_class)
                all_y_duration.append(y_duration)
                all_info.extend(info)
            
        except Exception as e:
            print(f"  Error processing {symbol}: {str(e)}")
            continue
    
    # Combine all data
    if all_X:
        X_combined = np.vstack(all_X)
        y_class_combined = np.hstack(all_y_class)
        y_duration_combined = np.hstack(all_y_duration)
        
        return X_combined, y_class_combined, y_duration_combined, all_info
    else:
        return None, None, None, []


def save_training_data(X, y_class, y_duration, info, output_dir, lookback_window):
    """Save training data to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numpy arrays
    np.save(os.path.join(output_dir, f'X_train_n{lookback_window}.npy'), X)
    np.save(os.path.join(output_dir, f'y_class_n{lookback_window}.npy'), y_class)
    np.save(os.path.join(output_dir, f'y_duration_n{lookback_window}.npy'), y_duration)
    
    # Save metadata
    with open(os.path.join(output_dir, f'sample_info_n{lookback_window}.json'), 'w') as f:
        json.dump(info, f, indent=2)
    
    # Save summary statistics
    summary = {
        'lookback_window': lookback_window,
        'n_samples': len(X),
        'n_features': X.shape[2] if len(X.shape) > 2 else 0,
        'feature_shape': list(X.shape),
        'class_distribution': {
            'bull': int(np.sum(y_class == 0)),
            'flat': int(np.sum(y_class == 1)),
            'bear': int(np.sum(y_class == 2))
        },
        'class_percentages': {
            'bull': float(np.sum(y_class == 0) / len(y_class) * 100),
            'flat': float(np.sum(y_class == 1) / len(y_class) * 100),
            'bear': float(np.sum(y_class == 2) / len(y_class) * 100)
        },
        'duration_stats': {
            'mean': float(np.mean(y_duration)),
            'std': float(np.std(y_duration)),
            'min': float(np.min(y_duration)),
            'max': float(np.max(y_duration)),
            'median': float(np.median(y_duration))
        }
    }
    
    with open(os.path.join(output_dir, f'summary_n{lookback_window}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Training data saved to {output_dir}")
    print(f"Files saved with prefix n{lookback_window}")
    print(f"\nSummary:")
    print(f"  Total samples: {summary['n_samples']}")
    print(f"  Feature shape: {summary['feature_shape']}")
    print(f"  Class distribution:")
    for regime in ['bull', 'flat', 'bear']:
        print(f"    {regime}: {summary['class_distribution'][regime]} ({summary['class_percentages'][regime]:.1f}%)")
    print(f"  Duration metric stats:")
    print(f"    Mean: {summary['duration_stats']['mean']:.2f}")
    print(f"    Std:  {summary['duration_stats']['std']:.2f}")
    print(f"    Range: [{summary['duration_stats']['min']:.2f}, {summary['duration_stats']['max']:.2f}]")


def main():
    parser = argparse.ArgumentParser(description='Generate training data for market regime prediction')
    parser.add_argument('n', type=int, help='Lookback window size')
    parser.add_argument('--data-dir', default='data/downloaded', help='Directory containing CSV files')
    parser.add_argument('--output-dir', default='data/training', help='Output directory for training data')
    
    args = parser.parse_args()
    
    print(f"Generating training data with lookback window n={args.n}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Process all instruments
    X, y_class, y_duration, info = process_all_instruments(args.data_dir, args.n)
    
    if X is not None:
        # Save training data
        save_training_data(X, y_class, y_duration, info, args.output_dir, args.n)
    else:
        print("No training data generated!")


if __name__ == "__main__":
    main()