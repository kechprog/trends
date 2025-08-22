import pandas as pd
import os
from datetime import time
import glob

def clean_and_merge_30min_data(output_file='voo_30min_clean.csv'):
    """
    Merge all 30-minute VOO data and keep only standard trading hours (9:30 AM - 4:00 PM ET)
    """
    print("Cleaning and merging 30-minute VOO data...")
    print("Keeping only standard trading hours: 9:30 AM - 4:00 PM ET")
    
    # Get all 30-min CSV files
    csv_files = sorted(glob.glob('voo_30min/VOO_*_30min.csv'))
    
    if not csv_files:
        print("No CSV files found in voo_30min directory")
        return
    
    print(f"Found {len(csv_files)} files to process")
    
    all_dfs = []
    total_records = 0
    kept_records = 0
    
    for i, filename in enumerate(csv_files):
        if (i + 1) % 12 == 0:  # Progress update every year
            print(f"Processing file {i+1}/{len(csv_files)}...")
        
        try:
            # Load the CSV
            df = pd.read_csv(filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df = df.sort_index()
            
            original_count = len(df)
            total_records += original_count
            
            # Filter to standard trading hours only
            # Standard hours: 9:30 AM - 4:00 PM ET
            # We keep times from 09:30 to 15:30 (inclusive) since 16:00 would be after close
            df = df.between_time(time(9, 30), time(15, 30))
            
            filtered_count = len(df)
            kept_records += filtered_count
            
            if filtered_count > 0:
                all_dfs.append(df)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    if not all_dfs:
        print("No valid data found after filtering")
        return
    
    # Combine all dataframes
    print("\nMerging all data...")
    df_combined = pd.concat(all_dfs)
    df_combined = df_combined.sort_index()
    
    # Remove duplicates if any
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
    
    print(f"\nData cleaning summary:")
    print(f"  Total records read: {total_records:,}")
    print(f"  Records kept (standard hours): {kept_records:,}")
    print(f"  Records removed (extended hours): {total_records - kept_records:,}")
    print(f"  Removal rate: {((total_records - kept_records) / total_records * 100):.1f}%")
    
    # Check for remaining anomalies
    df_combined['range_pct'] = ((df_combined['high'] - df_combined['low']) / df_combined['high']) * 100
    anomalies = df_combined[df_combined['range_pct'] > 5.0]
    
    if len(anomalies) > 0:
        print(f"\nWARNING: Still found {len(anomalies)} candles with >5% range after filtering")
        print("These may need manual inspection:")
        print(anomalies[['high', 'low', 'range_pct']].head(10))
    else:
        print("\n✓ No extreme anomalies (>5% range) found in cleaned data")
    
    # Save to CSV
    print(f"\nSaving cleaned data to {output_file}...")
    df_combined.to_csv(output_file)
    
    # Final statistics
    print(f"\nFinal dataset:")
    print(f"  Date range: {df_combined.index[0]} to {df_combined.index[-1]}")
    print(f"  Total records: {len(df_combined):,}")
    print(f"  File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    
    # Daily statistics
    df_combined['date'] = df_combined.index.date
    daily_counts = df_combined.groupby('date').size()
    print(f"  Trading days: {len(daily_counts):,}")
    print(f"  Avg candles per day: {daily_counts.mean():.1f}")
    print(f"  Expected candles per day: 13 (9:30 AM to 4:00 PM in 30-min intervals)")
    
    return df_combined

if __name__ == "__main__":
    # Run the cleaning process
    clean_df = clean_and_merge_30min_data('voo_30min_clean.csv')
    
    if clean_df is not None:
        print("\n" + "="*60)
        print("Data cleaning completed successfully!")
        print("You can now use 'voo_30min_clean.csv' for your analysis")
        print("="*60)