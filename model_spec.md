# Market Regime Prediction Model Specification

## Configuration
- Lookback window: Fixed (e.g., 60 days)
- Prediction horizon: 6-21 trading days

### Feature Set 1: Price Features (OHLC)
Normalized log returns for Open, High, Low, Close:
```python
returns = df[price_col].pct_change()
log_rets = np.log(1 + returns)
feature = (log_rets - log_rets.rolling(lookback).mean()) / log_rets.rolling(lookback).std()
```

### Feature Set 2: Volume Features
Z-score normalized log volume:
```python
volume_zscore = (np.log(volume) - np.log(volume).mean()) / np.log(volume).std()
```

## Feature Matrix Shape
- Input: `(n_samples, lookback_window, n_features)`
  - n_samples: Number of training examples
  - lookback_window: Fixed number of historical days per sample
  - n_features: 5 (open, high, low, close, volume)

## Model Output
The model predicts two values:
1. **Regime class**: One of 3 classes (bull=0, flat=1, bear=2)
2. **Trend duration**: Measured as sum of z-score normalized volumes

### Duration Calculation
For a trend spanning multiple days:
```python
# Using the same volume normalization as input features
log_volumes = np.log(segment_volumes)
volume_mean = np.log(all_volumes).mean()
volume_std = np.log(all_volumes).std()
volume_zscores = (log_volumes - volume_mean) / volume_std
duration_metric = volume_zscores.sum()
```

This approach weights duration by trading activity rather than pure time.