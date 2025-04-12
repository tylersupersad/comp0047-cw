import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

<<<<<<< Updated upstream
def load_data(file_path):
    # load a dataset and convert the first column (date) to datetime if applicable
    df = pd.read_csv(file_path)
    
    # auto-detect date column and convert
    if df.columns[0].lower() in ["date", "timestamp"]: 
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], errors='coerce')
    
    return df

def drop_high_missing(df, threshold=50):
    # drop columns with more than threshold percent missing values
    return df.dropna(axis=1, thresh=(threshold / 100) * len(df))
=======
def read_and_parse_csv(path):
    # load csv and convert any 'date' column to datetime
    print(f"Reading file: {path}")
    df = pd.read_csv(path)
    print(f"Loaded dataframe with shape: {df.shape} and columns: {list(df.columns)}")
    
    for col in df.columns:
        if col.lower() == 'date':
            print(f"Converting {col} to datetime and standardizing column name")
            df.rename(columns={col: 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
    
    return df

def reshape_and_merge_raw_data(close_df, volume_df, cap_df):
    print("\n===== Starting data reshaping and merging =====")
    print(f"Input shapes - Close: {close_df.shape}, Volume: {volume_df.shape}, Market Cap: {cap_df.shape}")
    
    # convert wide to long format for merging
    close_long = close_df.reset_index().melt(id_vars='Date', var_name='ticker', value_name='close')
    print(f"Close data after melting: {close_long.shape}, unique tickers: {close_long['ticker'].nunique()}")
    
    volume_long = volume_df.reset_index().melt(id_vars='Date', var_name='ticker', value_name='volume')
    print(f"Volume data after melting: {volume_long.shape}, unique tickers: {volume_long['ticker'].nunique()}")
    
    cap_df['Date'] = pd.to_datetime(cap_df['Date'])
    print(f"Market cap data shape: {cap_df.shape}, unique tickers: {cap_df['ticker'].nunique()}")

    # outer join on date and ticker to preserve all info
    merged = (
        close_long
        .merge(volume_long, on=['Date', 'ticker'], how='outer')
        .merge(cap_df, on=['Date', 'ticker'], how='outer')
        .rename(columns={'Date': 'date', 'marketcap': 'market_cap'})
        .sort_values(['ticker', 'date'])
        .reset_index(drop=True)
    )
    
    print(f"Merged data shape: {merged.shape}, unique tickers: {merged['ticker'].nunique()}")
    print(f"Missing values after merge - Close: {merged['close'].isna().sum()}, Volume: {merged['volume'].isna().sum()}, Market Cap: {merged['market_cap'].isna().sum()}")
    
    final_df = merged[['date', 'ticker', 'close', 'volume', 'market_cap']]
    print(f"Final reshaped data: {final_df.shape}\n")
    
    return final_df

def clean_and_transform_data(
    df,
    max_missing_close=0.3,
    cap_rolling_window=5,
    normalize=False
):
    print("\n===== Starting data cleaning and transformation =====")
    print(f"Initial dataframe shape: {df.shape}, unique tickers: {df['ticker'].nunique()}")
    print(f"Initial missing values - Close: {df['close'].isna().sum()}, Volume: {df['volume'].isna().sum()}, Market Cap: {df['market_cap'].isna().sum()}")
    
    df = df.copy()

    # remove rows with all key values missing
    before_shape = df.shape
    df = df.dropna(subset=['close', 'volume', 'market_cap'], how='all')
    print(f"After removing rows with all values missing: {df.shape}, removed {before_shape[0] - df.shape[0]} rows")

    # sort for grouped operations
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    # interpolate close and volume values per ticker
    missing_close_before = df['close'].isna().sum()
    df['close'] = df.groupby('ticker')['close'].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both'))
    missing_close_after = df['close'].isna().sum()
    print(f"Close price interpolation: filled {missing_close_before - missing_close_after} of {missing_close_before} missing values")
    
    missing_volume_before = df['volume'].isna().sum()
    df['volume'] = df.groupby('ticker')['volume'].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both'))
    missing_volume_after = df['volume'].isna().sum()
    print(f"Volume interpolation: filled {missing_volume_before - missing_volume_after} of {missing_volume_before} missing values")

    # impute market cap using rolling median per ticker
    missing_cap_before = df['market_cap'].isna().sum()
    df['market_cap'] = df.groupby('ticker')['market_cap'].transform(
        lambda x: x.fillna(x.rolling(cap_rolling_window, min_periods=1).median()))
    missing_cap_after = df['market_cap'].isna().sum()
    print(f"Market cap imputation: filled {missing_cap_before - missing_cap_after} of {missing_cap_before} missing values using rolling window of {cap_rolling_window}")

    # drop tickers with too much missing close data
    missing_ratio = df.groupby('ticker')['close'].apply(lambda x: x.isna().mean())
    tickers_before = df['ticker'].nunique()
    valid_tickers = missing_ratio[missing_ratio <= max_missing_close].index
    dropped_tickers = set(df['ticker'].unique()) - set(valid_tickers)
    df = df[df['ticker'].isin(valid_tickers)]
    print(f"Dropped {tickers_before - df['ticker'].nunique()} tickers with >{max_missing_close*100}% missing close values: {list(dropped_tickers)}")

    # remove any remaining rows with missing critical values
    rows_before = df.shape[0]
    df = df.dropna(subset=['close', 'volume', 'market_cap'])
    print(f"Removed {rows_before - df.shape[0]} rows with remaining missing values")

    # normalize close, volume, and market cap per ticker (z-score)
    if normalize:
        print("Normalizing data (z-score) per ticker:")
        for col in ['close', 'volume', 'market_cap']:
            col_stats_before = df.groupby('ticker')[col].agg(['mean', 'std']).mean()
            df[col] = df.groupby('ticker')[col].transform(lambda x: (x - x.mean()) / x.std())
            col_stats_after = df.groupby('ticker')[col].agg(['mean', 'std']).mean()
            print(f"  - {col}: Mean before={col_stats_before['mean']:.4f}, after={col_stats_after['mean']:.4f}; Std before={col_stats_before['std']:.4f}, after={col_stats_after['std']:.4f}")

    # ensure consistency and no duplicate entries
    rows_before = df.shape[0]
    df = df.sort_values(['ticker', 'date']).drop_duplicates(subset=['ticker', 'date']).reset_index(drop=True)
    print(f"Removed {rows_before - df.shape[0]} duplicate entries")
    
    print(f"Final cleaned data: {df.shape}, unique tickers: {df['ticker'].nunique()}")
    print(f"Final date range: {df['date'].min()} to {df['date'].max()}")
    print("===== Cleaning and transformation complete =====\n")
>>>>>>> Stashed changes

def interpolate_missing(df):
    # apply spline interpolation to fill missing values smoothly (best for time-series data like closing prices and market cap)
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='spline', order=2)
    
    return df

def forward_fill_missing(df):
    # apply forward fill to handle structured missing values (best for volume data where interpolation is not ideal)
    df.ffill()
    return df

def mean_imputation_missing(df):
    # fill missing column values with column mean
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    return df 

def standardize_data(df):
    # standardize numerical data using z-score normalization
    scaler = StandardScaler()
    df.iloc[:, :] = scaler.fit_transform(df.iloc[:, :])
    return df

def save_data(df, file_path):
    # ensure the directory exists 
    os.makedirs(os.path.dirname(file_path), exist_ok=True) 
    # save the cleaned dataset to a csv file
    df.to_csv(file_path, index=False)
    
    print(f"Saved cleaned file: {file_path}")