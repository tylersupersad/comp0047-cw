import pandas as pd

def read_and_parse_csv(path):
    # load csv and convert any 'date' column to datetime
    df = pd.read_csv(path)
    for col in df.columns:
        if col.lower() == 'date':
            df.rename(columns={col: 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
    return df

def reshape_and_merge_raw_data(close_df, volume_df, cap_df):
    # convert wide to long format for merging
    close_long = close_df.reset_index().melt(id_vars='Date', var_name='ticker', value_name='close')
    volume_long = volume_df.reset_index().melt(id_vars='Date', var_name='ticker', value_name='volume')
    cap_df['Date'] = pd.to_datetime(cap_df['Date'])

    # outer join on date and ticker to preserve all info
    merged = (
        close_long
        .merge(volume_long, on=['Date', 'ticker'], how='outer')
        .merge(cap_df, on=['Date', 'ticker'], how='outer')
        .rename(columns={'Date': 'date', 'marketcap': 'market_cap'})
        .sort_values(['ticker', 'date'])
        .reset_index(drop=True)
    )

    return merged[['date', 'ticker', 'close', 'volume', 'market_cap']]

def clean_and_transform_data(
    df,
    max_missing_close=0.3,
    cap_rolling_window=5,
    normalize=False
):
    df = df.copy()

    # remove rows with all key values missing
    df = df.dropna(subset=['close', 'volume', 'market_cap'], how='all')

    # sort for grouped operations
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    # interpolate close and volume values per ticker
    df['close'] = df.groupby('ticker')['close'].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both'))
    df['volume'] = df.groupby('ticker')['volume'].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both'))

    # impute market cap using rolling median per ticker
    df['market_cap'] = df.groupby('ticker')['market_cap'].transform(
        lambda x: x.fillna(x.rolling(cap_rolling_window, min_periods=1).median()))

    # drop tickers with too much missing close data
    missing_ratio = df.groupby('ticker')['close'].apply(lambda x: x.isna().mean())
    valid_tickers = missing_ratio[missing_ratio <= max_missing_close].index
    df = df[df['ticker'].isin(valid_tickers)]

    # remove any remaining rows with missing critical values
    df = df.dropna(subset=['close', 'volume', 'market_cap'])

    # normalize close, volume, and market cap per ticker (z-score)
    if normalize:
        for col in ['close', 'volume', 'market_cap']:
            df[col] = df.groupby('ticker')[col].transform(lambda x: (x - x.mean()) / x.std())

    # ensure consistency and no duplicate entries
    df = df.sort_values(['ticker', 'date']).drop_duplicates(subset=['ticker', 'date']).reset_index(drop=True)

    return df