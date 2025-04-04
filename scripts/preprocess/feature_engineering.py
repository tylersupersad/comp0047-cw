import argparse
import numpy as np
import pandas as pd
from utils import compute_returns, compute_volatility

def generate_features(df, window=20, dropna=False):
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    # return and volatility
    df['return_t'] = df.groupby('ticker')['close'].transform(compute_returns)
    df['volatility_t'] = df.groupby('ticker')['return_t'].transform(lambda x: compute_volatility(x, window))

    # prediction targets
    df['return_t+1'] = df.groupby('ticker')['return_t'].shift(-1)
    df['volatility_t+1'] = df.groupby('ticker')['volatility_t'].shift(-1)

    # lagged returns and volatility
    df['return_t-1'] = df.groupby('ticker')['return_t'].shift(1)
    df['return_t-2'] = df.groupby('ticker')['return_t'].shift(2)
    df['volatility_t-1'] = df.groupby('ticker')['volatility_t'].shift(1)
    df['volatility_t-2'] = df.groupby('ticker')['volatility_t'].shift(2)

    # lagged volumes
    df['volume_t-1'] = df.groupby('ticker')['volume'].shift(1)
    df['volume_t-2'] = df.groupby('ticker')['volume'].shift(2)

    # rolling means
    df['volume_rolling_mean_5'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(5).mean())
    df['market_cap_rolling_mean_5'] = df.groupby('ticker')['market_cap'].transform(lambda x: x.rolling(5).mean())

    # log-transformed volume (required by model_train.py)
    df['volume_t'] = np.log1p(df['volume'])

    # optional: drop rows with any missing values in engineered features
    if dropna:
        required_cols = [
            'return_t', 'volatility_t', 'return_t+1', 'volatility_t+1',
            'return_t-1', 'return_t-2',
            'volatility_t-1', 'volatility_t-2',
            'volume_t', 'volume_t-1', 'volume_t-2',
            'volume_rolling_mean_5', 'market_cap_rolling_mean_5'
        ]
        df = df.dropna(subset=required_cols)

    return df

def summarize(df, sector):
    print(f"\n[✓] Summary for {sector}")
    print(f"• rows: {len(df):,}")
    print(f"• tickers: {df['ticker'].nunique()}")
    print(f"• date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print("• sample columns:", ', '.join(df.columns[:10]) + "...")
    print(df[['return_t', 'volatility_t', 'return_t+1', 'volatility_t+1']].describe().round(4))

def main():
    parser = argparse.ArgumentParser(description="Feature engineering for return and volatility prediction")
    parser.add_argument("--input", required=True, help="path to cleaned input CSV")
    parser.add_argument("--output", required=True, help="path to output engineered CSV")
    parser.add_argument("--window", type=int, default=20, help="rolling window size for volatility")
    parser.add_argument("--dropna", action="store_true", help="drop rows with NaNs in engineered features")
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=['date'])
    sector = args.input.split("/")[-1].replace("clean_", "").replace(".csv", "")

    print(f"[→] Generating features for {sector} (window = {args.window})...")
    df_feat = generate_features(df, window=args.window, dropna=args.dropna)

    summarize(df_feat, sector)
    df_feat.to_csv(args.output, index=False)
    print(f"\n[✓] Saved engineered dataset to {args.output}")

if __name__ == "__main__":
    # python3 feature_engineering.py --input ../../data/clean/energy/clean_energy.csv --output ../../data/features/features_energy.csv --window 20 --dropna
    # python3 feature_engineering.py --input ../../data/clean/technology/clean_technology.csv --output ../../data/features/features_technology.csv --window 20 --dropna
    main()