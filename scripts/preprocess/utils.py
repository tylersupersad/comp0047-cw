import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

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