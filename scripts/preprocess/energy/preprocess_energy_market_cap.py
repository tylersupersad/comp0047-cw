import os
import sys
import pandas as pd

# add the scripts/ directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_data, drop_high_missing, interpolate_missing, forward_fill_missing, save_data

# Define the file paths
input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/raw/energy/energy_market_cap.csv"))
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/clean/energy/clean_energy_market_cap.csv"))

# load the dataset
df = load_data(input_path)

# remove unnecessary index column if present
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)
    
# ensure date column is in the correct format
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# ensure dataset is sorted by ticker and date (descending)
df.sort_values(by=['ticker', 'date'], ascending=[True, False], inplace=True)

# drop columns with excessive missing values
# df = drop_high_missing(df)

# reset index to ensure ticker is a regular column
df = df.reset_index(drop=True)

# identify numeric columns (excluding ticker and date)
numeric_cols = df.select_dtypes(include=['number']).columns.difference(['ticker', 'date'])

# apply spline interpolation to fill missing values per ticker
df[numeric_cols] = df.groupby('ticker', group_keys=False)[numeric_cols].apply(lambda group: interpolate_missing(group))

# apply forward fill for any remaining missing values per ticker
df[numeric_cols] = df.groupby('ticker', group_keys=False)[numeric_cols].apply(lambda group: forward_fill_missing(group))

# save cleaned dataset
save_data(df, output_path)