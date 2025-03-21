import os
import sys

# add the scripts/ directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_data, drop_high_missing, interpolate_missing, forward_fill_missing, save_data

# define the file paths
input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/raw/energy/energy_close.csv"))
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/clean/energy/clean_energy_close.csv"))

# load dataset
df = load_data(input_path)

# ensure dataset is sorted by date (ascending)
df.sort_values(by="Date", inplace=True)

# drop columns with excessive missing values
df = drop_high_missing(df)

# apply spline interpolation for missing values (best for stock price trends)
df = interpolate_missing(df)

# apply forward fill for any remaining missing values
df = forward_fill_missing(df)

# save cleaned dataset
save_data(df, output_path)