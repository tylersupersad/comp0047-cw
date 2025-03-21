import os
import sys

# add the scripts/ directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_data, drop_high_missing, forward_fill_missing, mean_imputation_missing, save_data

# define the file paths
input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/raw/energy/energy_volume.csv"))
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/clean/energy/clean_energy_volume.csv"))

# load the dataset
df = load_data(input_path)

# ensure dataset is sorted by date (ascending)
df.sort_values(by="Date", inplace=True)

# drop columns with excessive missing values
df = drop_high_missing(df)

# apply forward fill for missing values (best for trading volume)
df = forward_fill_missing(df)

# apply mean imputation for remaining missing values
df = mean_imputation_missing(df)

# save cleaned dataset
save_data(df, output_path)