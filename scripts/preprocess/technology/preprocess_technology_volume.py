import os
import sys

# add the scripts/ directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_data, drop_high_missing, forward_fill_missing, mean_imputation_missing, save_data

# define the file paths
input_path = "../../../data/raw/technology/technology_volume.csv"
output_path = "../../../data/clean/technology/clean_technology_volume.csv"

# load the dataset
df = load_data(input_path)

# ensure dataset is sorted by date (ascending)
df.sort_values(by="Date", inplace=True)

# drop columns with excessive missing values
df = drop_high_missing(df)

# forward fill small gaps
df = forward_fill_missing(df)

# fill remaining missing values with column mean
df = mean_imputation_missing(df)

# save cleaned dataset
save_data(df, output_path)