import os
import sys

# add the scripts/ directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils import load_data, drop_high_missing, mean_imputation_missing, standardize_data, save_data

# define the file paths
input_path = "../../../data/raw/energy/energy.csv"
output_path = "../../../data/clean/energy/clean_energy.csv"

# load the dataset
df = load_data(input_path)

# drop columns with excessive missing values (>50%)
df = drop_high_missing(df)

# fill remaining missing values using column mean
mean_imputation_missing(df)

# standardize numerical values using z-score normalization
df = standardize_data(df)

# save cleaned dataset
save_data(df, output_path)