import os
import sys

# add the scripts/ directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_data, drop_high_missing, standardize_data, mean_imputation_missing, save_data

# define the file paths
input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/raw/technology/technology.csv"))
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/clean/technology/clean_technology.csv"))

# load the dataset
df = load_data(input_path)

# drop columns with excessive missing values (>50%)
df = drop_high_missing(df)

# fill remaining missing values using column mean
df = mean_imputation_missing(df)

# standardize numerical values using z-score normalization
df = standardize_data(df)

# save cleaned dataset
save_data(df, output_path)