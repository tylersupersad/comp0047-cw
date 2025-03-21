from utils import load_data, drop_high_missing, interpolate_missing, save_data

# define file paths
input_path = "../../../data/raw/technology/technology_close.csv"
output_path = "../../../data/clean/technology/clean_technology_close.csv"

# load dataset
df = load_data(input_path)

# ensure dataset is sorted by date (ascending)
df.sort_values(by=df.columns[0], inplace=True)

# drop columns with excessive missing values
df = drop_high_missing(df)

# apply spline interpolation for missing values (best for stock price trends)
df = interpolate_missing(df)

# save cleaned dataset
save_data(df, output_path)