import pandas as pd

# load dataset and filter to non-null target rows
def load_feature_dataset(path, target_col):
    df = pd.read_csv(path, parse_dates=['date'])

    # keep only rows where the prediction target is defined
    df = df.dropna(subset=[target_col])

    return df
