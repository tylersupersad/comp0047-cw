import os
from utils import read_and_parse_csv, reshape_and_merge_raw_data, clean_and_transform_data

def run_pipeline_for_sector(sector, root_dir="../../data/raw"):
    # load raw csvs
    close = read_and_parse_csv(os.path.join(root_dir, sector, f"{sector}_close.csv"))
    volume = read_and_parse_csv(os.path.join(root_dir, sector, f"{sector}_volume.csv"))
    cap = read_and_parse_csv(os.path.join(root_dir, sector, f"{sector}_market_cap.csv"))

    # merge raw data
    merged = reshape_and_merge_raw_data(close, volume, cap)
    merged.to_csv(f"{sector}_data_aligned.csv", index=False)

    # apply cleaning and save result
    cleaned = clean_and_transform_data(merged)
    cleaned.to_csv(f"clean_{sector}.csv", index=False)

    print(f"[✓] Finished preprocessing for {sector}: {len(cleaned):,} rows saved to clean_{sector}.csv")

if __name__ == "__main__":
    # preprocess energy data
    #run_pipeline_for_sector("energy")
    # preprocess technology data
    run_pipeline_for_sector("technology")