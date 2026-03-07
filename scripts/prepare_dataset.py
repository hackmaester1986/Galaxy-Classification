import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import argparse

from galaxy_classifier.data.download import (
    download_galaxy_zoo_dataset,
    load_mapping_and_paths,
    load_gz2_table,
)
from galaxy_classifier.data.build_dataset import (
    merge_mapping_with_labels,
    create_clean_labels,
    build_stl10_negatives,
    build_uploaded_astro_negatives,
    combine_datasets,
)
from galaxy_classifier.config import PROCESSED_DIR, RAW_DIR

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare galaxy classification dataset.")
    parser.add_argument("--astro-neg-dir", type=str, default="data/raw/astro_neg")
    parser.add_argument("--output-csv", type=str, default="data/processed/all_data.csv")
    parser.add_argument("--n-neg", type=int, default=20000)
    return parser.parse_args()
    
def main():
    args = parse_args()
    dataset_path = download_galaxy_zoo_dataset()
    mapping, images_dir, _ = load_mapping_and_paths(dataset_path)
    gz2 = load_gz2_table()

    merged = merge_mapping_with_labels(mapping, gz2, images_dir)
    gz_min = create_clean_labels(merged)

    stl_df = build_stl10_negatives(n_neg=args.n_neg)
    astro_df = build_uploaded_astro_negatives(Path(args.astro_neg_dir))

    neg_df = pd.concat([stl_df, astro_df], ignore_index=True)
    all_df = combine_datasets(gz_min, neg_df)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(output_csv, index=False)
    print("Saved:", output_csv)
    print(all_df["label"].value_counts())

if __name__ == "__main__":
    main()