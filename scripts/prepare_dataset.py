import pandas as pd

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

def main():
    dataset_path = download_galaxy_zoo_dataset()
    mapping, images_dir, _ = load_mapping_and_paths(dataset_path)
    gz2 = load_gz2_table()

    merged = merge_mapping_with_labels(mapping, gz2, images_dir)
    gz_min = create_clean_labels(merged)

    stl_df = build_stl10_negatives(n_neg=20000)
    astro_df = build_uploaded_astro_negatives(RAW_DIR / "astro_neg")

    neg_df = pd.concat([stl_df, astro_df], ignore_index=True)
    all_df = combine_datasets(gz_min, neg_df)

    all_df.to_csv(PROCESSED_DIR / "all_data.csv", index=False)
    print("Saved:", PROCESSED_DIR / "all_data.csv")
    print(all_df["label"].value_counts())

if __name__ == "__main__":
    main()