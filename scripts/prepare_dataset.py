import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import argparse
from pathlib import Path
import boto3
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


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare galaxy dataset.")
    parser.add_argument("--astro-neg-dir", type=str, default="data/raw/astro_neg")
    parser.add_argument("--output-csv", type=str, default="data/processed/all_data.csv")
    parser.add_argument("--n-neg", type=int, default=20000)

    parser.add_argument("--is-sagemaker", action="store_true")
    parser.add_argument("--s3-bucket", type=str, default="sagemaker-us-east-1-587403180437")
    parser.add_argument("--s3-image-prefix", type=str, default="galaxy-classifier/data/images")
    parser.add_argument(
        "--mapping-s3-uri",
        type=str,
        default="s3://sagemaker-us-east-1-587403180437/galaxy-classifier/data/raw/gz2_filename_mapping.csv",
    )
    parser.add_argument("--region", type=str, default="us-east-1")

    return parser.parse_args()


def build_s3_path(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key.lstrip('/')}"


def list_s3_files(bucket: str, prefix: str, region: str) -> list[str]:
    s3 = boto3.client("s3", region_name=region)
    paginator = s3.get_paginator("list_objects_v2")

    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/"):
                keys.append(key)

    return keys


def convert_galaxy_zoo_paths_to_s3(df: pd.DataFrame, bucket: str, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df["image_path"] = df["image_path"].apply(
        lambda p: build_s3_path(bucket, f"{prefix}/galaxy_zoo/{Path(p).name}")
    )
    return df


def build_df_from_s3_keys(keys: list[str], bucket: str, label: str, source: str) -> pd.DataFrame:
    return pd.DataFrame({
        "image_path": [build_s3_path(bucket, key) for key in keys],
        "label": label,
        "source": source,
    })


def main():
    args = parse_args()

    if args.is_sagemaker:
        # ---------- Cloud-native SageMaker mode ----------
        gz2 = load_gz2_table()
        mapping = pd.read_csv(args.mapping_s3_uri)

        # Create synthetic filenames from asset_id only, then rewrite to S3
        placeholder_dir = Path("/tmp/galaxy_zoo_placeholder")

        merged = mapping.merge(
            gz2,
            left_on="objid",
            right_on="dr7objid",
            how="inner"
        )

        merged["image_path"] = (
            placeholder_dir.as_posix() + "/" +
            merged["asset_id"].astype(str) + ".jpg"
        )

        gz_min = create_clean_labels(merged)
        gz_min = convert_galaxy_zoo_paths_to_s3(
            gz_min,
            args.s3_bucket,
            args.s3_image_prefix
        )

        # list actual STL10 objects from S3
        stl_prefix = f"{args.s3_image_prefix}/neg_stl10/"
        stl_keys = list_s3_files(args.s3_bucket, stl_prefix, args.region)

        if args.n_neg is not None:
            stl_keys = stl_keys[:args.n_neg]

        stl_df = build_df_from_s3_keys(
            stl_keys,
            args.s3_bucket,
            label="other",
            source="stl10"
        )

        # list actual astronomy negatives from S3
        astro_prefix = f"{args.s3_image_prefix}/astro_neg/"
        astro_keys = list_s3_files(args.s3_bucket, astro_prefix, args.region)

        astro_df = build_df_from_s3_keys(
            astro_keys,
            args.s3_bucket,
            label="other",
            source="astronomy"
        )

        neg_df = pd.concat([stl_df, astro_df], ignore_index=True)

        all_df = pd.concat(
            [gz_min[["image_path", "label"]], neg_df[["image_path", "label"]]],
            ignore_index=True
        )

        all_df["is_galaxy"] = all_df["label"].isin(["elliptical", "disk"]).astype("int64")
        all_df["morph"] = all_df["label"].map({"elliptical": 0, "disk": 1})

    else:
        # ---------- Local mode ----------
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