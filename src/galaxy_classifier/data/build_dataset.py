import os
from pathlib import Path
import numpy as np
import pandas as pd
from torchvision.datasets import STL10

from galaxy_classifier.config import T_GAL, T_STAR, RAW_DIR

def merge_mapping_with_labels(mapping: pd.DataFrame, gz2: pd.DataFrame, images_dir: Path) -> pd.DataFrame:
    merged = mapping.merge(
        gz2,
        left_on="objid",
        right_on="dr7objid",
        how="inner"
    )

    merged["image_path"] = (
        images_dir.as_posix() + "/" + merged["asset_id"].astype(str) + ".jpg"
    )
    merged["image_exists"] = merged["image_path"].apply(os.path.exists)
    merged = merged[merged["image_exists"]].copy()

    return merged

def create_clean_labels(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()

    df["p_smooth"] = df["t01_smooth_or_features_a01_smooth_fraction"].astype(float)
    df["p_disk"] = df["t01_smooth_or_features_a02_features_or_disk_fraction"].astype(float)
    df["p_star"] = df["t01_smooth_or_features_a03_star_or_artifact_fraction"].astype(float)

    df["label"] = "discard"
    df.loc[df["p_smooth"] >= T_GAL, "label"] = "elliptical"
    df.loc[df["p_disk"] >= T_GAL, "label"] = "disk"
    df.loc[df["p_star"] >= T_STAR, "label"] = "other"

    gz_min = df[df["label"] != "discard"][
        ["image_path", "label", "p_smooth", "p_disk", "p_star"]
    ].copy()

    return gz_min

def build_stl10_negatives(n_neg: int = 20000) -> pd.DataFrame:
    neg_dir = RAW_DIR / "neg_stl10"
    neg_dir.mkdir(parents=True, exist_ok=True)

    stl = STL10(root=str(neg_dir), split="train", download=True)
    n_neg = min(len(stl), n_neg)
    idx = np.random.choice(len(stl), size=n_neg, replace=False)

    stl_paths = []
    for j in idx:
        img, _ = stl[j]
        p = neg_dir / f"stl_{j}.png"
        img.save(p)
        stl_paths.append(str(p))

    return pd.DataFrame({
        "image_path": stl_paths,
        "label": "other",
        "source": "stl10"
    })

def build_uploaded_astro_negatives(upload_dir: Path) -> pd.DataFrame:
    upload_dir.mkdir(parents=True, exist_ok=True)

    astro_paths = [str(p) for p in upload_dir.glob("*") if p.is_file()]

    return pd.DataFrame({
        "image_path": astro_paths,
        "label": "other",
        "source": "astronomy"
    })

def combine_datasets(gz_min: pd.DataFrame, neg_df: pd.DataFrame) -> pd.DataFrame:
    all_df = pd.concat(
        [gz_min[["image_path", "label"]], neg_df[["image_path", "label"]]],
        ignore_index=True
    )

    all_df["exists"] = all_df["image_path"].apply(os.path.exists)
    all_df = all_df[all_df["exists"]].drop(columns=["exists"]).copy()

    all_df["is_galaxy"] = all_df["label"].isin(["elliptical", "disk"]).astype("int64")
    all_df["morph"] = all_df["label"].map({"elliptical": 0, "disk": 1})

    return all_df