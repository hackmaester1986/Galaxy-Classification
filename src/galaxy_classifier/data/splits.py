from sklearn.model_selection import train_test_split

def create_train_val_test_splits(df, test_size=0.1, val_size=0.2, random_state=42):
    training_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"]
    )

    train_df, val_df = train_test_split(
        training_df,
        test_size=val_size,
        random_state=random_state,
        stratify=training_df["label"]
    )

    return train_df, val_df, test_df

def build_stage_dfs(train_df, val_df, test_df):
    s1_train = train_df[["image_path", "is_galaxy"]].rename(columns={"is_galaxy": "y"}).copy()
    s1_val = val_df[["image_path", "is_galaxy"]].rename(columns={"is_galaxy": "y"}).copy()
    s1_test = test_df[["image_path", "is_galaxy"]].rename(columns={"is_galaxy": "y"}).copy()

    s2_train = train_df[train_df["is_galaxy"] == 1][["image_path", "morph"]].rename(columns={"morph": "y"}).copy()
    s2_val = val_df[val_df["is_galaxy"] == 1][["image_path", "morph"]].rename(columns={"morph": "y"}).copy()
    s2_test = test_df[test_df["is_galaxy"] == 1][["image_path", "morph"]].rename(columns={"morph": "y"}).copy()

    return {
        "s1_train": s1_train,
        "s1_val": s1_val,
        "s1_test": s1_test,
        "s2_train": s2_train,
        "s2_val": s2_val,
        "s2_test": s2_test,
    }