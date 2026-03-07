import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from galaxy_classifier.config import IMG_SIZE, BATCH_SIZE, NUM_WORKERS, MODELS_DIR
from galaxy_classifier.data.splits import create_train_val_test_splits, build_stage_dfs
from galaxy_classifier.datasets.image_dataset import ImgDFDataset
from galaxy_classifier.models.train import train_classifier

def main():
    df = pd.read_csv("data/processed/all_data.csv")

    train_df, val_df, test_df = create_train_val_test_splits(df)
    stage_dfs = build_stage_dfs(train_df, val_df, test_df)

    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    dl_s1_train = DataLoader(ImgDFDataset(stage_dfs["s1_train"], train_tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    dl_s1_val = DataLoader(ImgDFDataset(stage_dfs["s1_val"], val_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    dl_s2_train = DataLoader(ImgDFDataset(stage_dfs["s2_train"], train_tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    dl_s2_val = DataLoader(ImgDFDataset(stage_dfs["s2_val"], val_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    s1_id_to_label = {0: "other", 1: "galaxy"}
    s2_id_to_label = {0: "elliptical", 1: "disk"}

    train_classifier(
        run_name="stage1_is_galaxy",
        num_classes=2,
        dl_train=dl_s1_train,
        dl_val=dl_s1_val,
        id_to_label=s1_id_to_label,
        epochs=2,
        save_path_resnet=str(MODELS_DIR / "resnet18_stage1_best.pt"),
        save_path_galaxycnn=str(MODELS_DIR / "galaxycnn_stage1_best.pt"),
    )

    train_classifier(
        run_name="stage2_morph",
        num_classes=2,
        dl_train=dl_s2_train,
        dl_val=dl_s2_val,
        id_to_label=s2_id_to_label,
        epochs=5,
        save_path_resnet=str(MODELS_DIR / "resnet18_stage2_best.pt"),
        save_path_galaxycnn=str(MODELS_DIR / "galaxycnn_stage2_best.pt"),
    )

if __name__ == "__main__":
    main()