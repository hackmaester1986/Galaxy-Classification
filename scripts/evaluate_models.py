import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from galaxy_classifier.config import IMG_SIZE, BATCH_SIZE, NUM_WORKERS, MODELS_DIR, EVAL_DIR
from galaxy_classifier.data.splits import create_train_val_test_splits, build_stage_dfs
from galaxy_classifier.datasets.image_dataset import ImgDFDataset
from galaxy_classifier.models.resnet import make_resnet18
from galaxy_classifier.models.cnn import GalaxyCNN
from galaxy_classifier.models.evaluate import eval_model_full

def main():
    df = pd.read_csv("data/processed/all_data.csv")
    train_df, val_df, test_df = create_train_val_test_splits(df)
    stage_dfs = build_stage_dfs(train_df, val_df, test_df)

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    s1_test_dl = DataLoader(ImgDFDataset(stage_dfs["s1_test"], val_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    s2_test_dl = DataLoader(ImgDFDataset(stage_dfs["s2_test"], val_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    s1_id_to_label = {0: "other", 1: "galaxy"}
    s2_id_to_label = {0: "elliptical", 1: "disk"}

    ckpt_s1 = torch.load(MODELS_DIR / "resnet18_stage1_best.pt", map_location="cpu")
    model_s1 = make_resnet18(num_classes=2)
    model_s1.load_state_dict(ckpt_s1["model_state"])

    ckpt_s2 = torch.load(MODELS_DIR / "resnet18_stage2_best.pt", map_location="cpu")
    model_s2 = make_resnet18(num_classes=2)
    model_s2.load_state_dict(ckpt_s2["model_state"])

    res_s1 = eval_model_full(model_s1, s1_test_dl, s1_id_to_label)
    res_s2 = eval_model_full(model_s2, s2_test_dl, s2_id_to_label)

    res_s1.to_csv(EVAL_DIR / "stage1_resnet_eval.csv", index=False)
    res_s2.to_csv(EVAL_DIR / "stage2_resnet_eval.csv", index=False)

    print("Saved evaluation CSVs.")

if __name__ == "__main__":
    main()