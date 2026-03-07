import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from galaxy_classifier.data.splits import create_train_val_test_splits, build_stage_dfs
from galaxy_classifier.datasets.image_dataset import ImgDFDataset
from galaxy_classifier.models.resnet import make_resnet18
from galaxy_classifier.models.evaluate import eval_model_full

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained galaxy models.")
    parser.add_argument("--input-csv", type=str, default="data/processed/all_data.csv")
    parser.add_argument("--model-dir", type=str, default="artifacts/models")
    parser.add_argument("--eval-dir", type=str, default="artifacts/evaluations")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()

def main(args):
    input_csv = Path(args.input_csv)
    model_dir = Path(args.model_dir)
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    train_df, val_df, test_df = create_train_val_test_splits(
        df,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )
    stage_dfs = build_stage_dfs(train_df, val_df, test_df)

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    s1_test_dl = DataLoader(
        ImgDFDataset(stage_dfs["s1_test"], val_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    s2_test_dl = DataLoader(
        ImgDFDataset(stage_dfs["s2_test"], val_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    s1_id_to_label = {0: "other", 1: "galaxy"}
    s2_id_to_label = {0: "elliptical", 1: "disk"}

    ckpt_s1 = torch.load(
        model_dir / "resnet18_stage1_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    model_s1 = make_resnet18(num_classes=2)
    model_s1.load_state_dict(ckpt_s1["model_state"])

    ckpt_s2 = torch.load(
    model_dir / "resnet18_stage2_best.pt",
    map_location="cpu",
    weights_only=False,
)
    model_s2 = make_resnet18(num_classes=2)
    model_s2.load_state_dict(ckpt_s2["model_state"])

    res_s1 = eval_model_full(model_s1, s1_test_dl, s1_id_to_label)
    res_s2 = eval_model_full(model_s2, s2_test_dl, s2_id_to_label)

    res_s1.to_csv(eval_dir / "stage1_resnet_eval.csv", index=False)
    res_s2.to_csv(eval_dir / "stage2_resnet_eval.csv", index=False)

    print(f"Saved: {eval_dir / 'stage1_resnet_eval.csv'}")
    print(f"Saved: {eval_dir / 'stage2_resnet_eval.csv'}")

if __name__ == "__main__":
    args = parse_args()
    main(args)