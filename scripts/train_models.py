import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from galaxy_classifier.data.splits import create_train_val_test_splits, build_stage_dfs
from galaxy_classifier.datasets.image_dataset import ImgDFDataset
from galaxy_classifier.models.train import train_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train stage 1 and stage 2 galaxy classifiers.")
    parser.add_argument("--input-csv", type=str, default="/opt/ml/input/data/train/all_data_s3.csv")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/output/data")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs-stage1", type=int, default=2)
    parser.add_argument("--epochs-stage2", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main(args):
    print("Training args:")
    print(json.dumps(vars(args), indent=2))


    input_path = Path(args.input_csv)

    if input_path.is_dir():
        csv_files = sorted(input_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {input_path}")
        input_csv = csv_files[0]
    else:
        input_csv = input_path

    print(f"Reading training CSV from: {input_csv}")
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    if args.max_rows is not None:
        df = df.sample(min(args.max_rows, len(df)), random_state=args.random_state).copy()

    train_df, val_df, test_df = create_train_val_test_splits(
        df,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )
    stage_dfs = build_stage_dfs(train_df, val_df, test_df)

    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dl_s1_train = DataLoader(
        ImgDFDataset(stage_dfs["s1_train"], train_tf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    dl_s1_val = DataLoader(
        ImgDFDataset(stage_dfs["s1_val"], val_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    dl_s2_train = DataLoader(
        ImgDFDataset(stage_dfs["s2_train"], train_tf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    dl_s2_val = DataLoader(
        ImgDFDataset(stage_dfs["s2_val"], val_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    s1_id_to_label = {0: "other", 1: "galaxy"}
    s2_id_to_label = {0: "elliptical", 1: "disk"}

    _, best_s1, _, best_gal_s1, hist_s1 = train_classifier(
        run_name="stage1_is_galaxy",
        num_classes=2,
        dl_train=dl_s1_train,
        dl_val=dl_s1_val,
        id_to_label=s1_id_to_label,
        epochs=args.epochs_stage1,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_path_resnet=str(model_dir / "resnet18_stage1_best.pt"),
        save_path_galaxycnn=str(model_dir / "galaxycnn_stage1_best.pt"),
    )

    _, best_s2, _, best_gal_s2, hist_s2 = train_classifier(
        run_name="stage2_morph",
        num_classes=2,
        dl_train=dl_s2_train,
        dl_val=dl_s2_val,
        id_to_label=s2_id_to_label,
        epochs=args.epochs_stage2,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_path_resnet=str(model_dir / "resnet18_stage2_best.pt"),
        save_path_galaxycnn=str(model_dir / "galaxycnn_stage2_best.pt"),
    )

    summary = {
        "stage1_resnet_best_val_acc": float(best_s1),
        "stage1_galaxycnn_best_val_acc": float(best_gal_s1),
        "stage2_resnet_best_val_acc": float(best_s2),
        "stage2_galaxycnn_best_val_acc": float(best_gal_s2),
        "hist_s1": hist_s1,
        "hist_s2": hist_s2,
    }

    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved training summary to {output_dir / 'training_summary.json'}")

    saved_models = sorted(model_dir.glob("*"))
    print("Model dir:", model_dir)
    print("Model dir contents:", [p.name for p in saved_models])

    if not saved_models:
        raise RuntimeError(f"No files were saved to model_dir: {model_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)