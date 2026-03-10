import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import pandas as pd
import torch
import tarfile
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms

from galaxy_classifier.data.splits import create_train_val_test_splits, build_stage_dfs
from galaxy_classifier.datasets.image_dataset import ImgDFDataset
from galaxy_classifier.models.resnet import make_resnet18
from galaxy_classifier.models.cnn import GalaxyCNN
from galaxy_classifier.models.evaluate import eval_model_full


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained galaxy models.")
    parser.add_argument("--input-csv", type=str, default="/opt/ml/processing/input/data/all_data.csv")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/processing/model")
    parser.add_argument("--eval-dir", type=str, default="/opt/ml/processing/evaluation")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def compute_accuracy(results_df: pd.DataFrame) -> float:
    return float(results_df["correct"].mean())


def load_resnet_checkpoint(path: Path, num_classes: int):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = make_resnet18(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_galaxycnn_checkpoint(path: Path, num_classes: int):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = GalaxyCNN(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def extract_model_artifact(model_dir: Path) -> None:
    tar_path = model_dir / "model.tar.gz"
    if tar_path.exists():
        print(f"Extracting model artifact: {tar_path}")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=model_dir)
        print("Model dir after extract:", [p.name for p in model_dir.iterdir()])
    else:
        print("No model.tar.gz found. Model dir contents:", [p.name for p in model_dir.iterdir()])


def main(args):
    print("Evaluation args:")
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
    extract_model_artifact(model_dir)
    eval_dir = Path(args.eval_dir)

    eval_dir.mkdir(parents=True, exist_ok=True)

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
        num_workers=args.num_workers,
    )

    s2_test_dl = DataLoader(
        ImgDFDataset(stage_dfs["s2_test"], val_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    s1_id_to_label = {0: "other", 1: "galaxy"}
    s2_id_to_label = {0: "elliptical", 1: "disk"}

    model_s1_resnet = load_resnet_checkpoint(model_dir / "resnet18_stage1_best.pt", 2)
    model_s2_resnet = load_resnet_checkpoint(model_dir / "resnet18_stage2_best.pt", 2)
    model_s1_gal = load_galaxycnn_checkpoint(model_dir / "galaxycnn_stage1_best.pt", 2)
    model_s2_gal = load_galaxycnn_checkpoint(model_dir / "galaxycnn_stage2_best.pt", 2)

    res_s1_resnet = eval_model_full(model_s1_resnet, s1_test_dl, s1_id_to_label)
    res_s2_resnet = eval_model_full(model_s2_resnet, s2_test_dl, s2_id_to_label)
    res_s1_gal = eval_model_full(model_s1_gal, s1_test_dl, s1_id_to_label)
    res_s2_gal = eval_model_full(model_s2_gal, s2_test_dl, s2_id_to_label)

    res_s1_resnet.to_csv(eval_dir / "stage1_resnet_eval.csv", index=False)
    res_s2_resnet.to_csv(eval_dir / "stage2_resnet_eval.csv", index=False)
    res_s1_gal.to_csv(eval_dir / "stage1_galaxycnn_eval.csv", index=False)
    res_s2_gal.to_csv(eval_dir / "stage2_galaxycnn_eval.csv", index=False)

    metrics = {
        "stage1_resnet_accuracy": compute_accuracy(res_s1_resnet),
        "stage2_resnet_accuracy": compute_accuracy(res_s2_resnet),
        "stage1_galaxycnn_accuracy": compute_accuracy(res_s1_gal),
        "stage2_galaxycnn_accuracy": compute_accuracy(res_s2_gal),
        "num_stage1_test_rows": int(len(res_s1_resnet)),
        "num_stage2_test_rows": int(len(res_s2_resnet)),
    }

    with open(eval_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved evaluation outputs:")
    print(eval_dir / "stage1_resnet_eval.csv")
    print(eval_dir / "stage2_resnet_eval.csv")
    print(eval_dir / "stage1_galaxycnn_eval.csv")
    print(eval_dir / "stage2_galaxycnn_eval.csv")
    print(eval_dir / "metrics.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    args = parse_args()
    main(args)