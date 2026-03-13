import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import argparse
from pathlib import Path as _Path

import pandas as pd
import torch
from torchvision import transforms

from galaxy_classifier.models.resnet import make_resnet18_eval
from galaxy_classifier.visualization.gradcam_utils import run_gradcam


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations from evaluation outputs."
    )

    parser.add_argument(
        "--eval-csv",
        type=str,
        default="artifacts/evaluations/stage2_resnet_eval.csv",
        help="Path to evaluation CSV.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/models/resnet18_stage2_best.pt",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/figures/gradcam",
        help="Directory to save Grad-CAM images.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Image size used during training/inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of Grad-CAM examples to generate.",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="incorrect",
        choices=["all", "correct", "incorrect"],
        help="Which rows to visualize.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=None,
        help="Optional explicit row index from the evaluation CSV. If set, overrides selection filters.",
    )
    parser.add_argument(
        "--pred-label",
        type=str,
        default=None,
        help="Optional filter by predicted label.",
    )
    parser.add_argument(
        "--true-label",
        type=str,
        default=None,
        help="Optional filter by true label.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="pred_conf",
        choices=["pred_conf", "image_path"],
        help="Column to sort selected examples by.",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending instead of descending.",
    )

    return parser.parse_args()


def build_transforms(img_size: int):
    model_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    cam_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    return model_tf, cam_tf


def load_model(model_path: str, device: torch.device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = make_resnet18_eval(num_classes=2).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def filter_rows(df: pd.DataFrame, args) -> pd.DataFrame:
    out = df.copy()

    if args.row_index is not None:
        if args.row_index < 0 or args.row_index >= len(out):
            raise ValueError(
                f"--row-index {args.row_index} is out of range for dataframe of size {len(out)}"
            )
        return out.iloc[[args.row_index]].copy()

    if args.selection == "correct":
        out = out.loc[out["correct"] == True].copy()
    elif args.selection == "incorrect":
        out = out.loc[out["correct"] == False].copy()

    if args.pred_label is not None:
        out = out.loc[out["pred_label"] == args.pred_label].copy()

    if args.true_label is not None:
        out = out.loc[out["true_label"] == args.true_label].copy()

    if out.empty:
        raise ValueError("No rows matched the requested filters.")

    if args.sort_by in out.columns:
        out = out.sort_values(args.sort_by, ascending=args.ascending).copy()

    return out.head(args.num_examples).copy()


def safe_name(value: str) -> str:
    keep = []
    for ch in str(value):
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "item"


def main(args):
    device = torch.device(args.device)
    output_dir = _Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Grad-CAM args:")
    print(vars(args))

    model_tf, cam_tf = build_transforms(args.img_size)

    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path, device)

    print(f"Loading evaluation CSV from: {args.eval_csv}")
    df = pd.read_csv(args.eval_csv)
    print(f"Loaded {len(df)} evaluation rows")

    selected = filter_rows(df, args)
    print(f"Selected {len(selected)} row(s) for Grad-CAM generation")

    # Adjust this label map if you use the script for stage1 later.
    label_map = {
        0: "elliptical",
        1: "disk",
    }

    for i, (_, row) in enumerate(selected.iterrows(), start=1):
        image_path = row["image_path"]
        pred_label = row.get("pred_label", "unknown")
        true_label = row.get("true_label", "unknown")
        pred_conf = row.get("pred_conf", None)

        print(
            f"[{i}/{len(selected)}] image_path={image_path} "
            f"true={true_label} pred={pred_label} conf={pred_conf}"
        )

        out_name = (
            f"{i:02d}_true-{safe_name(true_label)}"
            f"_pred-{safe_name(pred_label)}"
            f"_conf-{float(pred_conf):.4f}.png"
            if pred_conf is not None
            else f"{i:02d}_true-{safe_name(true_label)}_pred-{safe_name(pred_label)}.png"
        )
        out_path = output_dir / out_name

        run_gradcam(
            model=model,
            row=row,
            model_tf=model_tf,
            cam_tf=cam_tf,
            label_map=label_map,
            target_layer=model.layer4[-1],
            device=device,
            save_path=str(out_path),
        )

        print(f"Saved Grad-CAM to: {out_path}")

    print("Done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)