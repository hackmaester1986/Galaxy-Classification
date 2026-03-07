import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch
from torchvision import transforms

from galaxy_classifier.config import IMG_SIZE, DEVICE, MODELS_DIR, EVAL_DIR
from galaxy_classifier.models.resnet import make_resnet18
from galaxy_classifier.visualization.gradcam_utils import run_gradcam

def main():
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    cam_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
    ])

    s2_id_to_label = {0: "elliptical", 1: "disk"}

    ckpt = torch.load(MODELS_DIR / "resnet18_stage2_best.pt", map_location=DEVICE)
    model = make_resnet18(num_classes=2).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])

    res_s2 = pd.read_csv(EVAL_DIR / "stage2_resnet_eval.csv")
    row = res_s2.iloc[0]

    run_gradcam(
        model=model,
        row=row,
        model_tf=val_tf,
        cam_tf=cam_tf,
        label_map=s2_id_to_label,
        target_layer=model.layer4[-1],
        device=DEVICE
    )

if __name__ == "__main__":
    main()