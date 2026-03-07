import numpy as np
import torch
import torch.nn.functional as F
from galaxy_classifier.config import DEVICE

@torch.no_grad()
def predict_two_stage(img_pil, model_s1, model_s2, tf, galaxy_thresh=0.70):
    x = tf(img_pil.convert("RGB")).unsqueeze(0).to(DEVICE)

    logits1 = model_s1(x)
    p1 = F.softmax(logits1, dim=1)[0].cpu().numpy()
    p_galaxy = float(p1[1])

    if p_galaxy < galaxy_thresh:
        return {"label": "other", "p_galaxy": p_galaxy}

    logits2 = model_s2(x)
    p2 = F.softmax(logits2, dim=1)[0].cpu().numpy()
    pred2 = int(np.argmax(p2))
    label = "elliptical" if pred2 == 0 else "disk"

    return {
        "label": label,
        "p_galaxy": p_galaxy,
        "p_elliptical": float(p2[0]),
        "p_disk": float(p2[1]),
    }