import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from galaxy_classifier.config import DEVICE

@torch.no_grad()
def eval_model_full(model, dl, id_to_label):
    model.eval()

    all_true = []
    all_pred = []
    all_probs = []
    all_paths = []

    for x, y, paths in dl:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)

        all_true.extend(y.cpu().numpy())
        all_pred.extend(pred.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_paths.extend(paths)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_probs = np.array(all_probs)

    results_df = pd.DataFrame({
        "image_path": all_paths,
        "y_true": all_true,
        "y_pred": all_pred,
        "true_label": [id_to_label[i] for i in all_true],
        "pred_label": [id_to_label[i] for i in all_pred],
        "correct": all_true == all_pred,
        "pred_conf": all_probs.max(axis=1)
    })

    for class_id, class_name in id_to_label.items():
        results_df[f"prob_{class_name}"] = all_probs[:, class_id]

    return results_df