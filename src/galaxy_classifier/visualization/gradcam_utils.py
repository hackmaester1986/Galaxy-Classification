import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def run_gradcam(
    model,
    row,
    model_tf,
    cam_tf,
    label_map,
    target_layer,
    device,
    target_class=None,
    figsize=(12, 4),
    return_outputs=False,
    show=True,
    save_path=None,
):
    model.eval()

    img_path = row["image_path"]
    if not Path(img_path).exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = Image.open(img_path).convert("RGB")

    # Model input (normalized)
    input_tensor = model_tf(img).unsqueeze(0).to(device)

    # Display image / CAM base image (not normalized)
    display_tensor = cam_tf(img)
    display_img = np.transpose(display_tensor.numpy(), (1, 2, 0)).astype(np.float32)

    # Clamp just in case
    display_img = np.clip(display_img, 0.0, 1.0)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        pred_class = int(np.argmax(probs))
        pred_conf = float(probs[pred_class])

    if target_class is None:
        if "y_true" in row:
            target_class = int(row["y_true"])
        else:
            target_class = pred_class

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    visualization = show_cam_on_image(display_img, grayscale_cam, use_rgb=True)

    pred_label = label_map.get(pred_class, str(pred_class))
    target_label = label_map.get(target_class, str(target_class))

    fig = plt.figure(figsize=figsize)

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(display_img)
    plt.title("Model Input")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(visualization)
    plt.title(
        f"Grad-CAM\npred={pred_label} ({pred_conf:.2f})\ntarget={target_label}"
    )
    plt.axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_outputs:
        return {
            "image_path": img_path,
            "pred_class": pred_class,
            "pred_label": pred_label,
            "pred_conf": pred_conf,
            "probs": probs,
            "target_class": target_class,
            "target_label": target_label,
            "grayscale_cam": grayscale_cam,
            "overlay": visualization,
            "display_img": display_img,
            "save_path": str(save_path) if save_path is not None else None,
        }