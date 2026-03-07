import numpy as np
import matplotlib.pyplot as plt
import torch
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
    return_outputs=False
):
    model.eval()
    img_path = row["image_path"]
    img = Image.open(img_path).convert("RGB")

    input_tensor = model_tf(img).unsqueeze(0).to(device)
    display_tensor = cam_tf(img)
    display_img = np.transpose(display_tensor.numpy(), (1, 2, 0)).astype(np.float32)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_class = int(np.argmax(probs))
        pred_conf = float(probs[pred_class])

    if target_class is None:
        target_class = row["y_true"]

    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    visualization = show_cam_on_image(display_img, grayscale_cam, use_rgb=True)

    pred_label = label_map[pred_class]
    target_label = label_map[target_class]

    plt.figure(figsize=figsize)

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
    plt.title(f"Grad-CAM\npred={pred_label} ({pred_conf:.2f})\ntarget={target_label}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    if return_outputs:
        return {
            "pred_class": pred_class,
            "pred_label": pred_label,
            "pred_conf": pred_conf,
            "probs": probs,
            "target_class": target_class,
            "target_label": target_label,
            "grayscale_cam": grayscale_cam,
            "overlay": visualization,
            "display_img": display_img,
        }