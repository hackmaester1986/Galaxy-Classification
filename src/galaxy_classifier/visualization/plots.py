import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import textwrap

def show_images(df, n=12):
    sample = df.sample(n)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    for ax, (_, row) in zip(axes.flatten(), sample.iterrows()):
        img = Image.open(row["image_path"])
        ax.imshow(img)
        ax.axis("off")

        label = row.get("label", "unknown")
        p_smooth = row.get("t01_smooth_or_features_a01_smooth_fraction", 0)
        p_disk = row.get("t01_smooth_or_features_a02_features_or_disk_fraction", 0)
        p_star = row.get("t01_smooth_or_features_a03_star_or_artifact_fraction", 0)
        p_edge = row.get("t02_edgeon_a04_yes_fraction", 0)
        p_bar = row.get("t03_bar_a06_bar_fraction", 0)
        p_spiral = row.get("t04_spiral_a08_spiral_fraction", 0)

        title = f"""
label: {label}

Q1
smooth: {p_smooth:.2f}
disk: {p_disk:.2f}
star: {p_star:.2f}

edge-on: {p_edge:.2f}
bar: {p_bar:.2f}
spiral: {p_spiral:.2f}
"""
        ax.set_title(textwrap.dedent(title), fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_history(history, title="Training History"):
    epochs = range(1, len(history["resnet18"]["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["resnet18"]["train_loss"], label="ResNet18")
    plt.plot(epochs, history["galaxycnn"]["train_loss"], label="GalaxyCNN")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title(f"{title} - Train Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["resnet18"]["val_acc"], label="ResNet18")
    plt.plot(epochs, history["galaxycnn"]["val_acc"], label="GalaxyCNN")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(f"{title} - Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confidence_hist(results_df, title="Confidence Histogram"):
    plt.figure(figsize=(8, 5))
    plt.hist(results_df.loc[results_df["correct"], "pred_conf"], bins=30, alpha=0.6, label="Correct")
    plt.hist(results_df.loc[~results_df["correct"], "pred_conf"], bins=30, alpha=0.6, label="Wrong")
    plt.xlabel("Prediction Confidence")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.show()

def show_misclassified(results_df, n=8):
    sample = results_df[~results_df["correct"]].sample(min(n, len(results_df[~results_df["correct"]])))
    cols = 4
    rows = (len(sample) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = np.array(axes).reshape(-1)

    for ax in axes[len(sample):]:
        ax.axis("off")

    for ax, (_, row) in zip(axes, sample.iterrows()):
        img = Image.open(row["image_path"]).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        title = f"""
true: {row['true_label']}
pred: {row['pred_label']}
conf: {row['pred_conf']:.2f}
"""
        ax.set_title(textwrap.dedent(title), fontsize=10)

    plt.tight_layout()
    plt.show()