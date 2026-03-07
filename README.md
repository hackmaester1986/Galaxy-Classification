# Galaxy Classifier

A deep learning project for classifying galaxy images using a two-stage image classification pipeline built with PyTorch.

## Project Overview

This project uses Galaxy Zoo 2 image data to classify astronomical images in two stages:

1. **Stage 1:** Determine whether an image is a galaxy or not
2. **Stage 2:** For galaxy images, classify morphology as:
   - **elliptical**
   - **disk**

The project also compares two model architectures:

- **ResNet18**
- **Custom GalaxyCNN**

In addition, the repository includes evaluation utilities, visualization tools, and Grad-CAM explainability support for interpreting model predictions.

---

## Dataset Sources

This project uses:

- **Galaxy Zoo 2 image dataset**
- **Galaxy Zoo 2 main sample spectroscopic table**
- **STL10 images** as negative / non-galaxy examples
- Optional manually uploaded astronomy negatives

### Labeling Strategy

Galaxy Zoo vote fractions are used to create high-confidence labels:

- `elliptical` if `p_smooth >= 0.80`
- `disk` if `p_disk >= 0.80`
- `other` if `p_star >= 0.40`

Low-confidence rows are discarded from the supervised dataset.

---

## Repository Structure

```text
galaxy-classifier/
├── README.md
├── .gitignore
├── requirements.txt
├── .env.example
├── notebooks/
│   └── galaxy_exploration.ipynb
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── artifacts/
│   ├── models/
│   ├── evaluations/
│   └── figures/
├── scripts/
│   ├── prepare_dataset.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── gradcam_demo.py
└── src/
    └── galaxy_classifier/
        ├── config.py
        ├── paths.py
        ├── data/
        ├── datasets/
        ├── models/
        ├── visualization/
        └── utils/