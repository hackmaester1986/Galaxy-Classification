import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from galaxy_classifier.config import DEVICE
from galaxy_classifier.models.resnet import make_resnet18
from galaxy_classifier.models.cnn import GalaxyCNN

@torch.no_grad()
def eval_model(model, dl):
    model.eval()
    ys, ps = [], []

    for x, y, _ in dl:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        pred = logits.argmax(dim=1)

        ys.append(y.cpu().numpy())
        ps.append(pred.cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return y_true, y_pred

def train_classifier(
    *,
    run_name,
    num_classes,
    dl_train,
    dl_val,
    id_to_label,
    epochs=5,
    lr=3e-4,
    weight_decay=1e-4,
    save_path_resnet="best_resnet.pt",
    save_path_galaxycnn="best_galaxycnn.pt",
):
    model = make_resnet18(num_classes).to(DEVICE)
    galaxycnn = GalaxyCNN(num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_gal = torch.optim.AdamW(galaxycnn.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = -1.0
    best_acc_gal = -1.0

    history = {
        "resnet18": {"train_loss": [], "val_acc": []},
        "galaxycnn": {"train_loss": [], "val_acc": []}
    }

    for epoch in range(1, epochs + 1):
        model.train()
        galaxycnn.train()

        running_loss = 0.0
        running_loss_gal = 0.0
        n_samples = 0

        pbar = tqdm(dl_train, desc=f"{run_name} epoch {epoch}/{epochs}")

        for x, y, _ in pbar:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            batch_size = x.size(0)
            n_samples += batch_size

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            optimizer_gal.zero_grad(set_to_none=True)
            logits_gal = galaxycnn(x)
            loss_gal = criterion(logits_gal, y)
            loss_gal.backward()
            optimizer_gal.step()

            running_loss += loss.item() * batch_size
            running_loss_gal += loss_gal.item() * batch_size

            pbar.set_postfix(
                resnet_loss=float(loss.detach().cpu()),
                galaxycnn_loss=float(loss_gal.detach().cpu())
            )

        epoch_loss = running_loss / n_samples
        epoch_loss_gal = running_loss_gal / n_samples

        history["resnet18"]["train_loss"].append(epoch_loss)
        history["galaxycnn"]["train_loss"].append(epoch_loss_gal)

        y_true, y_pred = eval_model(model, dl_val)
        y_true_gal, y_pred_gal = eval_model(galaxycnn, dl_val)

        acc = (y_true == y_pred).mean()
        acc_gal = (y_true_gal == y_pred_gal).mean()

        history["resnet18"]["val_acc"].append(acc)
        history["galaxycnn"]["val_acc"].append(acc_gal)

        print(f"\n{run_name} ResNet18 Epoch {epoch} val_acc={acc:.4f}")
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(
            y_true, y_pred,
            target_names=[id_to_label[i] for i in range(num_classes)],
            digits=4
        ))

        print(f"\n{run_name} GalaxyCNN Epoch {epoch} val_acc={acc_gal:.4f}")
        print(confusion_matrix(y_true_gal, y_pred_gal))
        print(classification_report(
            y_true_gal, y_pred_gal,
            target_names=[id_to_label[i] for i in range(num_classes)],
            digits=4
        ))

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state": model.state_dict(),
                "num_classes": num_classes,
                "id_to_label": id_to_label,
                "model_type": "resnet18",
                "history": history,
            }, save_path_resnet)
            print(f"Saved ResNet checkpoint to {save_path_resnet}")

        if acc_gal > best_acc_gal:
            best_acc_gal = acc_gal
            torch.save({
                "model_state": galaxycnn.state_dict(),
                "num_classes": num_classes,
                "id_to_label": id_to_label,
                "model_type": "galaxycnn",
                "history": history,
            }, save_path_galaxycnn)
            print(f"Saved GalaxyCNN checkpoint to {save_path_galaxycnn}")

    return model, best_acc, galaxycnn, best_acc_gal, history