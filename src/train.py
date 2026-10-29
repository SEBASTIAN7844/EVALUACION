"""
Pregunta 4 - Entrenamiento (CrossEntropyLoss + Adam), métricas, curvas y matriz de confusión.
Uso:  python src/train.py
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import CLASSES, SEED, get_dataloaders, set_seed  # noqa: E402
from model import CNN, validate_forward_pass  # noqa: E402

# ----------------------------- Hiperparámetros -----------------------------
EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, optimizer=None):
    """Una pasada por el loader. Si hay optimizer -> entrena; si no -> evalúa."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def plot_curves(h, path):
    ep = range(1, len(h["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(ep, h["train_loss"], label="Train")
    ax[0].plot(ep, h["val_loss"], label="Validación")
    ax[0].set(title="Loss (CrossEntropy)", xlabel="Época", ylabel="Loss")
    ax[1].plot(ep, h["train_acc"], label="Train")
    ax[1].plot(ep, h["val_acc"], label="Validación")
    ax[1].set(title="Accuracy", xlabel="Época", ylabel="Accuracy")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(f"Curvas de aprendizaje (semilla {SEED})")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicción", ylabel="Clase real",
           title=f"Matriz de confusión - Test (semilla {SEED})")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thr = cm.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thr else "black")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def diagnose(h, test_acc):
    """Diagnóstico cuantitativo de overfitting / underfitting."""
    tr_acc, va_acc = h["train_acc"][-1], h["val_acc"][-1]
    gap = tr_acc - va_acc
    best_ep = int(np.argmin(h["val_loss"])) + 1
    loss_gap = h["val_loss"][-1] - h["train_loss"][-1]
    if tr_acc < 0.70 and va_acc < 0.70:
        verdict = "UNDERFITTING: el modelo no alcanza buen ajuste ni en entrenamiento."
    elif gap > 0.10 or (h["val_loss"][-1] - min(h["val_loss"])) > 0.10:
        verdict = "OVERFITTING: brecha train-val significativa / la val loss sube tras su mínimo."
    elif gap > 0.05:
        verdict = "Sobreajuste LEVE: brecha moderada entre train y validación."
    else:
        verdict = "Buen ajuste: brecha train-val pequeña, sin síntomas relevantes de overfitting."
    return {"train_acc_final": tr_acc, "val_acc_final": va_acc, "test_acc": test_acc,
            "gap_acc_train_val": gap, "gap_loss_val_train": loss_gap,
            "epoca_min_val_loss": best_ep, "min_val_loss": min(h["val_loss"]), "veredicto": verdict}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)
    print(f"Dispositivo: {DEVICE} | Semilla: {SEED}")

    validate_forward_pass()  # Pregunta 3: validación de shapes antes de entrenar

    train_loader, val_loader, test_loader, _ = get_dataloaders(BATCH_SIZE)

    set_seed(SEED)  # pesos iniciales reproducibles
    model = CNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val, ckpt = 0.0, os.path.join(OUT_DIR, "best_model.pt")

    for epoch in range(EPOCHS):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer)
        vl, va = run_epoch(model, val_loader, criterion)
        for k, v in zip(hist, (tl, ta, vl, va)):
            hist[k].append(v)
        if va > best_val:
            best_val = va
            torch.save(model.state_dict(), ckpt)
        print(f"Época {epoch + 1:02d}/{EPOCHS} | train loss {tl:.4f} acc {ta:.4f} | "
              f"val loss {vl:.4f} acc {va:.4f}")

    # ------------- Evaluación final en TEST con el mejor checkpoint -------------
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.append(model(x.to(DEVICE)).argmax(1).cpu())
            labels.append(y)
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    test_acc = float((preds == labels).mean())
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=CLASSES, digits=4)
    print(f"\nAccuracy en TEST: {test_acc:.4f}\n{report}")

    plot_curves(hist, os.path.join(OUT_DIR, "loss_accuracy.png"))
    plot_confusion(cm, os.path.join(OUT_DIR, "confusion_matrix.png"))

    summary = diagnose(hist, test_acc)
    summary.update({"seed": SEED, "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
                    "history": hist})
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    print("\n=== DIAGNÓSTICO ===")
    for k in ("train_acc_final", "val_acc_final", "test_acc", "gap_acc_train_val",
              "gap_loss_val_train", "epoca_min_val_loss", "min_val_loss"):
        v = summary[k]
        print(f"{k:<20}: {v:.4f}" if isinstance(v, float) else f"{k:<20}: {v}")
    print(summary["veredicto"])
    print(f"\nArtefactos guardados en {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
