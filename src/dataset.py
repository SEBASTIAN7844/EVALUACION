"""
Pregunta 1 - Carga, visualización, transformaciones y partición estratificada de CIFAR-10.
Semilla individual: código de matrícula 022100575k -> últimos dos dígitos = 75
"""
import os
import random

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # permite guardar figuras sin pantalla (el notebook usa plt.show())
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ----------------------------------------------------------------------------
# 1. SEMILLA INDIVIDUAL
# ----------------------------------------------------------------------------
SEED = 75

CLASSES = ["avion", "automovil", "pajaro", "gato", "ciervo",
           "perro", "rana", "caballo", "barco", "camion"]

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def set_seed(seed: int = SEED) -> None:
    """Fija todas las fuentes de aleatoriedad (mezcla, partición, pesos iniciales)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id: int) -> None:
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)


# ----------------------------------------------------------------------------
# 3. PIPELINE DE TRANSFORMACIONES
# ----------------------------------------------------------------------------
def get_transforms():
    """Devuelve (train_tf, eval_tf).

    train_tf: Data Augmentation (RandomCrop + RandomHorizontalFlip) + ToTensor + Normalize.
    eval_tf : solo ToTensor + Normalize (validación/test NO se aumentan).
    """
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),                          # (H,W,C) uint8 -> (C,H,W) float [0,1]
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),    # normalización por canal RGB
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_tf, eval_tf


# ----------------------------------------------------------------------------
# 4. PARTICIÓN ESTRATIFICADA
# ----------------------------------------------------------------------------
def stratified_split(targets, val_ratio: float = 0.1, seed: int = SEED):
    """Partición estratificada: cada clase aporta la misma proporción a validación."""
    rng = np.random.RandomState(seed)
    targets = np.asarray(targets)
    train_idx, val_idx = [], []
    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_ratio))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


# ----------------------------------------------------------------------------
# 1. CARGA
# ----------------------------------------------------------------------------
def get_dataloaders(batch_size: int = 128, val_ratio: float = 0.1,
                    num_workers: int = 2, data_dir: str = DATA_DIR, verbose: bool = True):
    set_seed(SEED)
    train_tf, eval_tf = get_transforms()

    # Dos vistas del train set: una con augmentation (train) y otra sin (validación)
    full_train_aug = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    full_train_eval = datasets.CIFAR10(data_dir, train=True, download=True, transform=eval_tf)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_tf)

    train_idx, val_idx = stratified_split(full_train_aug.targets, val_ratio, SEED)
    train_set = Subset(full_train_aug, train_idx)
    val_set = Subset(full_train_eval, val_idx)

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=g,
                              num_workers=num_workers, worker_init_fn=_worker_init_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if verbose:
        print(f"[SEED = {SEED}]")
        print(f"Entrenamiento : {len(train_set):>6} imágenes -> tensor (N, 3, 32, 32)")
        print(f"Validación    : {len(val_set):>6} imágenes -> tensor (N, 3, 32, 32)")
        print(f"Prueba (test) : {len(test_set):>6} imágenes -> tensor (N, 3, 32, 32)")
        xb, yb = next(iter(train_loader))
        print(f"Batch de entrenamiento: X {tuple(xb.shape)}  y {tuple(yb.shape)}")
        counts = np.bincount(np.asarray(full_train_aug.targets)[val_idx], minlength=10)
        print("Imágenes por clase en validación (estratificado):", counts.tolist())

    return train_loader, val_loader, test_loader, (train_idx, val_idx)


# ----------------------------------------------------------------------------
# 2. VISUALIZACIÓN
# ----------------------------------------------------------------------------
def visualize_samples(n: int = 16, save_path: str = None, data_dir: str = DATA_DIR):
    """Cuadrícula de n imágenes (sin normalizar) con su etiqueta real."""
    set_seed(SEED)
    raw = datasets.CIFAR10(data_dir, train=True, download=True)  # imágenes PIL originales
    idx = np.random.RandomState(SEED).choice(len(raw), n, replace=False)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows))
    for ax, i in zip(np.ravel(axes), idx):
        img, label = raw[i]
        ax.imshow(img)
        ax.set_title(f"Real: {CLASSES[label]}", fontsize=10)
        ax.axis("off")
    for ax in np.ravel(axes)[len(idx):]:
        ax.axis("off")
    fig.suptitle(f"Muestras de CIFAR-10 (semilla {SEED})", fontsize=13)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    return fig


if __name__ == "__main__":
    get_dataloaders()
    visualize_samples(16, save_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "outputs", "samples_grid.png"))
    print("Cuadrícula guardada en outputs/samples_grid.png")
