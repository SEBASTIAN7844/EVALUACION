"""
Pregunta 3 - Arquitectura CNN en PyTorch y validación del flujo de tensores 4D.
Formato de entrada: (Batch_Size, Channels, Height, Width) = (N, 3, 32, 32)
"""
import torch
import torch.nn as nn


class CNN(nn.Module):
    """3 bloques convolucionales (Conv2d -> BN -> ReLU -> MaxPool2d) + clasificador denso.

    Flujo de shapes para entrada (N, 3, 32, 32):
      Bloque 1: Conv2d(3 ->32,  3x3, pad=1) -> (N, 32, 32, 32) -> MaxPool2d(2) -> (N, 32, 16, 16)
      Bloque 2: Conv2d(32->64,  3x3, pad=1) -> (N, 64, 16, 16) -> MaxPool2d(2) -> (N, 64,  8,  8)
      Bloque 3: Conv2d(64->128, 3x3, pad=1) -> (N,128,  8,  8) -> MaxPool2d(2) -> (N,128,  4,  4)
      Flatten                                                   -> (N, 2048)
      Linear(2048->256) -> ReLU -> Dropout(0.5)                 -> (N, 256)
      Linear(256 -> 10)                                         -> (N, 10)   (logits)
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3, dropout: float = 0.5):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.block1 = block(in_channels, 32)
        self.block2 = block(32, 64)
        self.block3 = block(64, 128)

        # Conexión conv -> densa: Flatten + tamaño calculado dinámicamente
        self.flatten = nn.Flatten()
        flat_dim = self._infer_flat_dim(in_channels)

        self.fc1 = nn.Linear(flat_dim, 256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)

    def _infer_flat_dim(self, in_channels: int, size: int = 32) -> int:
        """Cálculo dinámico del tamaño de aplanado con un tensor 4D ficticio."""
        with torch.no_grad():
            was_training = self.training
            self.eval()  # evita que BatchNorm actualice estadísticas
            x = torch.zeros(1, in_channels, size, size)
            x = self.block3(self.block2(self.block1(x)))
            self.train(was_training)
        return x.numel()

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        assert x.ndim == 4, f"Se esperaba tensor 4D (N,C,H,W), llegó {tuple(x.shape)}"
        if verbose:
            print(f"{'Entrada':<30}: {tuple(x.shape)}")
        x = self.block1(x)
        if verbose:
            print(f"{'Bloque 1 (Conv-BN-ReLU-Pool)':<30}: {tuple(x.shape)}")
        x = self.block2(x)
        if verbose:
            print(f"{'Bloque 2 (Conv-BN-ReLU-Pool)':<30}: {tuple(x.shape)}")
        x = self.block3(x)
        if verbose:
            print(f"{'Bloque 3 (Conv-BN-ReLU-Pool)':<30}: {tuple(x.shape)}")
        x = self.flatten(x)
        if verbose:
            print(f"{'Flatten':<30}: {tuple(x.shape)}")
        x = self.dropout(self.relu(self.fc1(x)))
        if verbose:
            print(f"{'Linear 2048->256 + ReLU':<30}: {tuple(x.shape)}")
        x = self.fc2(x)
        if verbose:
            print(f"{'Linear 256->10 (logits)':<30}: {tuple(x.shape)}")
        return x


def validate_forward_pass(batch_size: int = 8, num_classes: int = 10) -> None:
    """Pasa un lote simulado por el modelo, imprime shapes y verifica con aserciones."""
    model = CNN(num_classes=num_classes)
    model.eval()
    dummy = torch.randn(batch_size, 3, 32, 32)
    print("=" * 58)
    print("VALIDACIÓN DEL FORWARD PASS (tensores 4D)")
    print("=" * 58)
    with torch.no_grad():
        out = model(dummy, verbose=True)
    assert out.shape == (batch_size, num_classes), f"Shape de salida inválido: {tuple(out.shape)}"
    assert model.fc1.in_features == 128 * 4 * 4, "Dimensión de Flatten inconsistente"
    n_params = sum(p.numel() for p in model.parameters())
    print("-" * 58)
    print(f"OK -> salida {tuple(out.shape)} == ({batch_size}, {num_classes}) | parámetros: {n_params:,}")


if __name__ == "__main__":
    torch.manual_seed(75)
    validate_forward_pass()
