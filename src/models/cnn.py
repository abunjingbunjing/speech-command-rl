import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from data_pipeline import COMMANDS, RESULTS_DIR, SEED

torch.manual_seed(SEED)


class SpectrogramCNN(nn.Module):
    """
    A small CNN that classifies speech commands from mel-spectrograms.
    Built from scratch as required by project specs.
    Input shape: [batch, 1, 64, 101] (mel-spectrogram image)
    """

    def __init__(self, num_classes=10):
        super(SpectrogramCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1,  32,  kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64,  kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.bn1     = nn.BatchNorm2d(32)
        self.bn2     = nn.BatchNorm2d(64)
        self.bn3     = nn.BatchNorm2d(128)

        # After 3 poolings: 64->32->16->8, 101->50->25->12
        self.fc1 = nn.Linear(128 * 8 * 12, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # → [b, 32,  32, 50]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # → [b, 64,  16, 25]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # → [b, 128,  8, 12]
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing SpectrogramCNN on {device}\n")

    # --- Build model ---
    model = SpectrogramCNN(num_classes=len(COMMANDS)).to(device)

    # --- Print architecture ---
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # --- Forward pass test with dummy input ---
    # Simulates one batch of 4 mel-spectrograms
    dummy_input = torch.randn(4, 1, 64, 101).to(device)
    print(f"\nDummy input shape : {dummy_input.shape}")

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output shape      : {output.shape}")   # should be [4, 10]
    print(f"Expected          : [4, {len(COMMANDS)}]")
    assert output.shape == (4, len(COMMANDS)), "Output shape mismatch!"
    print("\nForward pass test passed!")

    # --- Save model summary as JSON ---
    model_info = {
        "architecture"       : "SpectrogramCNN",
        "num_classes"        : len(COMMANDS),
        "input_shape"        : [1, 64, 101],
        "total_params"       : total_params,
        "trainable_params"   : trainable_params,
        "layers": {
            "conv1"  : "Conv2d(1, 32, kernel=3, padding=1) + BN + ReLU + MaxPool",
            "conv2"  : "Conv2d(32, 64, kernel=3, padding=1) + BN + ReLU + MaxPool",
            "conv3"  : "Conv2d(64, 128, kernel=3, padding=1) + BN + ReLU + MaxPool",
            "fc1"    : "Linear(128*8*12, 256) + Dropout(0.3) + ReLU",
            "fc2"    : "Linear(256, 10) — output logits",
        }
    }
    with open(RESULTS_DIR / "cnn_model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"\nModel info saved to {RESULTS_DIR}/cnn_model_info.json")
    print("cnn.py complete.")
