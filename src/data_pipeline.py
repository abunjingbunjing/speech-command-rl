import os
import random
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

COMMANDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
DATA_DIR  = Path('/content/speech_commands')
RESULTS_DIR = Path('experiments/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class SpeechCommandDataset(Dataset):
    def __init__(self, file_paths, labels, target_length=101):
        self.file_paths = file_paths
        self.labels = labels
        self.target_length = target_length
        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160, n_mels=64
        )
        self.db_transform = T.AmplitudeToDB()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.file_paths[idx])

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        if waveform.shape[1] < 16000:
            waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
        else:
            waveform = waveform[:, :16000]

        mel = self.mel_transform(waveform)
        mel = self.db_transform(mel)
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
        return mel, self.labels[idx]


def build_loaders(data_dir=DATA_DIR, max_per_class=800, batch_size=64):
    label_to_idx = {cmd: i for i, cmd in enumerate(COMMANDS)}
    all_files, all_labels = [], []

    for cmd in COMMANDS:
        files = list((data_dir / cmd).glob('*.wav'))[:max_per_class]
        for f in files:
            all_files.append(str(f))
            all_labels.append(label_to_idx[cmd])

    X_train, X_temp, y_train, y_temp = train_test_split(
        all_files, all_labels, test_size=0.30,
        random_state=SEED, stratify=all_labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50,
        random_state=SEED, stratify=y_temp
    )

    train_loader = DataLoader(SpeechCommandDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(SpeechCommandDataset(X_val,   y_val),
                              batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(SpeechCommandDataset(X_test,  y_test),
                              batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import subprocess

    # --- Download dataset if not already present ---
    if not DATA_DIR.exists():
        print("Downloading Google Speech Commands dataset...")
        subprocess.run([
            "wget", "-q",
            "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
        ], check=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "tar", "-xzf", "speech_commands_v0.02.tar.gz", "-C", str(DATA_DIR)
        ], check=True)
        print("Download complete.")
    else:
        print("Dataset already exists, skipping download.")

    # --- Verify folders ---
    print("\nVerifying command folders...")
    for cmd in COMMANDS:
        count = len(list((DATA_DIR / cmd).glob('*.wav')))
        print(f"  {cmd}: {count} files")

    # --- Build loaders and save split info ---
    print("\nBuilding data loaders...")
    train_loader, val_loader, test_loader = build_loaders()
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")

    # --- Save split sizes for other scripts to read ---
    split_info = {
        "train_size": len(train_loader.dataset),
        "val_size"  : len(val_loader.dataset),
        "test_size" : len(test_loader.dataset),
        "commands"  : COMMANDS,
        "seed"      : SEED,
    }
    import json
    with open(RESULTS_DIR / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nSplit info saved to {RESULTS_DIR}/split_info.json")
    print("data_pipeline.py complete.")
