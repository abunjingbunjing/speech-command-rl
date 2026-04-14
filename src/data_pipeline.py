# Install lib
!pip install torchaudio librosa seaborn -q

# lib
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import random

# Seed
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print("PyTorch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# data prep
!wget -q https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
!mkdir -p /content/speech_commands
!tar -xzf speech_commands_v0.02.tar.gz -C /content/speech_commands
print("Done! Dataset downloaded.")

# commands available
import os
commands = [d for d in os.listdir('/content/speech_commands')
            if os.path.isdir(f'/content/speech_commands/{d}')
            and not d.startswith('_')]
print(f"Found {len(commands)} command folders:")
print(sorted(commands))

# chosen commands
COMMANDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
DATA_DIR = Path('/content/speech_commands')
print("We will classify these commands:", COMMANDS)

# sample check per command
counts = {}
for cmd in COMMANDS:
    files = list((DATA_DIR / cmd).glob('*.wav'))
    counts[cmd] = len(files)
    print(f"{cmd}: {len(files)} samples")

# Plot class distribution
plt.figure(figsize=(10, 4))
plt.bar(counts.keys(), counts.values(), color='steelblue')
plt.title('Sample count per command class')
plt.xlabel('Command')
plt.ylabel('Number of audio files')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()
print("Class distribution looks balanced — good!")

import librosa
import librosa.display

sample_path = list((DATA_DIR / 'yes').glob('*.wav'))[0]
waveform, sample_rate = torchaudio.load(sample_path)
print(f"Sample rate: {sample_rate} Hz")
print(f"Duration: {waveform.shape[1]/sample_rate:.2f} seconds")
print(f"Shape: {waveform.shape}")  # [channels, samples]

# Plot the waveform
plt.figure(figsize=(10, 3))
plt.plot(waveform[0].numpy())
plt.title('Waveform of "yes"')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()

# CONVERSION TO MEL-SPECTOGRAM
mel_transform = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    hop_length=160,
    n_mels=64
)
db_transform = T.AmplitudeToDB()

mel_spec = mel_transform(waveform)
mel_db = db_transform(mel_spec)

plt.figure(figsize=(8, 4))
plt.imshow(mel_db[0].numpy(), origin='lower', aspect='auto', cmap='viridis')
plt.title('Mel-spectrogram of "yes" (this is what the CNN sees)')
plt.xlabel('Time frames')
plt.ylabel('Mel frequency')
plt.colorbar(label='dB')
plt.show()

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class SpeechCommandDataset(Dataset):
    """Loads audio files and converts them to mel-spectrograms."""

    def __init__(self, file_paths, labels, transform=None, target_length=101):
        self.file_paths = file_paths
        self.labels = labels
        self.target_length = target_length  # standardize time dimension
        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160, n_mels=64
        )
        self.db_transform = T.AmplitudeToDB()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load audio
        waveform, sr = torchaudio.load(self.file_paths[idx])

        # Make sure it's mono (1 channel) and 16kHz
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        # Pad or trim to exactly 1 second (16000 samples)
        if waveform.shape[1] < 16000:
            waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
        else:
            waveform = waveform[:, :16000]

        # Convert to mel-spectrogram
        mel = self.mel_transform(waveform)
        mel = self.db_transform(mel)

        # Normalize to [0, 1]
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)

        return mel, self.labels[idx]

# Collect all file paths and labels
all_files = []
all_labels = []
label_to_idx = {cmd: i for i, cmd in enumerate(COMMANDS)}

for cmd in COMMANDS:
    files = list((DATA_DIR / cmd).glob('*.wav'))[:800]  # 800 per class for speed
    for f in files:
        all_files.append(str(f))
        all_labels.append(label_to_idx[cmd])

print(f"Total samples: {len(all_files)}")

# Split into train (70%), val (15%), test (15%) — stratified so classes are balanced
X_train, X_temp, y_train, y_temp = train_test_split(
    all_files, all_labels, test_size=0.30, random_state=SEED, stratify=all_labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Create datasets and dataloaders
train_dataset = SpeechCommandDataset(X_train, y_train)
val_dataset   = SpeechCommandDataset(X_val, y_val)
test_dataset  = SpeechCommandDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=2)

print("Data loaders ready!")
