import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from data_pipeline import build_loaders, COMMANDS, RESULTS_DIR, SEED
from models.cnn import SpectrogramCNN

# --- Setup ---
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for spectrograms, labels in tqdm(loader, desc="Training", leave=False):
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(spectrograms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    return running_loss / len(loader), correct / total


def validate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for spectrograms, labels in tqdm(loader, desc="Validating", leave=False):
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            outputs = model(spectrograms)
            loss    = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

    return running_loss / len(loader), correct / total


def save_learning_curves(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label='Train loss')
    ax1.plot(val_losses,   label='Val loss')
    ax1.set_title('Loss curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(train_accs, label='Train acc')
    ax2.plot(val_accs,   label='Val acc')
    ax2.set_title('Accuracy curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'learning_curves.png')
    plt.close()
    print(f"Learning curves saved to {RESULTS_DIR}/learning_curves.png")


if __name__ == "__main__":
    # --- Load data ---
    print("Loading data...")
    train_loader, val_loader, test_loader = build_loaders()
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches  : {len(val_loader)}")
    print(f"  Test batches : {len(test_loader)}\n")

    # --- Build model ---
    model     = SpectrogramCNN(num_classes=len(COMMANDS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    # --- Training loop ---
    NUM_EPOCHS   = 15
    best_val_acc = 0
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    print("Starting training...\n")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc = validate(model, val_loader, criterion)

        scheduler.step(val_loss)

        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc);    val_accs.append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), RESULTS_DIR / 'best_model.pth')

        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} minutes.")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to {RESULTS_DIR}/best_model.pth")

    # --- Save learning curves ---
    save_learning_curves(train_losses, val_losses, train_accs, val_accs)

    # --- Run final evaluation on test set and save probabilities ---
    print("\nRunning final evaluation on test set...")
    model.load_state_dict(
        torch.load(RESULTS_DIR / 'best_model.pth', map_location=device)
    )
    model.eval()

    all_preds, all_true, all_probs = [], [], []
    with torch.no_grad():
        for spectrograms, labels in tqdm(test_loader, desc="Testing"):
            spectrograms = spectrograms.to(device)
            outputs      = model(spectrograms)
            probs        = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)
    all_probs = np.array(all_probs)

    # Save for eval.py and rl_agent.py to use
    np.save(RESULTS_DIR / 'test_probs.npy',  all_probs)
    np.save(RESULTS_DIR / 'test_labels.npy', all_true)
    np.save(RESULTS_DIR / 'test_preds.npy',  all_preds)
    print(f"Test probabilities saved to {RESULTS_DIR}/test_probs.npy")

    # --- Save training summary ---
    summary = {
        "best_val_accuracy" : float(best_val_acc),
        "epochs_trained"    : NUM_EPOCHS,
        "training_time_min" : round(elapsed / 60, 2),
        "optimizer"         : "Adam",
        "learning_rate"     : 0.001,
        "batch_size"        : 64,
        "seed"              : SEED,
        "device"            : str(device),
    }
    with open(RESULTS_DIR / 'train_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Training summary saved to {RESULTS_DIR}/train_summary.json")
    print("\ntrain.py complete.")
