import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# --- Import from sibling files ---
import sys
sys.path.append(str(Path(__file__).parent))
from data_pipeline import build_loaders, COMMANDS, RESULTS_DIR, SEED
from models.cnn import SpectrogramCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, test_loader):
    """Run model on test set, return predictions and probabilities."""
    model.eval()
    all_preds, all_true, all_probs = [], [], []

    with torch.no_grad():
        for spectrograms, labels in test_loader:
            spectrograms = spectrograms.to(device)
            outputs = model(spectrograms)
            probs   = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return (np.array(all_preds),
            np.array(all_true),
            np.array(all_probs))


def save_confusion_matrix(all_true, all_preds, save_path):
    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=COMMANDS, yticklabels=COMMANDS, cmap='Blues')
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def save_per_class_accuracy(all_true, all_preds, save_path):
    per_class_acc = {}
    for i, cmd in enumerate(COMMANDS):
        mask = (all_true == i)
        per_class_acc[cmd] = (all_preds[mask] == all_true[mask]).mean()

    # Bar chart
    plt.figure(figsize=(10, 4))
    plt.bar(per_class_acc.keys(), per_class_acc.values(), color='steelblue')
    plt.ylim(0, 1)
    plt.title('Per-class accuracy (error/slice analysis)')
    plt.xlabel('Command')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Per-class accuracy plot saved to {save_path}")
    return per_class_acc


if __name__ == "__main__":
    import json

    # --- Load data ---
    print("Loading test data...")
    _, _, test_loader = build_loaders()

    # --- Load trained model ---
    model_path = RESULTS_DIR / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found at {model_path}. Run train.py first."
        )

    print(f"Loading model from {model_path}...")
    model = SpectrogramCNN(num_classes=len(COMMANDS)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # --- Run evaluation ---
    print("Running evaluation on test set...")
    all_preds, all_true, all_probs = evaluate_model(model, test_loader)

    # --- Print classification report ---
    print("\n=== Classification Report ===")
    print(classification_report(all_true, all_preds, target_names=COMMANDS))

    # --- Save plots ---
    save_confusion_matrix(all_true, all_preds,
                          RESULTS_DIR / "confusion_matrix.png")
    per_class_acc = save_per_class_accuracy(all_true, all_preds,
                                            RESULTS_DIR / "per_class_accuracy.png")

    # --- Save probabilities and labels for rl_agent.py to use ---
    np.save(RESULTS_DIR / "test_probs.npy",  all_probs)
    np.save(RESULTS_DIR / "test_labels.npy", all_true)
    np.save(RESULTS_DIR / "test_preds.npy",  all_preds)
    print(f"\nTest probabilities saved to {RESULTS_DIR}/test_probs.npy")
    print("(rl_agent.py will load these automatically)")

    # --- Save metrics summary as JSON ---
    from sklearn.metrics import accuracy_score, f1_score
    metrics = {
        "accuracy"       : float(accuracy_score(all_true, all_preds)),
        "macro_f1"       : float(f1_score(all_true, all_preds, average='macro')),
        "per_class_acc"  : {k: float(v) for k, v in per_class_acc.items()},
    }
    with open(RESULTS_DIR / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics summary saved to {RESULTS_DIR}/eval_metrics.json")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Macro F1 : {metrics['macro_f1']:.4f}")
    print("eval.py complete.")
