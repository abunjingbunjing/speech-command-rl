from sklearn.metrics import classification_report, confusion_matrix

# Load best model weights
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

all_preds, all_true, all_probs = [], [], []

with torch.no_grad():
    for spectrograms, labels in test_loader:
        spectrograms = spectrograms.to(device)
        outputs = model(spectrograms)
        probs = torch.softmax(outputs, dim=1)  # convert logits to probabilities
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_true.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_true  = np.array(all_true)
all_probs = np.array(all_probs)

print("=== Classification Report ===")
print(classification_report(all_true, all_preds, target_names=COMMANDS))

# Confusion matrix
cm = confusion_matrix(all_true, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=COMMANDS, yticklabels=COMMANDS, cmap='Blues')
plt.title('Confusion matrix'); plt.ylabel('True label'); plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
