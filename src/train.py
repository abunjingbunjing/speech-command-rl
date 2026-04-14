import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

NUM_EPOCHS = 15
best_val_acc = 0
train_losses, val_losses, train_accs, val_accs = [], [], [], []

print("Starting training...\n")
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    # --- Training phase ---
    model.train()
    running_loss, correct, total = 0, 0, 0

    for spectrograms, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()           # clear old gradients
        outputs = model(spectrograms)   # forward pass
        loss = criterion(outputs, labels)
        loss.backward()                 # compute gradients
        optimizer.step()                # update weights

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # --- Validation phase ---
    model.eval()
    val_loss_sum, val_correct, val_total = 0, 0, 0

    with torch.no_grad():  # no gradient computation needed for evaluation
        for spectrograms, labels in val_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            val_loss_sum += loss.item()
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss_sum / len(val_loader)
    val_acc  = val_correct / val_total

    train_losses.append(train_loss); val_losses.append(val_loss)
    train_accs.append(train_acc);     val_accs.append(val_acc)
    scheduler.step(val_loss)

    # Save the best model (early stopping logic)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')

    print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

elapsed = time.time() - start_time
print(f"\nTraining complete in {elapsed/60:.1f} minutes. Best val acc: {best_val_acc:.4f}")
