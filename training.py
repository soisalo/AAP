import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from dataset_classes import PickleAudioDataset, SimpleAudioCNN, calculate_hierarchical_metrics, NUM_CLASSES

def train_model_pickle(train_dir="./data/train_pkl", val_dir="./data/val_pkl", save_path="audio_cnn_model.pth"):
    # Collect .pkl files
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".pkl")]
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith(".pkl")]

    # Dummy files if none exist
    if len(train_files) == 0:
        print("Warning: No pickle files found. Creating dummy tensors for demonstration.")
        dummy_mel = torch.randn(1, 64, 431)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        train_files = [os.path.join(train_dir, f"dummy_{i}.pkl") for i in range(100)]
        val_files = [os.path.join(val_dir, f"dummy_{i}.pkl") for i in range(20)]
        for f in train_files:
            with open(f, "wb") as pf:
                pickle.dump({'mel': dummy_mel, 'label': np.random.randint(0, NUM_CLASSES)}, pf)
        for f in val_files:
            with open(f, "wb") as pf:
                pickle.dump({'mel': dummy_mel, 'label': np.random.randint(0, NUM_CLASSES)}, pf)

    train_dataset = PickleAudioDataset(train_files)
    val_dataset = PickleAudioDataset(val_files)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleAudioCNN(num_classes=NUM_CLASSES).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training on {device}...")

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(device), labels.to(device)
                outputs = model(specs)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_acc = np.mean(np.array(val_preds) == np.array(val_targets))
        hF, hP, hR = calculate_hierarchical_metrics(val_preds, val_targets, lambda_val=0.5)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f} | hF: {hF:.4f} (P:{hP:.4f}, R:{hR:.4f})")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_model_pickle()
