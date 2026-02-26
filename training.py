import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from dataset_classes import PickleAudioDataset, SimpleCLAPClassifier, calculate_hierarchical_metrics, NUM_CLASSES
from plotting_utils import plot_training_metrics

"""
Author: Eemil Soisalo
email: eemil.soisalo@tuni.fi

This script trains a simple CNN model on precomputed Mel spectrograms stored in .pkl files.
The training process includes:
- Loading datasets from specified directories
- Training the model for a set number of epochs
- Evaluating on a validation set each epoch
- Saving the trained model to disk
- Plotting training loss and validation metrics over epochs

2026-06-01: Initial version created.
"""
def hierarchical_loss(outputs, targets, child_to_parent_idx, penalty_weight=2.0):
    """
    outputs: [batch, NUM_CLASSES]
    targets: [batch] (the sub-class indices)
    """
    # 1. Get the standard per-sample loss
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    base_loss = criterion(outputs, targets)

    # 2. Get the predicted class index
    _, preds = torch.max(outputs, 1)

    # 3. Get Parent IDs for both predictions and targets
    pred_parents = child_to_parent_idx[preds]
    target_parents = child_to_parent_idx[targets]

    # 4. Determine where the top-category is wrong
    # True if parents are different, False if they are the same
    wrong_parent_mask = (pred_parents != target_parents).float()

    # 5. Apply Logic: 
    # If same parent: loss becomes 0 (as requested)
    # If different parent: apply a penalty multiplier
    final_loss = base_loss * wrong_parent_mask * penalty_weight

    return final_loss.mean()

def train_model_pickle(train_dir="./dataset/train", val_dir="./dataset/val", save_path="audio_cnn_model.pth"):
    # --- 1. Collect .pkl files ---
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".pkl")]
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith(".pkl")]

    # --- 2. Dummy files if none exist ---
    if len(train_files) == 0:
        print("Warning: No pickle files found. Creating dummy tensors for demonstration.")
        dummy_mel = torch.randn(1, 64, 431)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        train_files = [os.path.join(train_dir, f"dummy_{i}.pkl") for i in range(100)]
        val_files = [os.path.join(val_dir, f"dummy_{i}.pkl") for i in range(20)]
        for f in train_files:
            with open(f, "wb") as pf:
                pickle.dump({'mel': dummy_mel, 'label': np.random.randint(0, NUM_CLASSES), 'confidence': np.random.randint(1,6)}, pf)
        for f in val_files:
            with open(f, "wb") as pf:
                pickle.dump({'mel': dummy_mel, 'label': np.random.randint(0, NUM_CLASSES), 'confidence': np.random.randint(1,6)}, pf)

    # --- 3. Create datasets and loaders ---
    train_dataset = PickleAudioDataset(train_files)
    val_dataset = PickleAudioDataset(val_files)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # --- 4. Setup model, device, optimizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCLAPClassifier(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Metrics lists
    train_losses, val_accuracies, val_hFs = [], [], []

    print(f"Training on {device}...")

    num_epochs = 10
    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        running_loss = 0.0
        top_class_penalty = 1.5
        
        for specs, labels,  weights, top_classes in train_loader:
            #print(f"Labels: {labels}, {type(labels)}")
            #print(f"specs shape: {specs.shape}, {type(specs)}")
            specs = specs.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            top_classes = top_classes.to(device)

            #Zero the gradients
            optimizer.zero_grad()

            outputs = model(specs)
            
            # 2. Get the predicted SUB-class index
            _, predicted_sub_indices = torch.max(outputs, 1)

            # 3. Map the PREDICTED sub-class to its PARENT class
            # We still need that mapping table we built earlier for the model's prediction
            pred_top_classes = child_to_parent_idx[predicted_sub_indices]

            loss = torch.nn.CrossEntropyLoss(reduction='none')(outputs, labels)

            #Give a bigger penalty if the predicted top-level class is wrong
            hierarchy_mask = (pred_top_classes != top_classes).float()
            
            # 1 if same parent, penalty if different
            hierarchy_penalty = hierarchy_mask * top_class_penalty + (1 - hierarchy_mask)  

            #Apply the confidence-based weights and the hierarchy penalty to the loss
            weighted_loss = (loss * weights * hierarchy_penalty).mean()
            
            weighted_loss.backward()
            optimizer.step()

            running_loss += weighted_loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # --- Validation ---
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for specs, labels, _, top_classes in val_loader:
                specs = specs.to(device)
                labels = labels.to(device)  # Use sub_labels (second column)
                outputs = model(specs)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_acc = np.mean(np.array(val_preds) == np.array(val_targets))
        hF, hP, hR = calculate_hierarchical_metrics(val_preds, val_targets, lambda_val=0.5)
        val_accuracies.append(val_acc)
        val_hFs.append(hF)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"hF: {hF:.4f} (P:{hP:.4f}, R:{hR:.4f})")

    # --- 5. Save trained model ---
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # --- 6. Plot metrics ---
    plot_training_metrics(train_losses, val_accuracies, val_hFs, save_path="training_metrics.png")


if __name__ == "__main__":
    train_model_pickle()
