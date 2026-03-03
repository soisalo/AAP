from sklearn.utils import compute_class_weight
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import pickle
from dataset_classes import PickleAudioDataset, SimpleCLAPClassifier, calculate_hierarchical_metrics, NUM_CLASSES, IDX_PARENT_MAP
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

def hierarchical_loss(outputs, targets, mapping_tensor, penalty_weight=2.0):
    """
    outputs: [batch, NUM_CLASSES]
    targets: [batch] (the sub-class indices)
    """

    # 1. Get the predicted class index (highest logit)
    _, preds = torch.max(outputs, 1)

    # 2. Look up Parent IDs for both Predictions and Targets
    # This is a vectorized operation on the GPU
    pred_parents = mapping_tensor[preds]
    target_parents = mapping_tensor[targets]

    # 3. Apply the logic:
    # If parents are DIFFERENT: Apply penalty_weight
    # If parents are THE SAME: Apply 0.0 (per your request)
    mask = torch.where(pred_parents != target_parents, 
                       float(penalty_weight), 
                       0.0)
    
    return mask

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
    model = SimpleCLAPClassifier(num_classes=NUM_CLASSES, num_parents=5).to(device)

    # Collect all labels first
    all_labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])

    # Include class weights to handle class imbalance
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
    class_weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Metrics lists
    train_losses, val_losses, val_accuracies, val_hFs = [], [], [], []

    print(f"Training on {device}...")

    num_epochs = 20
    patience = 5

    # Create the mapping tensor from your dictionary
    # Index = Child Class ID, Value = Parent Class ID
    mapping_list = [IDX_PARENT_MAP[i] for i in range(len(IDX_PARENT_MAP))]
    mapping_tensor = torch.tensor(mapping_list).to(device)

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        running_loss = 0.0
        
        for specs, labels,  weights, top_classes in train_loader:
            #print(f"Labels: {labels}, {type(labels)}")
            #print(f"specs shape: {specs.shape}, {type(specs)}")
            specs = specs.to(device).squeeze(1)  # Add channel dimension if needed
            labels = labels.to(device)
            weights = weights.to(device)
            top_classes = top_classes.to(device)

            #Zero the gradients
            optimizer.zero_grad()

            #New implementation with hierarchical loss and confidence weighting:
            child_out, parent_out = model(specs)     

            # 1. Child Loss (Weighted by confidence and hierarchy)
            raw_child_loss = criterion(child_out, labels)
            h_penalty = hierarchical_loss(child_out, labels, mapping_tensor)

            # Add 1.0 to penalty so it scales up errors (e.g., * 2.0) rather than zeroing them
            child_loss = (raw_child_loss * weights * (1.0 + h_penalty)).mean()

            # 2. Parent Loss (Helping the model learn the broad category)
            parent_loss = criterion(parent_out, top_classes).mean()

            # 3. Total Loss (Alpha=0.3 is a good starting point for the parent auxiliary task)
            total_loss = child_loss + 0.3 * parent_loss

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # --- Validation ---
        model.eval()
        val_preds, val_targets = [], []
        epoch_val_loss = 0.0
        with torch.no_grad():
            for specs, labels, weights, top_classes in val_loader:
                specs = specs.to(device).squeeze(1)
                labels = labels.to(device)  # Use sub_labels (second column)
                weights = weights.to(device)
                #outputs = model(specs)

                child_out, _ = model(specs)
                v_loss = criterion(child_out, labels).mean()
                epoch_val_loss += v_loss.item()

                _, predicted = torch.max(child_out, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

                #Implement patience-based early stopping based on validation accuracy
                if epoch > 0 and val_acc < val_accuracies[-1]:
                    patience_count += 1
                    if patience_count >= patience:  # Stop if no improvement for 5 epochs
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break
                else:
                    patience_count = 0
                

        val_acc = np.mean(np.array(val_preds) == np.array(val_targets))
        val_loss = epoch_val_loss / len(val_loader)
        hF, hP, hR = calculate_hierarchical_metrics(val_preds, val_targets, lambda_val=0.5)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)
        val_hFs.append(hF)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"hF: {hF:.4f} (P:{hP:.4f}, R:{hR:.4f})")

    # --- 5. Save trained model ---
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # --- 6. Plot metrics ---
    plot_training_metrics(train_losses, val_accuracies, val_losses, val_hFs, save_path="training_metrics.png")


def main():
    
    train_model_pickle()

if __name__ == "__main__":
    main()