import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import pickle
from dataset_classes import PickleAudioDataset, SimpleCLAPClassifier, calculate_hierarchical_metrics, NUM_CLASSES, IDX_PARENT_MAP
from plotting_utils import plot_training_metrics

"""
Authors: Eemil Soisalo, Tommi Salonen, Venla Numminen
email: eemil.soisalo@tuni.fi, tommi.salonen@tuni.fi, venla.numminen@tuni.fi

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
    Custom loss function that applies a penalty when the predicted class and true class belong to different parent categories.
    outputs: [batch, NUM_CLASSES]
    targets: [batch] (the sub-class indices)
    """

    #Get the predicted class index (highest logit)
    _, preds = torch.max(outputs, 1)

    # Look up Parent IDs for both Predictions and Targets
    pred_parents = mapping_tensor[preds]
    target_parents = mapping_tensor[targets]

    # 3. Apply the logic:
    # If parents are DIFFERENT: Apply penalty_weight
    # If parents are THE SAME: Apply 0.0
    mask = torch.where(pred_parents != target_parents, 
                       float(penalty_weight), 
                       0.0)
    
    return mask

def train_model_pickle(train_dir="./dataset/train", val_dir="./dataset/val", save_path="audio_cnn_model.pth"):
    """
    Trains a simple CNN model on precomputed Mel spectrograms stored in .pkl files.
    The training process includes:
        - Loading datasets from specified directories
        - Training the model for a set number of epochs
        - Evaluating on a validation set each epoch
        - Saving the trained model to disk
        - Plotting training loss and validation metrics over epochs

        :param train_dir: Directory containing the training .pkl files.
        :type train_dir: str
        :param val_dir: Directory containing the validation .pkl files.
        :type val_dir: str
        :param save_path: Path to save the trained model.
        :type save_path: str
    """
    #Collect .pkl files from the training and validation directories
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".pkl")]
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith(".pkl")]

    #Create datasets and loaders
    train_dataset = PickleAudioDataset(train_files)
    val_dataset = PickleAudioDataset(val_files)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    #Setup model, device, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCLAPClassifier(num_classes=NUM_CLASSES, num_parents=5).to(device)

    # For simplicity, we will use unweighted loss functions for both child and parent classifications
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    #Learning rate scheduler that reduces LR by half if validation accuracy doesn't improve for 2 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    #Metrics lists for plotting
    train_losses, val_losses, val_accuracies, val_hFs, val_parent_accuracies = [], [], [], [], []

    print(f"Training on {device}...")

    #Hyperparameters for training
    num_epochs = 20
    patience = 5
    best_val_acc = 0.0
    patience_count = 0

    # Create the mapping tensor from your dictionary
    # Index = Child Class ID, Value = Parent Class ID
    mapping_list = [IDX_PARENT_MAP[i] for i in range(len(IDX_PARENT_MAP))]
    mapping_tensor = torch.tensor(mapping_list).to(device)

    for epoch in range(num_epochs):

        # --- Training ---
        model.train()
        running_loss = 0.0
        
        for specs, labels,  weights, top_classes in train_loader:
            specs = specs.to(device).squeeze(1)
            labels = labels.to(device)
            weights = weights.to(device)
            top_classes = top_classes.to(device)

            #Zero the gradients and perform a forward pass
            optimizer.zero_grad()
            child_out, parent_out = model(specs)     

            # Child Loss (Weighted by confidence and hierarchy)
            raw_child_loss = criterion(child_out, labels)
            h_penalty = hierarchical_loss(child_out, labels, mapping_tensor)

            # Apply confidence weights and hierarchical penalty to the child loss
            child_loss = (raw_child_loss * weights * (1.0 + h_penalty)).mean()

            # Parent Loss (Broad category)
            parent_loss = criterion(parent_out, top_classes).mean()

            # Total Loss with a weight on the parent loss (e.g., 0.5)
            total_loss = child_loss + 0.5 * parent_loss

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # --- Validation ---
        model.eval()
        val_preds, val_targets = [], []
        val_parent_preds, val_parent_targets = [], []
        epoch_val_loss = 0.0
        with torch.no_grad():
            for specs, labels, weights, top_classes in val_loader:
                specs = specs.to(device).squeeze(1)
                labels = labels.to(device)  # Use sub_labels (second column)
                weights = weights.to(device)
                top_classes = top_classes.to(device)  # Use top_classes (third column)

                child_out, parent_out = model(specs)
                v_loss = criterion(child_out, labels).mean()
                epoch_val_loss += v_loss.item()
                
                #Child predictions
                _, predicted = torch.max(child_out, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

                #Parent predictions
                _, predicted_parent = torch.max(parent_out, 1)
                val_parent_preds.extend(predicted_parent.cpu().numpy())
                val_parent_targets.extend(top_classes.cpu().numpy())

               
                
        #Validation metrics
        val_acc = np.mean(np.array(val_preds) == np.array(val_targets))
        val_loss = epoch_val_loss / len(val_loader)
        val_parent_acc = np.mean(np.array(val_parent_preds) == np.array(val_parent_targets)) # NEW

        hF, hP, hR = calculate_hierarchical_metrics(val_preds, val_targets, lambda_val=0.5)
        
        # Append metrics to lists for plotting
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)
        val_parent_accuracies.append(val_parent_acc)
        val_hFs.append(hF)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Child Acc: {val_acc:.4f} | "
              f"Val Parent Acc: {val_parent_acc:.4f} | "
              f"hF: {hF:.4f} (P:{hP:.4f}, R:{hR:.4f})")
        
        #Step the scheduler with the validation accuracy, reduces learning rate if it doesn't improve for 2 epochs
        scheduler.step(val_acc)

        #Early Stopping Logic (Patience)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_count = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # --- Save trained model ---
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # --- Plot metrics ---
    plot_training_metrics(train_losses, val_accuracies, val_losses, val_hFs, save_path="training_metrics.png")


def main():
    
    train_model_pickle()

if __name__ == "__main__":
    main()