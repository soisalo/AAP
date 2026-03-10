"""
Evaluation module for classification tasks.
This module provides accuracy computation, per-class accuracy, confusion matrix computation and plotting,
hierarchical evaluation metrics.

Author: Venla Numminen
email: venla.numminen@tuni.fi
"""
from py_compile import main
import os
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from dataset_classes import (
    PickleAudioDataset,
    SimpleCLAPClassifier,
    NUM_CLASSES,
    CLASSES
)




def compute_accuracy(predictions, targets):
    """
    Compute overall classification accuracy.

    """
    # Convert to numpy arrays if they are lists
    predictions = np.array(predictions)
    targets = np.array(targets)
    # Check if lengths match
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length.")

    return (predictions == targets).mean()


def compute_per_class_accuracy(predictions, targets, class_names):
    """
    Compute accuracy separately for each class.
    This measures how well the model performs on each individual class.
    Uses confusion matrix for calculation.
    """
    #making confusion matrix
    cm = confusion_matrix(targets, predictions)

    # Calculate per-class accuracy using confusion matrix
    per_class = {
        class_names[i]: (
            cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else None
        )
        for i in range(len(class_names))
    }

    return per_class


def compute_confusion_matrix(targets, predictions):
    """Return confusion matrix."""
    return confusion_matrix(targets, predictions)

def compute_precision_recall_f1(cm, class_names):
    """"
    Compute precision, recall and F1-score for each class based on confusion matrix.
    """
    #making dictionary for metrics
    metrics_dict = {}
    # Loop through each class and calculate metrics
    for i, class_name in enumerate(class_names):
        true_positive = cm[i, i]
        false_positive = cm[:, i].sum() - true_positive
        false_negative = cm[i, :].sum() - true_positive

        #Calculate precision, recall and F1-score with checks to avoid division by zero
        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0 else 0
        )

        #Calculate recall with checks to avoid division by zero
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0 else 0
        )

        #Calculate F1-score with checks to avoid division by zero
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0
        )

        #Store metrics in dictionary
        metrics_dict[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    return metrics_dict


def evaluate_model(model, dataloader, device, class_names, lambda_val=0.5):
    """
    Evaluate a trained PyTorch model on a dataset.

    This function:
    - Runs inference on the entire dataset
    - Collects predictions
    - Computes all evaluation metrics
    - Returns them in a dictionary
    """
    # Set model to evaluation mode
    model.eval()
    predictions, targets = [], []

    # Run inference without gradient calculation
    with torch.no_grad():
        for batch in dataloader:
            specs = batch[0].to(device)
            labels = batch[1].to(device)
            
            outputs = model(specs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    return evaluate_predictions(
        predictions,
        targets,
        class_names ,
        lambda_val=lambda_val
    )



def evaluate_predictions(predictions, targets, class_names, lambda_val: float = 0.5):
    """
    Compute all evaluation metrics from predictions and targets.

    Metrics included:
    - Overall accuracy
    - Per-class accuracy
    - Confusion matrix
    - Hierarchical metrics
        - Precision / Recall / F1-score for each class

    """

    results = {}

    # Accuracy
    results["accuracy"] = compute_accuracy(predictions, targets)

    # Per-class accuracy
    results["per_class_accuracy"] = compute_per_class_accuracy(
        predictions, targets, class_names
    )

    # Confusion matrix
    results["confusion_matrix"] = compute_confusion_matrix(
        targets, predictions
    )

    # Hierarchical metrics - F1, Precision, Recall
    metrics = compute_precision_recall_f1(
        results["confusion_matrix"], class_names
    )

    results["metrics"] = metrics


    return results

def plot_confusion_matrix(cm, class_names,figsize=(12, 10), save_path=None):

    """
    Plot confusion matrix using a heatmap.
    """

    plt.figure(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()

def main():

    # Define test dataset and dataloader
    test_dir = "./dataset/test"

    # Collect all pickle files in the test directory
    test_files = [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.endswith(".pkl")
    ]

    #send the dataset to the PickleAudioDataset class
    test_dataset = PickleAudioDataset(test_files)

    # Create DataLoader for test dataset
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )
    # Set device for evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCLAPClassifier(num_classes=NUM_CLASSES)
    model.load_state_dict(
        torch.load("audio_cnn_model.pth", map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    # Evaluate the model and get results
    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=CLASSES
    )
    #plot confusion matrix and print metrics
    plot_confusion_matrix(
        results["confusion_matrix"],
        class_names=CLASSES,
        figsize=(12, 10),
        save_path="confusion_matrix.png"
    )

    print("Accuracy:", results["accuracy"])
    for class_name, scores in results["metrics"].items():
        print(f"  {class_name}:")
        print(f"    Precision: {scores['precision']:.4f}")
        print(f"    Recall:    {scores['recall']:.4f}")
        print(f"    F1-score:  {scores['f1_score']:.4f}")


if __name__ == "__main__":
    main()