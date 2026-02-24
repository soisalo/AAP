"""
Evaluation module for classification tasks.
This module provides accuracy computation, per-class accuracy, confusion matrix computation and plotting,
hierarchical evaluation metrics and full model evaluation pipeline

Author: Venla Numminen
email: venla.numminen@tuni.fi
"""
import numpy as np
import torch
import pickle
from dataset_classes import SimpleAudioCNN, NUM_CLASSES, CLASSES
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_classes import calculate_hierarchical_metrics


def compute_accuracy(predictions, targets):
    """
    Compute overall classification accuracy.
    Args:
        predictions: Model predicted class labels.
        targets: Ground truth class labels.
    Returns:
        Float accuracy value between 0 and 1.
    """

    predictions = np.array(predictions)
    targets = np.array(targets)

    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length.")

    return (predictions == targets).mean()


def compute_per_class_accuracy(predictions, targets, class_names):
    """
    Compute accuracy separately for each class.
    This measures how well the model performs on each individual class.
    Uses confusion matrix for calculation.
    Args:
        predictions: Model predictions.
        targets: Ground truth labels.
        class_names: List of class label names.
    Returns:
        Dictionary mapping class name -> accuracy.
        If a class has no samples, returns None for that class.
    """
    cm = confusion_matrix(targets, predictions)

    per_class = {
        class_names[i]: (
            cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else None
        )
        for i in range(len(class_names))
    }

    return per_class


def compute_confusion_matrix(predictions, targets):
    """Return confusion matrix."""
    return confusion_matrix(targets, predictions)


def evaluate_model(model, dataloader, device, class_names, lambda_val=0.5):
    """
    Evaluate a trained PyTorch model on a dataset.

    This function:
    - Runs inference on the entire dataset
    - Collects predictions
    - Computes all evaluation metrics
    - Returns them in a dictionary
    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader for validation/test data.
        device: CPU or CUDA device.
        class_names: List of class names.
        lambda_val: Weight parameter for hierarchical metric.
    Returns:
        Dictionary containing all evaluation results.
    """

    model.eval()
    predictions, targets = [], []

    with torch.no_grad():
        for specs, labels, _ in dataloader:
            specs = specs.to(device)
            labels = labels.to(device)

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

    This function is independent of PyTorch and can be used
    for any classification results.

    Metrics included:
    - Overall accuracy
    - Per-class accuracy
    - Confusion matrix
    - Precision / Recall / F1-score
    - Hierarchical metrics

    Returns:
        Dictionary with all computed metrics.
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
        predictions, targets
    )

    # Standard classification metrics
    results["classification_report"] = classification_report(
        targets,
        predictions,
        target_names=class_names,
        output_dict=True
    )

    # Hierarchical metrics 
    hF, hP, hR = calculate_hierarchical_metrics(
        predictions,
        targets,
        lambda_val=lambda_val
    )

    results["hierarchical_F"] = hF
    results["hierarchical_P"] = hP
    results["hierarchical_R"] = hR

    return results

def plot_confusion_matrix(cm, class_names,figsize=(12, 10), save_path=None):

    """
    Plot confusion matrix using a heatmap.
    Args:
        cm: Confusion matrix as a 2D numpy array.
        class_names: List of class names for axes labels.
        figsize: Size of the plot.
        save_path: If provided, saves the plot to this path.
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


if __name__ == "__main__":


    #results = evaluate_model(model, val_loader, device, class_names)

    #print(results["accuracy"])
    #print(results["hierarchical_F"])

    #plot_confusion_matrix(
        #results["confusion_matrix"],
        #class_names)

    pass