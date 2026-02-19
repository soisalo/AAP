"""
Evaluation module for classification tasks.

Author: Venla Numminen
email: venla.numminen@tuni.fi
"""
import numpy as np
from training import train_model_pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Compute overall accuracy
def compute_overall_accuracy(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)
    return (predictions == targets).mean()



# Compute per-class accuracy
def compute_per_class_accuracy(predictions, targets, class_names):
    predictions = np.array(predictions)
    targets = np.array(targets)

    per_class = {}
    for i, cls in enumerate(class_names):
        idx = np.where(targets == i)[0]
        if len(idx) == 0:
            per_class[cls] = None
        else:
            per_class[cls] = (predictions[idx] == targets[idx]).mean()
    return per_class



# Confusion matrix + plotting
def plot_confusion_matrix(predictions, targets, class_names, figsize=(12, 10)):
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
    return cm



# Optional: Hierarchy-based scoring
# hierarchy should be dict:  class_index -> parent_category
# e.g., hierarchy = {0:"human", 1:"human", 2:"machine", ...}
def compute_hierarchical_score(predictions, targets, hierarchy):
    scores = []
    for p, t in zip(predictions, targets):
        if p == t:
            scores.append(1.0)
        elif hierarchy[p] == hierarchy[t]:
            scores.append(0.5)  # mild penalty
        else:
            scores.append(0.0)  # full penalty
    return sum(scores) / len(scores)



# Full evaluation wrapper
def evaluate(predictions, targets, class_names, hierarchy=None, plot_cm=True):
    results = {}

    # overall accuracy
    results["overall_accuracy"] = compute_overall_accuracy(predictions, targets)

    # per-class accuracy
    results["per_class_accuracy"] = compute_per_class_accuracy(
        predictions, targets, class_names)

    # confusion matrix
    if plot_cm:
        results["confusion_matrix"] = plot_confusion_matrix(
            predictions, targets, class_names)
    else:
        results["confusion_matrix"] = confusion_matrix(targets, predictions)

    # optional hierarchy score
    if hierarchy is not None:
        results["hierarchical_score"] = compute_hierarchical_score(
            predictions, targets, hierarchy)

    return results



if __name__ == "__main__":



    #results = evaluate(predictions, targets, class_names, hierarchy)

    #print("--- Evaluation Results ---")
    #print("Overall accuracy:", results["overall_accuracy"])
    #print("Per-class accuracy:", results["per_class_accuracy"])
    #if "hierarchical_score" in results:
    #    print("Hierarchical score:", results["hierarchical_score"])

    pass