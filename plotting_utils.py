import matplotlib.pyplot as plt

def plot_training_metrics(train_losses, val_accuracies, val_hFs, save_path=None):
    """
    Plots training loss, validation accuracy, and hierarchical F-score.

    Args:
        train_losses (list[float]): Training loss per epoch.
        val_accuracies (list[float]): Validation accuracy per epoch.
        val_hFs (list[float]): Validation hierarchical F-score per epoch.
        save_path (str, optional): Path to save the figure as PNG. If None, just shows the plot.
    """
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(12,5))

    # Training loss
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, marker='o', color='tab:blue', label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Validation metrics
    plt.subplot(1,2,2)
    plt.plot(epochs, val_accuracies, marker='o', color='tab:green', label='Val Accuracy')
    plt.plot(epochs, val_hFs, marker='x', color='tab:red', label='Hierarchical F-score')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
