import torch.nn as nn
from torch.utils.data import Dataset
import pickle
import numpy as np


"""
Authors: Eemil Soisalo, Tommi Salonen, Venla Numminen
email: eemil.soisalo@tuni.fi, tommi.salonen@tuni.fi, venla.numminen@tuni.fi

This module defines the dataset class for loading precomputed Mel spectrograms and labels from .pkl files,
as well as the model architecture and hierarchical metrics calculation for the audio classification task.

2026-06-01: Initial version created.
"""

# --- CONFIGURATION & HIERARCHY DEFINITION ---

HIERARCHY = {
    "Music": ["Solo percussion", "Solo instrument", "Multiple instruments"],
    "Instrument Samples": ["Percussion", "String", "Wind", "Piano/Keyboard instruments", "Synth/Electronic"],
    "Speech": ["Solo speech", "Conversation/Crowd", "Processed/Synthetic"],
    "Sound Effects": ["Objects/House appliances", "Vehicles", "Other mechanisms, engines, machines", 
                      "Animals", "Human sounds and actions", "Natural elements and explosions", 
                      "Experimental", "Electronic/Design"],
    "Soundscapes": ["Nature", "Indoors", "Urban", "Synthetic/Artificial"]
}


# Build the PARENT_TO_IDX mapping
# This maps Child Index (0-21) -> Parent Index (0-4)
PARENT_TO_IDX = {name: i for i, name in enumerate(HIERARCHY.keys())}
IDX_PARENT_MAP = {}
CLASSES = []

for parent_name, children in HIERARCHY.items():
    parent_idx = PARENT_TO_IDX[parent_name]
    for child_name in children:
        child_idx = len(CLASSES)
        CLASSES.append(child_name)
        IDX_PARENT_MAP[child_idx] = parent_idx


#Verification
#print("Child Index to Parent Index Map:")
#print(IDX_PARENT_MAP)

NUM_CLASSES = len(CLASSES)
print(f"Initialized {NUM_CLASSES} classes under {len(HIERARCHY)} parents.")

# --- PICKLE DATASET CLASS ---

class PickleAudioDataset(Dataset):
    """
    Dataset class for loading precomputed Mel spectrograms and labels from .pkl files.
    Each .pkl file is expected to contain a dictionary with keys:
    """
    def __init__(self, pkl_files):
        # pkl_files: List of paths to .pkl files containing the data
        self.pkl_files = pkl_files

    def __len__(self):
        # The length of the dataset is the number of .pkl files
        return len(self.pkl_files)

    def __getitem__(self, idx):
        """
        Loads the Mel spectrogram and label from the specified .pkl file.
        Returns:
        - mel: The precomputed Mel spectrogram (as a numpy array)
        - label: The integer class label
        - weight: The confidence-based weight for the sample (normalized to [0,1])
        - top_class: The integer index of the top-level class (parent category)
        """
        with open(self.pkl_files[idx], "rb") as f:
            data = pickle.load(f)
        mel = data['features']               # precomputed Mel spectrogram
        label = int(data['class_idx']) # integer label
        top_class = int(data.get('top_class', -1))  # Optional top-level class
        confidence = int(data.get('confidence_score', 0))# default to 5 if missing
        weight = (confidence - 1) / 4.0     # Normalize 1-5 → 0-1

        return mel, label, weight, top_class
    
# --- MODEL ARCHITECTURE ---

class SimpleCLAPClassifier(nn.Module):
    """
    A simple feedforward neural network for classifying audio samples based on precomputed CLAP embeddings.
    The model consists of:  
    - A fully connected layer that reduces the embedding dimension to 256, followed by batch normalization, ReLU activation, and dropout.
    - Another fully connected layer that reduces the dimension to 128, followed by batch normalization, ReLU activation, and dropout.
    - Two separate output heads: one for fine-grained classification (22 classes) and one for coarse-grained classification (5 parent classes).
    """
    def __init__(self, embedding_dim=512, num_classes=NUM_CLASSES, num_parents=5):
        super().__init__()
        self.fc = nn.Sequential(

            #First fully connected layer to reduce dimensionality
            nn.Linear(embedding_dim, 256),
            #Batch normalization
            nn.BatchNorm1d(256),
            #ReLU activation and dropout for regularization
            nn.ReLU(),
            nn.Dropout(0.4),

            #Second fully connected layer to further reduce dimensionality
            nn.Linear(256, 128),
            #Batch normalization
            nn.BatchNorm1d(128),
            #ReLU activation and dropout for regularization
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        #Fine grain classification for all 22 classes
        self.child_head = nn.Linear(128, num_classes)

        #Coarse grain classification for 5 parent classes
        self.parent_head = nn.Linear(128, num_parents)

    def forward(self, x):
        x = self.fc(x)
        return self.child_head(x), self.parent_head(x)

# --- HIERARCHICAL METRICS ---

def calculate_hierarchical_metrics(preds, targets, lambda_val=0.5):
    """
    Calculates hierarchical precision, recall, and F1-score based on the predicted and true class indices.
    The function applies a penalty when the predicted class and true class belong to different parent categories,
    and a smaller penalty when they belong to the same parent category but are different classes.

    :param preds: List or array of predicted class indices.
    :type preds: list[int] or numpy.ndarray
    :param targets: List or array of true class indices.
    :type targets: list[int] or numpy.ndarray
    :param lambda_val: Penalty weight for predictions that are in the same parent category but different classes (default=0.5).
    :type lambda_val: float
    :return: A tuple containing the global hierarchical F1-score, precision, and recall.
    :rtype: tuple[float, float, float]

    """
    preds = np.array(preds)
    targets = np.array(targets)
    n_samples = len(preds)
    h_precisions, h_recalls = [], []

    for i in range(n_samples):
        pred_cls = preds[i]
        true_cls = targets[i]
        if pred_cls == true_cls:
            w_ij = 1.0
        elif IDX_PARENT_MAP[pred_cls] == IDX_PARENT_MAP[true_cls]:
            w_ij = lambda_val
        else:
            w_ij = 0.0
        h_precisions.append(w_ij)
        h_recalls.append(w_ij)

    global_h_prec = np.mean(h_precisions)
    global_h_rec = np.mean(h_recalls)
    global_hf = 2 * (global_h_prec * global_h_rec) / (global_h_prec + global_h_rec) if (global_h_prec + global_h_rec) > 0 else 0
    return global_hf, global_h_prec, global_h_rec
