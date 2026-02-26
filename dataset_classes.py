import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import pickle
import numpy as np

# --- 1. CONFIGURATION & HIERARCHY DEFINITION ---

HIERARCHY = {
    "Music": ["Solo percussion", "Solo instrument", "Multiple instruments"],
    "Instrument Samples": ["Percussion", "String", "Wind", "Piano/Keyboard instruments", "Synth/Electronic"],
    "Speech": ["Solo speech", "Conversation/Crowd", "Processed/Synthetic"],
    "Sound Effects": ["Objects/House appliances", "Vehicles", "Other mechanisms, engines, machines", 
                      "Animals", "Human sounds and actions", "Natural elements and explosions", 
                      "Experimental", "Electronic/Design"],
    "Soundscapes": ["Nature", "Indoors", "Urban", "Synthetic/Artificial"]
}

CLASSES = []
PARENT_MAP = {}  # Maps child index to parent name
for parent, children in HIERARCHY.items():
    for child in children:
        PARENT_MAP[len(CLASSES)] = parent
        CLASSES.append(child)

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

print(f"Initialized {NUM_CLASSES} classes under {len(HIERARCHY)} parents.")

# --- 2. PICKLE DATASET CLASS ---

class PickleAudioDataset(Dataset):
    """
    NOTE: 
    Loads precomputed Mel spectrograms and labels from .pkl files.
    Each .pkl file should contain a dict with keys:
    'mel': Tensor of shape [1, n_mels, time]
    'label': Integer class index
    """
    def __init__(self, pkl_files):
        self.pkl_files = pkl_files

    def __len__(self):
        return len(self.pkl_files)

    def __getitem__(self, idx):
        with open(self.pkl_files[idx], "rb") as f:
            data = pickle.load(f)
        mel = data['features']               # precomputed Mel spectrogram
        label = int(data['class_idx']) # integer label
        top_class = int(data.get('top_class', -1))  # Optional top-level class
        confidence = int(data.get('confidence_score', 0))# default to 5 if missing
        weight = (confidence - 1) / 4.0     # Normalize 1-5 → 0-1

        return mel, label, weight, top_class
    
class SimpleCLAPClassifier(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# --- 3. HIERARCHICAL METRICS ---

def calculate_hierarchical_metrics(preds, targets, lambda_val=0.5):
    """Calculates hierarchical F1, precision, and recall based on the 
    defined hierarchy and lambda penalty."""
    preds = np.array(preds)
    targets = np.array(targets)
    n_samples = len(preds)
    h_precisions, h_recalls = [], []

    for i in range(n_samples):
        pred_cls = preds[i]
        true_cls = targets[i]
        if pred_cls == true_cls:
            w_ij = 1.0
        elif PARENT_MAP[pred_cls] == PARENT_MAP[true_cls]:
            w_ij = lambda_val
        else:
            w_ij = 0.0
        h_precisions.append(w_ij)
        h_recalls.append(w_ij)

    global_h_prec = np.mean(h_precisions)
    global_h_rec = np.mean(h_recalls)
    global_hf = 2 * (global_h_prec * global_h_rec) / (global_h_prec + global_h_rec) if (global_h_prec + global_h_rec) > 0 else 0
    return global_hf, global_h_prec, global_h_rec
