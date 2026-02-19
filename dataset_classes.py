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
    Loads precomputed Mel spectrograms and labels from .pkl files.
    Each .pkl file should contain a dict: {'mel': torch.Tensor, 'label': int}
    """
    def __init__(self, pkl_files):
        self.pkl_files = pkl_files

    def __len__(self):
        return len(self.pkl_files)

    def __getitem__(self, idx):
        with open(self.pkl_files[idx], "rb") as f:
            data = pickle.load(f)
        mel = data['mel']        # Shape: [1, n_mels, time]
        label = data['label']    # Already an integer
        return mel, label

# --- 3. SIMPLE AUDIO CNN MODEL ---

class SimpleAudioCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleAudioCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = self.fc_layers(x)
        return x

# --- 4. HIERARCHICAL METRICS ---

def calculate_hierarchical_metrics(preds, targets, lambda_val=0.5):
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
