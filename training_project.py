import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

# --- 1. CONFIGURATION & HIERARCHY DEFINITION ---
# Based on the taxonomy in the project slides [cite: 62-91]

HIERARCHY = {
    "Music": ["Solo percussion", "Solo instrument", "Multiple instruments"],
    "Instrument Samples": ["Percussion", "String", "Wind", "Piano/Keyboard instruments", "Synth/Electronic"],
    "Speech": ["Solo speech", "Conversation/Crowd", "Processed/Synthetic"],
    "Sound Effects": ["Objects/House appliances", "Vehicles", "Other mechanisms, engines, machines", 
                      "Animals", "Human sounds and actions", "Natural elements and explosions", 
                      "Experimental", "Electronic/Design"],
    "Soundscapes": ["Nature", "Indoors", "Urban", "Synthetic/Artificial"]
}

# Flatten classes to create a mapping: Label String -> Integer Index
CLASSES = []
PARENT_MAP = {} # Maps child index to parent name
for parent, children in HIERARCHY.items():
    for child in children:
        PARENT_MAP[len(CLASSES)] = parent
        CLASSES.append(child)

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES) # Should be 23 [cite: 3]

# Audio Config
SAMPLE_RATE = 44100  # Adjust based on your actual wav files
N_MELS = 64
FIXED_LENGTH = SAMPLE_RATE * 5 # e.g., 5 seconds of audio

print(f"Initialized {NUM_CLASSES} classes under {len(HIERARCHY)} parents.")

# --- 2. DATASET CLASS ---

class HeterogeneousAudioDataset(Dataset):
    """
    Handles loading BSD10k and BSD35k audio files and converting to Mel Spectrograms.
    """
    def __init__(self, metadata_df, audio_dir, transform=None, target_sample_rate=SAMPLE_RATE):
        self.metadata = metadata_df
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        # Standard Mel Spectrogram transform
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_mels=N_MELS
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Assumes metadata has 'filename' and 'class_label' columns
        row = self.metadata.iloc[idx]
        file_path = os.path.join(self.audio_dir, row['filename'])
        label_str = row['class_label']
        label = CLASS_TO_IDX[label_str]

        # Load Audio
        waveform, sr = torchaudio.load(file_path)

        # Resample if necessary
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            waveform = resampler(waveform)

        # Mix down to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or Truncate to fixed length
        if waveform.shape[1] < FIXED_LENGTH:
            pad_amount = FIXED_LENGTH - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        else:
            waveform = waveform[:, :FIXED_LENGTH]

        # Convert to Mel Spectrogram
        spec = self.mel_spectrogram(waveform)
        spec = self.amplitude_to_db(spec)

        return spec, label

# --- 3. MODEL ARCHITECTURE ---
# Simple CNN as suggested in "Simplest case: CNN-based supervised classification" 

class SimpleAudioCNN(nn.Module):
    def __init__(self, num_classes=23):
        super(SimpleAudioCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Layer 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Adaptive pool allows us to handle slightly different input sizes smoothly
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

# --- 4. HIERARCHICAL EVALUATION METRICS ---
# Implementing hPrecision, hRecall, and hF-score 

def calculate_hierarchical_metrics(preds, targets, lambda_val=0.5):
    """
    preds: List or Tensor of predicted class indices
    targets: List or Tensor of true class indices
    lambda_val: Weight for sibling classes (usually between 0 and 1)
    """
    preds = np.array(preds)
    targets = np.array(targets)
    
    total_h_precision = 0
    total_h_recall = 0
    
    # We iterate per sample to apply weights
    # Note: The formula in the PDF sums over j (predicted class) 
    # This implementation is a vectorized version of that logic
    
    n_samples = len(preds)
    h_precisions = []
    h_recalls = []

    for i in range(n_samples):
        pred_cls = preds[i]
        true_cls = targets[i]
        
        # Determine Weight w_ij
        if pred_cls == true_cls:
            w_ij = 1.0
        elif PARENT_MAP[pred_cls] == PARENT_MAP[true_cls]:
            w_ij = lambda_val # Same parent, different class
        else:
            w_ij = 0.0 # Different parent
            
        # hPrecision calculation (simplified for single-label case)
        # In single label, sum(TP+FP) is always 1 (the one prediction made)
        # So hPrecision is just the weight of the prediction
        h_precision = w_ij 
        
        # hRecall calculation
        # In single label, sum(TP+FN) is always 1 (the one true label)
        # So hRecall is also just the weight
        h_recall = w_ij
        
        h_precisions.append(h_precision)
        h_recalls.append(h_recall)

    # Global averaging
    global_h_prec = np.mean(h_precisions)
    global_h_rec = np.mean(h_recalls)
    
    # Avoid div by zero
    if (global_h_prec + global_h_rec) == 0:
        global_hf = 0
    else:
        global_hf = 2 * (global_h_prec * global_h_rec) / (global_h_prec + global_h_rec)
        
    return global_hf, global_h_prec, global_h_rec

# --- 5. MAIN TRAINING LOOP SKELETON ---

def train_model():
    # PLACEHOLDERS: Replace these with your actual file loading logic
    # You need to parse the provided text files/CSVs to build these DataFrames
    # Training Data: BSDNoisy35k + BSD10k (50% split) [cite: 15]
    train_df = pd.DataFrame(columns=['filename', 'class_label']) 
    # Validation Data: BSD10k (10% split) [cite: 19]
    val_df = pd.DataFrame(columns=['filename', 'class_label'])   
    
    # Check if empty (for the sake of the script running without data)
    if len(train_df) == 0:
        print("Warning: No data loaded. Creating dummy data for demonstration.")
        # Create dummy data
        train_df = pd.DataFrame({
            'filename': ['dummy.wav'] * 100, 
            'class_label': [CLASSES[np.random.randint(0, 23)] for _ in range(100)]
        })
        val_df = pd.DataFrame({
            'filename': ['dummy.wav'] * 20, 
            'class_label': [CLASSES[np.random.randint(0, 23)] for _ in range(20)]
        })

    # Create Datasets
    # Note: "audio_dir" should point to where you unzipped the BSD data
    train_dataset = HeterogeneousAudioDataset(train_df, audio_dir="./data/train")
    val_dataset = HeterogeneousAudioDataset(val_df, audio_dir="./data/val")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleAudioCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training on {device}...")

    # Training Loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validation Step
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(device), labels.to(device)
                outputs = model(specs)
                _, predicted = torch.max(outputs.data, 1)
                
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        # Calculate Metrics
        # Standard Accuracy
        val_acc = np.mean(np.array(val_preds) == np.array(val_targets))
        
        # Hierarchical Metrics 
        hF, hP, hR = calculate_hierarchical_metrics(val_preds, val_targets, lambda_val=0.5)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Loss: {running_loss/len(train_loader):.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"Hierarchical F-Score: {hF:.4f} (P: {hP:.4f}, R: {hR:.4f})")
        print("-" * 20)

    print("Training Complete. Remember: Do not touch the test subset until done! [cite: 24]")

if __name__ == "__main__":
    # To run this, you need actual audio files and metadata lists
    # train_model()
    pass