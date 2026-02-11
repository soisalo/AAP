import torch
import torch.nn as nn
import torch.nn.functional as F

class SSLClassifier(nn.Module):
    def __init__(self, input_channels=4, num_classes=4, freq_bins=1025):
        """
        CNN classifier for binaural sound source localization.
        
        :param input_channels: Number of input channels (default 4: 2 mag, 2 phase) 
        :param num_classes: Output classes (front, left, back, right) [cite: 261]
        :param freq_bins: Number of frequency bins from STFT (K) [cite: 301]
        """
        super(SSLClassifier, self).__init__()
        
        # 3 Convolutional Blocks [cite: 318]
        # Each: 2D Conv (3x3, 128 ch) -> MaxPool (stride 4 freq) -> BatchNorm -> ReLU
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Block 2
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Block 3
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Calculate remaining frequency bins after three stride-4 poolings
        # K_final = K // (4 * 4 * 4)
        self.final_freq_bins = freq_bins // 64
        
        # Multilayer Perceptron (MLP) [cite: 320]
        # Hidden size 128, dropout, ReLU
        self.mlp = nn.Sequential(
            nn.Linear(128 * self.final_freq_bins, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes) # No activation here 
        )

    def forward(self, x):
        """
        Forward pass.
        Input x shape: (Batch, C, T, K) [cite: 301]
        """
        # Apply convolutional blocks
        x = self.conv_blocks(x)
        
        # Aggressive average pooling over the whole time axis 
        # x shape after conv: (Batch, 128, T, K_final)
        x = torch.mean(x, dim=2) # Result shape: (Batch, 128, K_final)
        
        # Flatten for MLP [cite: 320]
        x = x.view(x.size(0), -1)
        
        # Output layer (Logits)
        x = self.mlp(x)
        
        return x

# Example of how to name the file: model_<your_name>.py [cite: 324]