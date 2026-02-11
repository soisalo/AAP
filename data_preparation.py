"""
Docstring for data_preparation

Author: Tommi Salonen
Email: tommi.t.salonen@tuni.fi
"""

import os
import numpy as np

def split_data(audio_files, split_file):
    """
    Split the data into training, validation, and test sets based on the provided csv file.
    Audio files are moved to dataset folder folders named 'train', 'val', and 'test' based
    on the split information in the csv file.

    Parameters:
    audio_files (list): List of audio file paths.
    split_file (str): Path to the split file containing the split information.
    """
    # Read the split file
    with open(split_file, 'r') as f:
        lines = f.readlines()

    # Create directories for train, val, and test sets
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join('dataset', split), exist_ok=True)
    
    # Process each line in the split file