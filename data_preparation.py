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
    split_file (str): Path to the split file containing the splits information.
    """
    # Read the split file
    with open(split_file, 'r') as f:
        lines = f.readlines()

    # Create directory for this split
    if 'train' in split_file:
        split_dir = 'dataset/train'
    elif 'val' in split_file:
        split_dir = 'dataset/val'
    elif 'test' in split_file:
        split_dir = 'dataset/test'
    else:
        raise ValueError("Split file name must contain 'train', 'val', or 'test' to determine the split type.")
    
    os.makedirs(split_dir, exist_ok=True)

    split_file_names = []

    # Process each line in the split file
    for line in lines[1:]:  # Skip the header
        file_name = line.strip().split(',')[0]  # Assuming the first column contains the file names
        split_file_names.append(file_name)
    
    # Move the audio files to the corresponding split directory
    for audio_file in audio_files:
        if os.path.basename(audio_file) in split_file_names:
            os.rename(audio_file, os.path.join(split_dir, os.path.basename(audio_file)))

def main():
    # Define paths
    audio_dir = 'audio_files'
    split_dir = 'splits'

    # Get list of audio files
    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]

    # Process each split file
    for split_file in os.listdir(split_dir):
        if split_file.endswith('.csv'):
            split_file_path = os.path.join(split_dir, split_file)
            split_data(audio_files, split_file_path)


if __name__ == "__main__":
    main()