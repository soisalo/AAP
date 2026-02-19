"""
This module contains functions for preparing the data for training, validation, and testing.
It includes functions for extracting features from audio files, splitting the data into training,
validation, and test sets based on provided csv files, and serializing the features and metadata
to files in the corresponding split directories.

Author: Tommi Salonen
Email: tommi.t.salonen@tuni.fi
"""

import os
from typing import MutableMapping, Union
import numpy as np
import librosa
import pickle

def serialize_features_and_metadata(file: str, features_and_classes: MutableMapping[str, Union[np.ndarray, int]])\
        -> None:
    """Serializes the features and classes.

    :param file: File to dump the serialized features
    :type file: str
    :param features_and_classes: Features and classes.
    :type features_and_classes: dict[str, numpy.ndarray|int]
    """
    with open(file, 'wb') as pkl_file:
        pickle.dump(features_and_classes, pkl_file)

def extract_features(audio_files, split_file):
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

    # Process each line in the split file
    for line in lines[1:]:  # Skip the header
        file_name = line.strip().split(',')[0]  # Assuming the first column contains the file names
        class_idx = line.strip().split(',')[2]  # Assuming the second column contains the class labels
        top_class = line.strip().split(',')[3]  # Assuming the third column contains the top class predictions
        confidence_score = line.strip().split(',')[4]  # Assuming the fifth column contains the confidence scores

        # Extract the corresponding audio file
        audio_file = next((f for f in audio_files if os.path.basename(f) == file_name), None)

        # Calculate the Mel spectrogram for the audio file
        if audio_file:
            mel = calculate_mel_spectrogram(audio_file)
        else:
            print(f"Audio file {file_name} not found in the provided audio files list.")
            continue

        # Serialize the features and metadata
        features_and_metadata = {   
                                    'features': mel,
                                    'class_idx': class_idx,
                                    'top_class': top_class ,
                                    'confidence_score': confidence_score
                                }

        # Serialize the features and metadata to a file in the corresponding split directory
        serialize_features_and_metadata(os.path.join(split_dir, file_name + '.pkl'), features_and_metadata)


def calculate_mel_spectrogram(audio_file):
    """
    Calculate the Mel spectrogram for a given audio file.

    Parameters:
    audio_file (str): Path to the audio file.

    Returns:
    np.ndarray: Mel spectrogram of the audio file.
    """
    y, sr = librosa.load(audio_file, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return mel_spectrogram_db

def main():
    # Define paths
    audio_dir = 'audio_files'
    split_dir = 'splits' # Directory containing the split csv files (train.csv, val.csv, test.csv)

    # Get list of audio files
    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]

    # Process each split file
    for split_file in os.listdir(split_dir):
        if split_file.endswith('.csv'):
            split_file_path = os.path.join(split_dir, split_file)
            extract_features(audio_files, split_file_path)


if __name__ == "__main__":
    main()