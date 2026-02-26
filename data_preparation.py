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


def process_files(fetures_dir, split_file):
    """
    Use precomputed audio embeddings to prepare the data for training, validation, and testing.
    The function reads the split file, loads the corresponding feature vectors from the .npy files
    and serializes the features and metadata to files in the corresponding split directories.

    :param fetures_dir: Directory containing the precomputed audio embeddings in .npy format.
    :type fetures_dir: str
    :param split_file: Path to the csv file containing the split information (train.csv, val.csv, test.csv).
    :type split_file: str
    """
    # Read the split file
    with open(split_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Get the base directory of the project

    base_dir = f"{base_dir}/Advanced Audio Processing Project" # Define the base directory for the dataset
    
    # Create directory for this split
    if 'train' in split_file:
        split_dir = os.path.join(base_dir, 'dataset/train')
    elif 'val' in split_file:
        split_dir = os.path.join(base_dir, 'dataset/val')
    elif 'test' in split_file:
        split_dir = os.path.join(base_dir, 'dataset/test')
    else:
        raise ValueError("Split file name must contain 'train', 'val', or 'test' to determine the split type.")
    
    os.makedirs(split_dir, exist_ok=True)

    # Process each line in the split file
    for line in lines[1:]:  # Skip the header
        file_name = line.strip().split(',')[0]  # Assuming the first column contains the file names
        class_idx = line.strip().split(',')[2]  # Assuming the second column contains the class labels
        top_class = line.strip().split(',')[3]  # Assuming the third column contains the top class predictions
        confidence_score = line.strip().split(',')[4]  # Assuming the fifth column contains the confidence scores

        feature_vector = np.load(os.path.join(fetures_dir, file_name + '.npy'))  # Load the feature vector from the .npy file

        # Serialize the features and metadata
        features_and_metadata = {   
                                    'features': feature_vector,
                                    'class_idx': class_idx,
                                    'top_class': top_class,
                                    'confidence_score': confidence_score
                                }

        # Serialize the features and metadata to a file in the corresponding split directory
        serialize_features_and_metadata(os.path.join(split_dir, file_name + '.pkl'), features_and_metadata)


def main():
    # Define paths
    split_dir = 'bsdk10k-splits' # Directory containing the split csv files (train.csv, val.csv, test.csv)
    audio_embeddings_dir = 'data/features/clap_audio_embeddings' # Directory containing the audio files

    # Process each split file
    for split_file in os.listdir(split_dir):
        if split_file.endswith('.csv'):
            split_file_path = os.path.join(split_dir, split_file)
            process_files(audio_embeddings_dir, split_file_path)


if __name__ == "__main__":
    main()