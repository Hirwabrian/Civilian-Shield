import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def extract_features(file_path):
    """
    Loads an audio file and converts it into MFCCs (Mel-Frequency Cepstral Coefficients).
    """
    try:
        # Load audio with librosa
        # res_type='kaiser_fast' speeds up processing
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
        
        # Generate MFCCs (40 features is a standard for environmental sound)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Take the mean of the features across time to get a fixed size input
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        
        return mfccs_scaled
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def get_data(dataset_path, metadata_file, folds_to_use=None):
    """
    Main function to load data, extract features, and prepare X and y for the model.
    """
    # Load the metadata CSV
    metadata = pd.read_csv(metadata_file)
    
    features = []
    labels = []
    
    # If no specific folds are requested, use all of them
    if folds_to_use is None:
        folds_to_use = range(1, 11)
        
    print(f"Starting extraction for folds: {list(folds_to_use)}")
    
    # Iterate through the CSV
    for index, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
        fold = row['fold']
        
        if fold in folds_to_use:
            file_name = row['slice_file_name']
            class_label = row['class']
            
            # Construct path to the audio file
            # Assumes structure: dataset_path/fold1/filename.wav
            file_path = os.path.join(dataset_path, f"fold{fold}", file_name)
            
            data = extract_features(file_path)
            
            if data is not None:
                features.append(data)
                labels.append(class_label)
                
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    # Encode the labels (e.g., "siren" -> 1, "dog_bark" -> 2)
    le = LabelEncoder()
    y_encoded = to_categorical(le.fit_transform(y))
    
    print("Data extraction complete.")
    print(f"Features shape: {X.shape}")
    return X, y_encoded, le