import librosa
import numpy as np
import tensorflow as tf
import pickle
import os

def predict_audio(file_path, model_path='../models/audio_classifier.h5', encoder_path='../models/label_encoder.pkl'):
    """
    Takes a raw audio file path, processes it, and returns the prediction.
    """
    try:
        # 1. Load the Model
        # We use compile=False because we only need it for prediction, not training
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # 2. Load the Label Encoder
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
            
        # 3. Preprocess (Must be identical to training!)
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        mfccs_processed = mfccs_scaled.reshape(1, -1) # Reshape to (1, 40)
        
        # 4. Predict
        prediction_probabilities = model.predict(mfccs_processed)
        
        # 5. Decode the Result
        predicted_class_index = np.argmax(prediction_probabilities, axis=1)
        predicted_label = encoder.inverse_transform(predicted_class_index)[0]
        confidence = prediction_probabilities[0][predicted_class_index[0]]
        
        return predicted_label, confidence, prediction_probabilities[0]

    except Exception as e:
        print(f"Error predicting file: {e}")
        return None, 0, None