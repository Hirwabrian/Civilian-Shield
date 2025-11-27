import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
import datetime

# Forces TensorFlow to run operations immediately (Eagerly) rather than building graphs.
# This prevents specific errors in Streamlit/Jupyter when retraining repeatedly.
tf.config.run_functions_eagerly(True)

def create_model(input_shape, num_classes):
    """
    Defines the Neural Network architecture with Regularization.
    """
    model = Sequential()
    
    # LAYER 1
    # Input: 40 MFCC features
    # Dropout(0.5): Regularization technique to prevent overfitting
    model.add(Dense(256, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dropout(0.5)) 
    
    # LAYER 2
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    # LAYER 3
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # OUTPUT LAYER
    # Softmax for multi-class probability
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # OPTIMIZER
    # Adam is used for adaptive learning rates
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    return model

def train_and_save_model(X, y, save_path='../models/audio_classifier.h5', epochs=50, batch_size=32):
    """
    Splits data, trains the model using optimization callbacks, evaluates with 
    multiple metrics (Precision, Recall, F1), and saves the result.
    """
    # DATA SANITIZATION
    # Ensure inputs are pure Numpy float32 arrays to prevent Tensor conflicts
    X = np.array(X).astype('float32')
    y = np.array(y).astype('float32')

    # Initialize test sets to None to ensure safety if split doesn't happen
    X_test = None
    y_test = None

    # 1. Validation Split
    # We only split if we have enough data (>10 samples)
    if len(X) > 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        validation_data = (X_test, y_test)
        print(f">> Split Data: {len(X_train)} Train, {len(X_test)} Test")
    else:
        print(">> Small batch detected. Training on full batch without split.")
        X_train, y_train = X, y
        validation_data = (X, y)
    
    # 2. Load Existing or Create New Model
    if os.path.exists(save_path):
        print(f"🔄 Loading existing model from {save_path} for fine-tuning...")
        try:
            model = load_model(save_path)
            # Recompile to reset optimizer state and prevent variable conflicts
            print(">> Recompiling model with fresh optimizer...")
            model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
        except:
            print("Error loading model. Creating a fresh one.")
            input_shape = (X.shape[1],) 
            num_classes = y.shape[1] 
            model = create_model(input_shape, num_classes)
    else:
        print("🆕 Creating a fresh model...")
        input_shape = (X.shape[1],)
        num_classes = y.shape[1]
        model = create_model(input_shape, num_classes)
    
    # OPTIMIZATION TECHNIQUES
    # 1. EarlyStopping: Stop training if validation loss doesn't improve for 5 epochs
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    
    # 2. ReduceLROnPlateau: Reduce learning rate if validation loss plateaus
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    
    print(f"Training on {len(X_train)} samples...")
    
    # 3. Train
    history = model.fit(X_train, y_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        validation_data=validation_data,
                        callbacks=[early_stopping, reduce_lr], 
                        verbose=1)
    
    # 4. Detailed Evaluation (Criteria Check: Accuracy, Precision, Recall, F1)
    print("\n" + "="*40)
    print("🔬 COMPREHENSIVE EVALUATION REPORT 🔬")
    print("="*40)
    
    if X_test is not None:
        # Basic Metrics
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"TEST LOSS:     {loss:.4f}")
        print(f"TEST ACCURACY: {accuracy*100:.2f}%")
        
        # Advanced Metrics
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        print("\n--- CLASSIFICATION REPORT (Precision, Recall, F1) ---")
        # specific classification metrics
        print(classification_report(y_true, y_pred))
        
        print("\n--- CONFUSION MATRIX ---")
        print(confusion_matrix(y_true, y_pred))
        
        score = accuracy
    else:
        # Fallback for small batches
        score = model.evaluate(X_train, y_train, verbose=0)[1]
        print("Batch too small for validation split. Accuracy calculated on training set.")
        
    # 5. Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\nModel saved to {save_path}")
    
    return model, history, score