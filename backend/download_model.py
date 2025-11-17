"""
Script to download and prepare a pre-trained plant disease detection model.
This uses a simple CNN model trained on the PlantVillage dataset.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

def create_model(num_classes=25):
    """Create a simple CNN model for plant disease classification"""
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=(224, 224, 3)),
        
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def download_pretrained_model():
    """
    Download a pre-trained model from TensorFlow Hub or create a new one.
    For simplicity, we'll create a model structure that can be trained later.
    """
    try:
        # Try to download a pre-trained model (if available)
        # For now, we'll create a model architecture
        model = create_model(num_classes=25)
        
        # Save the model
        model_path = 'models/plant_disease_model.h5'
        os.makedirs('models', exist_ok=True)
        model.save(model_path)
        
        print(f"Model created and saved to {model_path}")
        print("Note: This is an untrained model. For production use, you should:")
        print("1. Download the PlantVillage dataset")
        print("2. Train the model on this dataset")
        print("3. Or use a pre-trained model from TensorFlow Hub")
        
        return model
    
    except Exception as e:
        print(f"Error creating model: {e}")
        return None

if __name__ == '__main__':
    print("Creating plant disease detection model...")
    model = download_pretrained_model()
    
    if model:
        print("\nModel summary:")
        model.summary()
