"""
CNN Training Script for Plant Disease Detection
This creates and trains a Convolutional Neural Network on plant disease images.
"""
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

print("=" * 60)
print("🧠 CNN Model Training - Plant Disease Detection")
print("=" * 60)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 25

# Model save path
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'plant_disease_model.h5')
os.makedirs(MODEL_DIR, exist_ok=True)

def create_cnn_model():
    """Create a CNN model using MobileNetV2 with transfer learning"""
    print("\n🏗️  Building CNN Architecture...")
    
    # Use MobileNetV2 as base (pre-trained on ImageNet)
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n📊 Model Architecture:")
    model.summary()
    
    return model

def create_data_generators(data_dir):
    """Create data generators for training"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation'
    )
    
    return train_generator, val_generator

def train_model():
    """Main training function"""
    
    # Check if dataset exists
    DATA_DIR = 'dataset/plant_diseases'
    
    if not os.path.exists(DATA_DIR):
        print("\n❌ Dataset not found!")
        print(f"Expected location: {DATA_DIR}")
        print("\n📥 Please download the PlantVillage dataset:")
        print("   1. Visit: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
        print("   2. Download and extract to 'backend/dataset/plant_diseases/'")
        print("   3. Run this script again")
        return
    
    print(f"\n✅ Dataset found at: {DATA_DIR}")
    
    # Create model
    model = create_cnn_model()
    
    # Prepare data
    print("\n📁 Loading dataset...")
    train_gen, val_gen = create_data_generators(DATA_DIR)
    
    print(f"\nTraining samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Classes: {len(train_gen.class_indices)}")
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    print("=" * 60)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("📊 Final Results:")
    print(f"Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
    print(f"Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
    
    # Save final model
    model.save(MODEL_PATH)
    print(f"\n✅ Model saved to: {MODEL_PATH}")
    
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print("You can now use app.py with CNN predictions")
    print("=" * 60)

if __name__ == '__main__':
    try:
        train_model()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        print("Please ensure TensorFlow is installed and dataset is available")
