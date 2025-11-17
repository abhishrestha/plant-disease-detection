from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import io

# Try to import TensorFlow, but make it optional for faster startup
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    print("TensorFlow loaded successfully")
except Exception as e:
    TF_AVAILABLE = False
    print(f"TensorFlow not loaded: {e}")
    print("Running in demo mode with dummy predictions")

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'models/plant_disease_model.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Plant disease classes
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Load model (will be created if doesn't exist)
model = None
if TF_AVAILABLE:
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully")
        else:
            print("Model not found. Using dummy predictions.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("TensorFlow not available, using dummy predictions")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_file):
    """Preprocess image for model prediction"""
    img = Image.open(io.BytesIO(image_file.read()))
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_disease(image_file):
    """Predict plant disease from image"""
    try:
        # Preprocess image
        img_array = preprocess_image(image_file)
        
        if model is not None:
            # Make prediction
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = CLASS_NAMES[predicted_class_idx]
        else:
            # Dummy prediction if model doesn't exist
            predicted_class_idx = np.random.randint(0, len(CLASS_NAMES))
            predicted_class = CLASS_NAMES[predicted_class_idx]
            confidence = np.random.uniform(0.7, 0.99)
        
        # Parse the disease name
        parts = predicted_class.split('___')
        plant_name = parts[0].replace('_', ' ')
        disease_name = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'
        
        return {
            'plant': plant_name,
            'disease': disease_name,
            'confidence': confidence,
            'is_healthy': 'healthy' in disease_name.lower()
        }
    except Exception as e:
        raise Exception(f"Error predicting disease: {str(e)}")

@app.route('/')
def home():
    return jsonify({'message': 'Plant Disease Detection API', 'status': 'running'})

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict disease
        with open(filepath, 'rb') as f:
            result = predict_disease(f)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'tensorflow_available': TF_AVAILABLE,
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
