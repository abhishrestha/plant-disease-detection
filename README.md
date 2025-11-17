# 🌿 Plant Disease Detection Web App

A simple and effective web application that uses machine learning to detect plant diseases from leaf images. Upload a photo of a plant leaf, and the app will identify the plant type and any diseases present.

## 🚀 Features

- **Simple Image Upload**: Drag and drop or click to upload plant leaf images
- **AI-Powered Detection**: Uses a Convolutional Neural Network (CNN) to identify diseases
- **Real-time Results**: Get instant predictions with confidence scores
- **Comprehensive Plant Information**: 
  - Scientific name and botanical family
  - Plant type and climate requirements
  - Common diseases for each plant
- **Detailed Disease Analysis**:
  - Disease severity level
  - Causative agent (fungus, bacteria, virus, or pest)
  - Symptoms description
  - Treatment recommendations
- **Clean UI**: Modern, responsive interface built with React
- **Multiple Plant Support**: Detects diseases in Apple, Corn, Grape, Potato, and Tomato plants

## 🛠️ Tech Stack

**Frontend:**
- React.js
- Axios for API calls
- CSS3 for styling

**Backend:**
- Python 3.8+
- Flask (REST API)
- TensorFlow/Keras (Machine Learning)
- Pillow (Image Processing)

## 📋 Prerequisites

Before you begin, ensure you have installed:
- Node.js (v14 or higher)
- Python (v3.8 or higher)
- pip (Python package manager)

## 🔧 Installation & Setup

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Create a trained model:
```bash
python download_model.py
```
Note: This creates a model structure. For production, you should train it on the PlantVillage dataset or use a pre-trained model.

5. Start the Flask server:
```bash
python app.py
```

The backend will run on `http://localhost:5000`

### Frontend Setup

1. Open a new terminal and navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node dependencies:
```bash
npm install
```

3. Start the React development server:
```bash
npm start
```

The frontend will run on `http://localhost:3000`

## 🎯 Usage

1. Make sure both backend and frontend servers are running
2. Open your browser and go to `http://localhost:3000`
3. Click on the upload area or drag and drop a plant leaf image
4. Click "Detect Disease" button
5. View the results showing:
   - Plant type
   - Disease name (or "Healthy" status)
   - Confidence score

## 📁 Project Structure

```
PlantDiseaseDetection/
├── backend/
│   ├── app.py                 # Flask API server
│   ├── download_model.py      # Model creation script
│   ├── requirements.txt       # Python dependencies
│   ├── models/               # ML model storage
│   └── uploads/              # Temporary image uploads
│
└── frontend/
    ├── public/
    │   └── index.html        # HTML template
    ├── src/
    │   ├── App.js           # Main React component
    │   ├── App.css          # Styling
    │   ├── index.js         # React entry point
    │   └── index.css        # Global styles
    └── package.json         # Node dependencies
```

## 🤖 Supported Plants & Diseases

The model can detect diseases in:

- **Apple**: Apple scab, Black rot, Cedar apple rust, Healthy
- **Corn**: Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy
- **Grape**: Black rot, Esca, Leaf blight, Healthy
- **Potato**: Early blight, Late blight, Healthy
- **Tomato**: Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, Mosaic virus, Healthy

## 🔄 API Endpoints

### `GET /`
Health check endpoint

### `POST /api/predict`
Upload image for disease detection
- **Request**: multipart/form-data with 'file' field
- **Response**: JSON with plant type, disease name, and confidence score

### `GET /api/health`
Check if server and model are running properly

## 📝 Notes

- The current model is a basic structure. For production use, train it on a proper dataset like PlantVillage
- Images should be clear photos of plant leaves
- Supported formats: PNG, JPG, JPEG
- Maximum file size: 16MB

## 🚀 Future Enhancements

- Train model on actual PlantVillage dataset
- Add treatment recommendations
- Support more plant species
- Mobile app version
- History of predictions
- User authentication

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements!

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- PlantVillage Dataset for disease images
- TensorFlow team for the ML framework
- React team for the frontend framework

---

Made with ❤️ and 🤖 Machine Learning
