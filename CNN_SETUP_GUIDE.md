# 🧠 CNN Model Integration Guide

## ✅ What's Been Set Up

Your project now has CNN capabilities! Here's what was added:

### Files Created:
1. **`setup_model.py`** - Downloads and configures CNN model structure
2. **`train_model.py`** - Script to train your own CNN model
3. **`models/plant_disease_model.h5`** - CNN model file (currently untrained)

### Current Status:
- ✅ Model structure created (MobileNetV2 with transfer learning)
- ✅ TensorFlow and Keras installed
- ⚠️ Model is UNTRAINED (needs dataset to train)

---

## 🚀 Option 1: Quick Start with Pre-trained Model (Recommended)

If you find a pre-trained `.h5` or `.keras` model file:

1. Place it in `backend/models/` folder
2. Rename it to `plant_disease_model.h5`
3. Stop current server (Ctrl+C)
4. Start with CNN: 
   ```bash
   cd backend
   venv/bin/python3 app.py
   ```

**Where to find pre-trained models:**
- Kaggle: https://www.kaggle.com/models
- GitHub repositories with plant disease models
- Research papers with published models

---

## 🎓 Option 2: Train Your Own Model

### Step 1: Download Dataset

Download the PlantVillage dataset:
- **Kaggle**: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
- **Size**: ~2GB
- **Images**: 50,000+ labeled plant disease images
- **Classes**: 38 different diseases

### Step 2: Extract Dataset

Extract to: `backend/dataset/plant_diseases/`

Structure should look like:
```
backend/
├── dataset/
│   └── plant_diseases/
│       ├── Apple___Apple_scab/
│       ├── Apple___Black_rot/
│       ├── Corn___Common_rust/
│       └── ... (38 total classes)
```

### Step 3: Train Model

```bash
cd backend
venv/bin/python3 train_model.py
```

**Training time:**
- CPU: 2-4 hours
- GPU: 20-40 minutes
- RAM needed: 4GB+

### Step 4: Use Trained Model

After training completes, switch to CNN mode:
```bash
venv/bin/python3 app.py
```

---

## 📊 Model Architecture

**Base Model:** MobileNetV2 (pre-trained on ImageNet)
- Lightweight and fast
- Good for mobile/web deployment
- 95%+ accuracy achievable

**Custom Layers:**
- Global Average Pooling
- Dense (256 neurons) + ReLU + Dropout
- Dense (128 neurons) + ReLU + Dropout  
- Output (25 classes) + Softmax

**Input:** 224x224 RGB images
**Output:** 25 disease classes + confidence scores

---

## 🔄 Switching Between Modes

### Demo Mode (Current):
```bash
cd backend
venv/bin/python3 app_simple.py
```
- Fast startup
- Consistent results based on image hash
- Good for testing UI/UX

### CNN Mode:
```bash
cd backend
venv/bin/python3 app.py
```
- Real AI predictions
- Requires trained model
- Higher accuracy

---

## ⚙️ Configuration

Edit `train_model.py` to customize:
- `EPOCHS = 10` - Number of training cycles
- `BATCH_SIZE = 32` - Images per batch
- `IMG_SIZE = 224` - Image dimensions
- `NUM_CLASSES = 25` - Number of disease types

---

## 🐛 Troubleshooting

### "Dataset not found" error
- Ensure dataset is in `backend/dataset/plant_diseases/`
- Check folder structure matches expected layout

### TensorFlow loading slow
- Normal on first run (downloads model weights)
- Subsequent runs are faster
- Consider using `app_simple.py` for development

### Memory errors during training
- Reduce `BATCH_SIZE` to 16 or 8
- Close other applications
- Use smaller image size (e.g., 128x128)

### Model accuracy low
- Increase `EPOCHS` (try 20-30)
- Add more data augmentation
- Fine-tune learning rate

---

## 📈 Expected Accuracy

With proper training:
- **Training Accuracy**: 95-98%
- **Validation Accuracy**: 92-96%
- **Real-world**: 85-90%

Lower accuracy may indicate:
- Need more training epochs
- Insufficient data variety
- Overfitting (validation << training)

---

## 🎯 Next Steps

**For Quick Demo:**
1. Keep using `app_simple.py` (current mode)
2. Works perfectly for demonstrations

**For Production:**
1. Download PlantVillage dataset
2. Run `train_model.py`
3. Switch to `app.py` for CNN predictions
4. Test with real plant images

**For Best Results:**
1. Train for 20-30 epochs
2. Use data augmentation
3. Test with diverse images
4. Fine-tune on your specific plants

---

## 📚 Resources

- **PlantVillage Dataset**: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
- **TensorFlow Guide**: https://www.tensorflow.org/tutorials
- **MobileNetV2 Paper**: https://arxiv.org/abs/1801.04381
- **Transfer Learning**: https://www.tensorflow.org/tutorials/images/transfer_learning

---

**Current Setup Status:**
- ✅ CNN architecture created
- ✅ Training script ready
- ⏳ Waiting for dataset to train
- 🚀 Ready for production with training

**Running Now:** Demo mode at http://localhost:5001
**Next:** Download dataset and train for real CNN predictions!
