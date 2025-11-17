import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://localhost:5001/api/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        setResult(response.data.result);
      } else {
        setError('Failed to get prediction');
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Error connecting to server. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🌿 Plant Disease Detection Using Machine Learning</h1>
        <p>Upload a plant leaf image to detect diseases</p>
      </header>

      <main className="App-main">
        <div className="upload-container">
          {!preview ? (
            <div className="upload-area">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                id="file-input"
                style={{ display: 'none' }}
              />
              <label htmlFor="file-input" className="upload-label">
                <div className="upload-content">
                  <span className="upload-icon">📁</span>
                  <p>Click to select an image</p>
                  <span className="upload-hint">PNG, JPG, or JPEG</span>
                </div>
              </label>
            </div>
          ) : (
            <div className="preview-container">
              <img src={preview} alt="Preview" className="preview-image" />
              <div className="button-group">
                <button onClick={handleUpload} disabled={loading} className="btn btn-primary">
                  {loading ? 'Analyzing...' : 'Detect Disease'}
                </button>
                <button onClick={handleReset} className="btn btn-secondary">
                  Choose Another Image
                </button>
              </div>
            </div>
          )}
        </div>

        {error && (
          <div className="error-message">
            <span>⚠️</span> {error}
          </div>
        )}

        {result && (
          <div className="result-container">
            <h2>Detection Result</h2>
            <div className={`result-card ${result.is_healthy ? 'healthy' : 'diseased'}`}>
              <div className="result-item">
                <span className="result-label">Status:</span>
                <span className="result-value">
                  {result.is_healthy ? '✅ Healthy Plant' : `🦠 ${result.disease}`}
                </span>
              </div>
              <div className="result-item">
                <span className="result-label">Confidence:</span>
                <span className="result-value">{(result.confidence * 100).toFixed(2)}%</span>
              </div>
            </div>

            {result.disease_info && (
              <div className="info-section">
                <h3>{result.is_healthy ? '💚 Health Status' : '🔬 Disease Information'}</h3>
                <div className="info-card">
                  <div className="info-item">
                    <span className="info-label">Severity:</span>
                    <span className={`severity-badge severity-${result.disease_info.severity.toLowerCase().replace(' ', '-')}`}>
                      {result.disease_info.severity}
                    </span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Cause:</span>
                    <span className="info-value">{result.disease_info.cause}</span>
                  </div>
                  <div className="info-item full-width">
                    <span className="info-label">Symptoms:</span>
                    <p className="info-description">{result.disease_info.symptoms}</p>
                  </div>
                  <div className="info-item full-width">
                    <span className="info-label">Treatment:</span>
                    <p className="info-description">{result.disease_info.treatment}</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="App-footer">
        <p>Group 25 </p>
      </footer>
    </div>
  );
}

export default App;
