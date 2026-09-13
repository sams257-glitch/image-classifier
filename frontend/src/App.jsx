import { useEffect, useState } from "react";
import "./App.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  function handleFileChange(event) {
    const file = event.target.files[0];

    if (file) {
      setSelectedFile(file);
      setResults(null);
      setError(null);
    }
  }

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }

    const url = URL.createObjectURL(selectedFile);

    setPreviewUrl(url);

    return () => {
      URL.revokeObjectURL(url);
    };
  }, [selectedFile]);

  async function handlePredict() {
    if (!selectedFile) {
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error("Prediction request failed.");
      }

      const data = await response.json();

      setResults(data.predictions);
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app">
      <div className="container">
        <header className="header">
          <p className="eyebrow">COMPUTER VISION</p>

          <h1>Natural Scene<br />Image Classifier</h1>

          <p className="subtitle">
            Upload a scene and let a ResNet-18 model classify it.
          </p>
        </header>

        <section className="card">
          <label className="upload-area">
            <span className="upload-icon">↑</span>

            <span className="upload-title">
              Choose an image
            </span>

            <span className="upload-description">
              JPG, JPEG, PNG or other image files
            </span>

            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
            />
          </label>

          {selectedFile && (
            <div className="preview-section">
              <p className="filename">
                {selectedFile.name}
              </p>

              {previewUrl && (
                <img
                  className="preview-image"
                  src={previewUrl}
                  alt="Selected scene"
                />
              )}

              <button
                className="predict-button"
                onClick={handlePredict}
                disabled={loading}
              >
                {loading ? "Analyzing..." : "Predict scene"}
              </button>
            </div>
          )}

          {error && (
            <p className="error">
              {error}
            </p>
          )}
        </section>

        {results && (
          <section className="results-card">
            <div className="results-header">
              <div>
                <p className="eyebrow">MODEL OUTPUT</p>
                <h2>Predictions</h2>
              </div>
            </div>

            <div className="results">
              {results.map((result, index) => (
                <div
                  className="result"
                  key={result.class}
                >
                  <div className="result-top">
                    <span className="class-name">
                      {result.class}
                    </span>

                    <span className="confidence">
                      {(result.confidence * 100).toFixed(2)}%
                    </span>
                  </div>

                  <div className="confidence-track">
                    <div
                      className={`confidence-bar ${
                        index === 0 ? "primary" : ""
                      }`}
                      style={{
                        width: `${result.confidence * 100}%`,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        <footer>
          <span>PyTorch</span>
          <span>•</span>
          <span>ResNet-18</span>
          <span>•</span>
          <span>FastAPI</span>
          <span>•</span>
          <span>React</span>
        </footer>
      </div>
    </main>
  );
}

export default App;