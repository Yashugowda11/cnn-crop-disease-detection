import { useState } from "react";
import "./index.css";
import "./App.css";

const API_URL = "http://127.0.0.1:5000";

const diseaseGuidance = {
  "Apple___Apple_scab": { title: "Apple scab", solution: "Remove affected leaves and fruit, improve airflow by pruning, and clear fallen leaves around the tree.", pesticide: "Use a locally approved fungicide containing captan or myclobutanil. Apply only according to the product label." },
  "Apple___Black_rot": { title: "Apple black rot", solution: "Prune dead or infected branches, remove mummified fruit, and disinfect pruning tools between cuts.", pesticide: "A copper-based fungicide or captan may help when used at the correct growth stage and label rate." },
  "Corn___Common_rust": { title: "Corn common rust", solution: "Use resistant varieties, remove heavily infected plant debris, and monitor nearby plants for new lesions.", pesticide: "If infection is severe, use an approved triazole or strobilurin fungicide for corn according to its label." },
  "Grape___Leaf_blight": { title: "Grape leaf blight", solution: "Remove infected leaves, reduce leaf wetness, improve canopy ventilation, and avoid overhead irrigation.", pesticide: "A labeled copper, mancozeb, or other approved fungicide can be used as part of an integrated treatment plan." },
  "Potato___Early_blight": { title: "Potato early blight", solution: "Remove infected foliage, rotate crops, mulch the soil, and water at the base of the plant.", pesticide: "Approved chlorothalonil, mancozeb, or copper fungicides may help prevent spread when used as directed." },
  "Tomato___Leaf_Mold": { title: "Tomato leaf mold", solution: "Increase ventilation, space plants properly, remove infected leaves, and avoid wetting the foliage.", pesticide: "Use an approved copper or chlorothalonil fungicide if necessary, following the label and harvest interval." },
  "Tomato___Healthy": { title: "Healthy tomato leaf", solution: "No visible disease was detected. Continue regular scouting, balanced watering, and good garden hygiene.", pesticide: "No pesticide is recommended for a healthy leaf. Treat only after confirming a disease and its cause." },
};

function readableDisease(value) {
  return diseaseGuidance[value]?.title || value?.replaceAll("___", " - ").replaceAll("_", " ") || "Unknown result";
}

export default function App() {
  const [image, setImage] = useState(null);
  const [fileName, setFileName] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  async function uploadImage(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setResult({ error: "Please choose a valid image file." });
      return;
    }

    setImage(URL.createObjectURL(file));
    setFileName(file.name);
    setResult(null);
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/predict`, { method: "POST", body: formData });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "The image could not be analyzed.");
      setResult(data);
    } catch (error) {
      setResult({ error: error.message || "Backend connection failed. Check that Flask is running." });
    } finally {
      setLoading(false);
    }
  }

  const guidance = result && diseaseGuidance[result.predicted_class];
  const confidence = result && Number(result.confidence || 0) * 100;
  const distribution = result?.dataset_distribution || [];
  const largestClassCount = Math.max(...distribution.map((item) => item.count), 1);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-mark">✦</div>
        <div><p className="eyebrow">AI-POWERED PLANT CARE</p><h1>LeafLens</h1></div>
        <span className="status-pill"><span /> Backend ready</span>
      </header>
      <main className="content">
        <section className="intro"><p className="eyebrow">CROP HEALTH ASSISTANT</p><h2>Understand what your leaf is telling you.</h2><p>Upload a clear leaf photo to identify a possible infection and get practical next steps for treatment.</p></section>
        <div className="workspace">
          <section className="panel upload-panel">
            <div className="panel-heading"><div><span className="step-number">01</span><h3>Upload a leaf</h3></div><span className="file-type">JPG · PNG</span></div>
            <label className={`drop-zone ${image ? "has-image" : ""}`} htmlFor="leaf-upload">
              {image ? <img className="preview" src={image} alt="Selected leaf" /> : <div className="upload-placeholder"><span className="upload-icon">↑</span><strong>Choose a leaf image</strong><span>or drag and drop it here</span></div>}
              <input id="leaf-upload" type="file" accept="image/*" onChange={uploadImage} />
            </label>
            {fileName && <p className="file-name">{fileName}</p>}
            <p className="upload-tip">For the best result, use a well-lit image with one leaf in focus.</p>
          </section>
          <section className="panel results-panel">
            <div className="panel-heading"><div><span className="step-number">02</span><h3>Health report</h3></div>{result && !result.error && <span className="result-dot">Analysis complete</span>}</div>
            {loading && <div className="loading-state"><span className="spinner" /><strong>Analyzing your leaf...</strong><span>Checking visual patterns and severity</span></div>}
            {!loading && !result && <div className="empty-state"><span>◌</span><strong>Your result will appear here</strong><p>Upload an image to start the analysis.</p></div>}
            {result?.error && <div className="error-box"><strong>Analysis unavailable</strong><p>{result.error}</p><small>Make sure the backend is running at {API_URL}.</small></div>}
            {!loading && result && !result.error && <div className="report">
              <div className="diagnosis-card"><div><span className="label">POSSIBLE CONDITION</span><h4>{readableDisease(result.predicted_class)}</h4></div><span className={`severity ${result.severity}`}>{result.severity}</span></div>
              <section className="severity-card">
                <div className="severity-heading">
                  <div><span className="label">SEVERITY ANALYSIS</span><h5>How much of the leaf appears affected?</h5></div>
                  <span className={`severity-icon ${result.severity}`}>◒</span>
                </div>
                <div className="severity-summary">
                  <div><strong>{Number(result.affected_area_percent || 0).toFixed(1)}%</strong><span>estimated affected area</span></div>
                  <span className={`severity ${result.severity}`}>{result.severity}</span>
                </div>
                <div className="severity-track"><div className={`severity-value ${result.severity}`} style={{ width: `${Math.min(Number(result.affected_area_percent || 0), 100)}%` }} /></div>
                <p className="severity-note">This is an approximate visual estimate based on leaf color and lesion segmentation. It is separate from the CNN disease classification.</p>
              </section>
              <div className="confidence-block"><div className="metric-heading"><span>Prediction confidence</span><strong>{confidence.toFixed(1)}%</strong></div><div className="progress-track"><div className="progress-value" style={{ width: `${Math.min(confidence, 100)}%` }} /></div><p>This is the CNN&apos;s estimated probability for the detected condition.</p>{result.top_predictions?.length > 1 && <div className="probability-list"><span className="label">OTHER POSSIBILITIES</span>{result.top_predictions.slice(1, 4).map((item) => <div className="probability-row" key={item.class}><span>{readableDisease(item.class)}</span><strong>{(item.probability * 100).toFixed(1)}%</strong></div>)}</div>}</div>
              <div className="guidance-grid"><article className="guidance-card"><span className="guidance-icon green">✓</span><div><span className="label">SOLUTION FOR THIS PREDICTION</span><p>{result.recommendation || guidance?.solution || "Consult a local agricultural expert to confirm the diagnosis."}</p></div></article><article className="guidance-card"><span className="guidance-icon amber">⌁</span><div><span className="label">PESTICIDE FOR THIS PREDICTION</span><p>{result.pesticide || guidance?.pesticide || "Consult a local agricultural expert for a crop-approved treatment."}</p></div></article></div>
              {distribution.length > 0 && <section className="distribution-card"><div className="distribution-heading"><div><span className="label">DATASET DISTRIBUTION</span><h5>How common is this class in the dataset?</h5></div><span className="chart-badge">PlantVillage</span></div><div className="distribution-chart">{distribution.map((item) => <div className={`distribution-row ${item.class === result.predicted_class ? "selected" : ""}`} key={item.class}><div className="distribution-label"><span>{readableDisease(item.class)}</span><strong>{item.count.toLocaleString()}</strong></div><div className="distribution-track"><div className="distribution-value" style={{ width: `${Math.max((item.count / largestClassCount) * 100, 3)}%` }} /></div></div>)}</div><p className="chart-note">This shows labeled images in the uploaded dataset, not disease incidence in real-world fields.</p></section>}
              <p className="disclaimer">Treatment products vary by region. Always confirm the diagnosis and follow the pesticide label, protective equipment guidance, and harvest interval.</p>
            </div>}
          </section>
        </div>
      </main>
      <footer>LeafLens · A decision-support tool for crop monitoring, not a substitute for professional agricultural advice.</footer>
    </div>
  );
}
