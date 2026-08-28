import { useCallback, useEffect, useMemo, useState } from "react";
import { analyzeImage, fetchAnalysis, fetchHistory } from "./api.js";
import HistoryPanel from "./components/HistoryPanel.jsx";
import ResultPanel from "./components/ResultPanel.jsx";
import UploadPanel from "./components/UploadPanel.jsx";
import "./App.css";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [historyError, setHistoryError] = useState("");

  const loadHistory = useCallback(async () => {
    try {
      setHistoryError("");
      setHistory(await fetchHistory());
    } catch (err) {
      setHistoryError(err.message || "Could not load history");
    }
  }, []);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  useEffect(() => {
    if (!file) {
      setPreview("");
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const onAnalyze = async () => {
    if (!file) {
      setError("Choose an image first.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const data = await analyzeImage(file);
      setResult(data);
      await loadHistory();
    } catch (err) {
      setResult(null);
      setError(err.message || "Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  const onSelectHistory = async (id) => {
    setLoading(true);
    setError("");
    try {
      setResult(await fetchAnalysis(id));
    } catch (err) {
      setError(err.message || "Could not open analysis");
    } finally {
      setLoading(false);
    }
  };

  const status = useMemo(() => {
    if (loading) return "loading";
    if (error) return "error";
    if (result) return "success";
    return "idle";
  }, [loading, error, result]);

  return (
    <div className="page">
      <header className="hero">
        <p className="eyebrow">Internship technical assessment</p>
        <h1>AI-Powered Image Quality & Defect Detection</h1>
        <p className="lede">
          Local computer-vision features and a trained Random Forest. No external vision APIs.
        </p>
      </header>

      <main className="layout">
        <section className="card">
          <UploadPanel
            file={file}
            preview={preview}
            loading={loading}
            onFile={setFile}
            onAnalyze={onAnalyze}
          />
          {status === "error" && (
            <div className="banner error" role="alert">
              {error}
            </div>
          )}
          {status === "loading" && (
            <div className="banner loading" role="status">
              Analyzing image…
            </div>
          )}
        </section>

        <section className="card">
          <ResultPanel result={result} />
        </section>

        <section className="card history-card">
          <HistoryPanel items={history} error={historyError} onSelect={onSelectHistory} />
        </section>
      </main>
    </div>
  );
}
