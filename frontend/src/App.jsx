import { useCallback, useEffect, useMemo, useState } from "react";
import { analyzeImage, fetchAnalysis, fetchHistory } from "./api.js";
import HistoryPanel from "./components/HistoryPanel.jsx";
import ResultPanel from "./components/ResultPanel.jsx";
import UploadPanel from "./components/UploadPanel.jsx";
import { isFirebaseConfigured, signInWithGoogle, signOutFromGoogle } from "./firebase.js";
import "./App.css";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [historyError, setHistoryError] = useState("");
  const [user, setUser] = useState(null);
  const [authLoading, setAuthLoading] = useState(false);
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);

  const loadHistory = useCallback(async (uid = user?.uid) => {
    if (!uid) {
      setHistory([]);
      return;
    }
    try {
      setHistoryError("");
      setHistory(await fetchHistory(uid));
    } catch (err) {
      setHistoryError(err.message || "Could not load history");
    }
  }, [user]);

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
    if (!user?.uid) {
      setError("Please sign in with Google first.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const data = await analyzeImage(file, user.uid);
      setResult(data);
      await loadHistory(user.uid);
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
      setResult(await fetchAnalysis(id, user?.uid));
    } catch (err) {
      setError(err.message || "Could not open analysis");
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    if (!isFirebaseConfigured()) {
      setError("Firebase is not configured. Add VITE_FIREBASE_* values in the root .env file and restart Vite.");
      return;
    }
    setAuthLoading(true);
    setError("");
    try {
      const signedInUser = await signInWithGoogle();
      setUser(signedInUser);
      await loadHistory(signedInUser.uid);
    } catch (err) {
      setError(err.message || "Could not sign in with Google");
    } finally {
      setAuthLoading(false);
    }
  };

  const handleGoogleSignOut = async () => {
    try {
      await signOutFromGoogle();
      setUser(null);
      setFile(null);
      setPreview("");
      setResult(null);
      setHistory([]);
      setHistoryError("");
      setProfileMenuOpen(false);
    } catch (err) {
      setError(err.message || "Could not sign out");
    }
  };

  useEffect(() => {
    if (!profileMenuOpen) return undefined;

    const handleOutsideClick = (event) => {
      if (!event.target.closest(".profile-menu-wrap")) {
        setProfileMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", handleOutsideClick);
    return () => document.removeEventListener("mousedown", handleOutsideClick);
  }, [profileMenuOpen]);

  const status = useMemo(() => {
    if (loading) return "loading";
    if (error) return "error";
    if (result) return "success";
    return "idle";
  }, [loading, error, result]);

  if (!user) {
    return (
      <div className="page auth-page">
        <div className="auth-card card">
          <div className="auth-badge">Secure access</div>
          <p className="eyebrow">Welcome</p>
          <h1>AI-Powered Image Quality & Defect Detection</h1>
          <p className="lede">
            Sign in with Google to keep each user history separate.
          </p>

          {isFirebaseConfigured() ? (
            <button className="primary auth-button google-auth-button" onClick={handleGoogleSignIn} disabled={authLoading}>
              <span className="google-icon" aria-hidden="true">G</span>
              <span>{authLoading ? "Signing in…" : "Continue with Google"}</span>
            </button>
          ) : (
            <div className="banner error" role="alert">
              Firebase is not configured.
            </div>
          )}
        </div>
      </div>
    );
  }

  const profileName = user.displayName || user.email || "User";
  const profileLetter = profileName.charAt(0).toUpperCase();

  return (
    <div className="page">
      <header className="hero topbar">
        <div>
          <p className="eyebrow">Internship technical assessment</p>
          <h1>AI-Powered Image Quality & Defect Detection</h1>
          <p className="lede">Logged in as {profileName}.</p>
        </div>

        <div className="profile-menu-wrap">
          <button
            className="profile-button"
            type="button"
            onClick={() => setProfileMenuOpen((open) => !open)}
            aria-expanded={profileMenuOpen}
          >
            {user.photoURL ? (
              <img className="profile-avatar" src={user.photoURL} alt={profileName} />
            ) : (
              <span className="profile-avatar profile-avatar-text">{profileLetter}</span>
            )}
            <span className="profile-name">{profileName}</span>
            <span className="profile-caret">▾</span>
          </button>

          {profileMenuOpen && (
            <div className="profile-dropdown">
              <div className="profile-summary">
                <div className="profile-summary-label">Signed in</div>
                <strong>{profileName}</strong>
                {user.email && <span>{user.email}</span>}
              </div>
              <button className="dropdown-logout" type="button" onClick={handleGoogleSignOut}>
                Logout
              </button>
            </div>
          )}
        </div>
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
