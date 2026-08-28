export default function UploadPanel({ file, preview, loading, onFile, onAnalyze }) {
  const handleFileChange = (event) => {
    const nextFile = event.target.files?.[0] || null;
    onFile(nextFile);
    event.target.value = "";
  };

  return (
    <div>
      <h2>Upload</h2>
      <p className="muted">JPEG, PNG, BMP, or WEBP. Max 8 MB.</p>
      <label className="file-label">
        <input
          type="file"
          accept="image/jpeg,image/png,image/bmp,image/webp"
          onChange={handleFileChange}
        />
        <span>Choose image</span>
      </label>
      {preview && (
        <div className="preview-wrap">
          <img src={preview} alt="Selected preview" className="preview" />
        </div>
      )}
      <button type="button" className="primary" disabled={loading || !file} onClick={onAnalyze}>
        {loading ? "Analyzing…" : "Analyze"}
      </button>
    </div>
  );
}
