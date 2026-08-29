# AI-Powered Image Quality & Defect Detection

Full-stack internship assessment application: upload an image, extract computer-vision quality features, run a **trained local Random Forest**, store the result in **PostgreSQL**, and browse history.

No external AI or cloud vision APIs are used. Inference runs entirely inside this repository.

## 🚀 Live Demo
Live Application:
https://ai-powered-image-zlzr.onrender.com/

Upload an image and the application will analyze:
- Blur / insufficient sharpness
- Underexposure
- Overexposure
- Image noise
- Severe degradation
- Potential visual defects

## 🎯 Assessment Coverage

| Requirement | Implementation |
|---|---|
| Image upload and analysis | React + FastAPI |
| Blur detection | Laplacian variance + issue RF |
| Underexposure | Brightness + dark-pixel ratio + issue RF |
| Overexposure | Brightness + bright-pixel ratio + issue RF |
| Image noise | Median residual + high-frequency energy + issue RF |
| Corruption | Image decoding validation + HTTP 400 |
| Severe degradation | Quality RF + severe degradation issue |
| Potential visual defect | Synthetic scratch/blob/stain detection + Isolation Forest |
| AI/ML decision component | Random Forest + Isolation Forest |
| Computer-vision features | 20 engineered image-quality features |
| REST API | FastAPI |
| Database | PostgreSQL |
| Analysis history | PostgreSQL + REST API |
| Explainability | Statistics + feature importance + confidence |
| Evaluation | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix |
| Unseen evaluation | Original-ID based train/validation/test split |
| Frontend | React + Vite |
| Containerization | Docker + Docker Compose |
| Online deployment | Render |
| Health endpoint | `/health` |
| Automated tests | Pytest + Vitest |

## Project overview

Photograph and scan pipelines fail for predictable visual reasons: blur, bad exposure, noise, file corruption, and localized defects. This project turns those signals into:

1. Engineered OpenCV/NumPy features
2. A supervised Random Forest quality classifier (`ACCEPTABLE` / `DEGRADED` / `POTENTIALLY_DEFECTIVE`)
3. A multi-output Random Forest that reports issue types
4. A supporting Isolation Forest trained on non-defect samples
5. A transparent quality score and textual explanation
6. A React UI and REST API with persistent history

## Problem statement

Given a user-uploaded image, the system must:

- Reject **file corruption** (undecodable bytes) at validation time
- Score **visual quality** of readable images
- Detect **blur, underexposure, overexposure, noise, severe degradation**
- Flag **potential visual defects** with a documented, limited formulation
- Explain the decision using the same features the model saw

## Required detection capabilities

| Capability | How it is handled |
|---|---|
| Blur / insufficient sharpness | Laplacian variance, edge density, gradient mean → issue head `blur` |
| Underexposure | Brightness, dark-pixel ratio, histogram skew → `underexposure` |
| Overexposure | Brightness, bright-pixel ratio → `overexposure` |
| Image noise | Median residual + high-frequency energy → `noise` |
| Corruption vs severe degradation | Undecodable files: HTTP 400 `corrupted_file`. Readable but collapsed images: class `DEGRADED` + `severe_degradation` |
| Potential visual defect | Localized scratches/blobs/stains in **synthetic training**; issue `visual_defect` + Isolation Forest support |

## Features

- Image upload, preview, loading / success / error states
- Quality score, quality label, issues (type, severity, confidence)
- Image statistics and explanation (including RF feature importances)
- Analysis history stored in PostgreSQL
- Docker Compose: frontend, backend, Postgres
- Reproducible dataset generation, training, and unseen-test evaluation

## Architecture


                    ┌─────────────────┐
                    │      User       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   React + Vite  │
                    └────────┬────────┘
                             │
                    Image Upload
                             │
                             ▼
                    ┌─────────────────┐
                    │ FastAPI Backend │
                    └────────┬────────┘
                             │
                    Validate / Decode
                             │
                             ▼
                    ┌─────────────────┐
                    │ OpenCV + NumPy  │
                    │ Feature Extract │
                    └────────┬────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │      ML Inference            │
              │                              │
              │ Quality Random Forest        │
              │ Issue Random Forest          │
              │ Isolation Forest             │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │ Score + Issues + Explanation │
              └──────────────┬───────────────┘
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
          ┌──────────────┐       ┌────────────┐
          │  PostgreSQL  │       │ React UI   │
          │   History    │       │   Result   │
          └──────────────┘       └────────────┘


## Technology stack

- Frontend: React, Vite, CSS
- Backend: Python, FastAPI
- CV: OpenCV, NumPy
- ML: scikit-learn Random Forest (primary), Isolation Forest (supporting)
- DB: PostgreSQL, SQLAlchemy, psycopg3
- Tests: Pytest, Vitest
- Deploy: Docker, Docker Compose, Render

## Computer vision features

Implemented in `backend/ml/feature_extraction/features.py`. For each feature: what it measures, why it is relevant, how it is calculated, and how it is used.

| Feature | Measures | Why | How | ML / explanation role |
|---|---|---|---|---|
| `sharpness_laplacian_var` | Focus / edge energy | Blur suppresses second derivatives | Variance of Laplacian on gray | Blur |
| `edge_density` | Canny edge fraction | Blur and wash-out remove edges | Mean of Canny map | Blur, severe degradation |
| `gradient_mean` | Texture | Blur flattens gradients | Mean Sobel magnitude | Texture / blur |
| `brightness_mean` | Luminance | Exposure | Mean gray | Under/overexposure |
| `brightness_p5` / `p95` | Shadow / highlight tails | Mean hides clipping | Percentiles | Exposure |
| `dark_pixel_ratio` | Near-black share | Underexposure | `gray < 20` | Underexposure |
| `bright_pixel_ratio` | Near-white share | Overexposure | `gray > 235` | Overexposure |
| `contrast_std` | Global contrast | Severe degradation reduces spread | Std of gray | Severe degradation |
| `histogram_entropy` | Intensity information | Clipped images have low entropy | Shannon entropy of 256-bin hist | Quality / wash-out |
| `histogram_skew` | Histogram asymmetry | Dark vs bright skew | 3rd standardized moment | Exposure |
| `noise_median_residual` | Grain | Noise remains after median filter | MAE vs `medianBlur(k=5)` | Noise |
| `high_freq_energy` | Fine detail + grain | Complements median residual | MAE vs Gaussian blur | Noise vs sharpness |
| `saturation_mean` / `std` | Color richness / variation | Wash-out vs stains | HSV S channel | Color / stains |
| `color_channel_imbalance` | Color cast | Stains / casts unbalance BGR | Max \|channel mean − gray mean\| | Defect / color |
| `local_mean_deviation` | Strongest tile anomaly | Localized defects vs global issues | Max z-score of 8×8 tile means | Visual defect |
| `tophat_residue` / `blackhat_residue` | Thin bright/dark structures | Scratches | Morphological top-hat / black-hat | Visual defect |
| `extreme_blob_ratio` | Compact extreme blobs | Spots vs global clip | Connected components in intensity tails with bounded area | Visual defect |

API `statistics` expose a compact subset: sharpness, brightness, contrast, noise, saturation, dark/bright ratios, local anomaly, edge density.

## Dataset

**Source:** procedurally generated 256×256 synthetic scenes (gradients, sinusoids, shapes, lines) via OpenCV/NumPy. Seed `42`. This is **not** a public photo dataset; it is a reproducible synthetic base so degradations are controlled and documented.

**Why synthetic:** the assessment allows a reproducible synthetic degradation strategy. Using generated clean frames avoids scraping and keeps splits leak-free.

## Dataset generation

```bash
python scripts/generate_dataset.py --n-originals 140 --seed 42
```

Outputs:

- `data/raw/originals/` — clean originals
- `data/processed/images/` — all variants
- `data/processed/manifest.csv`
- `data/processed/dataset_meta.json`
- copies into `samples/`

Current generation: **140 originals → 980 samples** (7 variants each).

## Synthetic degradation methodology

| Variant | Operation | Quality label | Issue labels |
|---|---|---|---|
| clean | none | ACCEPTABLE | none |
| blur | GaussianBlur kernel 11–25, σ 3–8 | DEGRADED | blur |
| underexposure | × 0.18–0.42 | DEGRADED | underexposure |
| overexposure | × 1.7–2.4 + bias 30–70, clip | DEGRADED | overexposure |
| noise | Gaussian σ 18–42, optional 1–4% salt-pepper | DEGRADED | noise |
| severe | blur + noise + contrast crush + optional darken | DEGRADED | severe_degradation, blur, noise |
| defect | localized scratch, compact blob, or color stain | POTENTIALLY_DEFECTIVE | visual_defect |

## Train / validation / test methodology

Split is **by original image id**, not by file:

- 98 originals → train
- 21 originals → validation
- 21 originals → unseen test

Every degraded twin of an original stays in the same split.

## Data leakage prevention

`scripts/generate_dataset.py` assigns `split` on `original_id` only. Training never sees a blur/noise/defect version of a test original.

## 🤖 AI / ML Approach

The application uses trained machine-learning models for the final quality
decision rather than relying solely on hard-coded computer-vision rules.

### Models

- **Quality Random Forest:** 3-class classification:
  `ACCEPTABLE`, `DEGRADED`, `POTENTIALLY_DEFECTIVE`
- **Issue Random Forest:** Multi-output classifier for the six issue types
- **Isolation Forest:** Supporting anomaly detector trained on non-defect
  training samples

### Input

The models operate on a 20-dimensional engineered image-quality feature
vector extracted using OpenCV and NumPy.

### Training

The models are trained offline using the generated dataset. The trained
pipeline is saved as:

`models/quality_pipeline.joblib`

At inference time, the trained model is loaded by the backend and used to
generate predictions for uploaded images.


## ML model selection

Two supervised models plus one anomaly model, saved together in `models/quality_pipeline.joblib`:

1. **Quality Random Forest** — 3-class (`ACCEPTABLE`, `DEGRADED`, `POTENTIALLY_DEFECTIVE`)
2. **Issue Random Forest** — `MultiOutputClassifier` of binary RFs for the six issue types
3. **Isolation Forest** — fit on **non-defect training** vectors; supports visual-defect when local anomaly features are strong

Predictions are **not** hard-coded. Thresholds used after the model (issue reporting at probability ≥ 0.40, severity bins) are documented scoring rules, not a substitute for the forest.

## Why Random Forest

- Tabular engineered features (not pixels)
- Nonlinear combinations of sharpness/exposure/noise/local cues
- Class-balanced trees for the minority ACCEPTABLE / DEFECTIVE classes
- Native `feature_importances_` for explanations
- Fast local inference without a GPU

## Training process

```bash
python scripts/train_model.py
```

1. Read manifest
2. Extract the 20-D feature vector per image
3. Fit quality RF (`n_estimators=220`, `max_depth=14`, `class_weight=balanced`)
4. Fit issue multi-output RF
5. Fit Isolation Forest on non-defect train rows
6. Set anomaly threshold to the 5th percentile of non-defect **validation** scores
7. Save `models/quality_pipeline.joblib` and `models/training_meta.json`

Training does **not** run at upload time.

Validation (not the test set): quality macro-F1 **0.789**. Issue F1: blur 0.963, underexposure 1.0, overexposure 1.0, noise 1.0, severe 0.923, visual_defect 0.706.

## Model saving / loading

- Save: `joblib.dump` of models, feature names, class lists, anomaly threshold, `model_version`
- Load: `QualityPredictor` in `backend/ml/inference/predictor.py`
- Production path: `MODEL_PATH` (default `models/quality_pipeline.joblib`)

## Inference process

`POST /api/analyze`: validate → `cv2.imdecode` → features → quality RF + issue RF + Isolation Forest → issues/severity → quality score → explanation → insert row → JSON.

## Quality score methodology

Transparent combination of class probabilities and issue penalties (clipped to 0–100):

```
score = 100 * P(ACCEPTABLE) + 55 * P(DEGRADED) + 25 * P(POTENTIALLY_DEFECTIVE)
        − {low: 4, medium: 10, high: 18} per reported issue
```

This is a **scoring layer on top of ML probabilities**, not a hidden rule-based classifier.

## Quality labels

| Label | Meaning in this project | How labels are created |
|---|---|---|
| ACCEPTABLE | Clean synthetic scene, no applied degradation | `variant=clean` |
| DEGRADED | Global quality problem (blur, exposure, noise, or combined collapse) | corresponding degradations |
| POTENTIALLY_DEFECTIVE | Localized synthetic defect on an otherwise intact scene | `variant=defect` |

At inference, the quality RF is primary. A high-confidence `visual_defect` issue can map the displayed label to `POTENTIALLY_DEFECTIVE`.

## Issue detection

Issue probabilities come from the multi-output RF. An issue is **reported** if P ≥ 0.40. Severity uses probability plus extreme feature checks (for example very low Laplacian variance → high blur).

`visual_defect` in this codebase means: **scratch / blob / stain patterns similar to the synthetic generator**, not arbitrary industrial defects on unseen products.

## Explainability

Each response includes:

- `explanation.summary`
- `explanation.contributing_factors` derived from statistics
- `explanation.feature_importances` from the quality RF
- per-issue confidence
- class probabilities

Grad-CAM is not used (no CNN).

## Evaluation methodology

```bash
python scripts/evaluate_model.py
```

Uses the **unseen test originals** only (`split=test`, 21 originals, 147 images). Metrics: accuracy, macro precision/recall/F1, confusion matrix, per-issue precision/recall/F1, and per-issue ROC-AUC (both classes present). Quality-level multi-class ROC-AUC was attempted; sklearn returned an error on the rounded probability matrix used in that script, so **quality ROC-AUC is not reported**. Per-issue ROC-AUC **is** reported from issue-head probabilities.

Full dump: `reports/evaluation.json`.

### 📊 Key Evaluation Results

Evaluation was performed on 147 unseen test images from 21 original
images that were not used during training.

| Metric | Result |
|---|---:|
| Accuracy | **89.1%** |
| Macro Precision | **75.4%** |
| Macro Recall | **74.6%** |
| Macro F1 | **74.7%** |

The evaluation also includes per-class F1, confusion matrix, per-issue
precision/recall/F1, ROC-AUC for issue detection, failure cases, and
uncertain predictions.




## Actual evaluation results (unseen test)

Generated by `scripts/evaluate_model.py` on 2026-08-27. Do not treat these as real-camera production metrics.

**Quality (3-class)**

- Accuracy: **0.891**
- Macro precision: **0.754**
- Macro recall: **0.746**
- Macro F1: **0.747**
- Per class F1: ACCEPTABLE 0.579, DEGRADED 0.995, POTENTIALLY_DEFECTIVE 0.667

**Confusion matrix** (rows = true, columns = predicted; order ACCEPTABLE, DEGRADED, POTENTIALLY_DEFECTIVE)

```
[[ 11,  1,  9],
 [  0,105,  0],
 [  6,  0, 15]]
```

**Issue heads (test)**

| Issue | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| blur | 0.953 | 0.976 | 0.965 | 0.999 |
| underexposure | 1.000 | 1.000 | 1.000 | 1.000 |
| overexposure | 1.000 | 1.000 | 1.000 | 1.000 |
| noise | 1.000 | 1.000 | 1.000 | 1.000 |
| severe_degradation | 0.833 | 0.952 | 0.889 | 0.997 |
| visual_defect | 0.548 | 0.810 | 0.654 | 0.950 |

Exposure and noise scores of 1.0 reflect **strong, controlled synthetic degradations**, not guaranteed performance on real photographs.

## Failure cases / incorrect predictions

16 / 147 test images were mislabeled:

- 10 **clean** images predicted `POTENTIALLY_DEFECTIVE` (and 1 clean → `DEGRADED`)
- 6 **defect** images predicted `ACCEPTABLE`

Geometric shapes in clean scenes can look like blobs/scratches to local-anomaly features. Faint stains can look like part of the synthetic scene. **No blur/exposure/noise/severe test sample was confused with another global class in this run.**

## Uncertain predictions

Confidence &lt; 0.55: **3** test images (see `reports/evaluation.json`). Low confidence is reported honestly rather than forced.

## Limitations & Generalization

- Training images are **synthetic**, not camera captures or a public defect dataset (for example NEU-DET). Global quality cues transfer better than defect cues.
- `POTENTIALLY_DEFECTIVE` is a **narrow** definition (scratches, blobs, stains in this generator).
- Perfect issue F1 on exposure/noise will **drop** on real data.
- Isolation Forest is supporting, not a full unsupervised defect detector.
- Images are not stored in PostgreSQL (metadata only).
- Quality multi-class ROC-AUC is omitted (see evaluation section).

## API documentation

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Process, model, database status |
| POST | `/api/analyze` | multipart `file` → analysis JSON |
| GET | `/api/analyses` | History (`limit`, default 50) |
| GET | `/api/analyses/{id}` | One analysis |

### Example response schema

```json
{
  "analysis_id": "1",
  "created_at": "2026-08-27T17:00:00+00:00",
  "filename": "photo.png",
  "quality_score": 82,
  "quality_label": "ACCEPTABLE",
  "quality_confidence": 0.71,
  "issues": [{ "type": "noise", "severity": "low", "confidence": 0.71 }],
  "statistics": {
    "sharpness": 421.5,
    "brightness": 128.2,
    "contrast": 57.4,
    "noise": 6.8,
    "saturation": 0.54,
    "dark_pixel_ratio": 0.01,
    "bright_pixel_ratio": 0.0,
    "local_anomaly": 0.8,
    "edge_density": 0.12
  },
  "explanation": {
    "summary": "...",
    "contributing_factors": ["..."],
    "feature_importances": [{ "feature": "contrast_std", "importance": 0.09 }]
  },
  "class_probabilities": {
    "ACCEPTABLE": 0.7,
    "DEGRADED": 0.2,
    "POTENTIALLY_DEFECTIVE": 0.1
  },
  "model_version": "1.0.0",
  "anomaly_score": 0.12
}
```

Error bodies use `{ "detail": { "message": "...", "code": "..." } }` (no stack traces). Codes include `empty_upload`, `invalid_file_type`, `file_too_large`, `corrupted_file`, `inference_error`, `database_error`, `not_found`.

### Example API requests

Health:

```bash
curl http://localhost:8000/health
```

Analyze:

```bash
curl -F "file=@samples/02_blurry.png" http://localhost:8000/api/analyze
```

History:

```bash
curl http://localhost:8000/api/analyses
curl http://localhost:8000/api/analyses/1
```

## PostgreSQL setup

Docker (recommended):

```bash
docker compose up -d postgres
```

Connection URL (local):

```
DATABASE_URL=postgresql+psycopg://iqd:iqd@localhost:5432/image_quality
```

Tables are created on API startup (`Base.metadata.create_all`). Model: `analyses` with JSONB `issues`, `statistics`, `explanation`, `class_probabilities`. **Uploaded pixels are not stored.**

PostgreSQL is **not a website**. Do not open `http://localhost:5432` in a browser (that is the database protocol port).

To browse tables in a browser, start pgAdmin (included in Compose):

```bash
docker compose up -d pgadmin
```

Then open [http://localhost:5050](http://localhost:5050)

- Email: `admin@localhost.com`
- Password: `admin`
- Pre-registered server **Image Quality Postgres** — when prompted, password is `iqd`

From the terminal:

```bash
docker compose exec postgres psql -U iqd -d image_quality -c "SELECT id, filename, quality_label FROM analyses;"
```

## Environment variables

Copy `.env.example` to `.env`. Do not commit secrets. Variables:

| Variable | Role |
|---|---|
| `DATABASE_URL` | SQLAlchemy URL (`postgresql+psycopg://...`) |
| `BACKEND_PORT` | Host port for API (Compose) |
| `FRONTEND_PORT` | Host port for nginx UI (Compose, default 8080) |
| `MODEL_PATH` | Joblib pipeline path |
| `MODEL_VERSION` | Recorded on each analysis |
| `MAX_UPLOAD_BYTES` | Upload cap (default 8 MiB) |
| `CORS_ORIGINS` | Comma-separated browser origins |
| `POSTGRES_USER` / `PASSWORD` / `DB` / `PORT` | Compose Postgres |

## Docker / Docker Compose

```bash
docker compose up --build
```

- UI: http://localhost:8080 (nginx proxies `/api` and `/health` to the backend)
- API: http://localhost:8000
- Postgres: localhost:5432

The backend image includes FastAPI, OpenCV, NumPy, scikit-learn, and `models/`. Compose also bind-mounts `./models`.

Healthcheck: Postgres `pg_isready`; backend `GET /health`.

Stop:

```bash
docker compose down
```

## Health endpoint

`GET /health` → `{ status, model_loaded, model_version, database }` where `database` is `up` or `down`. `status` is `ok` only if the model is loaded and the database answers.

## Local development

Requires Python 3.11+, Node 20+, and Postgres (Compose service is enough).

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
pip install -r backend/requirements.txt
docker compose up -d postgres
python scripts/generate_dataset.py   # if data/ is missing
python scripts/train_model.py        # if models/quality_pipeline.joblib is missing

# from repo root
.\.venv\Scripts\uvicorn app.main:app --reload --app-dir backend --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api` and `/health` to `http://127.0.0.1:8000`.

## Sample images

| File | Condition |
|---|---|
| `samples/01_acceptable_clean.png` | Clean / acceptable |
| `samples/02_blurry.png` | Blur |
| `samples/03_underexposed.png` | Underexposure |
| `samples/04_overexposed.png` | Overexposure |
| `samples/05_noisy.png` | Noise |
| `samples/06_severely_degraded.png` | Severe degradation |
| `samples/07_potential_visual_defect.png` | Localized synthetic defect |
| `samples/08_corrupted_not_an_image.png` | Invalid bytes (validation failure) |

## Testing

Backend (Postgres must be running, model file present):

```bash
cd backend
..\.venv\Scripts\pytest -q
```

Covers health, valid upload, invalid type, corrupted image, response schema, DB persistence, history, and CV feature sanity checks.

Frontend:

```bash
cd frontend
npm test
```

## Deployment instructions

## 🚀 Deployment

### Live Deployment

The application is deployed on Render.

**Live Application:**  
https://ai-powered-image-zlzr.onrender.com/

### Docker Compose Deployment

The complete application can also be run locally using Docker Compose.

1. Copy `.env.example` → `.env`
2. Set a strong `POSTGRES_PASSWORD`
3. Ensure `models/quality_pipeline.joblib` exists
4. Start the services:

```bash
docker compose up --build -d
```

## Project structure

```
frontend/          React + Vite + nginx
backend/app/      FastAPI, DB, validation, explainability
backend/ml/       feature extraction + inference
backend/tests/
scripts/           generate_dataset, train_model, evaluate_model
data/              generated dataset (raw originals gitignored)
models/            trained joblib
samples/           demo images
reports/           evaluation.json
docker-compose.yml
```

## Assessment requirement checklist

| Requirement | Implementation | Location | Verification |
|---|---|---|---|
| Blur | Laplacian + issue RF `blur` | `features.py`, `predictor.py` | Sample `02_blurry.png` → DEGRADED / blur |
| Underexposure | Brightness / dark ratio + `underexposure` | same | `03_underexposed.png` |
| Overexposure | Brightness / bright ratio + `overexposure` | same | `04_overexposed.png` |
| Noise | Median residual + `noise` | same | `05_noisy.png` |
| Corruption vs severe degradation | Decode fail vs `severe_degradation` | `image_validation.py` | `08_*.png` HTTP 400; `06_*.png` DEGRADED |
| Potential visual defect | Synthetic scratch/blob/stain + Isolation Forest support | `generate_dataset.py`, `predictor.py` | `07_*.png` POTENTIALLY_DEFECTIVE |
| Genuine ML | Trained RF joblib, not rules | `scripts/train_model.py`, `models/` | Training + evaluation scripts |
| CV features | 20 documented features | `backend/ml/feature_extraction/features.py` | pytest feature tests |
| Dataset / no leakage | Split by `original_id` | `scripts/generate_dataset.py` | `dataset_meta.json` |
| Unseen evaluation | Test originals only | `scripts/evaluate_model.py` | `reports/evaluation.json` |
| Explainability | Summary, factors, importances | `explainability.py` | Analyze JSON |
| REST API | health, analyze, history | `backend/app/api/` | curl / pytest |
| PostgreSQL history | SQLAlchemy JSONB | `backend/app/db/models.py` | pytest + Docker |
| React UI | upload, preview, result, history | `frontend/src/` | Vitest + http://localhost:8080 |
| Docker Compose | frontend, backend, postgres | `docker-compose.yml` | `docker compose ps` healthy |
| Tests | Pytest + Vitest | `backend/tests/`, `frontend/src/App.test.jsx` | 12 passed / 2 passed |

## ⭐ Optional / Bonus Features
### Implemented
- Model versioning
- Automated backend tests
- Automated frontend tests
- CI/CD workflow


