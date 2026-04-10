# 🛡️ Credit Card Fraud Detection — Production ML System

**Author:** Aniket Bhosale | `aniketbhosale2808@gmail.com`  
**Stack:** Python 3.12 · XGBoost · scikit-learn · PyTorch · MLflow · Pydantic v2 · Streamlit · Docker

---

## What This Is

Not a fraud detection model. Not a Jupyter notebook pipeline.

A **production-grade, modular ML pipeline framework** — built with explicit architectural boundaries, schema-first validation contracts, versioned artifact management, multi-model support (sklearn + PyTorch), and a full inference system.

The domain problem (fraud detection) is secondary. The primary signal is the system design.

---

## Architecture

```
src/
├── domain/          # Framework-agnostic core — entities, exceptions
├── infrastructure/  # External integrations — HuggingFace, MLflow, storage
└── application/     # Orchestration — preprocessor, trainer, predictor, pipeline
app/                 # Streamlit UI — multi-tab interface
configs/             # YAML configs — base, train, schema
tests/               # Unit tests — preprocessor, schema validation
```

### Layer Contracts

| Layer | Responsibility | What It Never Does |
|---|---|---|
| `domain/` | Entities, typed exceptions, pure logic | Imports from `application` or `infrastructure` |
| `application/` | Pipeline orchestration, phase sequencing | Directly accesses storage or external APIs |
| `infrastructure/` | MLflow, HuggingFace, disk I/O | Contains business logic |

No circular imports. No leakage. Every cross-layer call goes through an explicit interface.

---

## Pipeline Phases

| Phase | Description | Key Design Decision |
|---|---|---|
| Ingestion | Downloads CSV from HuggingFace with SHA-256 caching | Cache-first; no redundant downloads |
| Validation | Schema + quality checks via Pydantic / DataValidator | Fail-fast — invalid data never reaches the model |
| Preprocessing | StandardScaler fit on training data only | `transform()` is guarded; fails if `fit()` was never called |
| Splitting | Stratified 80/20 train/test split | Preserves fraud class imbalance for realistic evaluation |
| Training | XGBoost · RandomForest · PyTorch MLP | Unified sklearn-like interface across all model types |
| Evaluation | AUPRC (primary), Recall, F1, ROC-AUC | AUPRC is the correct metric for severe class imbalance |
| Model Selection | Auto-selects best model by AUPRC | No manual intervention; metric-driven |
| Registration | Versioned artifact storage (`models/v_{timestamp}/`) | Each version ships model + scaler + feature order |
| Reporting | MLflow experiment tracking, PR curve, feature importance | Full reproducibility of every run |

---

## Engineering Decisions

### 1. Clean Architecture with Enforced Layer Separation

The codebase is structured around three concentric layers: domain, application, and infrastructure. Dependency direction is strictly inward — infrastructure never leaks into domain, and application orchestrates without coupling to external services directly.

This is not stylistic. It means the core logic is testable without a running MLflow server, a network connection, or a filesystem.

### 2. Schema-First Validation

Two distinct validation layers:

- **`TransactionInput` / `PredictionOutput`** — Pydantic v2 models enforcing type safety and value constraints at the inference boundary. Malformed requests are rejected before any model code executes.
- **`DataValidator`** — Dataset-level schema enforcement during ingestion. Catches column mismatches, unexpected nulls, and type drift before preprocessing begins.

Validation failures raise typed exceptions from the domain layer — never silent failures, never generic `ValueError`.

### 3. Stateful Preprocessing with Scaler Safety

The `StandardScaler` is fit exclusively on training data. This is enforced at the code level: `transform()` raises a domain exception if `fit()` was never called on the current instance. Feature order is captured at fit time and serialized alongside the scaler artifact. At inference, feature alignment is verified before transform — preventing silent prediction errors caused by column reordering.

### 4. Multi-Model Support (sklearn + PyTorch)

The system supports heterogeneous model types through a unified interface:

- **XGBoost** and **RandomForest** — native sklearn estimators, no wrapper needed.
- **PyTorch MLP** — wrapped in a sklearn-compatible class exposing `fit()`, `predict()`, `predict_proba()`. The pipeline sees one interface regardless of the underlying framework.

Model persistence is handled conditionally: sklearn models serialize via `joblib`, PyTorch models via `.pt` state dictionaries. Loading is symmetric — `model_type` metadata in the artifact manifest drives the correct deserialization path.

The PyTorch MLP underperforms XGBoost on this dataset, as expected. Tabular fraud data with engineered PCA features (`V1`–`V28`) is not a regime where deep learning has a structural advantage. The MLP is included to demonstrate that the system architecture accommodates framework extension without pipeline modification — not to chase a leaderboard score.

### 5. Versioned Artifact System

Every trained model is stored as a self-contained versioned bundle:

```
models/
└── v_{timestamp}/
    ├── model          # serialized estimator (joblib or .pt)
    ├── scaler         # fitted StandardScaler
    └── feature_order  # ordered list of feature names
```

The inference system resolves the latest version automatically at startup. No manual artifact paths in production code.

### 6. Production-Style Inference System

The `Predictor` is implemented as a thread-safe singleton using double-checked locking. It loads the latest artifact bundle once at initialization and exposes two inference modes:

- **Single prediction** — accepts a JSON-serialized `TransactionInput`, returns a `PredictionOutput` with class label and fraud probability.
- **Batch prediction** — accepts a CSV path, runs predictions row-by-row through the same validation and inference path, writes results to a specified output file.

Both modes pass through the Pydantic validation layer. The batch path is not a shortcut around schema enforcement.

### 7. Typed Exception Hierarchy

```
ProjectError
├── IngestionError
├── ValidationError
├── PreprocessingError
├── TrainingError
├── EvaluationError
└── InferenceError
```

Every phase raises its own exception subclass. Error handling at the pipeline level is selective — callers catch exactly what they expect. No bare `except Exception` swallowing failures silently.

### 8. Phase Protocol Contracts

All pipeline phases implement `PhaseProtocol`. Compliance is verified at runtime with `assert isinstance(phase, PhaseProtocol)`. Adding a new phase that skips the protocol contract fails immediately at pipeline construction — not at execution time, not in production.

### 9. No Global State

All pipeline state flows through `PipelineContext`. No module-level singletons, no implicit shared mutable state between phases. Each run is isolated; parallel runs do not interfere.

---

## Evaluation Methodology

**Primary metric: AUPRC** (Area Under the Precision-Recall Curve)

ROC-AUC is misleading on severely imbalanced datasets — a classifier that predicts the majority class everywhere can achieve high AUC. AUPRC measures performance at the operating points that matter for fraud: high precision at recall thresholds where the system would actually be deployed.

Secondary metrics (Recall, F1, ROC-AUC) are logged for completeness but do not drive model selection.

---

## MLflow Integration

Every training run logs:

- Hyperparameters (model config, split ratio, random seed)
- Evaluation metrics (AUPRC, Recall, F1, ROC-AUC)
- Artifacts (PR curve plot, feature importance chart)
- Model registry entry (versioned, linked to run ID)

Runs are reproducible. Given the same data and config, the experiment can be reconstructed exactly.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy env template
cp .env.example .env

# 3. Train
python -m src.main --mode train

# 4. Single prediction
python -m src.main --mode predict --input '{"Time":0,"Amount":149.62,"V1":-1.36,"V2":-0.07,"V3":2.54,"V4":1.38,"V5":-0.34,"V6":0.46,"V7":0.24,"V8":0.10,"V9":0.36,"V10":0.09,"V11":-0.55,"V12":-0.62,"V13":-0.99,"V14":-0.31,"V15":1.47,"V16":-0.47,"V17":0.21,"V18":0.03,"V19":0.40,"V20":0.25,"V21":-0.02,"V22":0.28,"V23":-0.11,"V24":0.07,"V25":0.13,"V26":-0.19,"V27":0.13,"V28":-0.02}'

# 5. Batch prediction
python -m src.main --mode batch --input data/raw/creditcard.csv --output results.csv

# 6. Streamlit UI
streamlit run app/main_ui.py

# 7. Run tests
pytest tests/ -v --cov=src

# 8. Docker
docker build -t fraud-detector .
docker run -p 8501:8501 fraud-detector
```

---

## Streamlit UI

Four-tab interface:

| Tab | Function |
|---|---|
| Single Prediction | Input transaction fields, returns fraud probability with visualization |
| Batch Upload | Upload CSV, runs full batch inference, displays results table |
| Model Info | Displays loaded artifact version, feature list, model type |
| System Health | Confirms predictor initialization, scaler fit state, artifact integrity |

---

## Testing

Tests cover the components where silent failure is most dangerous:

| Test Suite | What It Verifies |
|---|---|
| `test_entities` | Domain entity validation — rejects malformed inputs at construction |
| `test_preprocessor` | Fit/transform lifecycle — transform before fit raises `PreprocessingError` |
| `test_schema` | DataValidator — catches column mismatches, null violations, type errors |
| `test_splitting` | Stratified split — verifies class ratio preservation across train/test sets |

Tests run against real domain logic, not mocks of mocks.

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Engineering Standards

- **PEP 8 strict** — double quotes, 4-space indent, enforced via linter
- **Full type annotations** — `from __future__ import annotations` in every module
- **No global state** — all state threaded through `PipelineContext`
- **Singleton predictor** — thread-safe, double-checked locking
- **Scaler safety** — fit only on training data; transform guarded against unfit state
- **Deterministic runs** — fixed random seeds in config; reproducible splits and training
- **No data leakage** — scaler is never fit on the full dataset or test set
- **Strict feature alignment** — feature order is serialized with the artifact and verified at inference

---

## Honest Assessment

XGBoost is the best model in this system. The PyTorch MLP is a valid addition to the architecture — it demonstrates framework-agnostic model support — but it does not close the performance gap. PCA-transformed tabular features with engineered statistics are XGBoost territory.

The system is designed to make adding a better deep learning model later a matter of implementing the interface, not restructuring the pipeline. That is the correct tradeoff.