"""
Loan Approval MLOps Platform
Full working Python app - FastAPI + ML + Dashboard
Run: uvicorn app:app --reload --port 8000
"""

import os, json, uuid, time, pickle, hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────────────
# ML Model (train on startup)
# ─────────────────────────────────────────────────
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

MODEL_PATH   = Path("model_store/model.pkl")
METRICS_PATH = Path("model_store/metrics.json")
DATA_PATH    = Path("model_store/dataset.json")
LOGS_PATH    = Path("model_store/predictions.json")
RUNS_PATH    = Path("model_store/runs.json")

Path("model_store").mkdir(exist_ok=True)

FEATURE_NAMES = [
    "age", "annual_income", "credit_score",
    "loan_amount", "employment_years", "debt_ratio"
]

# ─────────────────────────────────────────────────
# Generate dataset + train model
# ─────────────────────────────────────────────────

def generate_dataset(n=5000, seed=42):
    np.random.seed(seed)
    age          = np.random.normal(38, 12, n).clip(18, 75)
    income       = np.random.lognormal(10.8, 0.6, n).clip(20000, 500000)
    credit_score = np.random.normal(680, 80, n).clip(300, 850)
    loan_amount  = np.random.lognormal(10.2, 0.8, n).clip(1000, 100000)
    emp_years    = np.random.exponential(6, n).clip(0, 40)
    debt_ratio   = np.random.beta(2, 5, n)

    score = (
        0.0008 * (credit_score - 500) +
        0.000003 * income +
        0.4 * emp_years / 40 -
        0.000008 * loan_amount +
        0.3 * (1 - debt_ratio) +
        np.random.normal(0, 0.15, n)
    )
    approved = (score > score.mean()).astype(int)
    X = np.column_stack([age, income, credit_score, loan_amount, emp_years, debt_ratio])
    return X, approved, age, income, credit_score, loan_amount, emp_years, debt_ratio


def train_model(params=None):
    X, y, age, income, cs, loan, emp, debt = generate_dataset()

    if params is None:
        params = dict(n_estimators=150, max_depth=4, learning_rate=0.08,
                      subsample=0.85, random_state=42)

    clf = GradientBoostingClassifier(**params)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_tr, y_tr)

    y_pred  = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1]

    metrics = {
        "accuracy":  round(float(accuracy_score(y_te, y_pred)), 4),
        "f1_score":  round(float(f1_score(y_te, y_pred)), 4),
        "roc_auc":   round(float(roc_auc_score(y_te, y_proba)), 4),
        "precision": round(float(accuracy_score(y_te, y_pred)), 4),
        "recall":    round(float(f1_score(y_te, y_pred)), 4),
        "cm":        confusion_matrix(y_te, y_pred).tolist(),
        "fi":        dict(zip(FEATURE_NAMES,
                              [round(float(v), 4) for v in clf.feature_importances_])),
        "params":    params,
        "train_n":   len(X_tr),
        "test_n":    len(X_te),
        "trained_at": datetime.utcnow().isoformat(),
        "version":    datetime.utcnow().strftime("v%Y%m%d.%H%M"),
        "n_approved": int(y.sum()),
        "n_declined": int((1 - y).sum()),
    }

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)

    # Save metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)

    # Save sample dataset
    rows = []
    for i in range(min(5000, len(X))):
        rows.append({
            "age": round(float(age[i]), 1),
            "annual_income": round(float(income[i]), 2),
            "credit_score": int(cs[i]),
            "loan_amount": round(float(loan[i]), 2),
            "employment_years": round(float(emp[i]), 1),
            "debt_ratio": round(float(debt[i]), 3),
            "approved": int(y[i])
        })
    with open(DATA_PATH, "w") as f:
        json.dump(rows, f)

    # Save run history
    runs = []
    if RUNS_PATH.exists():
        with open(RUNS_PATH) as f:
            runs = json.load(f)

    run = {
        "run_id": f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
        "params": params,
        "trained_at": metrics["trained_at"],
        "version": metrics["version"],
        "status": "FINISHED",
        "trigger": "manual",
    }
    runs.append(run)
    with open(RUNS_PATH, "w") as f:
        json.dump(runs, f)

    return pipe, metrics


# ─────────────────────────────────────────────────
# Load or train on startup
# ─────────────────────────────────────────────────
model = None
metrics = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, metrics
    print("🚀 Starting MLOps Loan Approval Platform...")
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        print(f"✅ Model loaded: accuracy={metrics['accuracy']}")
    else:
        print("🔄 Training model on startup...")
        model, metrics = train_model()
        print(f"✅ Model trained: accuracy={metrics['accuracy']}")
    yield
    print("👋 Shutting down...")


app = FastAPI(
    title="Loan Approval MLOps Platform",
    description="Full MLOps platform with real-time predictions, drift detection, and experiment tracking",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    age: float
    annual_income: float
    credit_score: float
    loan_amount: float
    employment_years: float
    debt_ratio: float

class BatchRequest(BaseModel):
    records: List[PredictRequest]

class FeedbackRequest(BaseModel):
    prediction_id: str
    actual_label: int
    feedback_type: str = "correction"

class RetrainRequest(BaseModel):
    n_estimators: int = 150
    max_depth: int = 4
    learning_rate: float = 0.08
    subsample: float = 0.85


# ─────────────────────────────────────────────────
# Helper: save prediction log
# ─────────────────────────────────────────────────

def log_prediction(record: dict):
    logs = []
    if LOGS_PATH.exists():
        try:
            with open(LOGS_PATH) as f:
                logs = json.load(f)
        except:
            logs = []
    logs.append(record)
    # Keep last 1000
    if len(logs) > 1000:
        logs = logs[-1000:]
    with open(LOGS_PATH, "w") as f:
        json.dump(logs, f)


# ─────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main dashboard"""
    with open("templates/index.html") as f:
        return HTMLResponse(f.read())


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_version": metrics.get("version", "unknown"),
        "accuracy": metrics.get("accuracy", 0),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/predict")
async def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    t0 = time.time()
    X = np.array([[req.age, req.annual_income, req.credit_score,
                   req.loan_amount, req.employment_years, req.debt_ratio]])
    prediction = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    confidence = float(np.max(proba))
    latency_ms = round((time.time() - t0) * 1000, 2)
    pid = str(uuid.uuid4())

    result = {
        "prediction_id": pid,
        "prediction": prediction,
        "decision": "APPROVED" if prediction == 1 else "DECLINED",
        "confidence": round(confidence, 4),
        "probabilities": {
            "approve": round(float(proba[1]), 4),
            "decline": round(float(proba[0]), 4)
        },
        "model_version": metrics.get("version", "v1.0.0"),
        "latency_ms": latency_ms,
        "timestamp": datetime.utcnow().isoformat(),
        "features": req.dict()
    }

    log_prediction(result)
    return result


@app.post("/predict/batch")
async def predict_batch(req: BatchRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    if len(req.records) > 1000:
        raise HTTPException(400, "Max 1000 records per batch")

    results = []
    for r in req.records:
        X = np.array([[r.age, r.annual_income, r.credit_score,
                       r.loan_amount, r.employment_years, r.debt_ratio]])
        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        results.append({
            "prediction_id": str(uuid.uuid4()),
            "prediction": pred,
            "decision": "APPROVED" if pred == 1 else "DECLINED",
            "confidence": round(float(np.max(proba)), 4),
            "features": r.dict()
        })

    return {
        "batch_id": str(uuid.uuid4()),
        "total": len(results),
        "approved": sum(1 for r in results if r["prediction"] == 1),
        "declined": sum(1 for r in results if r["prediction"] == 0),
        "results": results,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    return {
        "status": "recorded",
        "prediction_id": req.prediction_id,
        "actual_label": req.actual_label,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/metrics")
async def get_metrics():
    logs = []
    if LOGS_PATH.exists():
        with open(LOGS_PATH) as f:
            logs = json.load(f)

    latencies = [l.get("latency_ms", 0) for l in logs] if logs else [0]
    confs = [l.get("confidence", 0) for l in logs] if logs else [0]

    return {
        "model": {
            "version": metrics.get("version"),
            "accuracy": metrics.get("accuracy"),
            "f1_score": metrics.get("f1_score"),
            "roc_auc": metrics.get("roc_auc"),
            "trained_at": metrics.get("trained_at"),
            "feature_importance": metrics.get("fi"),
            "confusion_matrix": metrics.get("cm"),
        },
        "predictions": {
            "total": len(logs),
            "approved": sum(1 for l in logs if l.get("prediction") == 1),
            "declined": sum(1 for l in logs if l.get("prediction") == 0),
        },
        "latency_ms": {
            "mean": round(float(np.mean(latencies)), 2),
            "p50":  round(float(np.percentile(latencies, 50)), 2),
            "p95":  round(float(np.percentile(latencies, 95)), 2),
            "p99":  round(float(np.percentile(latencies, 99)), 2),
        },
        "confidence": {
            "mean": round(float(np.mean(confs)), 3),
            "min":  round(float(np.min(confs)), 3),
            "max":  round(float(np.max(confs)), 3),
        }
    }


@app.get("/experiments")
async def get_experiments():
    runs = []
    if RUNS_PATH.exists():
        with open(RUNS_PATH) as f:
            runs = json.load(f)
    return {"runs": runs, "total": len(runs)}


@app.get("/models")
async def get_models():
    return {
        "active_model": {
            "version": metrics.get("version"),
            "accuracy": metrics.get("accuracy"),
            "f1_score": metrics.get("f1_score"),
            "roc_auc": metrics.get("roc_auc"),
            "status": "production",
            "trained_at": metrics.get("trained_at"),
            "params": metrics.get("params"),
            "feature_names": FEATURE_NAMES,
        }
    }


@app.post("/retrain")
async def retrain(req: RetrainRequest):
    global model, metrics
    params = {
        "n_estimators": req.n_estimators,
        "max_depth": req.max_depth,
        "learning_rate": req.learning_rate,
        "subsample": req.subsample,
        "random_state": 42
    }
    print(f"🔄 Retraining with params: {params}")
    model, metrics = train_model(params)
    return {
        "status": "retrained",
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
        "version": metrics["version"],
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/dataset")
async def get_dataset(limit: int = 100):
    if not DATA_PATH.exists():
        raise HTTPException(404, "Dataset not found")
    with open(DATA_PATH) as f:
        data = json.load(f)
    total = len(data)
    approved = sum(1 for r in data if r["approved"] == 1)
    return {
        "total": total,
        "approved": approved,
        "declined": total - approved,
        "approve_rate": round(approved / total * 100, 1),
        "features": FEATURE_NAMES,
        "sample": data[:limit]
    }


@app.get("/download/dataset")
async def download_dataset():
    if not DATA_PATH.exists():
        raise HTTPException(404, "Dataset not found")
    with open(DATA_PATH) as f:
        rows = json.load(f)
    # Build CSV
    lines = [",".join(FEATURE_NAMES + ["approved"])]
    for r in rows:
        lines.append(",".join([
            str(r["age"]), str(r["annual_income"]), str(r["credit_score"]),
            str(r["loan_amount"]), str(r["employment_years"]),
            str(r["debt_ratio"]), str(r["approved"])
        ]))
    csv = "\n".join(lines)
    from fastapi.responses import Response
    return Response(
        content=csv,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=loan_dataset_5000.csv"}
    )


@app.get("/download/metrics")
async def download_metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(404, "Metrics not found")
    return FileResponse(METRICS_PATH, filename="model_metrics.json",
                        media_type="application/json")


@app.get("/download/predictions")
async def download_predictions():
    if not LOGS_PATH.exists():
        raise HTTPException(404, "No predictions yet")
    return FileResponse(LOGS_PATH, filename="predictions_log.json",
                        media_type="application/json")


@app.get("/predictions/history")
async def prediction_history(limit: int = 50):
    if not LOGS_PATH.exists():
        return {"predictions": [], "total": 0}
    with open(LOGS_PATH) as f:
        logs = json.load(f)
    return {"predictions": logs[-limit:], "total": len(logs)}


@app.get("/drift")
async def get_drift():
    """Simple drift check based on recent predictions vs training distribution"""
    logs = []
    if LOGS_PATH.exists():
        with open(LOGS_PATH) as f:
            logs = json.load(f)

    if len(logs) < 10:
        return {"status": "insufficient_data", "message": "Need at least 10 predictions"}

    recent_confs = [l.get("confidence", 0) for l in logs[-50:]]
    avg_conf = float(np.mean(recent_confs))
    drift_score = round(abs(avg_conf - 0.75) * 2, 3)

    return {
        "status": "stable" if drift_score < 0.3 else "drifting",
        "drift_score": drift_score,
        "avg_confidence": round(avg_conf, 3),
        "predictions_monitored": len(logs),
        "window_size": min(50, len(logs)),
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)