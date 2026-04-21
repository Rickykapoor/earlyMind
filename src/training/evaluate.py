"""
src/training/evaluate.py
Full evaluation pipeline: metrics, reports, plots, baselines, ablations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, auc, brier_score_loss, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score, roc_curve, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.config import cfg
from src.utils.age_norms import age_to_band


# ---------------------------------------------------------------------------
# Threshold selection (maximize F1 weighted by sensitivity)
# ---------------------------------------------------------------------------

def find_optimal_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    sensitivity_weight: float = 2.0,
) -> float:
    """
    Find threshold that maximizes: F1 × (2×Sensitivity + Specificity) / 3.
    Rationale: missing a true case (low sensitivity) is more costly.
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    best_score = -1.0
    best_thr   = 0.5

    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        if preds.sum() == 0 or preds.sum() == len(preds):
            continue
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        f1   = f1_score(labels, preds, zero_division=0)
        score = f1 * (sensitivity_weight * sens + spec) / (sensitivity_weight + 1)
        if score > best_score:
            best_score = score
            best_thr   = thr

    return float(best_thr)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    pred_dq: Optional[np.ndarray] = None,
    true_dq: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute all classification and regression metrics.

    Returns dict with:
      AUC, F1, Accuracy, MCC, Brier, Sensitivity, Specificity,
      PPV (Precision), NPV, threshold, DQ_MAE, DQ_R2
    """
    if threshold is None:
        threshold = find_optimal_threshold(probs, labels)

    preds = (probs >= threshold).astype(int)

    try:
        auc_score = float(roc_auc_score(labels, probs))
    except Exception:
        auc_score = 0.5

    f1 = float(f1_score(labels, preds, zero_division=0))
    acc = float(accuracy_score(labels, preds))
    mcc = float(matthews_corrcoef(labels, preds))
    brier = float(brier_score_loss(labels, probs))

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    ppv = tp / (tp + fp + 1e-8)
    npv = tn / (tn + fn + 1e-8)

    metrics: Dict[str, float] = {
        "AUC":         auc_score,
        "F1":          f1,
        "Accuracy":    acc,
        "MCC":         mcc,
        "Brier":       brier,
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "PPV":         float(ppv),
        "NPV":         float(npv),
        "Threshold":   float(threshold),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }

    # DQ metrics
    if pred_dq is not None and true_dq is not None:
        id_mask = np.array(labels) == 1
        if id_mask.sum() > 0:
            mae = float(np.mean(np.abs(pred_dq[id_mask] - true_dq[id_mask])))
            ss_res = np.sum((true_dq[id_mask] - pred_dq[id_mask]) ** 2)
            ss_tot = np.sum((true_dq[id_mask] - true_dq[id_mask].mean()) ** 2)
            r2 = float(1 - ss_res / (ss_tot + 1e-8))
            metrics["DQ_MAE"] = mae
            metrics["DQ_R2"]  = r2
        else:
            metrics["DQ_MAE"] = float("nan")
            metrics["DQ_R2"]  = float("nan")

    return metrics


# ---------------------------------------------------------------------------
# Bootstrap CI for ROC curve
# ---------------------------------------------------------------------------

def bootstrap_auc_ci(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Return (lower, upper) confidence interval for AUC via bootstrap."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(labels), size=len(labels))
        if len(np.unique(labels[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(labels[idx], probs[idx]))
        except Exception:
            pass

    alpha = (1 - ci) / 2
    return (
        float(np.percentile(aucs, 100 * alpha)),
        float(np.percentile(aucs, 100 * (1 - alpha))),
    )


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Run inference on a DataLoader, return dicts with probs, preds, labels, dq.
    """
    model.eval()
    all_probs, all_labels, all_pred_dq, all_true_dq = [], [], [], []

    for batch in loader:
        batch_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        missing = batch_dev.get("missing_modalities", None)
        out = model(batch_dev, missing_modalities=missing)

        probs   = torch.softmax(out["logits"], dim=-1)[:, 1].cpu().numpy()
        labels  = batch_dev["label"].cpu().numpy()
        pred_dq = out["severity"].squeeze(-1).cpu().numpy()
        true_dq = batch_dev["dq"].cpu().numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())
        all_pred_dq.extend(pred_dq.tolist())
        all_true_dq.extend(true_dq.tolist())

    return {
        "probs":    np.array(all_probs),
        "labels":   np.array(all_labels, dtype=int),
        "pred_dq":  np.array(all_pred_dq),
        "true_dq":  np.array(all_true_dq),
    }


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def run_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
) -> pd.DataFrame:
    """
    Fit and evaluate standard tabular baselines on HPO features.
    Returns DataFrame with columns: Model, AUC, F1, Accuracy, Sensitivity, Specificity
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost":             XGBClassifier(n_estimators=100, random_state=42,
                                              eval_metric="logloss", verbosity=0),
    }
    rows = []
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        metrics = compute_classification_metrics(probs, y_test)
        rows.append({
            "Model":       name,
            "AUC":         round(metrics["AUC"],         3),
            "F1":          round(metrics["F1"],          3),
            "Accuracy":    round(metrics["Accuracy"],    3),
            "Sensitivity": round(metrics["Sensitivity"], 3),
            "Specificity": round(metrics["Specificity"], 3),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Modality ablation
# ---------------------------------------------------------------------------

@torch.no_grad()
def ablation_study(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    """
    Run inference on all 7 combinations of modality availability.
    Returns DataFrame showing AUC for each combination.
    """
    from itertools import combinations

    modalities = ["eeg", "mri", "hpo"]
    configs = []
    for r in range(1, len(modalities) + 1):
        for combo in combinations(modalities, r):
            configs.append(list(combo))

    model.eval()
    rows = []

    for keep in configs:
        missing = [m for m in modalities if m not in keep]
        all_probs, all_labels = [], []

        for batch in loader:
            batch_dev: dict = {}
            for k, v in batch.items():
                if k in ["eeg", "mri", "hpo"] and k not in keep:
                    continue   # drop this modality
                if isinstance(v, torch.Tensor):
                    batch_dev[k] = v.to(device)
                else:
                    batch_dev[k] = v

            missing_per_sample = [missing for _ in range(batch_dev["label"].shape[0])]
            out = model(batch_dev, missing_modalities=missing_per_sample)
            probs = torch.softmax(out["logits"], dim=-1)[:, 1].cpu().numpy()
            labels = batch_dev["label"].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

        try:
            auc_score = float(roc_auc_score(all_labels, all_probs))
        except Exception:
            auc_score = 0.5

        rows.append({
            "Modalities": " + ".join(sorted(keep)),
            "AUC":        round(auc_score, 3),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per age-band evaluation
# ---------------------------------------------------------------------------

def evaluate_by_age_band(
    probs: np.ndarray,
    labels: np.ndarray,
    age_months: np.ndarray,
    threshold: float = 0.35,
) -> pd.DataFrame:
    """
    Compute AUC and Sensitivity per age band.
    """
    bands = ["neonate", "infant", "toddler1", "toddler2"]
    rows  = []

    for band in bands:
        band_mask = np.array([
            age_to_band(float(a)) == band for a in age_months
        ])
        if band_mask.sum() < 5:
            continue
        p  = probs[band_mask]
        lb = labels[band_mask]
        if len(np.unique(lb)) < 2:
            continue

        try:
            auc_score = float(roc_auc_score(lb, p))
        except Exception:
            auc_score = float("nan")
        preds = (p >= threshold).astype(int)
        sens = recall_score(lb, preds, zero_division=0)

        rows.append({
            "Age Band":  band,
            "N":         int(band_mask.sum()),
            "AUC":       round(auc_score, 3),
            "Sensitivity": round(float(sens), 3),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full benchmark report generator
# ---------------------------------------------------------------------------

def generate_benchmark_report(
    model: nn.Module,
    test_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    X_train_hpo: Optional[np.ndarray] = None,
    y_train_hpo: Optional[np.ndarray] = None,
    X_test_hpo:  Optional[np.ndarray] = None,
    y_test_hpo:  Optional[np.ndarray] = None,
    report_dir: Optional[str] = None,
) -> str:
    """
    Generate full benchmark_report.md with all specifications from Section 5.

    Returns: path to written report.
    """
    if report_dir is None:
        report_dir = str(cfg.paths.reports)
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    print("Running inference on test set …")
    val_res  = run_inference(model, val_loader,  device)
    test_res = run_inference(model, test_loader, device)

    # Optimal threshold on validation set
    thr = find_optimal_threshold(val_res["probs"], val_res["labels"])
    print(f"  Optimal threshold: {thr:.3f}")

    # Test-set metrics
    metrics = compute_classification_metrics(
        test_res["probs"], test_res["labels"],
        pred_dq=test_res["pred_dq"], true_dq=test_res["true_dq"],
        threshold=thr,
    )
    auc_lo, auc_hi = bootstrap_auc_ci(test_res["probs"], test_res["labels"])

    # Ablation
    print("Running ablation study …")
    ablation_df = ablation_study(model, test_loader, device)

    # Baselines
    baseline_df = pd.DataFrame()
    if X_train_hpo is not None and X_test_hpo is not None:
        print("Running baseline models …")
        baseline_df = run_baselines(X_train_hpo, y_train_hpo, X_test_hpo, y_test_hpo)

    # Modality importance
    importance = model.modality_importance.detach().cpu()
    importance = torch.softmax(importance, dim=0).numpy()

    # ----------------------------------------------------------------
    # Write report
    # ----------------------------------------------------------------
    lines = [
        "# EarlyMind — Benchmark Report",
        "",
        "Generated by `src/training/evaluate.py`",
        "",
        "---",
        "",
        "## 1. Test-Set Classification Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| **AUC** | {metrics['AUC']:.4f} (95% CI: {auc_lo:.3f}–{auc_hi:.3f}) |",
        f"| **F1** | {metrics['F1']:.4f} |",
        f"| **Accuracy** | {metrics['Accuracy']:.4f} |",
        f"| **MCC** | {metrics['MCC']:.4f} |",
        f"| **Brier Score** | {metrics['Brier']:.4f} |",
        f"| Threshold | {metrics['Threshold']:.3f} |",
        "",
        "## 2. Clinical Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| **Sensitivity (Recall)** | {metrics['Sensitivity']:.4f} |",
        f"| **Specificity** | {metrics['Specificity']:.4f} |",
        f"| **PPV (Precision)** | {metrics['PPV']:.4f} |",
        f"| **NPV** | {metrics['NPV']:.4f} |",
        f"| TP | {metrics['TP']} | TN | {metrics['TN']} | FP | {metrics['FP']} | FN | {metrics['FN']} |",
        "",
        "## 3. DQ Severity Estimation",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| **DQ MAE** | {metrics.get('DQ_MAE', 'N/A')} |",
        f"| **DQ R²** | {metrics.get('DQ_R2', 'N/A')} |",
        "",
        "## 4. Modality Importance (Learned Weights)",
        "",
        f"| Modality | Weight |",
        f"|----------|--------|",
        f"| EEG | {importance[0]:.4f} |",
        f"| MRI | {importance[1]:.4f} |",
        f"| HPO | {importance[2]:.4f} |",
        "",
        "## 5. Modality Ablation Study",
        "",
        ablation_df.to_markdown(index=False) if len(ablation_df) > 0 else "_No ablation data_",
        "",
    ]

    if len(baseline_df) > 0:
        lines += [
            "## 6. Baseline Comparison (HPO features only)",
            "",
            baseline_df.to_markdown(index=False),
            "",
        ]

    lines += [
        "## 7. Minimum Acceptable Performance",
        "",
        f"| Criterion | Target | Achieved | Status |",
        f"|-----------|--------|----------|--------|",
        f"| AUC | ≥ 0.85 | {metrics['AUC']:.3f} | {'✅' if metrics['AUC'] >= 0.85 else '❌'} |",
        f"| Sensitivity | ≥ 0.80 | {metrics['Sensitivity']:.3f} | {'✅' if metrics['Sensitivity'] >= 0.80 else '❌'} |",
        f"| Specificity | ≥ 0.70 | {metrics['Specificity']:.3f} | {'✅' if metrics['Specificity'] >= 0.70 else '❌'} |",
        f"| F1 | ≥ 0.75 | {metrics['F1']:.3f} | {'✅' if metrics['F1'] >= 0.75 else '❌'} |",
        "",
        "---",
        "",
        "_EarlyMind is a research screening tool. Not FDA cleared._",
    ]

    report_text = "\n".join(lines)
    report_path = str(Path(report_dir) / "benchmark_report.md")
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"  Benchmark report saved → {report_path}")
    return report_path
