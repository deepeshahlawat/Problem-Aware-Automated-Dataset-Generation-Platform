"""Statistical and model-based relevance signals (design spec 16.1), on training rows only.

statistical: mutual information with the target (normalised to [0, 1]), plus
             column diagnostics: unique ratio, top-value share, single-feature CV score.
model-based: LightGBM baseline on all candidate columns;
             permutation importance on an internal validation split +
             mean |TreeSHAP| (LightGBM pred_contrib), each normalised, averaged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from dsagent.config import Settings
from dsagent.schemas.spec import DatasetSpec
from dsagent.stages.relevance.modeling import (
    Encoder, cat_indices, fit_model, kfold, make_model, score, target_array,
)


@dataclass
class ColumnStats:
    column: str
    mi: float = 0.0
    stat_score: float = 0.0
    unique_ratio: float = 0.0
    top_share: float = 0.0
    single_score: float = float("nan")  # CV R^2 / AUC of a model using only this column
    perm: float = 0.0
    shap: float = 0.0
    model_score: float = 0.0


def _norm(d: dict[str, float]) -> dict[str, float]:
    m = max([v for v in d.values() if np.isfinite(v)] + [0.0])
    return {k: (max(v, 0.0) / m if m > 0 and np.isfinite(v) else 0.0) for k, v in d.items()}


def _impute_for_mi(X: np.ndarray, cat: list[int]) -> np.ndarray:
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col)
        if mask.any():
            fill = -1.0 if j in cat else (np.nanmedian(col) if (~mask).any() else 0.0)
            col[mask] = fill
    return X


def statistical(train: pl.DataFrame, spec: DatasetSpec, enc: Encoder, settings: Settings) -> dict[str, ColumnStats]:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    cols = enc.columns
    X = enc.transform(train, cols)
    y, _ = target_array(train, spec)
    cat = cat_indices(enc, cols)
    seed = settings.selection.random_seed
    stats = {c: ColumnStats(column=c) for c in cols}

    Xi = _impute_for_mi(X, cat)
    mi_fn = mutual_info_classif if spec.task_type == "classification" else mutual_info_regression
    for j, c in enumerate(cols):
        is_cat = j in cat
        if is_cat and len(np.unique(Xi[:, j])) > 0.5 * len(y):
            stats[c].mi = 0.0  # high-cardinality categorical (ID-like): MI estimate meaningless
            continue
        try:
            stats[c].mi = float(mi_fn(Xi[:, [j]], y, discrete_features=[is_cat], random_state=seed)[0])
        except ValueError:
            stats[c].mi = 0.0
    for c, s in _norm({c: stats[c].mi for c in cols}).items():
        stats[c].stat_score = round(s, 4)

    small = {**settings.relevance.lgbm_params, "n_estimators": 60}
    for j, c in enumerate(cols):
        vals = train[c].drop_nulls()
        nn = vals.len()
        stats[c].unique_ratio = round(vals.n_unique() / nn, 4) if nn else 0.0
        stats[c].top_share = round(vals.value_counts(sort=True)["count"][0] / nn, 4) if nn else 1.0
        # single-feature predictive power (3-fold CV) for the leakage rule
        xj = X[:, [j]]
        scores = []
        for tr, te in kfold(spec, y, 3, seed):
            if len(np.unique(y[tr])) < 2:
                continue
            m = make_model(spec, small, seed)
            m.fit(xj[tr], y[tr], categorical_feature=[0] if j in cat else "auto")
            scores.append(score(spec, m, xj[te], y[te]))
        stats[c].single_score = round(float(np.nanmean(scores)), 4) if scores else float("nan")
    return stats


def model_based(
    train: pl.DataFrame, spec: DatasetSpec, enc: Encoder, stats: dict[str, ColumnStats], settings: Settings,
    columns: list[str] | None = None,
) -> None:
    """Fills perm / shap / model_score in `stats` (columns = candidates after exclusions)."""
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    cols = columns if columns is not None else enc.columns
    if not cols:
        return
    seed = settings.selection.random_seed
    X = enc.transform(train, cols)
    y, _ = target_array(train, spec)
    w = train["_sample_weight"].to_numpy() if "_sample_weight" in train.columns else None
    cat = cat_indices(enc, cols)
    idx = np.arange(len(y))
    strat = y if spec.task_type == "classification" and np.min(np.bincount(y)) >= 2 else None
    tr, va = train_test_split(idx, test_size=0.25, random_state=seed, stratify=strat)
    m = fit_model(spec, settings.relevance.lgbm_params, seed, X[tr], y[tr], None if w is None else w[tr], cat)

    scoring = "r2" if spec.task_type == "regression" else ("roc_auc" if len(m.classes_) == 2 else "f1_macro")
    pi = permutation_importance(m, X[va], y[va], scoring=scoring, n_repeats=settings.relevance.permutation_repeats,
                                random_state=seed)
    contrib = m.booster_.predict(X, pred_contrib=True)
    k = len(cols)
    contrib = np.asarray(contrib)
    if contrib.shape[1] == k + 1:
        shap_abs = np.abs(contrib[:, :k]).mean(axis=0)
    else:  # multiclass: blocks of (k + 1) per class
        blocks = contrib.reshape(contrib.shape[0], -1, k + 1)[:, :, :k]
        shap_abs = np.abs(blocks).mean(axis=(0, 1))

    perm = _norm({c: float(v) for c, v in zip(cols, pi.importances_mean)})
    shp = _norm({c: float(v) for c, v in zip(cols, shap_abs)})
    for c in cols:
        stats[c].perm, stats[c].shap = round(perm[c], 4), round(shp[c], 4)
        stats[c].model_score = round(0.5 * perm[c] + 0.5 * shp[c], 4)
