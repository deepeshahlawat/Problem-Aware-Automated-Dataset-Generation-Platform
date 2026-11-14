"""Shared modelling utilities for the relevance engine.

* Feature matrix from a polars table: numeric columns as float, categorical columns
  ordinal-encoded with the mapping learned on the *training* rows only (unseen -> NaN).
* LightGBM baseline (NaN handled natively, so no imputation is needed for modelling).
* One score per task: R^2 for regression, ROC-AUC (binary) / macro-F1 (multiclass).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from dsagent.schemas.spec import DatasetSpec


def is_feature_column(c: str) -> bool:
    return not c.startswith("_") and not c.endswith("__imputed")


@dataclass
class Encoder:
    numeric: list[str]
    categorical: dict[str, dict[str, int]]

    @property
    def columns(self) -> list[str]:
        return self.numeric + list(self.categorical)

    def transform(self, df: pl.DataFrame, columns: list[str] | None = None) -> np.ndarray:
        cols = columns or self.columns
        out = np.full((df.height, len(cols)), np.nan, dtype=float)
        for j, c in enumerate(cols):
            if c in self.categorical:
                m = self.categorical[c]
                out[:, j] = [m.get(v, np.nan) if v is not None else np.nan for v in df[c].to_list()]
            else:
                out[:, j] = df[c].cast(pl.Float64, strict=False).fill_null(np.nan).to_numpy()
        return out


def fit_encoder(train: pl.DataFrame, columns: list[str]) -> Encoder:
    numeric, cats = [], {}
    for c in columns:
        dt = train[c].dtype
        if dt.is_numeric() or dt == pl.Boolean:
            numeric.append(c)
        else:
            values = sorted({str(v) for v in train[c].drop_nulls().to_list()})
            cats[c] = {v: i for i, v in enumerate(values)}
    return Encoder(numeric, cats)


def target_array(df: pl.DataFrame, spec: DatasetSpec, classes: list | None = None) -> tuple[np.ndarray, list | None]:
    y = df[spec.target.name].to_list()
    if spec.task_type == "classification":
        classes = classes or sorted({str(v) for v in y if v is not None})
        idx = {c: i for i, c in enumerate(classes)}
        return np.array([idx.get(str(v), -1) for v in y]), classes
    return np.array(y, dtype=float), None


def make_model(spec: DatasetSpec, params: dict, seed: int, n_classes: int | None = None):
    import lightgbm as lgb

    p = {**params, "random_state": seed}
    if spec.task_type == "classification":
        return lgb.LGBMClassifier(**p)
    return lgb.LGBMRegressor(**p)


def score(spec: DatasetSpec, model, X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import f1_score, r2_score, roc_auc_score

    if spec.task_type == "classification":
        if len(model.classes_) == 2:
            if len(np.unique(y)) < 2:
                return float("nan")
            return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))
        return float(f1_score(y, model.predict(X), average="macro"))
    return float(r2_score(y, model.predict(X)))


def fit_model(spec, params, seed, X, y, w=None, cat_idx: list[int] | None = None):
    m = make_model(spec, params, seed)
    m.fit(X, y, sample_weight=w, categorical_feature=cat_idx if cat_idx else "auto")
    return m


def fit_score(spec, params, seed, Xtr, ytr, Xte, yte, wtr=None, cat_idx=None) -> tuple[float, object]:
    m = fit_model(spec, params, seed, Xtr, ytr, wtr, cat_idx)
    return score(spec, m, Xte, yte), m


def cat_indices(enc: Encoder, columns: list[str]) -> list[int]:
    return [i for i, c in enumerate(columns) if c in enc.categorical]


def split_holdout(df: pl.DataFrame, spec: DatasetSpec, share: float, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Final untouched test split (stratified for classification)."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(df.height)
    strat = None
    if spec.task_type == "classification":
        y = [str(v) for v in df[spec.target.name].to_list()]
        if min(y.count(c) for c in set(y)) >= 2:
            strat = y
    tr, te = train_test_split(idx, test_size=share, random_state=seed, stratify=strat)
    return df[sorted(tr)], df[sorted(te)]


def kfold(spec: DatasetSpec, y: np.ndarray, k: int, seed: int):
    from sklearn.model_selection import KFold, StratifiedKFold

    if spec.task_type == "classification" and np.min(np.bincount(y[y >= 0])) >= k:
        return StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
    return KFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)))
