"""Признаки и препроцессинг для Credit Card Fraud Detection (V20)."""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Доменные признаки: стабилизация логарифмом распределения сумм."""
    out = df.copy()
    if "Amount" in out.columns:
        out["log_amount"] = np.log1p(np.maximum(out["Amount"].fillna(0), 0))
    return out

def build_preprocessor(cfg: dict[str, Any]) -> ColumnTransformer:
    f = cfg.get("features", {})
    blocks = []
    num_cols = f.get("numeric") or f.get("numerical")
    if num_cols:
        target = cfg.get("data", {}).get("target", "Class")
        cols = [c for c in num_cols if c != target]
        blocks.append((
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            cols,
        ))
    else:
        blocks.append((
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            make_column_selector(dtype_include=[np.number]),
        ))

    cat_cols = f.get("categorical", [])
    if cat_cols:
        blocks.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False)),
            ]),
            cat_cols,
        ))

    return ColumnTransformer(blocks, remainder="drop", verbose_feature_names_out=True)

def make_pipeline(cfg: dict[str, Any], estimator: Any) -> Pipeline:
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", build_preprocessor(cfg)),
        ("model", estimator),
    ])

def feature_names(pipeline: Pipeline) -> list[str]:
    try:
        return [str(n) for n in pipeline.named_steps["prep"].get_feature_names_out()]
    except Exception:
        return []
