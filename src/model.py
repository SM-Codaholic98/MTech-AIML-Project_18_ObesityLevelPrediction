"""Classifier construction and SHAP-based global feature explanations."""

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


def build_models(random_state: int = 42) -> Dict[str, object]:
    """Return the two classifiers used in the supplied notebook."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=random_state, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, random_state=random_state, n_jobs=-1
        ),
    }


def fit_models(models: Dict[str, object], X_train: pd.DataFrame, y_train: pd.Series):
    """Fit each classifier on the same training partition."""
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def original_feature_name(name: str) -> str:
    """Map one-hot encoded columns back to their original feature."""
    for prefix in ("CAEC_", "CALC_", "MTRANS_"):
        if name.startswith(prefix):
            return prefix[:-1]
    return name


def _normalize_shap_output(values, n_features: int) -> np.ndarray:
    """Normalize SHAP output to (samples, features, classes)."""
    if isinstance(values, list):
        values = np.stack([np.asarray(value) for value in values], axis=-1)
    else:
        values = np.asarray(values)

    if values.ndim == 2:
        values = values[:, :, np.newaxis]
    elif values.ndim == 3 and values.shape[1] != n_features and values.shape[2] == n_features:
        # Compatibility with explainers returning (samples, classes, features).
        values = np.transpose(values, (0, 2, 1))

    if values.ndim != 3 or values.shape[1] != n_features:
        raise ValueError(
            f"Unexpected SHAP output shape {values.shape}; expected samples x {n_features} x classes."
        )
    return values


def calculate_grouped_shap_importance(
    model,
    X_data: pd.DataFrame,
) -> pd.Series:
    """Calculate grouped mean absolute SHAP values.

    For one-hot groups, encoded-column SHAP contributions are summed first;
    absolute values are then averaged across samples and output classes. This
    follows the grouped-SHAP procedure in notebook cells 28-30.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("Install SHAP with `pip install shap` to calculate explanations.") from exc

    explainer = shap.TreeExplainer(model)
    values = _normalize_shap_output(explainer.shap_values(X_data), X_data.shape[1])

    groups = {}
    for index, feature_name in enumerate(X_data.columns):
        groups.setdefault(original_feature_name(feature_name), []).append(index)

    importance = {}
    for feature_name, indices in groups.items():
        grouped_values = values[:, indices, :].sum(axis=1)
        importance[feature_name] = float(np.abs(grouped_values).mean())

    return pd.Series(importance, name="Mean_Absolute_Grouped_SHAP").sort_values(
        ascending=False
    )
