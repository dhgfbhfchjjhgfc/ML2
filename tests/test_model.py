"""Тесты модели (этапы 3–4). Запуск: uv run pytest -m model"""

import numpy as np
import pytest

from mlproject import data as data_mod
from mlproject.metrics import compute_all
from mlproject.train import predict

pytestmark = pytest.mark.model


def test_probabilities_are_valid(smoke_model, splits, cfg):
    X, _ = data_mod.xy(splits["test"], cfg)
    proba = smoke_model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1)


def test_smoke_model_beats_dummy(smoke_model, splits, cfg):
    X, y = data_mod.xy(splits["test"], cfg)
    pred, proba = predict(smoke_model, X, cfg)
    score = compute_all(cfg["task"]["type"], y, pred, proba)[cfg["metric"]["primary"]]
    assert score > 0.4, f"{cfg['metric']['primary']}={score:.3f} — модель почти не учится"


def test_prediction_ignores_id(smoke_model, splits, cfg):
    id_col = cfg["data"].get("id_column")
    if not id_col:
        pytest.skip("Колонка id отсутствует в датасете V20, проверка не требуется")
    X, _ = data_mod.xy(splits["test"].head(50), cfg)
    shuffled = X.copy()
    shuffled[id_col] = shuffled[id_col].sample(frac=1, random_state=0).to_numpy()
    np.testing.assert_allclose(smoke_model.predict_proba(X), smoke_model.predict_proba(shuffled))


def test_custom_prediction_proba_bounds(smoke_model, sample_df, cfg):
    """Свой тест 1: предсказанные вероятности лежат в отрезке [0, 1]."""
    X, _ = data_mod.xy(sample_df, cfg)
    probas = smoke_model.predict_proba(X)
    assert (probas >= 0.0).all() and (probas <= 1.0).all()


def test_custom_prediction_count_matches(smoke_model, sample_df, cfg):
    """Свой тест 2: размерность прогноза совпадает с числом входных строк."""
    X, _ = data_mod.xy(sample_df, cfg)
    preds = smoke_model.predict(X)
    assert len(preds) == len(X)
