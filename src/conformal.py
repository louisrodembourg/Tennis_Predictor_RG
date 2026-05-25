"""
Conformal Prediction pour les probabilités de victoire.

Algorithme : Inductive (Split) Conformal Prediction
  - Score de non-conformité : s_i = 1 - p_model(vraie_classe_i)
  - q_hat = quantile (1-alpha) des scores sur l'ensemble de calibration
  - Intervalle de prédiction : [max(0, p - q_hat), min(1, p + q_hat)]
  - Couverture garantie : >= (1-alpha) en espérance sur la distribution de calibration
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


class ConformalPredictor:
    """
    Wrapper conformal prediction sur un modèle calibré existant.

    Usage :
        cp = ConformalPredictor(coverage=0.80)
        cp.calibrate(cal_df, feature_cols)   # cal_df doit avoir p_blend et target
        low, high = cp.predict_interval(p_model)
    """

    def __init__(self, coverage: float = 0.80):
        self.coverage = coverage
        self._q_hat: Optional[float] = None

    def calibrate(self, cal_df: pd.DataFrame) -> None:
        """
        Calibre le prédicteur conformal sur un DataFrame de prédictions passées.

        Parameters
        ----------
        cal_df : DataFrame avec colonnes 'p_blend' (probabilité modèle) et 'target' (0 ou 1)
        """
        p = cal_df["p_blend"].values.astype(float)
        y = cal_df["target"].values.astype(int)
        # s_i = 1 - p(classe vraie) : grande si le modèle est peu confiant sur la bonne classe
        scores = np.where(y == 1, 1.0 - p, p)
        alpha = 1.0 - self.coverage
        # Quantile empirique avec correction de Laplace (n+1)/(n) pour garantie finie
        n = len(scores)
        level = np.ceil((n + 1) * (1.0 - alpha)) / n
        level = min(level, 1.0)
        self._q_hat = float(np.quantile(scores, level))

    def predict_interval(self, p: float) -> tuple[float, float]:
        """
        Retourne l'intervalle conformal [p_low, p_high] pour une prédiction p.

        Si non calibré, retourne [0, 1] par défaut.
        """
        if self._q_hat is None:
            return 0.0, 1.0
        return max(0.0, p - self._q_hat), min(1.0, p + self._q_hat)

    @property
    def q_hat(self) -> Optional[float]:
        return self._q_hat

    @property
    def is_calibrated(self) -> bool:
        return self._q_hat is not None

    def coverage_on(self, eval_df: pd.DataFrame) -> float:
        """
        Calcule la couverture empirique sur un DataFrame d'évaluation.
        La prédiction couvre si la vraie classe est dans l'intervalle.
        """
        if self._q_hat is None:
            return float("nan")
        p = eval_df["p_blend"].values.astype(float)
        y = eval_df["target"].values.astype(int)
        p_low  = np.maximum(0.0, p - self._q_hat)
        p_high = np.minimum(1.0, p + self._q_hat)
        # couvre si p_low <= true_class_prob <= p_high
        true_prob = np.where(y == 1, p, 1.0 - p)
        return float(np.mean((true_prob >= p_low) & (true_prob <= p_high)))
