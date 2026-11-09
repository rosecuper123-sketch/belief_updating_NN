"""Calibrated linear belief decoder for `formal_osla` networks.

Ported from `formal_batch/analysis/validated_decoder.py` (old 13-input codebase).
The method is unchanged; only the plumbing adapts to the new artifacts and to
hidden sizes other than 32.

What it does
------------
Fits a linear readout from the recurrent state `h_t` to the round's *stated and
true* probability, on **consistent rounds only** (cue == true, where the label is
unambiguous), and folds the `StandardScaler` into the `Ridge` so the result is a
single linear map on RAW hidden vectors:

    D(h) = w_eff^T h + c_eff

Selection is a nested 5-fold `GroupKFold` (groups = episode) over a log-spaced
alpha grid; the outer score is computed on episodes never seen by either the
scaler or the ridge.  Calibration `y_true = beta0 + beta1 * y_pred` is reported
because a decoder can rank well while being off in absolute scale, and every
downstream quantity (belief levels, prediction errors, learning rates) is read on
the absolute scale.

Scope and limits -- carry these into any write-up
-------------------------------------------------
* This recovers **the environment-probability content of `h_t` that a linear
  readout can extract**.  It is not a structurally identified belief, and the
  circularity question raised in the old analysis log (a fitted policy time-course
  tracked observed betting better than decoded belief) was never resolved there.
* A high R^2 shows the quantity is *linearly encodable*, not that the network
  *uses* it.
* Sources, if a consistent-round rollout is not available, must be supplied
  explicitly: the belief probe cannot be fit on conflict-only data, whose target
  is constant within a group.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-4, 3, 15)  # 1e-4 .. 1e3, same grid as the old Stage-1


@dataclass(slots=True)
class Decoder:
    """A fitted linear map on raw hidden states, with its held-out diagnostics."""

    network_id: str
    w_eff: np.ndarray  # (hidden_size,)
    c_eff: float
    target: str
    alpha: float
    inner_r2: float
    outer_r2: float
    rmse: float
    mae: float
    beta0: float
    beta1: float
    mean: np.ndarray = field(repr=False, default=None)  # scaler statistics, for reference
    scale: np.ndarray = field(repr=False, default=None)

    def __call__(self, hidden: np.ndarray) -> np.ndarray:
        """Apply to raw hidden states of any leading shape -> same shape minus last axis."""
        return np.asarray(hidden, dtype=np.float64) @ self.w_eff + self.c_eff

    def as_dict(self) -> dict:
        return {
            "network_id": self.network_id,
            "target": self.target,
            "alpha": self.alpha,
            "inner_r2": self.inner_r2,
            "outer_r2": self.outer_r2,
            "rmse": self.rmse,
            "mae": self.mae,
            "beta0": self.beta0,
            "beta1": self.beta1,
            "hidden_size": int(self.w_eff.shape[0]),
        }


def select_alpha_group_cv(
    x_train: np.ndarray, y_train: np.ndarray, groups: np.ndarray, alphas: np.ndarray
) -> tuple[float, float]:
    """Inner 5-fold GroupKFold (episode-level) mean R^2 over TRAIN episodes only."""
    gkf = GroupKFold(n_splits=5)
    best_alpha, best_score = float(alphas[0]), -np.inf
    for alpha in alphas:
        fold_r2 = []
        for tr, va in gkf.split(x_train, y_train, groups):
            model = Ridge(alpha=alpha).fit(x_train[tr], y_train[tr])
            fold_r2.append(r2_score(y_train[va], model.predict(x_train[va])))
        mean_r2 = float(np.mean(fold_r2))
        if mean_r2 > best_score:
            best_alpha, best_score = float(alpha), mean_r2
    return best_alpha, best_score


def calibrate(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Least-squares `y_true = beta0 + beta1 * y_pred`; beta1 == 1 means absolute scale."""
    design = np.vstack([np.ones_like(y_pred), y_pred]).T
    beta, *_ = np.linalg.lstsq(design, y_true, rcond=None)
    return float(beta[0]), float(beta[1])


def fit_decoder(
    hidden: np.ndarray,
    target: np.ndarray,
    episode_index: np.ndarray,
    train_mask: np.ndarray,
    *,
    network_id: str,
    target_name: str,
    alphas: np.ndarray = ALPHAS,
) -> Decoder:
    """Fit the calibrated decoder on RAW hidden states.

    Parameters
    ----------
    hidden : (n_samples, hidden_size) -- flattened per-trial states, RAW (unscaled).
    target : (n_samples,) -- the label for each sample.
    episode_index : (n_samples,) -- which episode each sample came from; the
        GroupKFold and the train/test split both operate on this, never on trials.
    train_mask : (n_episodes,) bool -- which episodes are trainable.  Held-out
        episodes are excluded from the scaler, the alpha search and the ridge fit.

    Raises rather than silently degrading when a fold cannot be formed, which is
    what happens if `train_mask` selects fewer episodes than folds.
    """
    hidden = np.asarray(hidden, dtype=np.float64)
    target = np.asarray(target, dtype=float)
    episode_index = np.asarray(episode_index)
    if hidden.ndim != 2:
        raise ValueError(f"hidden must be (n_samples, hidden_size), got {hidden.shape}")
    if not (len(hidden) == len(target) == len(episode_index)):
        raise ValueError("hidden, target and episode_index disagree on length")

    finite = np.isfinite(target)
    if not finite.all():
        hidden, target, episode_index = hidden[finite], target[finite], episode_index[finite]

    sample_train = train_mask[episode_index]
    n_train_episodes = len(np.unique(episode_index[sample_train]))
    if n_train_episodes < 5:
        raise ValueError(f"alpha search needs >=5 train episodes, got {n_train_episodes}")

    scaler = StandardScaler().fit(hidden[sample_train])
    x_scaled = scaler.transform(hidden)
    x_train, y_train = x_scaled[sample_train], target[sample_train]
    x_test, y_test = x_scaled[~sample_train], target[~sample_train]
    if len(x_test) == 0:
        raise ValueError("no held-out episodes; train_mask selects everything")

    alpha, inner_r2 = select_alpha_group_cv(x_train, y_train, episode_index[sample_train], alphas)
    model = Ridge(alpha=alpha).fit(x_train, y_train)
    y_pred = model.predict(x_test)

    beta0, beta1 = calibrate(y_test, y_pred)

    # Fold the scaler into the ridge so downstream code needs no scaler step.
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    w_eff = np.asarray(model.coef_, dtype=np.float64) / scale
    c_eff = float(model.intercept_) - float(np.sum(np.asarray(model.coef_) * mean / scale))

    return Decoder(
        network_id=network_id,
        w_eff=w_eff,
        c_eff=c_eff,
        target=target_name,
        alpha=alpha,
        inner_r2=inner_r2,
        outer_r2=float(r2_score(y_test, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_test, y_pred))),
        mae=float(mean_absolute_error(y_test, y_pred)),
        beta0=beta0,
        beta1=beta1,
        mean=mean,
        scale=scale,
    )


def save_decoder(decoder: Decoder, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        w_eff=decoder.w_eff.astype(np.float32),
        c_eff=np.float32(decoder.c_eff),
        target=decoder.target,
        alpha=decoder.alpha,
        inner_r2=decoder.inner_r2,
        outer_r2=decoder.outer_r2,
        rmse=decoder.rmse,
        mae=decoder.mae,
        beta0=decoder.beta0,
        beta1=decoder.beta1,
        mean=decoder.mean.astype(np.float32),
        scale=decoder.scale.astype(np.float32),
    )


def load_decoder(path) -> Decoder:
    with np.load(path, allow_pickle=False) as data:
        return Decoder(
            network_id=str(path.stem),
            w_eff=np.asarray(data["w_eff"], dtype=np.float64),
            c_eff=float(data["c_eff"]),
            target=str(data["target"]),
            alpha=float(data["alpha"]),
            inner_r2=float(data["inner_r2"]),
            outer_r2=float(data["outer_r2"]),
            rmse=float(data["rmse"]),
            mae=float(data["mae"]),
            beta0=float(data["beta0"]),
            beta1=float(data["beta1"]),
            mean=np.asarray(data["mean"], dtype=np.float64),
            scale=np.asarray(data["scale"], dtype=np.float64),
        )
