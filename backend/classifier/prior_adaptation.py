"""
LesionIQ -- Prior Shift Adaptation
====================================

Addresses the train/val vs test class-prior mismatch (in ISIC 2019, the
public test set has noticeably different class proportions than the
training set: MEL is ~17% in val but ~21% in test, and AK roughly
doubles). When a softmax classifier trained on one prior is evaluated
on data drawn from a different prior, the posterior probabilities are
miscalibrated -- decision thresholds tuned on val no longer hold.

The standard fix is post-hoc logit adjustment using the relationship

    P_test(y|x) is proportional to P(x|y) * P_test(y)
              is proportional to P_train(y|x) / P_train(y) * P_test(y)

Taking logarithms:

    log P_test(y|x) = log P_train(y|x) + log P_test(y) - log P_train(y)

so we can correct the logits at inference time by simply adding the
per-class log-ratio ``log(P_test) - log(P_train)``.

When the test prior is unknown, this module estimates it from the
unlabeled test predictions using the Saerens-Latinne-Decaestecker (SLD)
EM algorithm (Neural Computation, 2002).

Public API
----------
- ``effective_train_prior(model, train_loader)`` -- recover the prior
  the model actually learned (mean softmax over training set). This
  matters because LesionIQ trains with WeightedRandomSampler which
  produces a near-uniform sampling prior, NOT the raw class counts.
- ``estimate_test_prior_sld(probs, train_prior)`` -- SLD EM. Works on
  unlabeled test probabilities, no test labels needed.
- ``adjust_probs_for_prior(probs, train_prior, test_prior)`` --
  post-hoc correction applied to a probability matrix.
- ``adjust_logits_for_prior(logits, train_prior, test_prior)`` -- same
  correction but operating directly on pre-softmax logits.

All functions are pure numpy; they neither load model state nor touch
the live inference pipeline. The inference path opts in via
``--adapt-prior {none,sld,oracle}`` (default ``none``).
"""
from __future__ import annotations

from typing import Iterable
import numpy as np


# ---------------------------------------------------------------------------
#  Effective training prior  (mean softmax over the training set)
# ---------------------------------------------------------------------------

def effective_train_prior(softmax_outputs: np.ndarray) -> np.ndarray:
    """Return the prior the *model* actually learned to output.

    Naive class counts give the prior of the *dataset* before sampling.
    LesionIQ trains with ``WeightedRandomSampler`` which oversamples rare
    classes -- the model's effective prior is closer to uniform than the
    raw counts suggest. The honest way to recover that effective prior is
    to forward-pass the training set and average the softmax outputs:

        P_train_effective(y=k) = (1/N) * sum_i softmax(z_i)[k]

    Parameters
    ----------
    softmax_outputs : (N, K) array of softmax probabilities collected by
        forward-passing the *training* set with the trained model.

    Returns
    -------
    prior : (K,) array summing to 1.
    """
    p = np.asarray(softmax_outputs, dtype=np.float64).mean(axis=0)
    p = np.clip(p, 1e-12, None)
    return (p / p.sum()).astype(np.float32)


def empirical_prior_from_labels(labels: Iterable[int], n_classes: int) -> np.ndarray:
    """Simple class-count prior. Useful when an oracle prior is desired."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=n_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    return (counts / counts.sum()).astype(np.float32)


# ---------------------------------------------------------------------------
#  Logit / probability adjustment  (closed-form correction)
# ---------------------------------------------------------------------------

def _normalize_prior(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-12, None)
    return (p / p.sum()).astype(np.float32)


def adjust_logits_for_prior(logits: np.ndarray, train_prior: np.ndarray,
                            test_prior: np.ndarray) -> np.ndarray:
    """Add log(P_test) - log(P_train) to the logits.

    Parameters
    ----------
    logits : (N, K) raw pre-softmax logits.
    train_prior, test_prior : (K,) probability vectors.

    Returns
    -------
    adjusted : (N, K) corrected logits (no softmax applied).
    """
    train_prior = _normalize_prior(train_prior)
    test_prior  = _normalize_prior(test_prior)
    log_correction = np.log(test_prior) - np.log(train_prior)
    return logits + log_correction[np.newaxis, :]


def adjust_probs_for_prior(probs: np.ndarray, train_prior: np.ndarray,
                            test_prior: np.ndarray) -> np.ndarray:
    """Reweight an (N, K) probability matrix to a new test prior.

    Equivalent to ``softmax(adjust_logits_for_prior(log(probs), ...))``
    but numerically safer when ``probs`` may contain hard zeros.
    """
    train_prior = _normalize_prior(train_prior)
    test_prior  = _normalize_prior(test_prior)
    ratio = (test_prior / train_prior).astype(np.float64)
    adjusted = np.asarray(probs, dtype=np.float64) * ratio[np.newaxis, :]
    denom = adjusted.sum(axis=1, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    return (adjusted / denom).astype(np.float32)


# ---------------------------------------------------------------------------
#  Saerens-Latinne-Decaestecker  (EM-based test prior estimation)
# ---------------------------------------------------------------------------

def estimate_test_prior_sld(probs: np.ndarray, train_prior: np.ndarray,
                             *, max_iter: int = 200, tol: float = 1e-6,
                             init: str = "uniform") -> np.ndarray:
    """Estimate the test prior from unlabeled probabilities (SLD EM).

    Saerens, Latinne & Decaestecker (2002), "Adjusting the outputs of a
    classifier to new a priori probabilities". Iteratively refines the
    estimate by alternating between:

        E-step: re-weight probabilities under the current prior estimate
        M-step: average re-weighted probabilities -> new prior

    Converges in ~10-30 iterations on a few thousand samples.

    Parameters
    ----------
    probs : (N, K) softmax probabilities produced by the current model
        on the (unlabeled) target / test data.
    train_prior : (K,) prior the model was trained / calibrated against
        (use ``effective_train_prior`` to recover it).
    max_iter : maximum EM iterations.
    tol : L-infinity convergence threshold on the prior.
    init : ``"uniform"`` (default) or ``"train"`` to warm-start from
        ``train_prior``.

    Returns
    -------
    test_prior : (K,) estimated prior.
    """
    probs       = np.asarray(probs, dtype=np.float64)
    train_prior = _normalize_prior(train_prior).astype(np.float64)
    n_classes   = probs.shape[1]

    if init == "train":
        prior = train_prior.copy()
    else:
        prior = np.full(n_classes, 1.0 / n_classes, dtype=np.float64)

    for _ in range(max_iter):
        ratio    = prior / train_prior
        adjusted = probs * ratio[np.newaxis, :]
        adjusted /= np.clip(adjusted.sum(axis=1, keepdims=True), 1e-12, None)
        new_prior = adjusted.mean(axis=0)
        new_prior = np.clip(new_prior, 1e-12, None)
        new_prior /= new_prior.sum()

        if np.abs(new_prior - prior).max() < tol:
            prior = new_prior
            break
        prior = new_prior

    return prior.astype(np.float32)
