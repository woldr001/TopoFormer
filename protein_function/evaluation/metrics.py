"""Evaluation metrics for protein function prediction.

Standard CAFA (Critical Assessment of Functional Annotation) metrics:

- **Fmax**: protein-centric maximum F-measure over decision thresholds.
  This is the primary metric used in CAFA challenges.
- **AUPR**: area under the precision-recall curve (micro-averaged).

References:
    - Radivojac et al. (2013) "A large-scale evaluation of computational protein
      function prediction." Nature Methods.
    - Zhou et al. (2019) "The CAFA challenge reports improved protein function
      prediction and new functional annotations for hundreds of genes." Genome Biology.
"""

from typing import Tuple

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


# ---------------------------------------------------------------------------
# Fmax
# ---------------------------------------------------------------------------

def compute_fmax(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray = None,
) -> Tuple[float, float]:
    """Compute protein-centric Fmax over a range of classification thresholds.

    For each threshold t:
      - Predict label j for protein i when y_score[i, j] >= t.
      - Compute precision and recall for each protein that has at least one
        prediction above t.
      - Average over those proteins to get mean precision (P(t)) and recall (R(t)).
      - F(t) = 2 * P(t) * R(t) / (P(t) + R(t))
    Fmax = max over all t of F(t).

    Args:
        y_true:     Binary ground-truth matrix of shape [N, L].
        y_score:    Predicted probability matrix of shape [N, L].
        thresholds: 1-D array of thresholds to sweep. Defaults to
                    np.arange(0.0, 1.01, 0.01).

    Returns:
        (fmax, optimal_threshold) tuple.
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.01)

    y_true  = np.asarray(y_true,  dtype=np.float32)
    y_score = np.asarray(y_score, dtype=np.float32)

    best_fmax = 0.0
    best_t    = 0.0

    for t in thresholds:
        y_pred = (y_score >= t).astype(np.float32)

        # Per-protein precision and recall
        # Only consider proteins with at least one prediction
        pred_positive = y_pred.sum(axis=1) > 0
        true_positive_mat = (y_pred * y_true)

        precision_list, recall_list = [], []
        for i in range(len(y_true)):
            if not pred_positive[i] and y_true[i].sum() == 0:
                continue
            tp = true_positive_mat[i].sum()
            if pred_positive[i]:
                prec = tp / y_pred[i].sum()
            else:
                prec = 0.0
            rec = tp / y_true[i].sum() if y_true[i].sum() > 0 else 0.0
            precision_list.append(prec)
            recall_list.append(rec)

        if not precision_list:
            continue

        mean_prec = np.mean(precision_list)
        mean_rec  = np.mean(recall_list)

        if mean_prec + mean_rec > 0:
            f = 2.0 * mean_prec * mean_rec / (mean_prec + mean_rec)
        else:
            f = 0.0

        if f > best_fmax:
            best_fmax = f
            best_t    = t

    return float(best_fmax), float(best_t)


# ---------------------------------------------------------------------------
# AUPR
# ---------------------------------------------------------------------------

def compute_aupr(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Micro-averaged area under the precision-recall curve.

    Uses ``sklearn.metrics.average_precision_score`` with
    ``average='micro'``, which flattens all label-protein pairs.

    Args:
        y_true:  Binary ground-truth matrix of shape [N, L].
        y_score: Predicted probability matrix of shape [N, L].

    Returns:
        AUPR as a float in [0, 1].
    """
    y_true  = np.asarray(y_true,  dtype=np.float32)
    y_score = np.asarray(y_score, dtype=np.float32)

    # Drop terms that have no positive examples (would cause undefined AUPR)
    term_mask = y_true.sum(axis=0) > 0
    if term_mask.sum() == 0:
        return 0.0

    return float(
        average_precision_score(
            y_true[:, term_mask],
            y_score[:, term_mask],
            average="micro",
        )
    )


# ---------------------------------------------------------------------------
# HuggingFace Trainer-compatible wrapper
# ---------------------------------------------------------------------------

def compute_metrics_for_trainer(eval_pred) -> dict:
    """Compute Fmax and AUPR for use with HuggingFace ``Trainer.compute_metrics``.

    ``eval_pred.predictions`` contains raw logits; sigmoid is applied here.
    ``eval_pred.label_ids`` contains the float multi-hot ground-truth labels.

    Args:
        eval_pred: ``transformers.EvalPrediction`` with ``.predictions`` and
                   ``.label_ids`` attributes.

    Returns:
        Dict with keys ``"fmax"`` and ``"aupr"``.
    """
    logits  = eval_pred.predictions
    y_true  = eval_pred.label_ids

    # Apply sigmoid to convert logits → probabilities
    y_score = 1.0 / (1.0 + np.exp(-logits))

    fmax, _ = compute_fmax(y_true, y_score)
    aupr    = compute_aupr(y_true, y_score)

    return {"fmax": fmax, "aupr": aupr}
