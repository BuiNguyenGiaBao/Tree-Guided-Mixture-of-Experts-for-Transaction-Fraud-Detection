from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, matthews_corrcoef, average_precision_score, roc_auc_score, confusion_matrix


def read_pred(path: Path):
    df = pd.read_csv(path)
    y_col = 'isFraud' if 'isFraud' in df.columns else ('y_true' if 'y_true' in df.columns else None)
    p_col = 'prob' if 'prob' in df.columns else ('y_prob' if 'y_prob' in df.columns else None)
    if y_col is None or p_col is None:
        raise ValueError(f'{path}: expected label column isFraud/y_true and probability column prob/y_prob')
    return df, df[y_col].to_numpy(dtype=int), df[p_col].to_numpy(dtype=float)


def best_f1_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def p80_threshold(y, p, min_precision=0.80):
    precision, recall, thresholds = precision_recall_curve(y, p)
    valid = np.where(precision[:-1] >= min_precision)[0]
    if len(valid) == 0:
        return 1.0
    idx = valid[np.argmax(recall[:-1][valid])]
    return float(thresholds[idx])


def metrics_at(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'threshold': float(threshold),
        'PR_AUC': float(average_precision_score(y, p)),
        'ROC_AUC': float(roc_auc_score(y, p)),
        'Precision': float(precision_score(y, pred, zero_division=0)),
        'Recall': float(recall_score(y, pred, zero_division=0)),
        'F1': float(f1_score(y, pred, zero_division=0)),
        'MCC': float(matthews_corrcoef(y, pred)),
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
    }


def main():
    ap = argparse.ArgumentParser(description='Select thresholds only on validation, then evaluate the locked test set.')
    ap.add_argument('--val-pred', required=True, type=Path)
    ap.add_argument('--test-pred', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--min-precision', type=float, default=0.80)
    args = ap.parse_args()

    _, yv, pv = read_pred(args.val_pred)
    _, yt, pt = read_pred(args.test_pred)
    t_f1 = best_f1_threshold(yv, pv)
    t_p80 = p80_threshold(yv, pv, args.min_precision)

    rows = []
    r1 = metrics_at(yt, pt, t_f1); r1['selection_rule'] = 'best_f1_on_validation'; rows.append(r1)
    r2 = metrics_at(yt, pt, t_p80); r2['selection_rule'] = f'max_recall_on_validation_at_precision>={args.min_precision:.2f}'; rows.append(r2)

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
