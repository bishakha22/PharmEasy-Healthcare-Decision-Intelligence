"""Shared helpers for part1.py to part4.py. Importing this file runs nothing."""
import csv
import numpy as np


def load():
    """Return an 800 x 9 array: X1..X7 (cols 0-6), outcome (7), value (8)."""
    with open("data/records.csv") as f:
        rows = list(csv.reader(f))[1:]
    return np.array([[float(v) for v in r[1:]] for r in rows])


def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4f}"
    return str(v)


def write_results(path, d):
    with open(path, "w") as f:
        for k, v in d.items():
            f.write(f"{k} = {fmt(v)}\n")


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(fmt(v) for v in r) + "\n")


def split():
    """Stratified 70/30 split (Task 2.1) with training-based standardisation."""
    a = load()
    X = a[:, :7]
    y = a[:, 7].astype(int)
    tr, te = [], []
    for c in (0, 1):
        idx = np.where(y == c)[0]
        k = 7 * len(idx) // 10
        tr += list(idx[:k])
        te += list(idx[k:])
    tr, te = np.array(sorted(tr)), np.array(sorted(te))
    mu = X[tr].mean(axis=0)
    sd = X[tr].std(axis=0, ddof=1)
    return dict(Xtr_raw=X[tr], Xte_raw=X[te], ytr=y[tr], yte=y[te],
                Xtr=(X[tr] - mu) / sd, Xte=(X[te] - mu) / sd)


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(z / 2.0))


def bce(X, y, w, b):
    z = X @ w + b
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def train_logistic(X, y, eta, epochs, record=()):
    """Batch gradient descent from zero. Returns w, b and {epoch: loss}."""
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    losses = {0: bce(X, y, w, b)}
    for e in range(1, epochs + 1):
        err = sigmoid(X @ w + b) - y
        gw = X.T @ err / n
        gb = err.mean()
        w = w - eta * gw
        b = b - eta * gb
        if e in record:
            losses[e] = bce(X, y, w, b)
    return w, b, losses


def cells(y, pred):
    pred = np.asarray(pred).astype(bool)
    y = np.asarray(y).astype(bool)
    return (int((pred & y).sum()), int((pred & ~y).sum()),
            int((~pred & ~y).sum()), int((~pred & y).sum()))


def _div(a, b):
    return a / b if b else 0.0


def metrics(tp, fp, tn, fn):
    acc = _div(tp + tn, tp + fp + tn + fn)
    prec = _div(tp, tp + fp)
    rec = _div(tp, tp + fn)
    f1 = _div(2 * prec * rec, prec + rec)
    spec = _div(tn, tn + fp)
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, specificity=spec)


def knn_counts(Xref, yref, Xq, K):
    """Number of positive neighbours among the K nearest (stable tie-break)."""
    d = ((Xq[:, None, :] - Xref[None, :, :]) ** 2).sum(axis=2)
    order = np.argsort(d, axis=1, kind="stable")[:, :K]
    return yref[order].sum(axis=1)


def knn_predict(Xref, yref, Xq, K):
    cnt = knn_counts(Xref, yref, Xq, K)
    return cnt >= (K + 1) / 2, cnt / K


def knn_cv(Xtr, ytr):
    """5-fold CV (fold = position mod 5). Returns list of (K, mean F1)."""
    fold = np.arange(len(ytr)) % 5
    Ks = list(range(1, 22, 2))
    scores = {K: [] for K in Ks}
    for f in range(5):
        q, r = fold == f, fold != f
        d = ((Xtr[q][:, None, :] - Xtr[r][None, :, :]) ** 2).sum(axis=2)
        order = np.argsort(d, axis=1, kind="stable")
        yr = ytr[r]
        for K in Ks:
            cnt = yr[order[:, :K]].sum(axis=1)
            tp, fp, tn, fn = cells(ytr[q], cnt >= (K + 1) / 2)
            scores[K].append(metrics(tp, fp, tn, fn)["f1"])
    return [(K, float(np.mean(scores[K]))) for K in Ks]


def best_k(cv):
    best = None
    for K, s in cv:
        if best is None or s > best[1]:
            best = (K, s)
    return best[0]
