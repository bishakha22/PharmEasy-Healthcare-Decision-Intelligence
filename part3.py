import numpy as np
from common import load, write_results, write_csv

a = load()
X = a[:, :5]
outcome, value = a[:, 7], a[:, 8]
n = len(X)
Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
R = {}


def kmeans(Z, K):
    step = len(Z) // K
    cent = Z[[i * step for i in range(K)]].copy()
    labels = None
    while True:
        d = ((Z[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(d, axis=1)  # ties go to the lower-numbered centroid
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for k in range(K):
            mem = Z[labels == k]
            if len(mem):
                cent[k] = mem.mean(axis=0)
    wcss = float(((Z - cent[labels]) ** 2).sum())
    return labels, wcss


wc, lab3 = {}, None
for K in range(2, 7):
    lab, w = kmeans(Z, K)
    wc[K] = w
    R[f"wcss_k{K}"] = w
    if K == 3:
        lab3 = lab

# 3.2
elbow = 0
for K in (3, 4, 5):
    if wc[K] - wc[K + 1] < 0.5 * (wc[K - 1] - wc[K]):
        elbow = K
        break
R["elbow_k"] = elbow

# 3.3
order = np.argsort([X[lab3 == k, 2].mean() for k in range(3)])
seg = np.zeros(n, dtype=int)
rows = []
for s, k in enumerate(order):
    seg[lab3 == k] = s
    m = lab3 == k
    rows.append(["ABC"[s], int(m.sum())] + [X[m, j].mean() for j in range(5)]
                + [outcome[m].mean(), value[m].mean()])
    R[f"size_segment_{'ABC'[s]}"] = int(m.sum())
write_csv("outputs/part3_segments.csv",
          ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5",
           "outcome_rate", "mean_value"], rows)

# 3.4
D = np.array([[0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
              [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
              [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
              [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
              [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
              [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]])


def name(c):
    return "+".join(f"E{m + 1}" for m in sorted(c))


def agglo(rule):
    clusters = [[i] for i in range(6)]
    merges = []
    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                ds = [D[p, q] for p in clusters[i] for q in clusters[j]]
                h = min(ds) if rule == "single" else (max(ds) if rule == "complete" else sum(ds) / len(ds))
                if best is None or h < best[0]:
                    best = (h, i, j)
        h, i, j = best
        ca, cb = clusters[i], clusters[j]
        if min(ca) > min(cb):
            ca, cb = cb, ca
        merges.append((name(ca), name(cb), h))
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [sorted(ca + cb)]
    return merges


hrows = []
for rule in ("single", "complete", "average"):
    ms = agglo(rule)
    for t, (ca, cb, h) in enumerate(ms, 1):
        hrows.append([rule, t, ca, cb, h])
    hs = [m[2] for m in ms]
    gaps = [hs[i + 1] - hs[i] for i in range(4)]
    g = int(np.argmax(gaps))
    R[f"{rule}_clusters_at_cut"] = 6 - (g + 1)
write_csv("outputs/part3_hierarchical.csv", ["linkage", "merge_no", "cluster_a", "cluster_b", "height"], hrows)

# 3.5
mean_vec = Z.mean(axis=0)
Xc = Z - mean_vec
S = Xc.T @ Xc / (n - 1)
vals, vecs = np.linalg.eigh(S)
idx = np.argsort(vals)[::-1]
vals, vecs = vals[idx], vecs[:, idx]
for j in range(5):
    if vecs[np.argmax(np.abs(vecs[:, j])), j] < 0:
        vecs[:, j] = -vecs[:, j]
tot = vals.sum()
R["eigenvalue_sum"] = tot
for j in range(5):
    R[f"eigenvalue_{j + 1}"] = vals[j]
for j in range(5):
    R[f"explained_ratio_{j + 1}"] = vals[j] / tot
cum = np.cumsum(vals) / tot
for j in range(5):
    R[f"cumulative_ratio_{j + 1}"] = cum[j]
R["components_for_80pct"] = int(np.argmax(cum >= 0.8) + 1)
for pc in (1, 2):
    for j in range(5):
        R[f"pc{pc}_loading_X{j + 1}"] = vecs[j, pc - 1]
P = Xc @ vecs[:, :2]
for r in range(3):
    R[f"row{r + 1}_pc1"] = P[r, 0]
    R[f"row{r + 1}_pc2"] = P[r, 1]
for s in range(3):
    R[f"mean_pc1_segment_{'ABC'[s]}"] = P[seg == s, 0].mean()

write_results("outputs/part3_results.txt", R)
