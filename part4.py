import math
import numpy as np
from common import (split, train_logistic, sigmoid, cells, metrics, knn_predict, knn_cv,
                    best_k, write_results, write_csv)

S = split()
Xtr, ytr, Xte, yte = S["Xtr"], S["ytr"], S["Xte"], S["yte"]
Rtr, Rte = S["Xtr_raw"], S["Xte_raw"]
R = {}


def entropy(p):
    return -sum(q * math.log2(q) for q in (p, 1 - p) if q > 0)


def gini(p):
    return 1 - p * p - (1 - p) ** 2


# 4.1
n = len(ytr)
p0 = ytr.mean()
R["root_entropy"], R["root_gini"] = entropy(p0), gini(p0)
conds = {"S1": lambda X: X[:, 5] == 1, "S2": lambda X: X[:, 6] == 1,
         "S3": lambda X: X[:, 2] > 0.5, "S4": lambda X: X[:, 0] <= 24}
rows, gains = [], {}
for s, f in conds.items():
    m = f(Rtr)
    ny, py, nn, pn = int(m.sum()), int(ytr[m].sum()), int((~m).sum()), int(ytr[~m].sum())
    ig = entropy(p0) - ny / n * entropy(py / ny) - nn / n * entropy(pn / nn)
    gg = gini(p0) - ny / n * gini(py / ny) - nn / n * gini(pn / nn)
    gains[s] = (ig, gg)
    rows.append([s, ny, py, nn, pn, ig, gg])
write_csv("outputs/part4_splits.csv",
          ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"], rows)
bi = max(gains, key=lambda s: gains[s][0])
bg = max(gains, key=lambda s: gains[s][1])
R["best_split_by_info_gain"], R["best_split_by_gini_gain"] = bi, bg

# 4.2
m = conds[bi](Rtr)
rate_yes, rate_no = ytr[m].mean(), ytr[~m].mean()
R["stump_rate_yes"], R["stump_rate_no"] = rate_yes, rate_no
w, b, _ = train_logistic(Xtr, ytr, 0.1, 1000)
p_log = sigmoid(Xte @ w + b)
K = best_k(knn_cv(Xtr, ytr))
k_pred, p_knn = knn_predict(Xtr, ytr, Xte, K)
p_stump = np.where(conds[bi](Rte), rate_yes, rate_no)
pred = {"logistic": p_log >= 0.3, "knn": k_pred, "stump": p_stump >= 0.3}
votes = pred["logistic"].astype(int) + pred["knn"].astype(int) + pred["stump"].astype(int)
pred["hard_vote"] = votes >= 2
pred["soft_vote"] = (p_log + p_knn + p_stump) / 3 >= 0.3
vrows = []
for name in ("logistic", "knn", "stump", "hard_vote", "soft_vote"):
    tp, fp, tn, fn = cells(yte, pred[name])
    mt = metrics(tp, fp, tn, fn)
    vrows.append([name, tp, fp, tn, fn, mt["precision"], mt["recall"], mt["f1"]])
write_csv("outputs/part4_voting.csv", ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"], vrows)

# 4.3
y8 = np.array([1, 1, 1, -1, -1, -1, 1, -1])
h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])
w8 = np.full(8, 1 / 8)
arows, alphas = [], []
for t, h in ((1, h1), (2, h2)):
    eps = w8[h != y8].sum()
    alpha = 0.5 * math.log((1 - eps) / eps)
    un = w8 * np.exp(-alpha * y8 * h)
    new = un / un.sum()
    for i in range(8):
        arows.append([t, i + 1, w8[i], un[i], new[i]])
    R[f"round{t}_error"], R[f"round{t}_alpha"] = eps, alpha
    alphas.append(alpha)
    w8 = new
write_csv("outputs/part4_adaboost.csv",
          ["round", "point", "weight_before", "weight_unnormalised", "weight_after"], arows)
score = alphas[0] * h1 + alphas[1] * h2
R["ensemble_accuracy"] = float((np.sign(score) == y8).mean())

# 4.4
R["oob_fraction"] = (1 - 1 / 560) ** 560

# 4.5
arch = [("mlp_hidden1", 784 * 16 + 16), ("mlp_hidden2", 16 * 16 + 16), ("mlp_output", 16 * 10 + 10),
        ("cnn_conv1", 5 * 5 * 1 * 6 + 6), ("cnn_conv2", 5 * 5 * 6 * 16 + 16),
        ("cnn_dense1", (5 * 5 * 16) * 32 + 32), ("cnn_dense2", 32 * 10 + 10)]
write_csv("outputs/part4_architecture.csv", ["layer", "parameters"], arch)
R["mlp_total"] = sum(v for k, v in arch if k.startswith("mlp"))
R["cnn_total"] = sum(v for k, v in arch if k.startswith("cnn"))
R["fc_layer_params"] = 128 * 128 * 3 * 1024 + 1024
R["conv_layer_params"] = 3 * 3 * 3 * 64 + 64
R["rnn_total"] = 128 * 64 + 128 * 128 + 16 * 128 + 128 + 16

write_results("outputs/part4_results.txt", R)
