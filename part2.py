import numpy as np
from common import (split, train_logistic, bce, sigmoid, cells, metrics, knn_predict,
                    knn_cv, best_k, write_results, write_csv)

S = split()
Xtr, ytr, Xte, yte = S["Xtr"], S["ytr"], S["Xte"], S["yte"]
R = {}
R.update(train_rows=len(ytr), train_positives=int(ytr.sum()),
         test_rows=len(yte), test_positives=int(yte.sum()))

# 2.2
w, b, L = train_logistic(Xtr, ytr, 0.1, 1000, record=(1, 10, 100, 1000))
for e in (0, 1, 10, 100, 1000):
    R[f"loss_epoch_{e}"] = L[e]
for j in range(7):
    R[f"weight_X{j + 1}"] = w[j]
R["bias"] = b
R["test_bce"] = bce(Xte, yte, w, b)

# 2.3
for eta in (0.01, 0.1, 1.0):
    _, _, l = train_logistic(Xtr, ytr, eta, 200, record=(200,))
    R[f"loss_epoch_200_eta_{eta}"] = l[200]

# 2.4
for j in range(7):
    R[f"odds_ratio_X{j + 1}"] = float(np.exp(w[j]))

# 2.5
pte = sigmoid(Xte @ w + b)
ptr = sigmoid(Xtr @ w + b)
for tau in (0.5, 0.3):
    tp, fp, tn, fn = cells(yte, pte >= tau)
    m = metrics(tp, fp, tn, fn)
    R.update({f"tau_{tau}_tp": tp, f"tau_{tau}_fp": fp, f"tau_{tau}_tn": tn, f"tau_{tau}_fn": fn})
    for k in ("accuracy", "precision", "recall", "f1", "specificity"):
        R[f"tau_{tau}_{k}"] = m[k]
R["blind_accuracy"] = float((yte == 0).mean())

# 2.6
rows = []
for i in range(1, 10):
    tau = i / 10
    tp, fp, tn, fn = cells(yte, pte >= tau)
    rows.append([tau, tp, fp, tn, fn, tp / (tp + fn), fp / (fp + tn)])
write_csv("outputs/part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], rows)

# 2.7
cv = knn_cv(Xtr, ytr)
write_csv("outputs/part2_knn_cv.csv", ["K", "mean_cv_f1"], cv)
K = best_k(cv)
R["knn_best_k"] = K
pred, _ = knn_predict(Xtr, ytr, Xte, K)
tp, fp, tn, fn = cells(yte, pred)
m = metrics(tp, fp, tn, fn)
R.update(knn_tp=tp, knn_fp=fp, knn_tn=tn, knn_fn=fn, knn_accuracy=m["accuracy"],
         knn_precision=m["precision"], knn_recall=m["recall"], knn_f1=m["f1"])


# 2.8
def cost(y, p, tau):
    _, fp, _, fn = cells(y, p >= tau)
    return 50 * fp + 400 * fn


best = None
for k in range(1, 20):
    tau = k / 20
    c = cost(ytr, ptr, tau)
    if best is None or c < best[1]:
        best = (tau, c)
R["cost_tau"] = best[0]
R["cost_train_at_tau"] = best[1]
R["cost_test_at_tau"] = cost(yte, pte, best[0])
R["cost_test_at_0.5"] = cost(yte, pte, 0.5)
R["cost_test_flag_nobody"] = 400 * int(yte.sum())

write_results("outputs/part2_results.txt", R)
