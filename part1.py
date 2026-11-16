import math
import numpy as np
from common import load, write_results

a = load()
X4, X1, X2, X3, X5, X6, X7 = a[:, 3], a[:, 0], a[:, 1], a[:, 2], a[:, 4], a[:, 5], a[:, 6]
y = a[:, 8]
n = len(y)
R = {}

# 1.1
mean_value = y.mean()
sd_value = y.std(ddof=1)
se_value = sd_value / math.sqrt(n)
R.update(n=n, mean_value=mean_value, sd_value=sd_value, se_value=se_value,
         ci95_low=mean_value - 1.96 * se_value, ci95_high=mean_value + 1.96 * se_value)

# 1.2
z_stat = (mean_value - 8.0) / se_value
R.update(z_stat=z_stat, reject_h0=abs(z_stat) > 1.96)

# 1.3
R["n_required"] = int(math.ceil((1.96 * sd_value / 0.5) ** 2))


def simple_fit(x, t):
    slope = (((x - x.mean()) * (t - t.mean())).sum() / (len(x) - 1)) / x.var(ddof=1)
    return slope, t.mean() - slope * x.mean()


# 1.4
slope, icpt = simple_fit(X4, y)
fit = icpt + slope * X4
tss = ((y - y.mean()) ** 2).sum()
ess = ((fit - y.mean()) ** 2).sum()
rss = ((y - fit) ** 2).sum()
R.update(reg_slope=slope, reg_intercept=icpt, tss=tss, ess=ess, rss=rss,
         tss_equals_ess_plus_rss=abs(tss - ess - rss) < 1e-6 * tss,
         r2=1 - rss / tss, rmse=math.sqrt(rss / n), mae=np.abs(y - fit).mean())

# 1.5
sdx, mx = X4.std(ddof=1), X4.mean()
z = (X4 - mx) / sdx
m = c = 0.0
costs = {0: (y ** 2).mean()}
for t in range(1, 501):
    e = y - (m * z + c)
    gm = -2.0 / n * (z * e).sum()
    gc = -2.0 / n * e.sum()
    m -= 0.1 * gm
    c -= 0.1 * gc
    if t in (1, 10, 100, 500):
        costs[t] = ((y - (m * z + c)) ** 2).mean()
R.update(gd_J0=costs[0], gd_J1=costs[1], gd_J10=costs[10], gd_J100=costs[100], gd_J500=costs[500])
gd_slope = m / sdx
gd_intercept = c - m * mx / sdx
R.update(gd_slope=gd_slope, gd_intercept=gd_intercept, gd_slope_gap=abs(gd_slope - slope))


# 1.6
def halves(fitted, resid):
    o = np.argsort(fitted, kind="stable")
    lo, up = resid[o[:400]].std(ddof=1), resid[o[400:]].std(ddof=1)
    return lo, up, up / lo


lo, up, ratio = halves(fit, y - fit)
R.update(resid_sd_lower_raw=lo, resid_sd_upper_raw=up, resid_sd_ratio_raw=ratio)
lx, ly = np.log(X4), np.log(y)
ls, li = simple_fit(lx, ly)
lfit = li + ls * lx
lrss = ((ly - lfit) ** 2).sum()
ltss = ((ly - ly.mean()) ** 2).sum()
loglog_r2 = 1 - lrss / ltss
R.update(loglog_intercept=li, loglog_slope=ls, loglog_r2=loglog_r2, loglog_rmse=math.sqrt(lrss / n))
lo, up, ratio = halves(lfit, ly - lfit)
R.update(resid_sd_lower_log=lo, resid_sd_upper_log=up, resid_sd_ratio_log=ratio)

# 1.7
M = np.column_stack([np.ones(n), lx, X1, X2, X3, X5, X6, X7])
beta = np.linalg.solve(M.T @ M, M.T @ ly)
for i, b in enumerate(beta):
    R[f"mr_beta_{i}"] = b
mrss = ((ly - M @ beta) ** 2).sum()
mr_r2 = 1 - mrss / ltss
R["mr_r2"] = mr_r2
R["mr_adj_r2"] = 1 - (1 - mr_r2) * (n - 1) / (n - 7 - 1)
R["loglog_adj_r2"] = 1 - (1 - loglog_r2) * (n - 1) / (n - 1 - 1)

write_results("outputs/part1_results.txt", R)
