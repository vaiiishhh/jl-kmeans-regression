#!/usr/bin/env python3
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

SEED = 160705
K = 15
N_INIT = 5
M_LIST = [5, 10, 15, 20, 25]
N_REPEAT_B = 5            # repetitions in (b), (c), (d)
N_PAIRS = 10_000          # (e)
N_DRAWS_E = 20            # (e)(ii)

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
GRAPH_DIR = BASE / "question1_graphs"
D_PATH = DATA_DIR / "D.npy"
META_PATH = DATA_DIR / "D_meta.json"
DATA_DIR.mkdir(exist_ok=True)
GRAPH_DIR.mkdir(exist_ok=True)


# Data matrix D

def build_D():
    from sklearn.datasets import fetch_kddcup99
    kdd = fetch_kddcup99(subset=None, as_frame=True)
    X = kdd.data.copy()
    cat_cols = ["protocol_type", "service", "flag"]
    for c in cat_cols:  # bytes -> str
        X[c] = X[c].map(lambda v: v.decode() if isinstance(v, bytes) else str(v))
    onehot = pd.get_dummies(X[cat_cols], dtype=float)               # (i)one-hot
    num = X.drop(columns=cat_cols).astype(float)                    # (i) 38 numeric -> float
    n_num = num.shape[1]
    full = pd.concat([num, onehot], axis=1)
    d_before = full.shape[1]
    keep = full.columns[full.nunique() > 1]                         # (ii) drop constants
    dropped = [str(c) for c in full.columns if c not in set(keep)]
    full = full[keep].astype(np.float64)
    mn, mx = full.min(), full.max()
    full = (full - mn) / (mx - mn)                                  # (iii) min-max to [0,1]
    D = full.to_numpy(dtype=np.float32)
    meta = dict(n_numeric=n_num, n_onehot=int(onehot.shape[1]), d_before_drop=int(d_before),
                dropped=dropped, n=int(D.shape[0]), d=int(D.shape[1]))
    np.save(D_PATH, D)
    META_PATH.write_text(json.dumps(meta, indent=2))
    return D, meta


if D_PATH.exists():
    D = np.load(D_PATH).astype(np.float32, copy=False)
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else None
    d_source = f"loaded from cache {D_PATH.name}"
else:
    D, meta = build_D()
    d_source = f"built from fetch_kddcup99 and saved to {D_PATH.name}"
n, d = D.shape
print(f"D: n={n}, d={d} ({d_source})", flush=True)


def kmeans_cost(Dm, B, chunk=50_000):
    """cost(D,B) = sum_i min_{b in B} ||D_i - b||^2  (float64 accumulation, chunked)."""
    B = np.asarray(B, dtype=np.float64)
    bn = (B ** 2).sum(1)
    tot = 0.0
    for s in range(0, len(Dm), chunk):
        X = Dm[s:s + chunk].astype(np.float64)
        d2 = (X ** 2).sum(1)[:, None] - 2.0 * X @ B.T + bn[None, :]
        tot += float(np.maximum(d2.min(1), 0.0).sum())
    return tot


def centroids_from_labels(Dm, labels, k=K):
    """b_hat_j = mean of the ORIGINAL rows of D in cluster j."""
    out = np.zeros((k, Dm.shape[1]))
    for j in range(k):
        out[j] = Dm[labels == j].mean(axis=0, dtype=np.float64)
    return out


def make_partition_index(labels, k=K):
    return [np.where(labels == j)[0] for j in range(k)]


def partition_cost(X, idx_list, chunk=100_000):
    """cost_X(C) = sum_j sum_{i in C_j} ||X_i - mu_j(X)||^2 = sum_j [ sum||x||^2 - ||sum x||^2 / n_j ]."""
    tot = 0.0
    for idx in idx_list:
        if len(idx) == 0:
            continue
        s = np.zeros(X.shape[1])
        ss = 0.0
        for a in range(0, len(idx), chunk):
            blk = X[idx[a:a + chunk]].astype(np.float64)
            s += blk.sum(0)
            ss += float((blk ** 2).sum())
        tot += ss - float(s @ s) / len(idx)
    return tot


def draw_A(m, rng):
    """A in R^{m x d}, iid N(0, 1/m)."""
    return rng.normal(0.0, 1.0 / np.sqrt(m), size=(m, d))


def fit_kmeans(X, seed):
    return KMeans(n_clusters=K, n_init=N_INIT, random_state=seed).fit(X)

print("(a) k-means on D ...", flush=True)
t0 = time.perf_counter()
km_D = fit_kmeans(D, SEED)
time_a = time.perf_counter() - t0
B_D = km_D.cluster_centers_.astype(np.float64)
labels_D = km_D.labels_
cost_B = kmeans_cost(D, B_D)
idx_D = make_partition_index(labels_D)
cost_D_CD = partition_cost(D, idx_D)  # cost_D(C_D), used in (e)
print(f"    cost(D,B)={cost_B:.6g}  time={time_a:.2f}s", flush=True)

# (b), (c), (d) share the same runs
print("(b)(c)(d) projected k-means ...", flush=True)
runs = []  # dicts
for mi, m in enumerate(M_LIST):
    for rep in range(N_REPEAT_B):
        rng = np.random.default_rng([SEED, 1, m, rep])
        t0 = time.perf_counter()
        A = draw_A(m, rng)
        E = D @ A.astype(np.float32).T                       # E = D A^T, rows A D_i
        km = fit_kmeans(E, SEED + 1000 * m + rep)
        t_run = time.perf_counter() - t0                     # time for (d)
        # (b) B_hat: average ORIGINAL rows of D in each cluster found on E
        B_hat = centroids_from_labels(D, km.labels_)
        c_hat = kmeans_cost(D, B_hat)
        # (c) B_plus: pseudo-inverse map of the m-dim centres
        A_pinv = np.linalg.pinv(A)                           # d x m
        B_plus = km.cluster_centers_.astype(np.float64) @ A_pinv.T   # rows = A^+ a
        c_plus = kmeans_cost(D, B_plus)
        runs.append(dict(m=m, rep=rep, c_hat=c_hat, r_hat=c_hat / cost_B,
                         c_plus=c_plus, r_plus=c_plus / cost_B, t=t_run))
        print(f"    m={m:2d} rep={rep}: ratio B_hat={c_hat/cost_B:.4f}  "
              f"ratio B+={c_plus/cost_B:.3f}  time={t_run:.2f}s", flush=True)

df = pd.DataFrame(runs)
summ = df.groupby("m").agg(mean_hat=("r_hat", "mean"), min_hat=("r_hat", "min"), max_hat=("r_hat", "max"),
                           mean_plus=("r_plus", "mean"), min_plus=("r_plus", "min"), max_plus=("r_plus", "max"),
                           mean_t=("t", "mean")).reset_index()

# (e) distance distortion vs cost distortion
print("(e) pairwise distortion vs cost distortion ...", flush=True)
rng_pairs = np.random.default_rng([SEED, 2])
pi, pl = [], []
while len(pi) < N_PAIRS:
    i = rng_pairs.integers(0, n, size=20_000)
    l = rng_pairs.integers(0, n, size=20_000)
    ok = (i != l) & np.any(D[i] != D[l], axis=1)
    pi.extend(i[ok].tolist())
    pl.extend(l[ok].tolist())
pi = np.array(pi[:N_PAIRS])
pl = np.array(pl[:N_PAIRS])
Dd = D[pi].astype(np.float64) - D[pl].astype(np.float64)      # same pairs for every m
Dd_norm = (Dd ** 2).sum(1)

pair_ratios, cost_ratios = {}, {}
for m in M_LIST:
    rng = np.random.default_rng([SEED, 3, m])
    A = draw_A(m, rng)                                         # one draw for (i)
    Ed = Dd @ A.T
    pair_ratios[m] = (Ed ** 2).sum(1) / Dd_norm
    cr = []
    for t in range(N_DRAWS_E):                                 # 20 draws for (ii)
        rng2 = np.random.default_rng([SEED, 4, m, t])
        A2 = draw_A(m, rng2)
        E2 = D @ A2.astype(np.float32).T
        cr.append(partition_cost(E2, idx_D) / cost_D_CD)
    cost_ratios[m] = np.array(cr)
    print(f"    m={m}: pair std={pair_ratios[m].std():.3f}  cost-ratio std={cost_ratios[m].std():.4f}", flush=True)

# Graphs
print("graphs ...", flush=True)
x = np.arange(len(M_LIST))

# (b)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
yerr = np.vstack([summ.mean_hat - summ.min_hat, summ.max_hat - summ.mean_hat])
ax.bar(x, summ.mean_hat, yerr=yerr, capsize=5, color="#4c78a8", label="mean (bars: min-max)")
ax.axhline(1.0, color="k", ls="--", lw=1, label="ratio = 1")
ax.set_xticks(x)
ax.set_xticklabels(M_LIST)
ax.set_xlabel("m (projected dimension)")
ax.set_ylabel("cost(D, B_hat) / cost(D, B)")
ax.set_title("(b) k-means cost ratio, B_hat from clusters of E")
lo = min(1.0, summ.min_hat.min())
hi = max(1.0, summ.max_hat.max())
ax.set_ylim(max(0, lo - 0.1 * (hi - lo + 0.05)), hi + 0.15 * (hi - lo + 0.05))
ax.legend()
fig.tight_layout()
fig.savefig(GRAPH_DIR / "q1b_ratio.png", dpi=150)
plt.close(fig)

# (c)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
w = 0.38
ax.bar(x - w / 2, summ.mean_hat, w, color="#4c78a8", label="B_hat (cluster means in R^d)")
ax.bar(x + w / 2, summ.mean_plus, w, color="#e45756", label="B+ = A^+ a")
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(M_LIST)
ax.set_xlabel("m")
ax.set_ylabel("mean cost ratio to cost(D,B) (log scale)")
ax.set_title("(c) B_hat vs B+")
ax.legend()
fig.tight_layout()
fig.savefig(GRAPH_DIR / "q1c_compare.png", dpi=150)
plt.close(fig)

# (d)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(M_LIST, summ.mean_t, "o-", color="#4c78a8", label="draw A + E=DA^T + k-means on E")
ax.axhline(time_a, color="#e45756", ls="--", label=f"k-means on D ({time_a:.1f} s)")
ax.set_xlabel("m")
ax.set_ylabel("wall-clock time (s)")
ax.set_title("(d) running time vs m (mean of 5 runs)")
ax.set_xticks(M_LIST)
ax.legend()
fig.tight_layout()
fig.savefig(GRAPH_DIR / "q1d_time.png", dpi=150)
plt.close(fig)

# (e)
fig, ax = plt.subplots(figsize=(8, 4.6))
pos_a = np.arange(len(M_LIST)) * 3.0
pos_b = pos_a + 1.0
bp1 = ax.boxplot([pair_ratios[m] for m in M_LIST], positions=pos_a, widths=0.8, patch_artist=True,
                 showfliers=False, whis=(1, 99))
bp2 = ax.boxplot([cost_ratios[m] for m in M_LIST], positions=pos_b, widths=0.8, patch_artist=True,
                 showfliers=True)
for b in bp1["boxes"]:
    b.set_facecolor("#9ecae9")
for b in bp2["boxes"]:
    b.set_facecolor("#f4a582")
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_xticks(pos_a + 0.5)
ax.set_xticklabels(M_LIST)
ax.set_xlabel("m")
ax.set_ylabel("ratio")
ax.set_title("(e) pair-distance ratio (blue, 1 draw, 10^4 pairs; whiskers 1-99%) vs\n"
             "cost_E(C_D)/cost_D(C_D) (orange, 20 draws)")
ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["(i) pairwise squared distances", "(ii) k-means cost of C_D"],
          loc="upper right")
fig.tight_layout()
fig.savefig(GRAPH_DIR / "q1e_boxplot.png", dpi=150)
plt.close(fig)

print(f"Done. Graphs: {GRAPH_DIR}", flush=True)
