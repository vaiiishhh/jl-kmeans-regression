#!/usr/bin/env python3
"""
CSE558 Assignment, Question 1: JL for k-means clustering (KDD Cup 1999, 10%).

Run:   python q1_full.py
Needs: numpy, pandas, scikit-learn, matplotlib
Makes: data/D.npy (+ data/D_meta.json)   if D is not already there
       q1_graphs/*.png
       q1_report.txt                     (answers for parts a-e, with [insert x.png] markers)

All random numbers come from numpy Generators seeded from SEED, so every
number in the report can be reproduced (KMeans also gets a fixed random_state).
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
SEED = 558
K = 15
N_INIT = 5
M_LIST = [5, 10, 15, 20, 25]
N_REPEAT_B = 5            # repetitions in (b), (c), (d)
N_PAIRS = 10_000          # (e)
N_DRAWS_E = 20            # (e)(ii)

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
GRAPH_DIR = BASE / "q1_graphs"
REPORT = BASE / "q1_report.txt"
D_PATH = DATA_DIR / "D.npy"
META_PATH = DATA_DIR / "D_meta.json"
DATA_DIR.mkdir(exist_ok=True)
GRAPH_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------
# Data matrix D
# ----------------------------------------------------------------------------
def build_D():
    from sklearn.datasets import fetch_kddcup99
    kdd = fetch_kddcup99(subset=None, as_frame=True)
    X = kdd.data.copy()
    cat_cols = ["protocol_type", "service", "flag"]
    for c in cat_cols:  # bytes -> str
        X[c] = X[c].map(lambda v: v.decode() if isinstance(v, bytes) else str(v))
    onehot = pd.get_dummies(X[cat_cols], dtype=float)               # (i) one-hot
    num = X.drop(columns=cat_cols).astype(float)                    # (i) 38 numeric -> float
    n_num = num.shape[1]
    full = pd.concat([num, onehot], axis=1)
    d_before = full.shape[1]
    keep = full.columns[full.nunique() > 1]                         # (ii) drop constant columns
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


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Report helpers
# ----------------------------------------------------------------------------
R = []
W = 78


def line(s=""):
    R.append(s)


def head(title):
    line("")
    line("=" * W)
    line(title)
    line("=" * W)


def sub(title):
    line("")
    line("-- " + title + " " + "-" * max(0, W - 4 - len(title)))


def answer(tag, lines):
    line("")
    line("#" * W)
    line(f"#  >>>  ANSWER {tag}")
    line("#" * W)
    for s in lines:
        line("#  " + s)
    line("#" * W)


def para(s):
    line(s)


# ----------------------------------------------------------------------------
# Report: header + data
# ----------------------------------------------------------------------------
line("CSE558 Data Science - Assignment - Question 1: JL for k-means clustering")
line("Dataset: KDD Cup 1999 (10% version), sklearn.datasets.fetch_kddcup99")
line(f"Global random seed = {SEED}  (numpy Generators + KMeans random_state are derived from it)")
line("Note: group members' names / discussion partners / URLs used: [fill in before submitting]")

head("0. DATA MATRIX D")
para("What was done:")
para("  (i)   One-hot encoded protocol_type, service, flag (pandas.get_dummies); the other 38")
para("        features converted to float.")
para("  (ii)  Dropped every column that has the same value on all rows.")
para("  (iii) Rescaled every column to [0,1] with  x -> (x - min) / (max - min).")
para("  D is stored as float32 (memory); every cost / sum is accumulated in float64.")
if meta:
    para(f"  Columns after one-hot: {meta['d_before_drop']} "
         f"({meta['n_numeric']} numeric + {meta['n_onehot']} one-hot); "
         f"constant columns dropped: {len(meta['dropped'])} {meta['dropped']}")
para(f"  Source of D in this run: {d_source}")
answer("(data)", [f"n = {n} rows,   d = {d} columns      D is in R^(n x d)"])

# ----------------------------------------------------------------------------
# (a) k-means on D
# ----------------------------------------------------------------------------
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

head("PART (a)  k-means on the original data D")
para("What was done: sklearn KMeans(n_clusters=15, n_init=5, random_state=%d) fitted on D." % SEED)
para("Formula:  cost(D,B) = sum_i min_{b in B} ||D_i - b||_2^2, computed by assigning every row of")
para("          D to its nearest centre and adding squared distances (own routine, float64).")
para("Why: this is the baseline (B, C_D) against which all projected solutions are compared.")
para("Cross-check: sklearn inertia_ = %.6f (should agree with our cost up to float32 rounding)." % km_D.inertia_)
para("Cluster sizes: " + str(sorted([len(i) for i in idx_D], reverse=True)))
answer("(a)", [f"cost(D, B)        = {cost_B:.6f}",
               f"wall-clock time  = {time_a:.3f} s   (KMeans.fit on D only, k=15, n_init=5)"])

# ----------------------------------------------------------------------------
# (b), (c), (d) share the same runs
# ----------------------------------------------------------------------------
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

# ---- (b) report
head("PART (b)  k-means on E = D A^T, centres B_hat = mean of ORIGINAL rows per cluster")
para("What was done: for every m in {5,10,15,20,25} and each of 5 repetitions a fresh A (m x d,")
para("iid N(0,1/m) from numpy) was drawn, E = D A^T formed, KMeans(15, n_init=5) run on E, the")
para("clusters kept and centres rebuilt in R^d as  b_hat_j = (1/|C_j|) sum_{i in C_j} D_i.")
para("Formulas: cost(D,B_hat) = sum_i min_j ||D_i - b_hat_j||^2 ; ratio = cost(D,B_hat)/cost(D,B).")
para("Why: centres from E live in R^m and cannot be compared with rows of D, but the CLUSTERS")
para("carry over; the mean of a cluster is the best centre for that cluster, so B_hat is the natural")
para("way to turn a clustering found in low dimension into centres in the original space.")
sub("All 25 runs")
line(f"{'m':>3} {'rep':>3} {'cost(D,B_hat)':>18} {'ratio':>9}")
for r in runs:
    line(f"{r['m']:>3} {r['rep']:>3} {r['c_hat']:>18.4f} {r['r_hat']:>9.5f}")
sub("Summary of ratio cost(D,B_hat)/cost(D,B) over 5 runs")
line(f"{'m':>3} {'mean':>10} {'smallest':>10} {'largest':>10}")
for _, s in summ.iterrows():
    line(f"{int(s.m):>3} {s.mean_hat:>10.5f} {s.min_hat:>10.5f} {s.max_hat:>10.5f}")
n_below = int((df.r_hat < 1).sum())
below_list = df[df.r_hat < 1][["m", "rep", "r_hat"]]
mean_first, mean_last = summ.mean_hat.iloc[0], summ.mean_hat.iloc[-1]
width = (summ.max_hat - summ.min_hat).to_numpy()
answer("(b)", [
    "Table above (cost and ratio for every run; mean/smallest/largest ratio per m).",
    "Mean ratio per m: " + ", ".join(f"m={int(s.m)}: {s.mean_hat:.4f}" for _, s in summ.iterrows()),
    f"Mean ratio goes from {mean_first:.4f} (m=5) to {mean_last:.4f} (m=25); "
    f"min-max width goes from {width[0]:.4f} to {width[-1]:.4f}.",
    f"Number of runs with ratio < 1: {n_below} of {len(df)}.",
])
sub("Comment")
para("* Trend with m: larger m preserves distances better (JL: relative error ~ sqrt(1/m)), so the")
para("  projected clustering is closer to the clustering found in R^d: the mean ratio is expected")
para(f"  to approach 1 and the spread to shrink. Observed: mean {mean_first:.4f} -> {mean_last:.4f}, "
     f"spread {width[0]:.4f} -> {width[-1]:.4f}.")
para("  (Small m can still give ratios close to 1 here because the 15 clusters are very well")
para("  separated in KDD: attack types such as smurf/neptune form tight, far-apart groups.)")
if n_below:
    para(f"* Ratios below 1 ({n_below} runs): " + ", ".join(
        f"(m={int(r.m)}, rep={int(r.rep)}): {r.r_hat:.5f}" for _, r in below_list.iterrows()))
else:
    para("* No ratio below 1 was observed in this run, but one can occur (see next point).")
para("  A ratio < 1 does NOT contradict JL. The baseline B is only a local optimum of Lloyd's")
para("  algorithm (k-means is NP-hard; n_init=5 does not guarantee the global minimum), so")
para("  cost(D,B) >= OPT. A clustering found on the projected data can land in a different,")
para("  better local optimum and, after recomputing the exact means in R^d, beat B. JL only")
para("  promises cost(D,B_hat) <= (1+eps) * OPT, and OPT <= cost(D,B).")
line("")
line("[insert q1b_ratio.png]")

# ---- (c) report
head("PART (c)  Centres mapped back with the pseudo-inverse: B+ = {A^+ a}")
para("What was done: in the same 25 runs each centre a in R^m found by k-means on E was mapped")
para("to R^d by A^+ a, A^+ = numpy.linalg.pinv(A) (d x m). cost(D,B+) is evaluated exactly like")
para("cost(D,B) (nearest-centre assignment over the rows of D).")
para("Formula: ratio_+ = cost(D,B+)/cost(D,B).")
para("Why: B+ needs only the m-dim centres, no second pass over the clusters; we test whether")
para("that cheaper route is as good as recomputing means from the original rows (B_hat).")
sub("Summary: mean (smallest-largest) of cost(D,.)/cost(D,B) over 5 runs")
line(f"{'m':>3} | {'B_hat mean':>10} {'min':>9} {'max':>9} | {'B+ mean':>10} {'min':>9} {'max':>9}")
for _, s in summ.iterrows():
    line(f"{int(s.m):>3} | {s.mean_hat:>10.5f} {s.min_hat:>9.5f} {s.max_hat:>9.5f} | "
         f"{s.mean_plus:>10.4f} {s.min_plus:>9.4f} {s.max_plus:>9.4f}")
sub("All 25 runs (ratio_hat, ratio_plus)")
line(f"{'m':>3} {'rep':>3} {'ratio B_hat':>12} {'ratio B+':>12}")
for r in runs:
    line(f"{r['m']:>3} {r['rep']:>3} {r['r_hat']:>12.5f} {r['r_plus']:>12.4f}")
use_hat = df.r_hat.mean() <= df.r_plus.mean()
answer("(c)", [
    "Ratio cost(D,B+)/cost(D,B) (mean over 5 runs) next to the (b) ratio:",
    *[f"   m={int(s.m):2d}:  B_hat = {s.mean_hat:.4f}    B+ = {s.mean_plus:.4f}" for _, s in summ.iterrows()],
    f"I would use: {'B_hat' if use_hat else 'B+'}  (lower mean ratio over all 25 runs: "
    f"B_hat {df.r_hat.mean():.4f} vs B+ {df.r_plus.mean():.4f}).",
])
sub("Comment / reasoning")
para("* B_hat consists of the actual cluster means in R^d, i.e. the optimal centres for the partition")
para("  found, so its cost is the best any centre set can do for that partition.")
para("* B+ = A^+ a lies in the m-dimensional row space of A and has no component along the other")
para("  d-m directions in which the data D_i do vary; every row pays for that missing part, and")
para("  A^+ A is only a projection, so B+ is not an approximation of the cluster means in R^d.")
para("* B_hat costs one extra pass over D (O(nd)); computing pinv(A) costs O(d m^2). So B_hat is")
para("  both better and cheap. Use B_hat.")
line("")
line("[insert q1c_compare.png]")

# ---- (d) report
head("PART (d)  Running time: E plus k-means on E versus k-means on D")
para("What was done: in every run of (b) the wall-clock time of 'draw A, compute E = D A^T, run")
para("KMeans(15,n_init=5) on E' was recorded (perf_counter) and averaged over the 5 runs per m.")
para("The time of (a) is drawn as a horizontal line. (Time to form B_hat and to evaluate costs is")
para("NOT included, since the assignment asks for E plus k-means only.)")
para("Why: the point of JL here is speed - k-means costs O(n k d) per Lloyd iteration, and")
para("reducing d to m cuts this to O(n k m), while forming E costs O(n d m).")
sub("Average time per m")
line(f"{'m':>3} {'mean time (s)':>14} {'speed-up vs (a)':>16}")
for _, s in summ.iterrows():
    line(f"{int(s.m):>3} {s.mean_t:>14.3f} {time_a / s.mean_t:>16.2f}x")
answer("(d)", [
    f"Time of k-means on D (from (a)): {time_a:.3f} s",
    *[f"   m={int(s.m):2d}: mean time(E + k-means) = {s.mean_t:.3f} s  ({time_a / s.mean_t:.2f}x vs (a))"
      for _, s in summ.iterrows()],
])
para("Comment: times grow with m (more dimensions to process per distance) but stay "
     + ("below" if (summ.mean_t < time_a).all() else "not always below")
     + " the time on the full D. Number of Lloyd iterations also varies between runs, which adds noise.")
line("")
line("[insert q1d_time.png]")

# ----------------------------------------------------------------------------
# (e) distance distortion vs cost distortion
# ----------------------------------------------------------------------------
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

head("PART (e)  Distortion of single pair distances vs distortion of the k-means cost")
para(f"What was done: {N_PAIRS} random pairs (i,l) with D_i != D_l were drawn once (same pairs for every m).")
para("(i)  For ONE draw of A per m: r_pair = ||E_i - E_l||^2 / ||D_i - D_l||^2 for all pairs.")
para(f"(ii) For {N_DRAWS_E} independent draws of A per m: r_cost = cost_E(C_D) / cost_D(C_D), where C_D is the")
para("     clustering from (a) and cost_X(C) = sum_j sum_{i in C_j} ||X_i - mu_j(X)||^2 with mu_j the")
para("     cluster mean in X (computed as sum||x||^2 - ||sum x||^2/n_j, in float64).")
para(f"     cost_D(C_D) = {cost_D_CD:.4f}")
para("Why: JL says E||A x||^2 = ||x||^2, with ||A x||^2/||x||^2 ~ chi^2_m / m for a FIXED x, so")
para("a single pair has relative std sqrt(2/m). The k-means cost for a fixed partition is a sum of")
para("many squared distances (it equals sum_j 1/(2 n_j) sum_{i,l in C_j} ||X_i - X_l||^2), so errors of")
para("different pairs partly cancel and its ratio should be much more concentrated.")
sub("Summary statistics")
line(f"{'m':>3} | {'pairs: mean':>11} {'std':>8} {'IQR':>8} {'sqrt(2/m)':>10} | "
     f"{'cost: mean':>10} {'std':>8} {'IQR':>8}")
spread_rows = {}
for m in M_LIST:
    p, c = pair_ratios[m], cost_ratios[m]
    ip = np.subtract(*np.percentile(p, [75, 25]))
    ic = np.subtract(*np.percentile(c, [75, 25]))
    spread_rows[m] = (p.std(), c.std(), ip, ic)
    line(f"{m:>3} | {p.mean():>11.4f} {p.std():>8.4f} {ip:>8.4f} {np.sqrt(2 / m):>10.4f} | "
         f"{c.mean():>10.4f} {c.std():>8.4f} {ic:>8.4f}")
cost_smaller_all = all(spread_rows[m][1] < spread_rows[m][0] for m in M_LIST)
ps5, ps25 = spread_rows[M_LIST[0]][0], spread_rows[M_LIST[-1]][0]
cs5, cs25 = spread_rows[M_LIST[0]][1], spread_rows[M_LIST[-1]][1]
answer("(e)", [
    f"Smaller spread: {'the cost ratio r_cost (ii)' if cost_smaller_all else 'see table (cost ratio not smaller for every m)'}"
    f"  (std smaller than the pairwise ratio's std for "
    f"{sum(spread_rows[m][1] < spread_rows[m][0] for m in M_LIST)} of {len(M_LIST)} values of m).",
    f"Change with m: std of pair ratio {ps5:.4f} (m={M_LIST[0]}) -> {ps25:.4f} (m={M_LIST[-1]}), "
    f"close to sqrt(2/m) = {np.sqrt(2 / M_LIST[0]):.4f} -> {np.sqrt(2 / M_LIST[-1]):.4f}.",
    f"                std of cost ratio {cs5:.4f} (m={M_LIST[0]}) -> {cs25:.4f} (m={M_LIST[-1]}).",
    "Both are centred near 1 (JL is unbiased) and both shrink as m grows.",
])
sub("Comment")
para("* Pairwise ratios: each pair is distorted like chi^2_m/m, so the box is wide (std ~ sqrt(2/m))")
para("  and has a long upper tail for small m; it falls like 1/sqrt(m).")
para("* Cost ratio: an average over very many pairs inside clusters, so over-/under-stretched")
para("  pairs cancel; boxes are much narrower, and still narrow further as m grows. This is why")
para("  JL can preserve the k-means cost with a small m even though single distances are noisy.")
line("")
line("[insert q1e_boxplot.png]")

# ----------------------------------------------------------------------------
# Graphs
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# Write report
# ----------------------------------------------------------------------------
line("")
line("=" * W)
line("Files produced: q1_graphs/q1b_ratio.png, q1c_compare.png, q1d_time.png, q1e_boxplot.png")
line("Reproducibility: all draws use numpy Generators seeded from SEED=%d; KMeans random_state is fixed." % SEED)
line("(Wall-clock times naturally vary a little between machines/runs.)")
REPORT.write_text("\n".join(R) + "\n", encoding="utf-8")
print(f"Done. Report: {REPORT}\nGraphs: {GRAPH_DIR}", flush=True)