# JL for k-means Clustering and Linear Regression

**CSE558 – Data Science – Assignment**

## Contributors

- [Anouska Sahoo](https://github.com/anouskasahoo) | 2023096
- [Vaishnavi Srivastava](https://github.com/vaiiishhh) | 2023574

## About the project

This project studies random projection using the Johnson–Lindenstrauss (JL) lemma on the KDD Cup 1999 dataset. Question 1 reduces the number of features before k-means clustering. Question 2 reduces the number of rows before least-squares regression. Both evaluate the resulting solutions on the original data.

## Project structure

```text
jl-kmeans-regression/
├── data/
│   ├── D.npy                       # Shared data matrix; generated locally, ignored by Git
│   └── feature_names.npy           # Feature names saved by build_data_matrix.py
├── build_data_matrix.py            # Download, encode and scale the shared dataset
├── question1.py                    # Q1 calculations and four graphs
├── question1_graphs/
│   ├── q1b_ratio.png
│   ├── q1c_compare.png
│   ├── q1d_time.png
│   └── q1e_boxplot.png
├── question2.py                    # Q2 calculations, measurements and graphs
├── question2_results/
│   ├── question2_a_coefficients.npy
│   ├── question2_a_summary.csv
│   ├── question2_b_runs.csv
│   ├── question2_b_summary.csv
│   ├── question2_b_ratios.png
│   ├── question2_c_runs.csv
│   ├── question2_c_summary.csv
│   ├── question2_c_times.png
│   ├── question2_d_runs.csv
│   ├── question2_d_summary.csv
│   ├── question2_d_histograms.png
│   ├── question2_e_runs.csv
│   ├── question2_e_summary.csv
│   ├── question2_e_subset_indices.npy
│   ├── build_data_matrix.log
│   └── question2_run.log
└── README.md
```

## Steps to run

1. Install the dependencies:

   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```

2. Open the project folder:

   ```bash
   cd DSC_Assignmnet
   ```

3. If `data/D.npy` is missing, prepare the shared matrix:

   ```bash
   python build_data_matrix.py
   ```

4. Run the questions:

   ```bash
   python question1.py
   python question2.py
   ```

Question 1 also builds `D.npy` itself if it is missing. Question 2 requires the saved matrix and fetches the matching dataset labels. Fetching the dataset needs internet access if it is not already cached.

Question 1 saves four PNGs in `question1_graphs/` and prints progress and measurements to the terminal. Question 2 saves CSV measurements, NumPy arrays and three PNGs in `question2_results/`. The existing `.log` files are saved run records; the scripts print to the terminal and do not create these logs automatically.

Random seeds are fixed: 160705 for Q1, 42 for Q2's main generator and 123 for Q2's subset. Timings vary across machines and runs. The values below are from the saved experiments.

## Data matrix

The default 10% KDD Cup 1999 dataset has 494,021 rows and 41 original features. The three categorical features are one-hot encoded, constant columns are removed, and each remaining column is scaled to `[0, 1]`. Labels are excluded from the feature matrix. Rows are kept in their original order.

| Quantity | Value |
|---|---:|
| Rows `n` | 494,021 |
| Columns `d` after preprocessing | 116 |
| Numerical rank from the Q2 least-squares solve | 113 |
| Storage type | float32 |

## Q1 – JL for k-means clustering

Every clustering uses `KMeans(n_clusters=15, n_init=5)`. For each projected dimension `m ∈ {5, 10, 15, 20, 25}`, entries of `A` are independent `N(0, 1/m)` variables and `E = DAᵀ`. Costs are evaluated on the original data using nearest-centre squared distances:

```text
cost(D, B) = sum_i min_b ||D_i - b||²
```

### (a) Full-data baseline

Clustering `D` gives centres `B` and partition `C_D`.

| Quantity | Result |
|---|---:|
| `cost(D, B)` | 53,498.074603 |
| Time for `KMeans.fit` | 26.569 s |

The timer excludes data loading and cost evaluation. Costs are accumulated in float64.

### (b) Accuracy after projection

For each of five fresh projections per `m`, we cluster `E`, keep the cluster memberships and average the original rows in each cluster to form `B̂`. The mean minimises squared error within a fixed cluster. We then evaluate `cost(D, B̂)` by assigning each original row to its nearest centre in `B̂`.

| `m` | Mean cost ratio | Smallest ratio | Largest ratio |
|---:|---:|---:|---:|
| 5 | 1.15774 | 1.07059 | 1.23410 |
| 10 | 1.12764 | 1.04345 | 1.22498 |
| 15 | 1.03702 | 0.98505 | 1.09835 |
| 20 | 1.01184 | 0.98603 | 1.07255 |
| 25 | 1.05941 | 0.96150 | 1.18019 |

Ratios use `cost(D, B)` as the denominator. The mean improves overall, with the best mean at `m=20`, but rises again at `m=25`. The spread does not decrease steadily. Five of the 25 runs have a ratio below 1: the full-data k-means run is not guaranteed to find the global optimum, so a projected run can find a better partition.

Plot: [Cost ratios](question1_graphs/q1b_ratio.png).

### (c) Original cluster means versus pseudo-inverse centres

In the same runs, projected centres `a` are also mapped back as `A⁺a`, using `numpy.linalg.pinv`, to form `B⁺`.

| `m` | Mean ratio for `B̂` | Mean ratio for `B⁺` |
|---:|---:|---:|
| 5 | 1.15774 | 78.5663 |
| 10 | 1.12764 | 75.2877 |
| 15 | 1.03702 | 71.1662 |
| 20 | 1.01184 | 67.7447 |
| 25 | 1.05941 | 64.7102 |

We choose **`B̂`**. Its overall mean ratio is 1.0787, versus 71.4950 for `B⁺`. The pseudo-inverse retains only the component in the row space of `A`; it cannot recover the components lost during projection. Recomputing the original cluster means preserves those components and takes one extra pass over `D`.

Plot: [Centre comparison](question1_graphs/q1c_compare.png).

### (d) Running time

A single timer covers drawing `A`, forming `E` and clustering `E`. Rebuilding centres and evaluating costs are excluded. Times are averaged over the five runs in (b).

| `m` | Mean combined time (s) | Speedup over 26.569 s baseline |
|---:|---:|---:|
| 5 | 2.426 | 10.95× |
| 10 | 2.254 | 11.79× |
| 15 | 2.538 | 10.47× |
| 20 | 2.794 | 9.51× |
| 25 | 3.219 | 8.25× |

All tested projections are faster than full-data clustering. Times generally increase with `m`, though iteration counts vary and `m=10` is faster than `m=5`.

Plot: [Timing comparison](question1_graphs/q1d_time.png).

### (e) Distance distortion versus cost distortion

We use the same 10,000 random pairs of unequal rows for every `m`. One projection per dimension gives the pairwise squared-distance ratios. Separately, 20 independent projections per dimension give `cost_E(C_D) / cost_D(C_D)`, keeping the baseline partition fixed and recomputing its means in each space. Here `cost_D(C_D) = 53,498.0350`.

| `m` | Pair-ratio standard deviation | Cost-ratio standard deviation |
|---:|---:|---:|
| 5 | 0.3428 | 0.1724 |
| 10 | 0.3191 | 0.0944 |
| 15 | 0.3986 | 0.1053 |
| 20 | 0.2634 | 0.0607 |
| 25 | 0.1938 | 0.0688 |

The **cost ratio has the smaller spread at every `m`**, by both standard deviation and IQR. It combines many within-cluster distances, allowing some distortions to cancel. Both spreads decrease overall with fluctuations. Pair ratios share one projection, so they are correlated; their measured spread need not equal the theoretical `sqrt(2/m)` for repeated projections of one fixed vector.

Plot: [Both sets of box plots](question1_graphs/q1e_boxplot.png).

## Q2 – JL for linear regression

The target is 0 for a normal connection and 1 for an attack. We minimise `Loss(D, y, a) = ||Da - y||²` using `numpy.linalg.lstsq` with `rcond=None` and no intercept. CountSketch assigns each row an independent uniform bucket and random sign, then sums signed features and targets within each bucket. The dense sketch matrix is never formed.

### (a) Full-data baseline

| Quantity | Result |
|---|---:|
| Full-data loss `||Db - y||²` | 2,949.788815 |
| Time for the least-squares solve | 2.127519 s |
| Numerical rank of `D` | 113 |

### (b) Accuracy after CountSketch

For each `m`, five fresh sketches produce a solution `ã` from `SD` and `Sy`. We evaluate the loss on the full original dataset and divide by the baseline loss.

| `m` | Mean full-data loss ratio | Smallest ratio | Largest ratio | Mean sketched loss |
|---:|---:|---:|---:|---:|
| 50 | 3.7744 | 3.0140 | 5.6804 | 1.384 × 10⁻²³ |
| 100 | 15.1856 | 6.5828 | 27.8798 | 1.030 × 10⁻²³ |
| 200 | 2.3413 | 1.8277 | 2.7100 | 1274.739 |
| 500 | 1.2954 | 1.2648 | 1.3222 | 2289.017 |
| 1000 | 1.1209 | 1.0948 | 1.1319 | 2624.802 |
| 2000 | 1.0622 | 1.0525 | 1.0764 | 2827.870 |
| 5000 | 1.0229 | 1.0188 | 1.0264 | 2887.240 |

Below rank 113, the compressed problem can fit its targets almost exactly while giving large original-data errors. The mean ratio worsens from `m=50` to `m=100`. Above the rank, the observed mean ratio improves steadily, reaching 1.0229 at `m=5000`. Every ratio is at least 1 because the baseline minimises the original least-squares loss.

Plot: [Loss ratios](question2_results/question2_b_ratios.png).

### (c) Running time

The timer covers drawing the sketch, forming `SD` and `Sy`, and solving the compressed problem. Loss evaluation and plotting are excluded.

| `m` | Mean combined time (s) | Speedup over 2.127519 s baseline |
|---:|---:|---:|
| 50 | 0.419867 | 5.067× |
| 100 | 0.407707 | 5.218× |
| 200 | 0.448826 | 4.740× |
| 500 | 0.448678 | 4.742× |
| 1000 | 0.444807 | 4.783× |
| 2000 | 0.429447 | 4.954× |
| 5000 | 0.440870 | 4.826× |

All tested sizes are faster than the full-data solve, including sketch construction.

Plot: [Timing comparison](question2_results/question2_c_times.png).

### (d) Residual norm preservation

For the fixed residual `v = Db - y`, we draw 100 independent sketches per size and measure `||Sv||² / ||v||²` without solving regression again.

| `m` | Mean norm ratio | Sample standard deviation |
|---:|---:|---:|
| 50 | 0.987864 | 0.183561 |
| 500 | 1.002429 | 0.065296 |
| 5000 | 0.998220 | 0.020838 |

The means stay close to 1 and the spread decreases with `m`. Independent random signs make collision cross terms zero in expectation. For a fixed residual, the variance scales as `1/m`, so the standard deviation scales as `1/sqrt(m)`.

Plot: [Residual norm-ratio histograms](question2_results/question2_d_histograms.png).

### (e) CountSketch versus Gaussian sketching

Both methods use the same fixed subset of 50,000 rows, `m=500` and five trials. The subset baseline loss is 295.158682. Only drawing the sketch and forming the compressed features and targets are timed; solving and loss evaluation are excluded.

| Method | Mean construction time (s) | Mean subset loss ratio |
|---|---:|---:|
| CountSketch | 0.038449 | 3.074398 |
| Gaussian | 0.170562 | 1.263199 |

CountSketch construction is about 4.44 times faster. Gaussian sketches give better accuracy at this fixed size. Four CountSketch trials have ratios between 1.3183 and 1.3516; one has ratio 10.0506 and raises the mean. It remains included in the results.

For the full dataset, we choose **CountSketch for speed and memory**, while checking the resulting loss and increasing the sketch size if needed. A dense Gaussian sketch with `m=500` would require about 1.98 GB in float64 for the sketch matrix alone. Gaussian is the better accuracy choice in these fixed-size subset experiments.

Individual trials and summaries for Q2 are saved in `question2_results/`.


---

### We hope you enjoy exploring this project as much as we enjoyed creating it !

```
⠀⣀⢤⠲⠖⠒⠲⠖⠒⠲⠖⠒⠲⠖⠒⠘⠀⠀⠀⠀
⣆⠈⣤⠤⠤⠤⠤⠤⠤⠤⠤⠤⣤⠠⠦⣴⣀⠀⠀⠀
⣸⠀⣽⠀⠀⠀⣳⣖⠀⠀⠀⠀⣻⠀⠀⠀⠋⡾⣰⠀
⠃⡴⢃⣰⠀⠀⠀⠀⠀⠀⠀⠀⣯⠀⠀⠀⣀⠖⠏⣰
⠀⠀⠁⠉⠚⠔⠤⣤⡤⢤⡤⢤⡽⢀⣄⣳⠈⠀⠀⡇
⠀⠀⠀⠀⠀⠀⢢⡌⠱⡎⠀⠀⠀⠛⠎⢠⡀⣀⣀⢭
⠀⠀⠀⠀⠀⠀⠸⠇⠀⠇⠀⠀⠀⠀⠀⣇⠀⠀⠀⣽
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣇⠐⡠⣦⢻
⠀⠀⠀⠀⠀⠀⠀⡶⠴⣤⢀⠀⠀⣄⠢⠙⣎⠳⠈⠀
⠀⠀⠀⠀⠀⠀⠀⣧⣋⣀⣁⣹⣉⠤⠶⠓⠈⠀⠀⠀⠀⠀
```

