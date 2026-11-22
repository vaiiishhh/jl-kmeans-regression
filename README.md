# JL for k-means Clustering and Linear Regression

**CSE558 – Data Science – Assignment**

## Contributors

- [Anouska Sahoo](https://github.com/anouskasahoo) | 2023096
- [Vaishnavi Srivastava](https://github.com/vaiiishhh) | 2023574

---

## About the Project

This project studies **random projection** (the Johnson–Lindenstrauss, or JL, lemma) on the KDD Cup 1999 network-intrusion dataset (about 494k connections, each described by 41 features).

The JL lemma says that a random matrix can shrink data from many dimensions to a few while keeping distances between points almost unchanged. That lets us run expensive algorithms on a much smaller problem and still get a nearly as good answer. We test this in two settings:

- **Q1 – k-means clustering:** reduce the *columns* (features) with a Gaussian random matrix, cluster in the small space, then check how good the clustering is in the original space.
- **Q2 – linear regression:** reduce the *rows* (connections) with a sparse CountSketch, solve least squares on the small sketch, and compare against the exact solution.

---

## Project Structure

```
jl-kmeans-regression/
├── data/
│   └── D.npy              # data matrix D (loaded if present, otherwise built and saved)
├── q1_full.py             # Q1: runs all parts, makes graphs and report
├── q1_report.txt          # Q1: full write-up with answers
├── q1_graphs/             # Q1: plots (PNG)
├── q2_full.py             # Q2: runs all parts, makes graphs and report
├── q2_report.txt          # Q2: full write-up with answers
├── q2_graphs/             # Q2: plots (PNG)
└── README.md
```

---

## Steps to Run

1. **Install requirements** (Python 3.9+):
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```
2. **Open the project folder**:
   ```bash
   cd jl-kmeans-regression
   ```
3. **Run Q1** (needs internet the first time, to download the dataset via `fetch_kddcup99`):
   ```bash
   python q1_full.py
   ```
   On the first run `data/D.npy` is built and saved. Later runs load it directly.
4. **Run Q2**:
   ```bash
   python queston2.py
   ```
5. **Read the outputs**: reports are in `q1_report.txt` and `q2_report.txt`, plots are in `q1_graphs/` and `q2_graphs/`.

**Notes**
- All random seeds are fixed, so every reported number can be reproduced (wall-clock times will vary a little between machines).
- Q1 takes a few minutes because k-means is run about 26 times on roughly 494k rows.

---

## Data Matrix

| Quantity | Value |
|----------|-------|
| Rows `n` | `<n>` |
| Columns `d` | `<d>` |

---

## Q1 – JL for k-means (k = 15)

Gaussian projection `E = D Aᵀ` to `m ∈ {5, 10, 15, 20, 25}` dimensions, k-means on `E`, then centres rebuilt in the original space.

| Part | Final answer |
|------|--------------|
| **(a)** k-means on `D` | cost(D, B) = `<value>`, time = `<value> s` |
| **(b)** cost ratio of `B̂` | mean / min / max ratio per `m`: `<table or values>`. Trend with `m`: `<comment>`. Ratios below 1: `<comment>` |
| **(c)** `B̂` vs `B⁺` | ratio for `B̂` = `<value>`, for `B⁺` = `<value>`. Choice: `<B̂ or B⁺>` |
| **(d)** running time | time of E + k-means for each `m`: `<values>`, compared with `<time from (a)>` |
| **(e)** distortion spread | smaller spread: `<pair distances or k-means cost>`. Change with `m`: `<comment>` |

Plots: `q1_graphs/q1b_ratio.png`, `q1c_compare.png`, `q1d_time.png`, `q1e_boxplot.png`

For the method, formulas and full explanations, see **`q1_report.txt`**.

---

## Q2 – JL for linear regression

Predict whether a connection is an attack (`y ∈ {0,1}`) by least squares. The rows of `(D, y)` are sketched with a sparse **CountSketch** `S` (`m ≪ n`), and the small problem `min ‖SDa − Sy‖²` is solved instead.

| Part | Final answer |
|------|--------------|
| **(a)** exact solution `b` | Loss(D, y, b) = `<value>`, time = `<value> s` |
| **(b)** sketched solution `ã` | Loss ratio and sketched loss for `m = 50 … 5000`: `<table or values>`. Behaviour below / above `m = rank(D)`: `<comment>` |
| **(c)** running time | time per `m`: `<values>`, compared with `<time from (a)>` |
| **(d)** residual norm of sketch | mean / std of `‖Sv‖²/‖v‖²` for `m = 50, 500, 5000`: `<values>`. Spread vs `m`: `<comment>` |
| **(e)** CountSketch vs dense Gaussian | time: `<values>`, loss ratio: `<values>`. Choice for full `D`: `<CountSketch or dense>` |

Plots: `q2_graphs/<add plot names>`

For the method, formulas and full explanations, see **`q2_report.txt`**.

---

### We hope you enjoy exploring this project as much as we enjoyed creating it !

```
⠀⠀⠀⠀⠀⠀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣀⣤⣤⣀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢀⣀⡙⠻⢶⣶⣦⣴⣶⣶⣶⠾⠛⠛⠋⠉⠉⠉⠉⠙⠃⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠉⠉⠙⠛⠛⠋⠉⠉⠡⣤⣴⣶⣶⣾⣿⣿⣿⣛⣩⣤⡤⠖⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢠⣴⣾⠂⣴⣦⠈⣿⣿⣿⣿⣿⣿⠿⠛⣋⠁⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢀⣼⣿⣶⣄⡉⠻⣧⣌⣁⣴⣿⣿⣿⣿⣿⣿⡿⠛⠁⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣾⣿⣿⣿⣿⣿⣦⡈⢻⣿⣿⣿⣿⡿⠿⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⡀⢻⣿⣿⣿⣿⣿⣿⣿⡄⠙⠛⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⢠⣷⣄⡉⠻⢿⣿⣿⣿⠏⠠⢶⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⢸⣿⣿⣿⣶⣤⣈⠙⠁⠰⣦⣀⠉⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠘⢿⣿⣿⣿⣿⣿⡇⠠⣦⣄⠉⠳⣤⠈⠛⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⢠⣌⣉⡉⠉⣉⡁⠀⠀⠙⠗⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠹⢿⣿⣿⣿⣿⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠙⠻⣿⣿⠟⢀⣤⡀⠀⠀⠀⠀⠀⠀⣀⣀⣠⣤⣤⣤⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠛⠿⠿⡿⠂⣀⣠⣤⣤⣤⣀⣉⣉⠉⠉⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠙⠛⠛⠛⠛⠋⠉⠉⠁⠀⠀⠀⠀
```
