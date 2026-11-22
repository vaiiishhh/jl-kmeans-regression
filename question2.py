"""Question 2"""

# SETUP

import csv
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_kddcup99


def loss(X, y, a):
    residual = X @ a - y
    return np.sum(residual ** 2)


def count_sketch(X, y, m, rng):
    n, d = X.shape
    buckets = rng.integers(0, m, size=n)
    signs = rng.choice([-1.0, 1.0], size=n)

    SD = np.zeros((m, d), dtype=float)
    Sy = np.zeros(m, dtype=float)

    np.add.at(SD, buckets, signs[:, None] * X)
    np.add.at(Sy, buckets, signs * y)
    return SD, Sy


if __name__ == "__main__":
    # SETUP
    
    data_path = Path(__file__).resolve().parent / "data" / "D.npy"
    D = np.load(data_path)
    n, d = D.shape

    kdd = fetch_kddcup99(subset=None, as_frame=True)

    y = (kdd.target != b"normal.").to_numpy(dtype=float)

    if len(y) != n:
        raise ValueError("D and y must contain the same number of rows.")

    rng = np.random.default_rng(42)


    # PART (a): Full-data baseline.

    start = perf_counter()

    b, _, rank_D, _ = np.linalg.lstsq(D, y, rcond=None)

    baseline_time = perf_counter() - start

    baseline_loss = loss(D, y, b)

    print("Part (a): Full-data baseline")
    print(f"Full-data loss: {baseline_loss:.6f}")
    print(f"Solve time: {baseline_time:.6f} seconds")
    print(f"Rank of D: {rank_D}")

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    np.save(results_dir / "question2_a_coefficients.npy", b)
    with (results_dir / "question2_a_summary.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["n", "d", "rank", "full_loss", "solve_seconds"])
        writer.writerow([n, d, rank_D, baseline_loss, baseline_time])

    
    # PART (b): Accuracy after CountSketch.

    m_values = np.array([50, 100, 200, 500, 1000, 2000, 5000])
    repetitions = 5
    ratios = np.zeros((len(m_values), repetitions))
    sketched_losses = np.zeros_like(ratios)
    run_results = []
    sketch_solve_times = np.zeros_like(ratios)

    for i, m in enumerate(m_values):
        for trial in range(repetitions):
            start = perf_counter()

            SD, Sy = count_sketch(D, y, m, rng)

            a_tilde, _, _, _ = np.linalg.lstsq(SD, Sy, rcond=None)

            sketch_solve_times[i, trial] = perf_counter() - start

            full_loss = loss(D, y, a_tilde)
            ratio = full_loss / baseline_loss

            sketched_loss = loss(SD, Sy, a_tilde)
            ratios[i, trial] = ratio
            sketched_losses[i, trial] = sketched_loss

            run_results.append([m, trial + 1, full_loss, ratio, sketched_loss])

    mean_ratios = ratios.mean(axis=1)
    min_ratios = ratios.min(axis=1)
    max_ratios = ratios.max(axis=1)
    mean_sketched_losses = sketched_losses.mean(axis=1)
    summary = np.column_stack((
        m_values, mean_ratios, min_ratios, max_ratios, mean_sketched_losses
    ))

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        results_dir / "question2_b_runs.csv",
        run_results,
        delimiter=",",
        header="m,trial,full_loss,loss_ratio,sketched_loss",
        comments="",
        fmt=["%d", "%d", "%.10g", "%.10g", "%.10g"],
    )
    np.savetxt(
        results_dir / "question2_b_summary.csv",
        summary,
        delimiter=",",
        header="m,mean_ratio,min_ratio,max_ratio,mean_sketched_loss",
        comments="",
        fmt=["%d", "%.10g", "%.10g", "%.10g", "%.10g"],
    )

    print("\nPart (b): Accuracy after CountSketch")
    print("m | mean ratio | min ratio | max ratio | mean sketched loss")
    for m, mean_ratio, min_ratio, max_ratio, mean_sketch_loss in summary:
        print(
            f"{int(m)} | {mean_ratio:.6f} | {min_ratio:.6f} | "
            f"{max_ratio:.6f} | {mean_sketch_loss:.6g}"
        )

    error_bars = np.vstack((mean_ratios - min_ratios, max_ratios - mean_ratios))
    fig, ax = plt.subplots()
    ax.errorbar(m_values, mean_ratios, yerr=error_bars, fmt="o-", capsize=4,
                label="Mean ratio with min-max error bars")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1, color="gray", linestyle=":", label="Full-data baseline")
    ax.axvline(rank_D, color="red", linestyle="--", label=f"rank(D) = {rank_D}")
    ax.set_xlabel("Sketch size m")
    ax.set_ylabel("Full-data loss / baseline loss")
    ax.set_title("Question 2(b): CountSketch accuracy")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(results_dir / "question2_b_ratios.png", dpi=200)
    plt.close(fig)
    print(f"Part (b) results saved to: {results_dir}")

    
    # PART (c): Compare sketch-and-solve times with the full-data solve.

    mean_times = sketch_solve_times.mean(axis=1)

    speedups = baseline_time / mean_times

    timing_runs = np.column_stack((
        np.repeat(m_values, repetitions),
        np.tile(np.arange(1, repetitions + 1), len(m_values)),
        sketch_solve_times.ravel(),
    ))
    np.savetxt(
        results_dir / "question2_c_runs.csv",
        timing_runs,
        delimiter=",",
        header="m,trial,sketch_and_solve_seconds",
        comments="",
        fmt=["%d", "%d", "%.10g"],
    )

    timing_summary = np.column_stack((
        m_values, mean_times, np.full(len(m_values), baseline_time), speedups
    ))
    np.savetxt(
        results_dir / "question2_c_summary.csv",
        timing_summary,
        delimiter=",",
        header="m,mean_sketch_and_solve_seconds,baseline_seconds,speedup",
        comments="",
        fmt=["%d", "%.10g", "%.10g", "%.10g"],
    )

    print("\nPart (c): Timing comparison")
    print(f"Full-data solve time: {baseline_time:.6f} seconds")
    print("m | mean sketch-and-solve seconds | speedup")
    for m, mean_time, speedup in zip(m_values, mean_times, speedups):
        print(f"{m} | {mean_time:.6f} | {speedup:.3f}x")

    fig, ax = plt.subplots()
    ax.plot(m_values, mean_times, "o-", label="Mean sketch-and-solve time")
    ax.axhline(baseline_time, color="red", linestyle="--",
               label="Full-data solve time")
    ax.set_xscale("log")
    ax.set_xlabel("Sketch size m")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Question 2(c): Timing comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(results_dir / "question2_c_times.png", dpi=200)
    plt.close(fig)
    print(f"Part (c) results saved to: {results_dir}")

    
    # PART (d): Check how CountSketch preserves the residual's squared norm.

    v = D @ b - y
    v_norm_squared = np.sum(v ** 2)
    if v_norm_squared == 0:
        raise ValueError("Part (d)'s norm ratio is undefined for a zero residual.")

    residual_m_values = np.array([50, 500, 5000])
    residual_repetitions = 100
    norm_ratios = np.zeros((len(residual_m_values), residual_repetitions))

    for i, m in enumerate(residual_m_values):
        for trial in range(residual_repetitions):
            buckets = rng.integers(0, m, size=n)
            signs = rng.choice([-1.0, 1.0], size=n)

            Sv = np.zeros(m, dtype=float)
            np.add.at(Sv, buckets, signs * v)

            norm_ratios[i, trial] = np.sum(Sv ** 2) / v_norm_squared

    mean_norm_ratios = norm_ratios.mean(axis=1)
    std_norm_ratios = norm_ratios.std(axis=1, ddof=1)

    residual_runs = np.column_stack((
        np.repeat(residual_m_values, residual_repetitions),
        np.tile(np.arange(1, residual_repetitions + 1), len(residual_m_values)),
        norm_ratios.ravel(),
    ))
    np.savetxt(
        results_dir / "question2_d_runs.csv",
        residual_runs,
        delimiter=",",
        header="m,trial,squared_norm_ratio",
        comments="",
        fmt=["%d", "%d", "%.10g"],
    )

    residual_summary = np.column_stack((
        residual_m_values, mean_norm_ratios, std_norm_ratios
    ))
    np.savetxt(
        results_dir / "question2_d_summary.csv",
        residual_summary,
        delimiter=",",
        header="m,mean_norm_ratio,sample_standard_deviation",
        comments="",
        fmt=["%d", "%.10g", "%.10g"],
    )
    print("\nPart (d): Residual norm preservation")
    print("m | mean squared-norm ratio | sample standard deviation")
    for m, mean_ratio, std_ratio in residual_summary:
        print(f"{int(m)} | {mean_ratio:.6f} | {std_ratio:.6f}")

    bins = np.histogram_bin_edges(norm_ratios.ravel(), bins=20)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    for i, m in enumerate(residual_m_values):
        ax = axes[i]
        ax.hist(norm_ratios[i], bins=bins, edgecolor="black", alpha=0.75)
        ax.axvline(1, color="red", linestyle="--", label="Ratio = 1")
        ax.set_title(f"m = {m}")
        ax.set_xlabel("Squared-norm ratio")
        ax.legend()
    axes[0].set_ylabel("Number of trials")
    fig.suptitle("Question 2(d): Residual norm preservation")

    fig.tight_layout()
    fig.savefig(results_dir / "question2_d_histograms.png", dpi=200)
    plt.close(fig)
    print(f"Part (d) results saved to: {results_dir}")


    
    # PART (e): Compare CountSketch and a dense Gaussian sketch.

    subset_size = 50_000
    subset_rng = np.random.default_rng(123)
    subset_indices = subset_rng.choice(n, size=subset_size, replace=False)
    D_sub = D[subset_indices]
    y_sub = y[subset_indices]
    np.save(results_dir / "question2_e_subset_indices.npy", subset_indices)

    b_sub, _, _, _ = np.linalg.lstsq(D_sub, y_sub, rcond=None)
    subset_baseline_loss = loss(D_sub, y_sub, b_sub)

    comparison_m = 500
    comparison_repetitions = 5
    methods = ["CountSketch", "Gaussian"]
    construction_times = np.zeros((len(methods), comparison_repetitions))
    comparison_ratios = np.zeros_like(construction_times)
    comparison_runs = []

    for i, method in enumerate(methods):
        for trial in range(comparison_repetitions):
            start = perf_counter()
            if method == "CountSketch":
                SD_sub, Sy_sub = count_sketch(D_sub, y_sub, comparison_m, rng)
            else:
                S = rng.normal(
                    loc=0.0,
                    scale=1 / np.sqrt(comparison_m),
                    size=(comparison_m, subset_size),
                )
                SD_sub = S @ D_sub
                Sy_sub = S @ y_sub
            construction_time = perf_counter() - start

            if method == "Gaussian":
                del S

            a_sub, _, _, _ = np.linalg.lstsq(SD_sub, Sy_sub, rcond=None)

            subset_loss = loss(D_sub, y_sub, a_sub)
            subset_ratio = subset_loss / subset_baseline_loss
            construction_times[i, trial] = construction_time
            comparison_ratios[i, trial] = subset_ratio
            comparison_runs.append([
                method, trial + 1, construction_time, subset_loss, subset_ratio
            ])

    mean_construction_times = construction_times.mean(axis=1)
    mean_comparison_ratios = comparison_ratios.mean(axis=1)

    with (results_dir / "question2_e_runs.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "method", "trial", "construction_seconds", "subset_loss", "loss_ratio"
        ])
        writer.writerows(comparison_runs)

    with (results_dir / "question2_e_summary.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "method", "m", "subset_size", "subset_baseline_loss",
            "mean_construction_seconds", "mean_loss_ratio"
        ])
        for i, method in enumerate(methods):
            writer.writerow([
                method, comparison_m, subset_size, subset_baseline_loss,
                mean_construction_times[i], mean_comparison_ratios[i]
            ])

    print("\nPart (e): CountSketch versus Gaussian sketch")
    print(f"Subset baseline loss: {subset_baseline_loss:.6f}")
    print("Method | mean construction seconds | mean loss ratio")
    for i, method in enumerate(methods):
        print(
            f"{method} | {mean_construction_times[i]:.6f} | "
            f"{mean_comparison_ratios[i]:.6f}"
        )
    print(f"Part (e) results saved to: {results_dir}")

