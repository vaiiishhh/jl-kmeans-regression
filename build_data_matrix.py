"""Prepare the KDD Cup 1999 data matrix.

Requires numpy, pandas, and scikit-learn.
Running this file downloads the default 10% dataset if it is not cached,
then saves D.npy and feature_names.npy in a data folder beside this file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99


def build_data_matrix():
    # Fetch the default 10% dataset. Labels are not part of D.
    kdd = fetch_kddcup99(subset=None, as_frame=True)
    features = kdd.data.copy()

    # Convert the 38 numeric features to float32 to save memory.
    text_columns = ["protocol_type", "service", "flag"]
    numeric_columns = features.columns.difference(text_columns)
    features[numeric_columns] = features[numeric_columns].astype(np.float32)

    # Make one column for every category of each text feature.
    # Keep all categories; constant columns are removed in the next step.
    encoded = pd.get_dummies(features, columns=text_columns, dtype=np.float32)
    feature_names = encoded.columns.to_numpy(dtype=str)
    D = encoded.to_numpy(dtype=np.float32, copy=True)

    # Drop columns whose minimum and maximum are the same.
    column_min = D.min(axis=0)
    column_max = D.max(axis=0)
    keep = column_max > column_min
    D = D[:, keep]
    feature_names = feature_names[keep]
    column_min = column_min[keep]
    column_max = column_max[keep]

    # Rescale each remaining column to [0, 1].
    D -= column_min
    D /= column_max - column_min

    return D, feature_names


def main():
    D, feature_names = build_data_matrix()

    # Save the matrix once so both questions can use the same data.
    output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "D.npy", D)
    np.save(output_dir / "feature_names.npy", feature_names)

    n, d = D.shape
    print(f"Number of rows n: {n}")
    print(f"Number of columns d: {d}")
    print(f"Saved data matrix to: {output_dir / 'D.npy'}")


if __name__ == "__main__":
    main()

# In later scripts, load the saved matrix with:
# D = np.load("data/D.npy")  # Path relative to working directory.
