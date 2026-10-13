"""Reusable helpers for the SVM / Kernel Trick project."""
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

CMAP_BG = ListedColormap(["#cfe0f3", "#f8d9c4"])
CMAP_PTS = ListedColormap(["#1f5fa8", "#d9661f"])


def make_spirals(n_per_class=300, noise=0.5, turns=1.5, seed=42):
    """Two interleaved spiral arms. Class 1 is class 0 rotated by 180 degrees.

    The radius grows with sqrt(U) so points are spread fairly evenly along each arm.
    Gaussian noise is added to every point, so the arms overlap slightly.
    """
    rng = np.random.default_rng(seed)
    t = np.sqrt(rng.uniform(0, 1, n_per_class)) * turns * 2 * np.pi
    arm = np.c_[t * np.cos(t), t * np.sin(t)]
    X = np.vstack([arm, -arm]) + rng.normal(0, noise, (2 * n_per_class, 2))
    y = np.r_[np.zeros(n_per_class), np.ones(n_per_class)].astype(int)
    return X, y


def evaluate(model, X, y):
    """Return a dict of classification metrics for a fitted model."""
    pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "f1": f1_score(y, pred),
        "confusion": confusion_matrix(y, pred),
    }


def timed_fit(model, X, y):
    """Fit and return (model, seconds)."""
    t0 = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - t0


def n_support(pipe):
    """Total number of support vectors in a fitted pipeline ending in 'svc'."""
    return int(pipe.named_steps["svc"].n_support_.sum())


def plot_boundary(pipe, X, y, ax=None, title="", show_sv=True, grid_n=300):
    """Plot decision regions in the ORIGINAL feature space.

    The pipeline scales internally, so we predict on a grid of raw coordinates.
    Support vectors are stored in the scaled space, so we map them back with
    the fitted scaler's inverse_transform before plotting.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    pad = 1.0
    x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
    y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x0, x1, grid_n), np.linspace(y0, y1, grid_n))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = pipe.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, cmap=CMAP_BG, alpha=0.9)
    ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.2)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=CMAP_PTS, s=10, edgecolor="none")
    if show_sv:
        svc = pipe.named_steps["svc"]
        sv = pipe.named_steps["standardscaler"].inverse_transform(svc.support_vectors_)
        ax.scatter(sv[:, 0], sv[:, 1], s=55, facecolors="none", edgecolors="k",
                   linewidths=0.6, label=f"support vectors ({len(sv)})")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
