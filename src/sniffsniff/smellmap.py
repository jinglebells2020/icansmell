"""Render the 2-D "smell map" — a PCA scatter of the fitted :class:`SmellModel`.

This is the optional visual add-on for Milestone 2. It draws the PCA scores of a
:class:`~sniffsniff.dataset.Dataset` coloured by odor class, overlays each known
cluster centroid as an ``X`` marker, and (optionally) marks one new sample as a
star. It saves a PNG and returns the path.

matplotlib is imported *lazily* and forced onto the headless ``Agg`` backend, so
the module imports fine on machines without matplotlib and only fails — with a
clear message pointing at the ``sniffsniff[viz]`` extra — when you actually try
to render.
"""
from __future__ import annotations

import tempfile

import numpy as np

__all__ = ["render_map"]


def render_map(model, dataset=None, *, new_sample=None, path=None) -> str | None:
    """Render the PCA smell map to a PNG and return the saved path.

    Parameters
    ----------
    model:
        A fitted :class:`~sniffsniff.model.SmellModel` (needs ``transform`` and
        ``centroids_``).
    dataset:
        Optional :class:`~sniffsniff.dataset.Dataset`. When given, its sniffs are
        scattered in PCA space and coloured by class. When ``None``, only the
        model's cluster centroids are drawn.
    new_sample:
        Optional single raw ``(48,)`` feature vector; drawn as a star at its PCA
        coordinates.
    path:
        Where to write the PNG. Defaults to a fresh temp file.

    Returns
    -------
    str
        The filesystem path of the written PNG.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - exercised only without mpl
        raise ImportError(
            "matplotlib is required to render the smell map; "
            "install sniffsniff[viz]"
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        handle, path = tempfile.mkstemp(suffix=".png", prefix="smellmap_")
        # We only need the path; close the fd matplotlib will reopen it.
        import os

        os.close(handle)
    path = str(path)

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    try:
        classes = list(model.classes_)
        cmap = plt.get_cmap("tab10")
        color_of = {label: cmap(i % 10) for i, label in enumerate(classes)}

        # Scatter the dataset scores, coloured per class.
        if dataset is not None and getattr(dataset, "X", None) is not None \
                and np.asarray(dataset.X).size:
            scores = model.transform(dataset.X)
            y = np.asarray(dataset.y, dtype=str)
            for label in classes:
                mask = y == label
                if not mask.any():
                    continue
                pts = scores[mask]
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    s=40,
                    alpha=0.7,
                    color=color_of[label],
                    label=label,
                    edgecolors="none",
                )

        # Overlay cluster centroids as X markers.
        for label in classes:
            centroid = np.asarray(model.centroids_[label], dtype=np.float64)
            ax.scatter(
                centroid[0],
                centroid[1],
                marker="X",
                s=180,
                color=color_of[label],
                edgecolors="black",
                linewidths=1.2,
                zorder=5,
            )

        # Mark the new sample as a star.
        if new_sample is not None:
            new_sample = np.asarray(new_sample, dtype=np.float64).reshape(1, -1)
            coords = model.transform(new_sample)[0]
            ax.scatter(
                coords[0],
                coords[1],
                marker="*",
                s=320,
                color="gold",
                edgecolors="black",
                linewidths=1.2,
                zorder=6,
                label="new sample",
            )

        evr = np.asarray(model.explained_variance_ratio_, dtype=np.float64)
        ax.set_xlabel(f"PC1 ({evr[0] * 100:.0f}% var)")
        if evr.shape[0] > 1:
            ax.set_ylabel(f"PC2 ({evr[1] * 100:.0f}% var)")
        else:  # pragma: no cover - contract fits 2-D maps
            ax.set_ylabel("PC2")
        ax.set_title("Smell map (PCA)")
        # Only draw a legend when there are labeled artists (dataset scatter or
        # the new-sample star); centroids are unlabeled, so a centroids-only map
        # has nothing to legend.
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize="small")
        ax.grid(True, linestyle=":", alpha=0.4)

        fig.tight_layout()
        fig.savefig(path, dpi=110)
    finally:
        plt.close(fig)

    return path
