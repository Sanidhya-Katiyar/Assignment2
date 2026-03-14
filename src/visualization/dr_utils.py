"""
src/visualization/dr_utils.py
──────────────────────────────
Dimensionality reduction helpers for feature embedding visualization.

Provides:
* Standard scaling (zero-mean, unit-variance)
* PCA — deterministic, fast, used both as a plot target and t-SNE pre-step
* t-SNE — run multiple times with different seeds for stability assessment;
  pick the run with lowest final KL divergence
* UMAP — optional, silently skipped if ``umap-learn`` is not installed

All functions return plain NumPy arrays so they are plottable by any
matplotlib code without further transformation.

Intermediate results are saved to ``outputs/embeddings/`` so expensive
projections can be reloaded without recomputing.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def fit_scaler(features: np.ndarray) -> StandardScaler:
    """
    Fit a StandardScaler on *features* and return it.

    Args:
        features: Float array ``(N, D)``.

    Returns:
        Fitted ``StandardScaler``.
    """
    scaler = StandardScaler()
    scaler.fit(features)
    return scaler


def scale_features(
    features: np.ndarray,
    scaler:   Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Z-score normalise *features*.

    Args:
        features: Float array ``(N, D)``.
        scaler:   Pre-fitted scaler to apply; if ``None`` a new one is fitted.

    Returns:
        ``(scaled_features, scaler)`` tuple.
    """
    if scaler is None:
        scaler = fit_scaler(features)
    return scaler.transform(features).astype(np.float32), scaler


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def compute_pca(
    features:     np.ndarray,
    n_components: int = 50,
    scale:        bool = True,
) -> Tuple[PCA, np.ndarray, StandardScaler]:
    """
    Fit PCA on *features* and return the fitted objects.

    Args:
        features:     Float array ``(N, D)``.
        n_components: Number of principal components to retain.
        scale:        Whether to z-score features before PCA.

    Returns:
        ``(pca, projected, scaler)`` where *projected* has shape
        ``(N, n_components)``.
    """
    scaler  = None
    X       = features.astype(np.float64)
    if scale:
        X, scaler = scale_features(X)

    n_comp  = min(n_components, X.shape[0], X.shape[1])
    pca     = PCA(n_components=n_comp, random_state=0)
    proj    = pca.fit_transform(X).astype(np.float32)
    return pca, proj, scaler


def project_pca_2d(
    pca:      PCA,
    features: np.ndarray,
    scaler:   Optional[StandardScaler] = None,
) -> np.ndarray:
    """
    Project *features* onto the first two PCs of a fitted PCA object.

    Args:
        pca:      Fitted ``sklearn.decomposition.PCA``.
        features: Float array ``(N, D)``.
        scaler:   Scaler used when fitting *pca*; applied before projection.

    Returns:
        Float array ``(N, 2)`` — first two principal components.
    """
    X = features.astype(np.float64)
    if scaler is not None:
        X = scaler.transform(X)
    return pca.transform(X)[:, :2].astype(np.float32)


def compute_pca_2d(
    features: np.ndarray,
    scale:    bool = True,
) -> Tuple[np.ndarray, PCA, StandardScaler]:
    """
    Convenience wrapper: fit PCA and immediately return 2-D projection.

    Returns:
        ``(coords_2d, pca, scaler)``
    """
    pca, proj_full, scaler = compute_pca(features, n_components=2, scale=scale)
    coords = proj_full[:, :2]
    return coords, pca, scaler


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------

def compute_tsne(
    features:      np.ndarray,
    n_components:  int   = 2,
    perplexity:    float = 30.0,
    learning_rate: float = 200.0,
    n_iter:        int   = 1000,
    init:          str   = "pca",
    random_state:  int   = 42,
    pca_preflight: bool  = True,
    pca_dims:      int   = 50,
    scale:         bool  = True,
) -> Tuple[np.ndarray, float]:
    """
    Compute a single t-SNE projection.

    When ``pca_preflight=True`` and the feature dimension exceeds
    *pca_dims*, PCA is applied first for speed (standard practice for
    high-dimensional features).

    Args:
        features:      Float array ``(N, D)``.
        n_components:  Number of output dimensions (default 2).
        perplexity:    t-SNE perplexity parameter.
        learning_rate: t-SNE learning rate.
        n_iter:        Maximum number of iterations.
        init:          Initialisation strategy (``"pca"`` or ``"random"``).
        random_state:  Random seed (logged alongside results).
        pca_preflight: Apply PCA to *pca_dims* before t-SNE when D > pca_dims.
        pca_dims:      PCA target dimensionality for pre-flight step.
        scale:         Z-score features before PCA/t-SNE.

    Returns:
        ``(embedding, kl_divergence)`` — embedding shape ``(N, n_components)``,
        KL divergence as a Python float.
    """
    X = features.astype(np.float64)

    if scale:
        X, _ = scale_features(X)

    # Pre-flight PCA when features are very high-dimensional
    if pca_preflight and X.shape[1] > pca_dims:
        n_c  = min(pca_dims, X.shape[0] - 1, X.shape[1])
        pca  = PCA(n_components=n_c, random_state=0)
        X    = pca.fit_transform(X)

    tsne = TSNE(
        n_components  = n_components,
        perplexity    = perplexity,
        learning_rate = learning_rate,
        n_iter        = n_iter,
        init          = init,
        random_state  = random_state,
        n_jobs        = 1,          # deterministic with single thread
    )
    embedding = tsne.fit_transform(X).astype(np.float32)
    kl        = float(tsne.kl_divergence_)
    return embedding, kl


def compute_tsne_multi_run(
    features:     np.ndarray,
    n_runs:       int   = 3,
    seeds:        Optional[List[int]] = None,
    perplexity:   float = 30.0,
    **tsne_kwargs,
) -> List[Dict]:
    """
    Run t-SNE *n_runs* times with different seeds.

    The run with the lowest KL divergence is flagged as ``"best"``.

    Args:
        features:   Float array ``(N, D)``.
        n_runs:     Number of independent t-SNE runs.
        seeds:      List of seeds (length must match *n_runs*).  Defaults to
                    ``[42, 43, 44, ...]``.
        perplexity: t-SNE perplexity (passed through to ``compute_tsne``).
        **tsne_kwargs: Additional kwargs forwarded to ``compute_tsne``.

    Returns:
        List of dicts, each with keys:
        ``"run_id"`` (1-based), ``"seed"``, ``"embedding"`` (ndarray),
        ``"kl_divergence"`` (float), ``"is_best"`` (bool).
    """
    if seeds is None:
        seeds = list(range(42, 42 + n_runs))
    if len(seeds) != n_runs:
        raise ValueError(f"len(seeds)={len(seeds)} must equal n_runs={n_runs}.")

    runs: List[Dict] = []
    for run_id, seed in enumerate(seeds, start=1):
        print(f"    t-SNE run {run_id}/{n_runs}  seed={seed}  "
              f"perplexity={perplexity} …", end="", flush=True)
        emb, kl = compute_tsne(
            features    = features,
            perplexity  = perplexity,
            random_state= seed,
            **tsne_kwargs,
        )
        print(f"  KL={kl:.4f}")
        runs.append({
            "run_id":       run_id,
            "seed":         seed,
            "embedding":    emb,
            "kl_divergence": kl,
            "is_best":      False,
        })

    best_idx = int(np.argmin([r["kl_divergence"] for r in runs]))
    runs[best_idx]["is_best"] = True
    return runs


# ---------------------------------------------------------------------------
# UMAP (optional)
# ---------------------------------------------------------------------------

def compute_umap(
    features:    np.ndarray,
    n_neighbors: int   = 15,
    min_dist:    float = 0.1,
    n_components:int   = 2,
    random_state:int   = 42,
    scale:       bool  = True,
) -> Optional[np.ndarray]:
    """
    Compute a UMAP projection of *features*.

    Silently returns ``None`` if ``umap-learn`` is not installed.

    Args:
        features:     Float array ``(N, D)``.
        n_neighbors:  UMAP ``n_neighbors`` parameter.
        min_dist:     UMAP ``min_dist`` parameter.
        n_components: Output dimensionality.
        random_state: Random seed.
        scale:        Z-score features before UMAP.

    Returns:
        Float array ``(N, n_components)`` or ``None``.
    """
    try:
        import umap  # type: ignore
    except ImportError:
        warnings.warn(
            "umap-learn is not installed; UMAP projection skipped.  "
            "Install with: pip install umap-learn",
            stacklevel=2,
        )
        return None

    X = features.astype(np.float64)
    if scale:
        X, _ = scale_features(X)

    reducer = umap.UMAP(
        n_neighbors  = n_neighbors,
        min_dist     = min_dist,
        n_components = n_components,
        random_state = random_state,
    )
    return reducer.fit_transform(X).astype(np.float32)


# ---------------------------------------------------------------------------
# Cluster compactness metric
# ---------------------------------------------------------------------------

def intra_class_compactness(
    embedding:   np.ndarray,
    labels:      np.ndarray,
    num_classes: int,
) -> Dict[int, float]:
    """
    Compute mean intra-class pairwise L2 distance in 2-D embedding space.

    A lower value indicates a tighter, more compact cluster.

    Args:
        embedding:   2-D embedding coordinates ``(N, 2)``.
        labels:      Integer class labels ``(N,)``.
        num_classes: Total number of classes.

    Returns:
        Dict mapping class index → mean pairwise distance.
    """
    compactness: Dict[int, float] = {}
    for cls in range(num_classes):
        idx  = np.where(labels == cls)[0]
        pts  = embedding[idx]
        if len(pts) < 2:
            compactness[cls] = 0.0
            continue
        # Compute pairwise distances via broadcasting
        diff = pts[:, None, :] - pts[None, :, :]          # (n,n,2)
        dists= np.sqrt((diff ** 2).sum(axis=-1))           # (n,n)
        # Upper triangle only, excluding diagonal
        upper = dists[np.triu_indices(len(pts), k=1)]
        compactness[cls] = float(upper.mean()) if len(upper) > 0 else 0.0
    return compactness


# ---------------------------------------------------------------------------
# Embedding persistence helpers
# ---------------------------------------------------------------------------

def save_embedding_csv(
    output_path: Path | str,
    coords:      np.ndarray,
    labels:      np.ndarray,
    image_paths: Optional[List[str]] = None,
    extra_cols:  Optional[Dict[str, np.ndarray]] = None,
) -> Path:
    """
    Save 2-D embedding coordinates to a CSV file.

    Columns: ``x``, ``y``, ``label``, optionally ``image_path`` and any
    *extra_cols*.

    Args:
        output_path:  Destination CSV path.
        coords:       Float array ``(N, 2)``.
        labels:       Integer label array ``(N,)``.
        image_paths:  Optional list of image path strings.
        extra_cols:   Optional dict of column_name → array (shape ``(N,)``).

    Returns:
        Path of the written file.
    """
    import csv as _csv

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["x", "y", "label"]
    if image_paths is not None:
        fieldnames.append("image_path")
    if extra_cols:
        fieldnames.extend(sorted(extra_cols.keys()))

    with out.open("w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(coords)):
            row: Dict = {
                "x":     f"{coords[i, 0]:.6f}",
                "y":     f"{coords[i, 1]:.6f}",
                "label": int(labels[i]),
            }
            if image_paths is not None:
                row["image_path"] = image_paths[i]
            if extra_cols:
                for k, arr in extra_cols.items():
                    row[k] = arr[i]
            writer.writerow(row)

    return out


def load_embedding_csv(csv_path: Path | str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load a CSV produced by :func:`save_embedding_csv`.

    Returns:
        ``(coords, labels, image_paths)`` where *coords* has shape ``(N, 2)``
        and *image_paths* is an empty list if the column is absent.
    """
    import csv as _csv
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Embedding CSV not found: '{p}'")

    xs, ys, lbls, paths = [], [], [], []
    with p.open() as fh:
        for row in _csv.DictReader(fh):
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            lbls.append(int(row["label"]))
            if "image_path" in row:
                paths.append(row["image_path"])

    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    labels = np.array(lbls, dtype=np.int64)
    return coords, labels, paths
