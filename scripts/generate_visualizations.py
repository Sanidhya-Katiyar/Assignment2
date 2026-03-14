#!/usr/bin/env python3
"""
scripts/generate_visualizations.py
────────────────────────────────────
Generate PCA, t-SNE (and optionally UMAP) embedding visualizations for
each backbone × layer-depth combination.

Features are loaded from pre-computed ``.npz`` files in
``outputs/features/`` (produced by Block 3 linear probe or Block 5
layer probing).  If a feature file is absent, the script can recompute
it from a checkpoint (see ``--checkpoint``).

Usage
-----
    # All models, all layers, PCA + 3 t-SNE runs:
    python scripts/generate_visualizations.py \\
        --config configs/baseline.yaml \\
        --models resnet50 efficientnet_b0 convnext_tiny \\
        --layers early middle final \\
        --dr pca tsne \\
        --tsne-runs 3

    # Single model, t-SNE only with custom perplexity:
    python scripts/generate_visualizations.py \\
        --config configs/baseline.yaml \\
        --models resnet50 \\
        --dr tsne \\
        --perplexity 50

Output files
------------
    outputs/embeddings/<model>_<layer>_pca.csv
    outputs/embeddings/<model>_<layer>_tsne_run<N>.csv
    outputs/plots/pca_layers_comparison.png/.pdf
    outputs/plots/tsne_layers_comparison_run<N>.png/.pdf
    outputs/plots/cluster_compactness.png/.pdf
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from src.datasets.aid_dataset import AIDDataset, discover_dataset, get_transforms
from src.probing.feature_extractor import LAYER_REGISTRY, DEPTH_ORDER, extract_all_layers
from src.utils.config import load_config
from src.utils.seed   import set_seed
from src.visualization.dr_utils import (
    compute_pca_2d,
    compute_tsne_multi_run,
    compute_umap,
    intra_class_compactness,
    save_embedding_csv,
)
from src.visualization.plotting import (
    build_class_colormap,
    plot_cluster_compactness,
    plot_embedding_grid,
    save_colormap_json,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate PCA/t-SNE/UMAP visualizations of feature embeddings."
    )
    p.add_argument("--config",    required=True)
    p.add_argument("--models",    nargs="+",
                   default=["resnet50", "efficientnet_b0", "convnext_tiny"],
                   choices=["resnet50", "efficientnet_b0", "convnext_tiny"])
    p.add_argument("--layers",    nargs="+", default=["early", "middle", "final"],
                   choices=["early", "middle", "final"])
    p.add_argument("--dr",        nargs="+", default=["pca", "tsne"],
                   choices=["pca", "tsne", "umap", "all"])
    p.add_argument("--tsne-runs", type=int,   default=3,
                   help="Number of independent t-SNE runs.")
    p.add_argument("--perplexity",type=float, default=30.0)
    p.add_argument("--seed",      type=int,   default=None)
    p.add_argument("--checkpoint",nargs="*",  default=None,
                   help="Optional checkpoint paths aligned with --models.")
    p.add_argument("--outdir",    default="outputs",
                   help="Root output directory (default: outputs).")
    p.add_argument("--num-workers",type=int,  default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Feature loading helpers
# ---------------------------------------------------------------------------

def _find_feature_npz(
    features_dir: Path,
    model_name:   str,
    layer_tag:    str,
) -> Optional[Path]:
    """
    Scan ``features_dir`` for any .npz file matching model+layer.

    Looks for files produced by Block 3 (``<model>_linearprobe_<seed>_epoch<N>.npz``)
    or Block 5 feature cache (``<model>_<layer>_features.npz``).
    """
    # Block 5 cache first
    b5 = features_dir / "feature_cache" / f"{model_name}_{layer_tag}_features.npz"
    if b5.exists():
        return b5
    # Block 3 probe snapshot (final layer only)
    if layer_tag == "final":
        for p in sorted(features_dir.glob(f"{model_name}_linearprobe_*.npz")):
            return p
    return None


def _load_features_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load features/labels/paths from a .npz file."""
    npz    = np.load(path, allow_pickle=True)
    feats  = npz["features"].astype(np.float32)
    labels = npz["labels"].astype(np.int64)
    paths: List[str] = []
    if "paths" in npz:
        paths = npz["paths"].tolist()
    return feats, labels, paths


def _build_viz_loader(config, num_workers: int = 0):
    """Build a DataLoader from visualization_subset.csv."""
    from torch.utils.data import DataLoader
    viz_csv = _PROJECT_ROOT / "outputs" / "visualization_subset.csv"
    if not viz_csv.exists():
        return None, []
    img_paths, labels = [], []
    with viz_csv.open() as fh:
        for row in csv.DictReader(fh):
            img_paths.append(row["path"])
            labels.append(int(row["label_idx"]))
    _, _, class_to_idx = discover_dataset(config.dataset_path)
    transform = get_transforms("test", config.image_size)
    dataset   = AIDDataset(img_paths, labels, class_to_idx, transform=transform)
    loader    = DataLoader(dataset, batch_size=config.batch_size,
                           shuffle=False, num_workers=num_workers, pin_memory=False)
    return loader, img_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    config = load_config(args.config)
    seed   = args.seed if args.seed is not None else config.seed
    set_seed(seed)

    outdir        = _PROJECT_ROOT / args.outdir
    features_dir  = outdir / "features"
    embed_dir     = outdir / "embeddings"
    plots_dir     = outdir / "plots"
    for d in (embed_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    dr_methods = set(args.dr)
    if "all" in dr_methods:
        dr_methods = {"pca", "tsne", "umap"}

    # ── Discover classes ──────────────────────────────────────────────
    _, _, class_to_idx = discover_dataset(config.dataset_path)
    class_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
    num_classes  = len(class_to_idx)
    colormap     = build_class_colormap(class_names, n_classes=num_classes)
    save_colormap_json(colormap, outdir / "colormap.json", class_names=class_names)

    # ── Viz loader (for on-the-fly extraction) ────────────────────────
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    viz_loader, viz_paths = _build_viz_loader(config, args.num_workers)

    # ── Checkpoint alignment ──────────────────────────────────────────
    ckpt_map: Dict[str, Optional[str]] = {m: None for m in args.models}
    if args.checkpoint:
        if len(args.checkpoint) == 1:
            ckpt_map = {m: args.checkpoint[0] for m in args.models}
        elif len(args.checkpoint) == len(args.models):
            ckpt_map = dict(zip(args.models, args.checkpoint))

    # ── Gather features for every (model, layer) pair ─────────────────
    # Structure: {model_name: {layer_tag: (feats, labels, img_paths)}}
    all_features: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]] = {}

    for model_name in args.models:
        all_features[model_name] = {}
        for layer_tag in args.layers:
            # Try loading from disk first
            npz_path = _find_feature_npz(features_dir, model_name, layer_tag)
            if npz_path is not None:
                print(f"  Loading {model_name}/{layer_tag} from {npz_path.name} …")
                feats, lbls, paths = _load_features_npz(npz_path)
                all_features[model_name][layer_tag] = (feats, lbls, paths or viz_paths)
            elif viz_loader is not None:
                # On-the-fly extraction
                print(f"  Extracting {model_name}/{layer_tag} on-the-fly …")
                from src.models.model_factory import create_model
                from src.train.utils_checkpoint import load_checkpoint
                model = create_model(model_name, num_classes=num_classes, pretrained=True)
                model.model_name = model_name
                if ckpt_map.get(model_name):
                    load_checkpoint(ckpt_map[model_name], model, device=str(device), strict=False)
                model.to(device).eval()
                extracted = extract_all_layers(model, model_name, viz_loader, device, [layer_tag])
                if layer_tag in extracted:
                    feats, lbls = extracted[layer_tag]
                    all_features[model_name][layer_tag] = (feats, lbls, viz_paths)
                    # Cache
                    cache_dir = features_dir / "feature_cache"
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(cache_dir / f"{model_name}_{layer_tag}_features.npz",
                                        features=feats, labels=lbls)
                del model
            else:
                print(f"  [warn] No features for {model_name}/{layer_tag} — skipping.")

    # ── PCA ───────────────────────────────────────────────────────────
    pca_embeddings:  Dict[str, Dict[str, np.ndarray]] = {}
    pca_labels_dict: Dict[str, Dict[str, np.ndarray]] = {}
    compactness_data: Dict[str, Dict[str, Dict[int, float]]] = {}

    if "pca" in dr_methods:
        print("\n── PCA projections ──────────────────────────────────────")
        for model_name in args.models:
            pca_embeddings[model_name]  = {}
            pca_labels_dict[model_name] = {}
            compactness_data[model_name]= {}
            for layer_tag in args.layers:
                entry = all_features.get(model_name, {}).get(layer_tag)
                if entry is None:
                    continue
                feats, lbls, img_paths = entry
                print(f"  PCA: {model_name}/{layer_tag} …", end="", flush=True)
                coords, _, _ = compute_pca_2d(feats, scale=True)
                print(f"  var_explained={_pca_var(feats):.1%}")
                pca_embeddings[model_name][layer_tag]  = coords
                pca_labels_dict[model_name][layer_tag] = lbls
                # Save embedding CSV
                save_embedding_csv(
                    embed_dir / f"{model_name}_{layer_tag}_pca.csv",
                    coords, lbls, img_paths,
                )
                # Compactness
                compactness_data[model_name][layer_tag] = intra_class_compactness(
                    coords, lbls, num_classes
                )

        # Multi-panel PCA grid
        paths = plot_embedding_grid(
            embeddings   = pca_embeddings,
            labels_dict  = pca_labels_dict,
            colormap     = colormap,
            model_names  = args.models,
            depth_tags   = args.layers,
            method       = "PCA",
            output_stem  = plots_dir / "pca_layers_comparison",
            class_names  = class_names,
        )
        print(f"  Saved: {[str(p) for p in paths]}")

        # Cluster compactness plot
        if compactness_data:
            cp_paths = plot_cluster_compactness(
                compactness_data = compactness_data,
                model_names      = args.models,
                depth_tags       = args.layers,
                output_path      = plots_dir / "cluster_compactness.png",
                class_names      = class_names,
            )
            print(f"  Compactness plot: {[str(p) for p in cp_paths]}")

    # ── t-SNE ─────────────────────────────────────────────────────────
    if "tsne" in dr_methods:
        print("\n── t-SNE projections ────────────────────────────────────")
        for run_id in range(1, args.tsne_runs + 1):
            tsne_embed:  Dict[str, Dict[str, np.ndarray]] = {}
            tsne_labels: Dict[str, Dict[str, np.ndarray]] = {}

            for model_name in args.models:
                tsne_embed[model_name]  = {}
                tsne_labels[model_name] = {}
                for layer_tag in args.layers:
                    entry = all_features.get(model_name, {}).get(layer_tag)
                    if entry is None:
                        continue
                    feats, lbls, img_paths = entry
                    tsne_seed = seed + run_id - 1
                    print(f"  t-SNE: {model_name}/{layer_tag} run={run_id} seed={tsne_seed} …",
                          end="", flush=True)
                    from src.visualization.dr_utils import compute_tsne
                    emb, kl = compute_tsne(feats, perplexity=args.perplexity,
                                           random_state=tsne_seed)
                    print(f"  KL={kl:.4f}")
                    tsne_embed[model_name][layer_tag]  = emb
                    tsne_labels[model_name][layer_tag] = lbls
                    # Save embedding CSV
                    save_embedding_csv(
                        embed_dir / f"{model_name}_{layer_tag}_tsne_run{run_id}.csv",
                        emb, lbls, img_paths,
                        extra_cols={"kl_divergence": np.full(len(lbls), kl)},
                    )

            # Multi-panel t-SNE grid for this run
            paths = plot_embedding_grid(
                embeddings   = tsne_embed,
                labels_dict  = tsne_labels,
                colormap     = colormap,
                model_names  = args.models,
                depth_tags   = args.layers,
                method       = "t-SNE",
                output_stem  = plots_dir / "tsne_layers_comparison",
                class_names  = class_names,
                run_id       = run_id,
            )
            print(f"  Saved: {[str(p) for p in paths]}")

    # ── UMAP ──────────────────────────────────────────────────────────
    if "umap" in dr_methods:
        print("\n── UMAP projections ─────────────────────────────────────")
        for model_name in args.models:
            for layer_tag in args.layers:
                entry = all_features.get(model_name, {}).get(layer_tag)
                if entry is None:
                    continue
                feats, lbls, img_paths = entry
                print(f"  UMAP: {model_name}/{layer_tag} …", end="", flush=True)
                umap_coords = compute_umap(feats, random_state=seed)
                if umap_coords is None:
                    print("  skipped (umap-learn not installed)")
                    continue
                print("  done")
                save_embedding_csv(
                    embed_dir / f"{model_name}_{layer_tag}_umap.csv",
                    umap_coords, lbls, img_paths,
                )

    print("\n  Visualization generation complete.")
    print(f"  Plots  → {plots_dir}")
    print(f"  Embeds → {embed_dir}")


def _pca_var(features: np.ndarray) -> float:
    """Quick explained variance fraction for the first 2 PCs."""
    from sklearn.decomposition import PCA as _PCA
    from sklearn.preprocessing import StandardScaler as _SS
    X = _SS().fit_transform(features.astype(np.float64))
    pca = _PCA(n_components=min(2, X.shape[1], X.shape[0] - 1))
    pca.fit(X)
    return float(sum(pca.explained_variance_ratio_))


if __name__ == "__main__":
    main()
