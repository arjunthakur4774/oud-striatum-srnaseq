#!/usr/bin/env python
"""
Exploratory analysis of a single-nucleus RNA-seq dataset from human striatum
in opioid use disorder (OUD) vs. neurotypical controls.

Dataset: oud_striatum.h5ad  (98,848 nuclei x 30,818 genes, CELLxGENE schema)

The script covers:
  1. Loading the AnnData object and summarizing its structure
  2. Composition of the dataset by disease state and cell type
  3. UMAP embeddings colored by cell type and disease
  4. Per-cell-type subsetting (endothelial cells shown as the example)
  5. Geometry of the embedding: cluster centroids, pairwise centroid
     distances, and within-cluster spread
  6. A k-nearest-neighbor purity score quantifying how well each annotated
     cell type separates in UMAP space

Usage:
    python oud_striatum_analysis.py --data ../data/raw/oud_striatum.h5ad

Outputs figures to ./figures/ and tables to ./results/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

matplotlib.use("Agg")  # render to file; drop this line to plot interactively


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PLOT_STYLE = {
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "legend.fontsize": 18,
    "legend.title_fontsize": 15,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
}

EXAMPLE_CELL_TYPE = "endothelial cell"
N_NEIGHBORS = 20


# --------------------------------------------------------------------------- #
# Loading and description
# --------------------------------------------------------------------------- #

def load_data(data_path: Path) -> ad.AnnData:
    """Read the h5ad file and print a high-level description."""
    adata = ad.read_h5ad(data_path)

    print(f"Loaded {data_path}")
    print(adata)
    print(f"\nShape (cells x genes): {adata.shape}")
    print(f"Gene metadata columns:  {list(adata.var.columns)}")
    print(f"Embeddings available:   {list(adata.obsm.keys())}")
    if "X_umap" in adata.obsm:
        print(f"UMAP shape:             {adata.obsm['X_umap'].shape}")

    return adata


def describe_composition(adata: ad.AnnData) -> dict[str, pd.Series]:
    """Count cells by disease state and by annotated cell type."""
    disease_counts = adata.obs["disease"].value_counts()
    celltype_counts = adata.obs["cell_type"].value_counts()

    print("\nCells by disease state:")
    print(disease_counts.to_string())
    print("\nCells by cell type:")
    print(celltype_counts.to_string())

    return {"disease": disease_counts, "cell_type": celltype_counts}


# --------------------------------------------------------------------------- #
# UMAP visualization
# --------------------------------------------------------------------------- #

def plot_umaps(adata: ad.AnnData, fig_dir: Path, keys=("cell_type", "disease")) -> None:
    """Save one UMAP per metadata key."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(PLOT_STYLE):
        for key in keys:
            sc.pl.umap(adata, color=key, show=False)
            out = fig_dir / f"umap_{key}.png"
            plt.savefig(out)
            plt.close()
            print(f"Wrote {out}")


def subset_cell_type(adata: ad.AnnData, cell_type: str) -> ad.AnnData:
    """Return a copy of the object containing only one annotated cell type."""
    subset = adata[adata.obs["cell_type"] == cell_type].copy()
    print(f"\n{cell_type}: {subset.n_obs} cells")
    print(subset.obs["disease"].value_counts().to_string())
    return subset


# --------------------------------------------------------------------------- #
# Embedding geometry
# --------------------------------------------------------------------------- #

def umap_frame(adata: ad.AnnData) -> pd.DataFrame:
    """Build a tidy frame of UMAP coordinates plus the cell-type label."""
    df = adata.obs[["cell_type"]].copy()
    df[["UMAP1", "UMAP2"]] = adata.obsm["X_umap"]
    return df


def centroid_distances(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the centroid of each cell type in UMAP space and the matrix of
    pairwise Euclidean distances between those centroids.

    Note: observed=True is passed explicitly because cell_type is categorical;
    without it pandas emits a FutureWarning and includes unused categories.
    """
    grouped = df.groupby("cell_type", observed=True)[["UMAP1", "UMAP2"]]
    centroids = grouped.mean()

    dist = squareform(pdist(centroids.values))
    dist_df = pd.DataFrame(dist, index=centroids.index, columns=centroids.index)

    return centroids, dist_df


def cluster_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell-type variance along each UMAP axis, as a dispersion measure."""
    return df.groupby("cell_type", observed=True)[["UMAP1", "UMAP2"]].var()


def neighbor_purity(adata: ad.AnnData, n_neighbors: int = N_NEIGHBORS) -> pd.Series:
    """
    For each cell, find its k nearest neighbors in UMAP space and record the
    fraction sharing its cell-type label. Averaged per cell type, this gives a
    simple measure of how cleanly each population separates in the embedding:
    1.0 means every neighbor matches, lower values mean the cluster is mixed.

    The per-cell score is stored in adata.obs["celltype_nn_purity"].
    """
    X = adata.obsm["X_umap"]
    labels = adata.obs["cell_type"].values

    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    neighbors = nn.kneighbors(return_distance=False)

    purity = np.array([
        (labels[nbrs] == labels[i]).mean()
        for i, nbrs in enumerate(neighbors)
    ])
    adata.obs["celltype_nn_purity"] = purity

    by_type = (
        adata.obs
        .groupby("cell_type", observed=True)["celltype_nn_purity"]
        .mean()
        .sort_values(ascending=False)
    )
    return by_type


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("..") / "data" / "raw" / "oud_striatum.h5ad",
        help="Path to the oud_striatum.h5ad file",
    )
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--cell-type", default=EXAMPLE_CELL_TYPE)
    parser.add_argument("--n-neighbors", type=int, default=N_NEIGHBORS)
    args = parser.parse_args()

    args.results.mkdir(parents=True, exist_ok=True)

    # 1-2. Load and summarize
    adata = load_data(args.data)
    composition = describe_composition(adata)
    composition["cell_type"].to_csv(args.results / "celltype_counts.csv")
    composition["disease"].to_csv(args.results / "disease_counts.csv")

    # 3. UMAPs
    plot_umaps(adata, args.figures)

    # 4. Single cell type
    subset = subset_cell_type(adata, args.cell_type)
    with plt.rc_context(PLOT_STYLE):
        sc.pl.umap(subset, color="disease", show=False)
        slug = args.cell_type.replace(" ", "_")
        plt.savefig(args.figures / f"umap_{slug}_disease.png")
        plt.close()

    # 5. Embedding geometry
    df = umap_frame(adata)
    centroids, dist_df = centroid_distances(df)
    spread = cluster_spread(df)

    print("\nCell-type centroids in UMAP space:")
    print(centroids.to_string())
    print("\nPairwise centroid distances:")
    print(dist_df.round(2).to_string())
    print("\nWithin-cluster variance:")
    print(spread.to_string())

    centroids.to_csv(args.results / "celltype_centroids.csv")
    dist_df.to_csv(args.results / "centroid_distances.csv")
    spread.to_csv(args.results / "celltype_spread.csv")

    # 6. Neighborhood purity
    purity = neighbor_purity(adata, args.n_neighbors)
    print(f"\nMean {args.n_neighbors}-NN cell-type purity:")
    print(purity.round(3).to_string())
    purity.to_csv(args.results / "celltype_nn_purity.csv")

    print(f"\nDone. Figures in {args.figures}/, tables in {args.results}/")


if __name__ == "__main__":
    main()
