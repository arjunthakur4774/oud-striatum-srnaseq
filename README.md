# Single-nucleus RNA-seq analysis — human striatum in opioid use disorder

Exploratory analysis of a CELLxGENE-schema snRNA-seq dataset profiling 98,848
nuclei (30,818 genes) from human striatum, comparing donors with opiate
dependence to neurotypical controls.

## What this does

| Step | Output |
|---|---|
| Dataset composition by disease state and cell type | `results/celltype_counts.csv`, `results/disease_counts.csv` |
| UMAP colored by cell type and by disease | `figures/umap_cell_type.png`, `figures/umap_disease.png` |
| Per-cell-type subsetting (endothelial cells as example) | `figures/umap_endothelial_cell_disease.png` |
| Cluster centroids and pairwise distances in UMAP space | `results/celltype_centroids.csv`, `results/centroid_distances.csv` |
| Within-cluster dispersion | `results/celltype_spread.csv` |
| k-NN cell-type purity score | `results/celltype_nn_purity.csv` |

The purity score asks, for each nucleus, what fraction of its 20 nearest
neighbors in UMAP space carry the same cell-type label. Averaged per cell type
it gives a quick read on which annotated populations are cleanly separated and
which are smeared into their neighbors — useful before committing to any
downstream differential-expression comparison.

## Dataset

The `.h5ad` file is not tracked in this repository (it is several GB). Place it
at `data/raw/oud_striatum.h5ad` or pass `--data /path/to/file.h5ad`.

Key `obs` fields used: `cell_type`, `disease`, `donor_id`, plus the standard
CELLxGENE ontology and QC columns. The UMAP embedding ships with the object in
`obsm["X_umap"]`.

## Running

```bash
pip install -r requirements.txt
python oud_striatum_analysis.py --data data/raw/oud_striatum.h5ad
```

Options: `--cell-type` to change the example subset, `--n-neighbors` to change
the purity neighborhood size, `--figures` / `--results` for output locations.

## Notes

Cell types are heavily imbalanced — oligodendrocytes make up roughly 64% of
nuclei, while mural cells and endothelial cells number in the low hundreds.
Any per-cell-type comparison between disease states needs to account for this;
the endothelial subset, for example, has only 581 nuclei total.
