"""Standalone SMACOF-embedding VFE-landscape visualizations for the prioritized-memory-transfer
model. Self-contained: `embedding.py` is a pure (model-agnostic) SMACOF embed-and-drape toolkit;
`figures.py` glues it to the live `src` network. Nothing here is imported by `src`, so the package's
no-plotting-on-import contract is preserved. Drive everything from `nb_embedding_landscapes.py`.
"""
