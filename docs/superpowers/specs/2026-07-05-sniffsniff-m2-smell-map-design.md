# sniffsniff — Milestone 2: The Smell Map — Design

**Date:** 2026-07-05
**Status:** Approved (design), building
**Milestone:** 2 of 3 — "Smell Map" (dimensionality reduction + identity + novelty + geometry serialization)

## Purpose

Turn the labeled **48-D** feature vectors that Milestone 1 produces into:
1. a low-dimensional **"smell map"** (PCA) you can look at and cluster,
2. an **identity** classifier ("closest known smell" + probability),
3. a **novelty** score ("this doesn't belong to any known smell"), and
4. a **serialized geometry** (structured JSON) — the exact hand-off the Milestone 3
   LLM reasoner consumes to talk about position, distance, and axes in words.

Everything is testable **today** with no new hardware: the M1 simulator generates
labeled datasets, and the same code runs on real recordings the moment the board streams.

## Scope

### In scope
- `dataset`: load the M1 `.npz` + `manifest.csv` dataset into `(X[n,48], y[n], names)`;
  plus a simulator-backed synthetic dataset generator for tests/demo.
- `model` (`SmellModel`): `StandardScaler → PCA → classifier` with per-class Mahalanobis
  novelty; fit / transform / predict / predict_proba / novelty / save / load; plus an
  honest cross-validated accuracy helper (split by sniff, `StratifiedKFold`/`LeaveOneOut`).
- `geometry`: serialize the learned geometry (cluster centroids/radii, PCA axis
  interpretation from loadings, per-sample distances, novelty) to the M3 JSON schema.
- `smellmap` (viz, optional): render the 2D PCA map to a PNG via matplotlib (lazy import,
  degrades gracefully if matplotlib is absent).
- `cli`: `fit`, `map`, `identify` subcommands.
- Tests (pytest) via the simulator, including cross-validated separability and novelty.

### Out of scope (deferred)
- **Milestone 3:** the LLM reasoning loop, active sensing ("what to sniff next"),
  predict-then-check, cluster grounding to words. M2 only *produces* the geometry JSON.
- **SOM/UMAP:** PCA is the primary, interpretable map (its loadings feed the LLM). SOM/UMAP
  are a possible later visual add, not required for M2.
- Live streaming dashboards; servo/fan automation.

## Architecture

M2 consumes M1's dataset and adds a fitted model + geometry serializer:

```
 data/<label>/*.npz (48-D vectors)          simulator (labeled synthetic vectors)
             \                                   /
              ▼                                 ▼
        dataset.load_dataset / dataset.simulate_dataset  ──►  Dataset(X[n,48], y[n], names, ids)
                              │
                              ▼
        SmellModel.fit:  StandardScaler → PCA(k) → classifier      + per-class Mahalanobis stats
                              │
              ┌───────────────┼───────────────────────────┐
              ▼               ▼                           ▼
        transform→scores   predict / proba / novelty    geometry.serialize_geometry → JSON  ──► (M3 LLM)
              ▼
        smellmap.render_map → PNG (optional)
```

## Module interfaces (the contract the build follows)

### `dataset.py`
```python
@dataclass(frozen=True)
class Dataset:
    X: np.ndarray          # (n, 48) float64 feature matrix
    y: np.ndarray          # (n,) str labels
    feature_names: list[str]   # length 48
    ids: list[str]         # length n, sniff ids (for split-by-sniff / leakage control)
    @property
    def classes(self) -> list[str]      # sorted unique labels

def load_dataset(data_dir) -> Dataset
    # read every <label>/*.npz under data_dir (features + meta), stack into X/y/ids.
    # skip files with non-finite features (open-channel sniffs) with a warning.

def simulate_dataset(config, odors: list[str], reps: int, *, seed: int,
                     noise_counts: float = 1.0) -> Dataset
    # per odor, per rep: Simulator(config, seed+rep_offset).sniff_frames(odor)
    #   -> SniffRecorder(config).process(frames, odor).features  (no disk write)
    # deterministic for a given seed. Distinct rep seeds so reps vary.
```

### `model.py`
```python
class SmellModel:
    def __init__(self, n_components: int = 2, classifier: str = "knn",
                 novelty_alpha: float = 0.975): ...
        # classifier in {"knn","svm","rf","lda"}; novelty threshold = chi2.ppf(alpha, df=n_components)
    def fit(self, X, y) -> "SmellModel"
        # StandardScaler().fit → PCA(n_components).fit → classifier.fit on PCA scores.
        # store classes_, explained_variance_ratio_, loadings_ (n_components,48),
        # per-class centroid (n_components,) and pinv-covariance in PCA space, novelty_threshold_.
    def transform(self, X) -> np.ndarray            # (n, n_components) PCA scores
    def predict(self, X) -> np.ndarray              # (n,) predicted labels
    def predict_proba(self, X) -> tuple[list[str], np.ndarray]   # (classes, (n,C) proba)
    def mahalanobis(self, X) -> np.ndarray          # (n, C) per-class D in PCA space
    def novelty(self, X) -> np.ndarray              # (n,) min per-class Mahalanobis distance
    def is_novel(self, X) -> np.ndarray             # (n,) bool: novelty > novelty_threshold_
    def save(self, path) -> None                    # joblib
    @classmethod
    def load(cls, path) -> "SmellModel"

def cross_val_accuracy(X, y, *, n_components=2, classifier="knn", groups=None) -> tuple[float,float]
    # StratifiedKFold (or LeaveOneOut for tiny n); if `groups` (sniff ids) given, GroupKFold
    # to prevent same-sniff leakage. Returns (mean_accuracy, std). Pipeline scaled+PCA'd inside CV.
```
Novelty math: in PCA space, per class `c`, `D_c² = (x−μ_c)ᵀ Σ_c⁻¹ (x−μ_c)` (`Σ_c⁻¹`
via `np.linalg.pinv`; pooled covariance fallback when a class has too few samples to
estimate its own). `novelty(x) = min_c D_c`; novel if `> sqrt(chi2.ppf(alpha, df=k))`.

### `geometry.py`
```python
def axis_interpretation(model: SmellModel, *, top_k: int = 3) -> dict[str, str]
    # per PC, name it from the dominant feature loadings, e.g. {"PC1": "MQ3__peak(+), MQ2__peak(+)"}

def serialize_geometry(model, *, dataset=None, new_sample=None) -> dict
    # returns the M3 JSON (schema below). new_sample is one 48-D raw feature vector.
```
M3 JSON schema:
```json
{
  "pca": {"n_components": 2, "explained_variance_ratio": [0.71, 0.18]},
  "axis_interpretation": {"PC1": "MQ3__peak(+), MQ2__peak(+)", "PC2": "MQ135__peak(+)"},
  "known_clusters": {
    "coffee": {"centroid": [1.2, -0.5], "radius": 0.6, "n": 8, "distance": 0.63},
    "vinegar": {"centroid": [-2.1, 1.3], "radius": 0.5, "n": 8, "distance": 4.1}
  },
  "new_sample": {"pca_coords": [1.8, -0.4],
                 "predicted": {"label": "coffee", "proba": 0.72},
                 "top_features_z": {"MQ3__peak": 2.9, "MQ135__peak": 0.3}},
  "novelty": {"min_mahalanobis": 1.9, "threshold": 3.0, "is_novel": false, "nearest": "coffee"}
}
```
`radius` = std of member-to-centroid distances in PCA space. `distance` (only when
`new_sample` given) = Euclidean from the new sample's PCA coords to each centroid.
Everything JSON-serializable (plain floats/lists), so it drops straight into an M3 prompt.

### `smellmap.py` (optional viz)
```python
def render_map(model, dataset=None, *, new_sample=None, path=None) -> str | None
    # matplotlib scatter of PCA scores colored by class (+ centroids, + new point).
    # lazy `import matplotlib`; if unavailable, raise a clear ImportError telling the
    # user to `pip install sniffsniff[viz]`. Saves PNG to `path`, returns the path.
```

### `cli.py` (new subcommands)
- `sniffsniff fit --data DIR [--sim] [--odors a,b,c] [--reps N] [--classifier knn] [--out model.joblib]`
  — build a dataset (from disk, or `--sim` to synthesize), fit a `SmellModel`, print
  cross-validated accuracy, save the model.
- `sniffsniff map --model M (--data DIR | --sim ...) [--out map.png]` — render the smell map.
- `sniffsniff identify --model M (--sim --odor X | --port P) [--json]` — capture/simulate one
  sniff, print predicted label + probability + novelty, and (`--json`) the geometry blob.

## Tiny-data discipline (hackathon reality)

- ~5 classes × 8–10 reps ⇒ ~50 samples in 48-D: always reduce (PCA to `k≈2–5`) **before**
  classifying; never feed raw 48-D to a classifier at this n.
- Report **cross-validated** accuracy (`StratifiedKFold`/`LeaveOneOut`), mean ± std — never a
  single split. Prefer high-bias models (kNN, LDA) at tiny n.
- **Leakage guard:** split by *sniff id*, not by timepoint (the vectors are one-per-sniff here,
  so this is naturally satisfied, but `cross_val_accuracy` accepts `groups` for safety).
- Simulated near-perfect numbers are optimistic; the CLI labels sim vs real accuracy.

## Persistence

`SmellModel.save/load` via **joblib** (scaler + PCA + classifier + per-class stats +
metadata in one file). The geometry JSON is the separate, human/LLM-readable artifact.

## Testing (via the simulator; pytest)

- `test_dataset`: `simulate_dataset` deterministic per seed; `load_dataset` round-trips a
  recorder-written dataset; non-finite (open-channel) sniffs are skipped with a warning.
- `test_model`: on a simulated 4-class set, PCA PC1+PC2 explain a healthy fraction; the
  classifier's cross-validated accuracy is high (> 0.85 on clean sim data); `save/load`
  round-trips and predicts identically; Mahalanobis flags an **unseen** odor as novel and a
  held-out in-distribution sniff as not novel.
- `test_geometry`: `serialize_geometry` returns the schema above; is JSON-serializable; the
  nearest cluster to a sample of class `c` is `c`; axis_interpretation names real feature columns.
- `test_smellmap`: `render_map` writes a non-empty PNG (skip if matplotlib missing).
- `test_cli_m2`: `fit --sim` then `identify --sim` and `map --sim` run end-to-end (rc 0,
  model + PNG created, `--json` emits valid geometry JSON).

## Dependencies

Add **scikit-learn** (pulls scipy + joblib) to core deps; **matplotlib** as an optional
`[viz]` extra (lazy-imported by `smellmap`). Python 3.11+, numpy as before.

## Handoff to Milestone 3

`geometry.serialize_geometry(...)` is the contract: M3's LLM reasoner takes that JSON,
interprets the new sample's position relative to clusters using the axis interpretations,
decides what to sniff next, predicts-then-checks, and grounds unlabeled clusters to words.
