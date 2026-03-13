# Engram Architecture

## Four-Layer Design

```
┌─────────────────────────────────────────┐
│           Application Layer             │
│  EngramClient — public API surface      │
│  insert / query / predict_and_retrieve  │
├─────────────────────────────────────────┤
│           Retrieval Layer               │
│  predict.py — predict-then-retrieve     │
│  funnel.py  — multi-scale coarse→fine   │
├─────────────────────────────────────────┤
│        Indexing & Routing Layer         │
│  spatial/  — Hilbert encoding + buckets │
│  temporal/ — epoch sharding + decay     │
├─────────────────────────────────────────┤
│           Storage Layer                 │
│  Qdrant (collections per epoch)         │
└─────────────────────────────────────────┘
```

### Layer 1: Storage (Qdrant)

Each temporal epoch maps to a separate Qdrant collection (`engram_{epoch_id}`).
Collections are created lazily on first insert. Payload indices are created
on `hilbert_id` (integer) and `timestamp_ms` (integer) for efficient filtering.

### Layer 2: Indexing & Routing

**Spatial** — The 4D point (x, y, z, t_normalised) is mapped to a single int64
via a 4-dimensional Hilbert curve. The resolution order (default 4 → 16 bins
per axis) is deliberately low so that bounding-box expansion enumerates a
manageable number of bucket IDs for `MatchAny` filtering.

**Temporal** — `timestamp_ms // epoch_size_ms` determines the epoch. Queries
compute which epochs overlap the requested time window and fan out searches.

### Layer 3: Retrieval

**predict-then-retrieve** — Calls the user's predictor function to generate a
hypothetical future embedding, then runs a standard query filtered to
`[now, now + future_horizon_ms]`.

**funnel search** — Cascades through scale levels (sequence → frame → patch)
to progressively refine results when multi-scale embeddings are stored.

### Layer 4: Application

`EngramClient` is the sole public entry point. It orchestrates the layers
below, handles collection lifecycle, and translates between `WorldState`
dataclasses and Qdrant point structs.

## Data Flow

```
insert(WorldState)
  → compute epoch_id → ensure collection exists
  → normalise t within epoch → compute hilbert_id
  → upsert PointStruct to qdrant

query(vector, bounds, time_window)
  → determine epoch range → expand bounding box to hilbert IDs
  → fan-out search across collections with MatchAny + Range filters
  → apply temporal decay → re-rank → return WorldStates
```
