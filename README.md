# Engram

A 4D spatiotemporal vector database middleware for AI world models.

Existing vector databases treat all embedding dimensions equally and have no native
understanding of space + time structure. World models (V-JEPA 2, DreamerV3, GAIA-1)
produce embeddings where every vector has a 4D address: **(x, y, z, t)**. Engram is a
middleware layer on top of [Qdrant](https://qdrant.tech) that makes this structure
first-class.

## Key Features

- **Hilbert Curve Spatial Bucketing** — encode (x, y, z, t) as a single int64 Hilbert
  index for efficient spatial pre-filtering via `MatchAny`, replacing 3+ float-range
  payload filters.
- **Temporal Sharding** — automatic routing of vectors to time-partitioned Qdrant
  collections with configurable epoch size.
- **Predict-then-Retrieve** — atomic API call that runs a user-supplied predictor
  function to generate a hypothetical future state, then searches for nearest neighbours
  within a configurable future time window.
- **Temporal Decay** — recency-weighted scoring:
  `score = similarity * exp(-λ * age_ms)`.

## Quick Start

```bash
pip install engram-db
# or from source
pip install -e .
```

Requires a running Qdrant instance:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

```python
from engram import EngramClient, WorldState

client = EngramClient("http://localhost:6333", vector_size=128)

# Insert
state = WorldState(x=0.5, y=0.3, z=0.8, timestamp_ms=1700000000000,
                   vector=[0.1] * 128, scene_id="lab")
state_id = client.insert(state)

# Query with spatial + temporal filtering
results = client.query(
    vector=[0.1] * 128,
    spatial_bounds={"x_min": 0.2, "x_max": 0.8, "y_min": 0.0, "y_max": 1.0,
                    "z_min": 0.0, "z_max": 1.0},
    time_window_ms=(1700000000000, 1700000005000),
    limit=5,
)

# Predict-then-retrieve
results = client.predict_and_retrieve(
    context_vector=[0.1] * 128,
    predictor_fn=lambda v: [x + 0.01 for x in v],  # your world model here
    future_horizon_ms=2000,
)
```

## Benchmarks

```bash
python benchmarks/vs_naive_qdrant.py
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

Apache 2.0
