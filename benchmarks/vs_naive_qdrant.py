#!/usr/bin/env python3
"""Benchmark: Engram middleware (Hilbert bucketing) vs naive Qdrant float-range filters.

Requires a local Qdrant instance running on http://localhost:6333.
"""

from __future__ import annotations

import random
import time

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    PointStruct,
    Range,
    VectorParams,
)

from engram import EngramClient, WorldState

VECTOR_DIM = 128
NUM_VECTORS = 1000
NUM_QUERIES = 50


def setup_naive_collection(qdrant: QdrantClient) -> None:
    """Create a flat collection with float payload filters (no Hilbert)."""
    name = "benchmark_naive"
    try:
        qdrant.delete_collection(name)
    except Exception:
        pass
    qdrant.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )

    now_ms = int(time.time() * 1000)
    points = []
    for i in range(NUM_VECTORS):
        points.append(
            PointStruct(
                id=i,
                vector=[random.gauss(0, 1) for _ in range(VECTOR_DIM)],
                payload={
                    "x": random.random(),
                    "y": random.random(),
                    "z": random.random(),
                    "timestamp_ms": now_ms + i * 10,
                },
            )
        )
    qdrant.upsert(collection_name=name, points=points)
    return now_ms


def benchmark_naive(qdrant: QdrantClient, now_ms: int) -> float:
    """Time naive float-range queries."""
    start = time.perf_counter()
    for _ in range(NUM_QUERIES):
        qv = [random.gauss(0, 1) for _ in range(VECTOR_DIM)]
        qdrant.search(
            collection_name="benchmark_naive",
            query_vector=qv,
            query_filter=Filter(
                must=[
                    FieldCondition(key="x", range=Range(gte=0.2, lte=0.8)),
                    FieldCondition(key="y", range=Range(gte=0.2, lte=0.8)),
                    FieldCondition(key="z", range=Range(gte=0.0, lte=1.0)),
                    FieldCondition(
                        key="timestamp_ms",
                        range=Range(gte=now_ms, lte=now_ms + NUM_VECTORS * 10),
                    ),
                ]
            ),
            limit=10,
        )
    return time.perf_counter() - start


def benchmark_engram(client: EngramClient, now_ms: int) -> float:
    """Time Engram Hilbert-bucketed queries."""
    start = time.perf_counter()
    for _ in range(NUM_QUERIES):
        qv = [random.gauss(0, 1) for _ in range(VECTOR_DIM)]
        client.query(
            vector=qv,
            spatial_bounds={
                "x_min": 0.2, "x_max": 0.8,
                "y_min": 0.2, "y_max": 0.8,
                "z_min": 0.0, "z_max": 1.0,
            },
            time_window_ms=(now_ms, now_ms + NUM_VECTORS * 10),
            limit=10,
        )
    return time.perf_counter() - start


def main() -> None:
    qdrant_url = "http://localhost:6333"
    qdrant = QdrantClient(url=qdrant_url)
    client = EngramClient(
        qdrant_url=qdrant_url,
        epoch_size_ms=5000,
        spatial_resolution=4,
        vector_size=VECTOR_DIM,
    )

    # --- Setup ---
    print(f"Setting up {NUM_VECTORS} vectors...")
    now_ms = setup_naive_collection(qdrant)

    # Also insert into Engram
    now_ms_engram = int(time.time() * 1000)
    states = [
        WorldState(
            x=random.random(),
            y=random.random(),
            z=random.random(),
            timestamp_ms=now_ms_engram + i * 10,
            vector=[random.gauss(0, 1) for _ in range(VECTOR_DIM)],
        )
        for i in range(NUM_VECTORS)
    ]
    client.insert_batch(states)

    # --- Benchmark ---
    print(f"Running {NUM_QUERIES} queries each...\n")

    naive_time = benchmark_naive(qdrant, now_ms)
    engram_time = benchmark_engram(client, now_ms_engram)

    print(f"Naive Qdrant (3 float-range filters): {naive_time:.3f}s  ({naive_time/NUM_QUERIES*1000:.1f} ms/query)")
    print(f"Engram (Hilbert bucketing):            {engram_time:.3f}s  ({engram_time/NUM_QUERIES*1000:.1f} ms/query)")
    print(f"\nRatio: {naive_time/engram_time:.2f}x")


if __name__ == "__main__":
    main()
