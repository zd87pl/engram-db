#!/usr/bin/env python3
"""Predict-then-retrieve demo with a dummy predictor.

Requires a local Qdrant instance running on http://localhost:6333.
"""

from __future__ import annotations

import random
import time

from engram import EngramClient, WorldState

VECTOR_DIM = 128


def dummy_predictor(context_vector: list[float]) -> list[float]:
    """A trivial 'world model' that slightly perturbs the input."""
    return [v + random.gauss(0, 0.1) for v in context_vector]


def main() -> None:
    client = EngramClient(
        qdrant_url="http://localhost:6333",
        epoch_size_ms=5000,
        spatial_resolution=4,
        vector_size=VECTOR_DIM,
    )

    # Seed some future states
    now_ms = int(time.time() * 1000)
    for i in range(50):
        client.insert(
            WorldState(
                x=random.random(),
                y=random.random(),
                z=random.random(),
                timestamp_ms=now_ms + i * 100,
                vector=[random.gauss(0, 1) for _ in range(VECTOR_DIM)],
                scene_id="predict_demo",
            )
        )

    # Run predict-then-retrieve
    context = [random.gauss(0, 1) for _ in range(VECTOR_DIM)]
    print("Running predict-then-retrieve...")
    results = client.predict_and_retrieve(
        context_vector=context,
        predictor_fn=dummy_predictor,
        future_horizon_ms=5000,
        limit=5,
    )

    print(f"Got {len(results)} results matching predicted future state:")
    for r in results:
        print(f"  id={r.id}  pos=({r.x:.2f}, {r.y:.2f}, {r.z:.2f})  t={r.timestamp_ms}")


if __name__ == "__main__":
    main()
