#!/usr/bin/env python3
"""Basic Engram usage — insert + query 100 random world states.

Requires a local Qdrant instance running on http://localhost:6333.
Start one with:  docker run -p 6333:6333 qdrant/qdrant
"""

from __future__ import annotations

import random
import time

from engram import EngramClient, WorldState

VECTOR_DIM = 128
NUM_STATES = 100


def main() -> None:
    client = EngramClient(
        qdrant_url="http://localhost:6333",
        epoch_size_ms=5000,
        spatial_resolution=4,
        vector_size=VECTOR_DIM,
    )

    # --- Insert 100 random states ---
    now_ms = int(time.time() * 1000)
    states: list[WorldState] = []
    for i in range(NUM_STATES):
        state = WorldState(
            x=random.random(),
            y=random.random(),
            z=random.random(),
            timestamp_ms=now_ms + i * 50,
            vector=[random.gauss(0, 1) for _ in range(VECTOR_DIM)],
            scene_id="demo_scene",
            scale_level="patch",
            confidence=random.random(),
        )
        states.append(state)

    print(f"Inserting {NUM_STATES} states (batched)...")
    ids = client.insert_batch(states)
    print(f"Inserted {len(ids)} states.")

    # --- Query: nearest neighbours in a spatial region ---
    query_vec = [random.gauss(0, 1) for _ in range(VECTOR_DIM)]
    results = client.query(
        vector=query_vec,
        spatial_bounds={
            "x_min": 0.2, "x_max": 0.8,
            "y_min": 0.2, "y_max": 0.8,
            "z_min": 0.0, "z_max": 1.0,
        },
        time_window_ms=(now_ms, now_ms + NUM_STATES * 50),
        limit=5,
    )

    print(f"\nTop {len(results)} results:")
    for r in results:
        print(f"  id={r.id}  pos=({r.x:.2f}, {r.y:.2f}, {r.z:.2f})  t={r.timestamp_ms}")


if __name__ == "__main__":
    main()
