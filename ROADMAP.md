# Engram Roadmap

## v0.1 — Foundation (current)

- [x] WorldState data model
- [x] Hilbert curve spatial encoding (4D)
- [x] Temporal sharding with epoch-based collections
- [x] EngramClient: insert, insert_batch, query
- [x] Predict-then-retrieve primitive
- [x] Temporal decay scoring
- [x] Basic test suite
- [ ] CI pipeline (GitHub Actions)

## v0.2 — Robustness

- [ ] Async client (`EngramAsyncClient`)
- [ ] Connection pooling and retry logic
- [ ] Causal chain maintenance (auto-link prev/next on insert)
- [ ] Shard lifecycle: warm → cold migration policy
- [ ] Configurable distance metrics (cosine, dot, euclidean)
- [ ] Comprehensive integration test suite (with Qdrant testcontainer)

## v0.3 — Performance

- [ ] Adaptive Hilbert resolution (auto-tune based on data density)
- [ ] Parallel fan-out across shards (asyncio / threading)
- [ ] Result caching for repeated spatial queries
- [ ] Benchmarks against Milvus and Weaviate spatial filters

## v0.4 — Multi-Scale

- [ ] Full funnel search pipeline (sequence → frame → patch)
- [ ] Cross-scale causal linking
- [ ] Scale-aware temporal decay

## v1.0 — Production Ready

- [ ] gRPC transport option
- [ ] Authentication and multi-tenancy
- [ ] Observability (OpenTelemetry traces, Prometheus metrics)
- [ ] Helm chart for Kubernetes deployment
- [ ] Published to PyPI
