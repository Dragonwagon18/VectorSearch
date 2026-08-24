# VectorSearch

A vector search engine built from scratch to understand the **algorithms, mathematics, data structures, memory behavior, and systems trade-offs** behind modern vector databases and approximate nearest-neighbor (ANN) search engines.

The goal is not to build another wrapper around FAISS, HNSWLib, or a vector database.

The goal is to understand and implement the machinery ourselves.

---

## Vision

Modern AI systems depend heavily on vector search:

```text
Embedding Model
      │
      ▼
  Vector
      │
      ▼
┌─────────────────┐
│  Vector Search  │
└─────────────────┘
      │
      ▼
Top-k nearest vectors
      │
      ▼
RAG / Recommendation / Retrieval
```

This project progressively builds that system from the simplest possible implementation toward a production-oriented approximate nearest-neighbor engine.

The progression is intentional:

```text
Exact Search
     │
     ▼
Vectorized Exact Search
     │
     ▼
Memory / Cache Optimization
     │
     ▼
Benchmarking
     │
     ▼
HNSW
     │
     ▼
Recall / Latency Trade-offs
     │
     ▼
Persistence
     │
     ▼
Production-oriented Vector Index
```

The project should make the answer to questions like these intuitive:

* Why is brute-force vector search slow?
* Why does NumPy outperform a Python loop?
* What actually determines vector-search latency?
* How does memory layout affect search?
* Why do vector databases use ANN instead of exact search?
* How does HNSW achieve high recall with sublinear search?
* What does `efSearch` actually trade off?
* Why does increasing dimensionality hurt performance?
* How do recall, latency, throughput, and memory interact?
* What happens when the index no longer fits comfortably in cache?
* How do production vector indexes handle persistence and updates?

---

# Current Status

### Completed

* [x] Project structure and packaging
* [x] Vector distance metrics
* [x] Row normalization
* [x] Exact brute-force vector search
* [x] Input validation
* [x] Deterministic tie-breaking
* [x] Unit tests for metrics
* [x] Unit tests for exact search
* [x] Editable package installation
* [x] Pytest `src/` import configuration

### Current test status

```text
13 tests passed
```

---

# Project Structure

```text
vector-search/
│
├── pyproject.toml
├── README.md
│
├── src/
│   └── vector_search/
│       ├── __init__.py
│       ├── metrics.py
│       ├── exact_loop.py
│       ├── exact_numpy.py          # planned
│       └── hnsw.py                 # planned
│
├── tests/
│   ├── test_metrics.py
│   ├── test_exact.py
│   └── test_hnsw.py                # planned
│
├── benchmarks/
│   ├── run_exact.py
│   ├── run_hnsw.py
│   └── compare_faiss.py
│
├── results/
│   └── ...
│
└── plots/
    └── ...
```

---

# 1. Vector Metrics

The first layer is the mathematics of similarity search.

## Squared L2 Distance

For two vectors:

```text
q = query
x = database vector
```

we calculate:

```text
d(q, x) = Σ(qᵢ - xᵢ)²
```

We use **squared** L2 distance because the square root is unnecessary when only ranking vectors by distance.

If:

```text
d₁ < d₂
```

then:

```text
√d₁ < √d₂
```

so the ranking is identical.

Implemented in:

```text
src/vector_search/metrics.py
```

## Normalization

We also implemented row-wise vector normalization.

For a vector:

```text
x = [x₁, x₂, ..., xD]
```

its L2 norm is:

```text
||x||₂ = √(Σxᵢ²)
```

and the normalized vector is:

```text
x̂ = x / ||x||₂
```

This becomes important later when comparing cosine similarity with L2 distance.

---

# 2. Exact Brute-Force Search

The first search implementation is deliberately simple.

```text
Query
  │
  ▼
Compare against vector 1
Compare against vector 2
Compare against vector 3
       ...
Compare against vector N
  │
  ▼
Sort all distances
  │
  ▼
Return top-k
```

Implemented in:

```text
src/vector_search/exact_loop.py
```

The main class is:

```python
ExactLoopIndex
```

Example:

```python
index = ExactLoopIndex(dimension=3)

index.add([1, 0, 0], item_id=10)
index.add([0, 1, 0], item_id=11)
index.add([0, 0, 1], item_id=12)

result = index.search([1, 0, 0], k=2)
```

The result contains:

```text
ids
distances
```

---

# Exact Search Guarantees

The exact index intentionally provides strong correctness guarantees.

## Dimension validation

Vectors must have exactly the configured dimension.

```text
dimension = 3

[1, 2, 3]       ✓
[1, 2]          ✗
[1, 2, 3, 4]    ✗
```

## Finite-value validation

NaN and infinity are rejected.

```text
[1, 2, 3]        ✓
[1, NaN, 3]      ✗
[1, Inf, 3]      ✗
```

This prevents undefined or corrupted distance calculations.

## Duplicate IDs

Each item ID must be unique.

## Valid k

Search requires:

```text
1 <= k <= number_of_vectors
```

## Deterministic results

Results are sorted by:

```text
(distance, item_id)
```

This means equal-distance vectors have deterministic ordering.

For example:

```text
distance    ID

1.0         20
1.0         10
```

becomes:

```text
10
20
```

This is important for reproducible tests and benchmarks.

---

# Complexity

The current implementation performs a comparison against every vector.

For:

```text
N = number of vectors
D = vector dimension
```

distance computation costs approximately:

```text
O(ND)
```

We then sort all candidates:

```text
O(N log N)
```

Therefore:

```text
Search ≈ O(ND + N log N)
```

Memory usage is approximately:

```text
O(ND)
```

This implementation is our **correctness baseline**.

It is not intended to be the final high-performance implementation.

---

# 3. Testing

Every major component is developed test-first.

Current tests cover:

### Metrics

* known squared-L2 result
* normalized vectors have unit norm

### Exact Search

* correct nearest neighbors
* sorted results
* deterministic tie-breaking
* wrong vector dimension
* duplicate IDs
* empty index
* invalid `k`
* `k=1`
* `k=N`
* wrong query dimension
* NaN vectors
* infinite vectors
* NaN queries
* infinite queries

Current status:

```text
13 passed
```

Run the complete test suite with:

```bash
python -m pytest -v
```

---

# 4. Packaging

The project uses a `src/` layout:

```text
src/vector_search/
```

The package is installed in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Pytest is configured to understand the `src` layout through:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

This allows:

```bash
python -m pytest -v
```

without manually setting:

```bash
PYTHONPATH="$PWD/src"
```

---

# Roadmap

The project will evolve through increasingly realistic vector-search implementations.

## Phase 1 — Correctness Foundation

* [x] Distance metrics
* [x] Normalization
* [x] Exact Python-loop search
* [x] Validation
* [x] Deterministic results
* [x] Unit tests

---

## Phase 2 — Vectorized Exact Search

Next:

```text
src/vector_search/exact_numpy.py
```

Replace:

```python
for vector in vectors:
    distance = squared_l2(query, vector)
```

with vectorized NumPy operations.

Conceptually:

```python
distances = np.sum((vectors - query) ** 2, axis=1)
```

We will then verify:

```text
ExactLoop results
        ==
ExactNumPy results
```

for the same inputs.

### Goal

Understand why vectorization is dramatically faster than repeatedly executing Python-level loops.

---

# Phase 3 — Benchmarking

We will build reproducible benchmarks measuring:

* latency
* throughput
* dataset size
* dimensionality
* memory usage
* batch size
* top-k

Example experiment:

```text
N = 1K
N = 10K
N = 100K
N = 1M
```

and dimensions such as:

```text
D = 64
D = 128
D = 384
D = 768
D = 1536
```

We want to produce plots showing:

```text
Dataset size → latency
Dimension    → latency
Batch size   → throughput
```

The benchmark should answer questions empirically rather than theoretically.

---

# Phase 4 — Better Top-k Selection

Currently we sort every distance:

```python
sorted(distances)
```

But if we only need:

```text
top k
```

sorting all `N` elements is unnecessary.

We will investigate:

```text
full sorting
      vs
partial selection
```

For example:

```text
np.argpartition
```

This introduces an important systems principle:

> Do not perform work that the query does not require.

---

# Phase 5 — Memory and Cache Behavior

Once the algorithm is vectorized, the next bottleneck becomes more interesting.

We will investigate:

* contiguous arrays
* row-major layout
* dtype
* float32 vs float64
* memory bandwidth
* CPU cache behavior
* SIMD/vectorization
* allocation overhead
* batching

The goal is to understand why two mathematically identical implementations can have very different latency.

---

# Phase 6 — HNSW

After understanding exact search, we will implement:

```text
src/vector_search/hnsw.py
```

from scratch.

HNSW:

**Hierarchical Navigable Small World**

Instead of comparing the query against every vector:

```text
Query
  │
  ├── vector 1
  ├── vector 2
  ├── vector 3
  ├── ...
  └── vector N
```

we construct a graph:

```text
        Layer 2

          A
         / \
        B   C

        Layer 1

     A──B──C──D──E
      \    \  /
       F────G

        Layer 0

 A──B──C──D──E──F──G──H──I...
```

Search navigates through the graph instead of exhaustively evaluating every vector.

This introduces the fundamental ANN trade-off:

```text
                 Recall
                   ▲
                   │
                   │       ●
                   │    ●
                   │  ●
                   │ ●
                   └──────────────► Latency
```

We will study how parameters such as:

```text
M
efConstruction
efSearch
```

affect:

* recall
* latency
* memory
* build time

---

# Phase 7 — Recall Evaluation

Exact search becomes our ground truth.

For a query:

```text
Exact Search
     │
     ▼
Ground-truth top-k
```

and:

```text
HNSW
 │
 ▼
Approximate top-k
```

we compare the two.

A simple recall@k metric:

```text
recall@k =
|approximate top-k ∩ exact top-k|
----------------------------------
              k
```

This allows us to create real recall/latency curves.

---

# Phase 8 — Persistence

Eventually the index should survive process restarts.

We will investigate:

```text
build index
     │
     ▼
save
     │
     ▼
disk
     │
     ▼
load
     │
     ▼
search
```

This introduces:

* serialization
* binary formats
* metadata
* versioning
* compatibility
* memory mapping

---

# Phase 9 — Production-Oriented Features

Eventually we may explore:

* batch insertion
* batch search
* deletion
* updates
* persistence
* memory mapping
* concurrent search
* concurrent insertion
* sharding
* filtering
* hybrid search
* quantization
* compressed vectors
* observability
* benchmark automation

---

# Comparison Targets

Once our implementations are mature enough, we will compare against established libraries such as:

```text
FAISS
HNSWlib
```

The purpose is **not** to beat production libraries.

The purpose is to understand:

```text
Our implementation
       │
       ▼
What optimization are we missing?
       │
       ▼
What does a mature implementation do differently?
```

External libraries should therefore serve as reference points rather than shortcuts.

---

# Experimental Philosophy

Every optimization should follow:

```text
Implement
   │
   ▼
Test correctness
   │
   ▼
Benchmark
   │
   ▼
Profile
   │
   ▼
Understand bottleneck
   │
   ▼
Optimize
   │
   ▼
Benchmark again
```

We should avoid optimizing based purely on intuition.

For every major optimization, we want to answer:

1. What was the bottleneck?
2. What changed?
3. Why should it be faster?
4. Did the benchmark confirm it?
5. Did correctness remain unchanged?
6. What new bottleneck appeared?

---

# Guiding Principle

The project follows a simple progression:

```text
Correctness
     ↓
Clarity
     ↓
Measurement
     ↓
Optimization
     ↓
Approximation
     ↓
Systems engineering
```

We intentionally start with the simplest possible implementation.

A slow but obviously correct implementation is extremely valuable because it gives us a **ground-truth reference implementation** against which every future optimization can be tested.

---

# Current Milestone

### Milestone 1 — Exact Search Baseline

Status:

```text
COMPLETE
```

Implemented:

```text
✓ Metrics
✓ Normalization
✓ ExactLoopIndex
✓ Validation
✓ Deterministic ordering
✓ 13 tests
✓ Packaging
✓ Git history
```

Latest commit:

```text
Implement exact vector search index
```

---

# Next Milestone

### Milestone 2 — NumPy Vectorized Exact Search

We will implement:

```text
exact_numpy.py
```

and answer the first major performance question:

> How much faster can we make exact search without changing the algorithm at all?

After that:

```text
ExactLoop
    │
    ├──────────────┐
    ▼              ▼
ExactNumPy      Benchmark
    │              │
    └──────┬───────┘
           ▼
      Performance
       Analysis
```

The goal is to progressively move from **"I know how vector search works"** to **"I understand why a production vector search engine is designed this way."**
