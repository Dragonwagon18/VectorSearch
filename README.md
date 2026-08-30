# VectorSearch

A vector search engine built from scratch to understand the **algorithms, mathematics, data structures, memory behavior, and systems trade-offs** behind modern vector databases and approximate nearest-neighbor (ANN) search engines.

The goal is not to build another wrapper around FAISS, HNSWLib, or a vector database.

The goal is to understand and implement the machinery ourselves.

---

# Vision

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
Top-k Optimization
     │
     ▼
Memory / Cache Optimization
     │
     ▼
Dimensionality / Batch Experiments
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

## Completed

* [x] Project structure and packaging
* [x] Vector distance metrics
* [x] Row normalization
* [x] Exact brute-force Python-loop search
* [x] Exact vectorized NumPy search
* [x] Input validation
* [x] Deterministic tie-breaking
* [x] Unit tests for metrics
* [x] Unit tests for exact search
* [x] Unit tests comparing loop and NumPy implementations
* [x] Editable package installation
* [x] Pytest `src/` import configuration
* [x] Reproducible synthetic datasets
* [x] Exact-search benchmark
* [x] Latency measurements
* [x] Benchmark result serialization
* [x] Python-loop vs NumPy benchmark
* [x] Exact-search latency plot

## Current test status

```text
27 tests passed
```

Run the complete test suite with:

```bash
python -m pytest -v
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
│       ├── exact_numpy.py
│       └── hnsw.py                  # planned
│
├── tests/
│   ├── test_metrics.py
│   ├── test_exact.py
│   ├── test_exact_numpy.py
│   └── test_hnsw.py                 # planned
│
├── benchmarks/
│   ├── run_exact.py
│   ├── compare_exact.py
│   ├── run_hnsw.py                  # planned
│   ├── compare_faiss.py             # planned
│   │
│   └── results/
│       └── exact.json
│
└── plots/
    └── Exact search latency- Python loop vs NumPy.png
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
Rank candidates
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
index = ExactLoopIndex(3)

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

This implementation serves as the **correctness reference implementation**.

---

# Exact Search Guarantees

The exact index intentionally provides strong correctness guarantees.

## Dimension Validation

Vectors must have exactly the configured dimension.

```text
dimension = 3

[1, 2, 3]       ✓
[1, 2]          ✗
[1, 2, 3, 4]    ✗
```

## Finite-Value Validation

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

## Deterministic Results

Results are ordered by:

```text
(distance, item_id)
```

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

# 3. Vectorized Exact Search

The next implementation performs the same mathematical operation using vectorized NumPy operations.

Implemented in:

```text
src/vector_search/exact_numpy.py
```

The core computation is conceptually:

```python
distances = np.sum((vectors - query) ** 2, axis=1)
```

Instead of executing a Python loop over every vector, NumPy performs the computation over the complete matrix using optimized native numerical operations.

The important point is that **the algorithm has not changed**.

Both implementations perform exact search:

```text
ExactLoopIndex
      │
      │ same mathematical algorithm
      ▼
ExactNumPyIndex
```

The NumPy implementation therefore gives us a useful performance baseline before introducing approximate search.

---

# 4. Correctness Testing

Every major component is developed test-first.

## Metrics

* known squared-L2 result
* normalized vectors have unit norm

## Exact Search

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

## NumPy Exact Search

Additional tests verify:

* vector storage
* ID handling
* wrong ID count
* duplicate IDs
* NumPy and loop implementations produce identical nearest-neighbor results
* multiple randomized trials produce identical nearest-neighbor results
* deterministic tie-breaking at the `k` boundary

Current status:

```text
27 passed
```

---

# 5. Benchmarking

We now have a reproducible benchmark comparing two **exact nearest-neighbor implementations**:

```text
Exact Python Loop
        │
        │ same algorithm
        ▼
Exact NumPy Vectorization
```

The purpose of this experiment is to isolate the performance impact of **vectorization** without changing the underlying search algorithm.

The benchmark measures:

* query latency
* p50 latency
* p95 latency
* p99 latency
* queries per second
* index build time

## Benchmark Configuration

```text
dimension = 128
k = 10
queries = 200
dtype = float32
```

Dataset sizes:

```text
N = 1,000
N = 10,000
N = 100,000
```

## Current Results

|    N | Implementation |          p50 |          p95 |          p99 |       QPS |        Build |
| ---: | -------------- | -----------: | -----------: | -----------: | --------: | -----------: |
|   1K | Python Loop    |      2.28 ms |      2.34 ms |      2.38 ms |       437 |      7.77 ms |
|   1K | NumPy          |  **0.13 ms** |  **0.14 ms** |  **0.15 ms** | **7,397** |  **0.07 ms** |
|  10K | Python Loop    |     24.55 ms |     26.48 ms |     77.08 ms |      37.9 |    486.18 ms |
|  10K | NumPy          |  **1.88 ms** |  **1.93 ms** |  **2.03 ms** |   **528** |  **0.34 ms** |
| 100K | Python Loop    |    259.69 ms |    273.66 ms |    325.71 ms |      3.81 | 45,469.77 ms |
| 100K | NumPy          | **23.24 ms** | **23.86 ms** | **24.90 ms** |  **42.9** | **29.66 ms** |

## Vectorization Speedup

The NumPy implementation provides a substantial improvement while returning the **exact same nearest neighbors**.

|    N | QPS Speedup | p50 Speedup |
| ---: | ----------: | ----------: |
|   1K |   **16.9×** |   **17.2×** |
|  10K |   **13.9×** |   **13.0×** |
| 100K |   **11.3×** |   **11.2×** |

The important observation is:

> **The algorithm did not change. The implementation changed.**

Both implementations still examine every vector and therefore have the same fundamental complexity.

```text
Python Loop
O(ND)

     │
     │ vectorization
     ▼

NumPy
O(ND)
```

NumPy moves the computational work from the Python interpreter into optimized native numerical operations, dramatically reducing per-vector interpreter overhead.

The benchmark therefore gives us our first concrete systems lesson:

> **Algorithmic complexity alone does not determine performance. Implementation and hardware utilization matter.**

Results are stored in:

```text
benchmarks/results/exact.json
```

---

# Exact Search Latency

The benchmark demonstrates the performance difference between a straightforward Python implementation and a vectorized NumPy implementation.

![Exact Search Latency — Python Loop vs NumPy](plots/Exact%20search%20latency-%20Python%20loop%20vs%20NumPy.png)

The Python-loop implementation becomes increasingly expensive as the dataset grows.

At `N = 100K`:

```text
Python Loop
p50 ≈ 259.7 ms
QPS ≈ 3.8

        vs

NumPy
p50 ≈ 23.2 ms
QPS ≈ 42.9
```

The NumPy implementation is approximately **11× faster** at this dataset size.

Conceptually:

```text
Same algorithm
      │
      ├───────────────┐
      │               │
      ▼               ▼
Python Loop         NumPy
      │               │
Python interpreter   Native vectorized operations
      │               │
      ▼               ▼
Higher overhead      Lower overhead
      │               │
      └───────┬───────┘
              ▼
       Exact top-k search
```

This is our first measured demonstration that:

> **A performance optimization does not necessarily require changing the algorithm.**

However, NumPy does **not** change the fundamental scaling behavior.

Every query still examines all `N` vectors:

```text
N increases
    │
    ▼
More distance computations
    │
    ▼
More memory accessed
    │
    ▼
Higher latency
```

This motivates the next experiments: reducing unnecessary computation, understanding memory behavior, and eventually replacing exhaustive search with ANN algorithms such as HNSW.

---

# 6. Top-k Selection

The current exact implementation uses partial selection rather than fully sorting all `N` distances.

If:

```text
N = 1,000,000
k = 10
```

we only need the 10 closest vectors.

Fully sorting one million distances is unnecessary.

The implementation therefore uses:

```python
np.argpartition(...)
```

to identify the `k` best candidates without completely sorting the entire distance array.

The selected candidates are then sorted to guarantee deterministic final ordering by:

```text
(distance, item_id)
```

Conceptually:

```text
N distances
     │
     ▼
np.argpartition
     │
     ▼
k candidates
     │
     ▼
small final sort
     │
     ▼
ordered top-k
```

This introduces an important systems principle:

> **Do not perform work that the query does not require.**

The goal is to preserve exact results while reducing the cost of selecting the nearest `k` vectors.

---

# 7. Memory and Cache Behavior

Once the algorithm is vectorized, memory behavior becomes an important bottleneck.

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

An important experiment will be determining what happens as the dataset grows through and beyond different levels of the CPU cache hierarchy.

Conceptually:

```text
Vectorized computation
        │
        ▼
CPU needs vector data
        │
        ▼
Where is the data?
        │
        ├── CPU registers
        ├── L1 cache
        ├── L2 cache
        ├── L3 / system cache
        └── DRAM
```

As the working set grows, more of the dataset must be served from slower levels of the memory hierarchy.

This makes cache behavior especially important for vector search because exact search repeatedly scans the vector matrix.

---

# 8. Dimensionality and Batch Experiments

We will benchmark the effect of vector dimensionality.

Potential dimensions:

```text
D = 64
D = 128
D = 384
D = 768
D = 1536
```

We will also investigate:

```text
batch size → throughput
```

The objective is to measure rather than assume how dimensionality and batching affect performance.

Expected questions include:

* Is latency proportional to `D`?
* When does memory bandwidth dominate?
* Does batching improve hardware utilization?
* When do allocations become significant?
* How does float32 compare with float64?
* Does higher dimensionality change the relative benefit of vectorization?

The experiments will help separate:

```text
Compute cost
     vs
Memory movement
     vs
Python / allocation overhead
```

---

# 9. HNSW

After understanding exact search and its systems bottlenecks, we will implement:

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

we construct a navigable graph:

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

We will study how:

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

The exact search implementation will remain our ground-truth oracle.

---

# 10. Recall Evaluation

Exact search becomes our ground truth.

For each query:

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

we compare the results.

A simple recall@k metric is:

```text
recall@k =

|approximate top-k ∩ exact top-k|
---------------------------------
                k
```

This allows us to create real recall/latency curves.

The exact implementation is therefore not merely the first implementation — it remains useful throughout the project as the **ground-truth oracle**.

---

# 11. Persistence

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

Persistence will also allow us to investigate the relationship between:

```text
Disk
 │
 ▼
Memory mapping
 │
 ▼
Page cache
 │
 ▼
Search latency
```

---

# 12. Production-Oriented Features

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

These features will move the project from an educational implementation toward a small production-oriented vector indexing system.

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

External libraries therefore serve as reference points rather than shortcuts.

The comparison should help answer questions such as:

* How much faster are optimized implementations?
* Which optimizations matter most?
* How do they organize memory?
* How do they exploit SIMD?
* How do they implement graph traversal?
* How much memory does the index require?
* What engineering complexity is hidden behind a simple search API?

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

This makes every optimization an experiment rather than a guess.

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

The objective is not merely to implement vector search.

The objective is to understand the engineering decisions that make vector search fast.

---

# Current Milestones

## Milestone 1 — Exact Search Baseline

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
✓ Unit tests
✓ Packaging
```

---

## Milestone 2 — Vectorized Exact Search

```text
COMPLETE
```

Implemented:

```text
✓ ExactNumPyIndex
✓ Vectorized distance computation
✓ Input validation
✓ Deterministic ordering
✓ NumPy vs loop correctness tests
✓ Randomized equivalence tests
✓ Reproducible benchmark
✓ Latency measurements
✓ Benchmark JSON output
✓ Python-loop vs NumPy comparison
✓ Latency visualization
```

Current test status:

```text
27 passed
```

The key result from this milestone is that we now have a **measured exact-search baseline**.

At `N = 100K`, vectorization reduced p50 query latency from approximately:

```text
259.7 ms
```

to:

```text
23.2 ms
```

while preserving exact results.

That is approximately:

```text
11.2× p50 speedup
```

---

# Next Milestone

## Milestone 3 — Top-k Optimization + Systems Benchmarking

We will now investigate how much performance can be gained without changing the search algorithm itself.

First:

```text
Full sorting
     │
     ▼
Partial selection
     │
     ▼
np.argpartition
```

Then:

```text
Dimension

   │
   ├── 64
   ├── 128
   ├── 384
   ├── 768
   └── 1536
```

Dataset size:

```text
   │
   ├── 1K
   ├── 10K
   ├── 100K
   └── 1M
```

Batch size:

```text
   │
   ▼
Throughput
```

Memory:

```text
   │
   ▼
Cache behavior
   │
   ▼
Memory bandwidth
```

The immediate next experiment is therefore:

```text
Exact Search
     │
     ▼
Memory / Cache Experiment
     │
     ▼
Understand working-set size
     │
     ▼
Measure cache effects
     │
     ▼
Optimize memory behavior
```

Only after understanding these exact-search bottlenecks will we move to HNSW.

The goal is to progressively move from:

> **"I know how vector search works."**

to:

> **"I understand why a production vector search engine is designed this way."**
