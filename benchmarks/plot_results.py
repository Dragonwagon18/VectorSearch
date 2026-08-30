from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
PLOTS = ROOT / "plots"


def load_json(filename: str):
    path = RESULTS / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark result: {path}")

    with path.open() as f:
        return json.load(f)


def rows(data):
    """Return the list containing benchmark rows."""
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("results", "runs", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError("Could not find benchmark result rows")


def value(row, *keys):
    """Return the first matching key from a row."""
    for key in keys:
        if key in row:
            return row[key]

    raise KeyError(
        f"None of these keys found: {keys}\n"
        f"Available keys: {list(row.keys())}"
    )


def save_plot(filename: str) -> None:
    PLOTS.mkdir(exist_ok=True)

    plt.tight_layout()
    plt.savefig(
        PLOTS / filename,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


# ============================================================
# Exact search
# ============================================================

def plot_exact_scaling() -> None:
    data = rows(load_json("exact.json"))

    n = [value(r, "n") for r in data]
    p50 = [value(r, "p50_ms", "median_p50_ms") for r in data]

    plt.figure(figsize=(8, 5))
    plt.plot(n, p50, marker="o")

    plt.xscale("log")
    plt.xlabel("Dataset size (N)")
    plt.ylabel("p50 latency (ms)")
    plt.title("Exact Search Scaling")
    plt.grid(True, alpha=0.3)

    save_plot("exact_scaling.png")


def plot_exact_qps() -> None:
    data = rows(load_json("exact.json"))

    n = [value(r, "n") for r in data]
    qps = [value(r, "qps", "median_qps") for r in data]

    plt.figure(figsize=(8, 5))
    plt.plot(n, qps, marker="o")

    plt.xscale("log")
    plt.xlabel("Dataset size (N)")
    plt.ylabel("Queries per second")
    plt.title("Exact Search Throughput")
    plt.grid(True, alpha=0.3)

    save_plot("exact_qps.png")


# ============================================================
# Batch search
# ============================================================

def plot_batch_scaling() -> None:
    data = rows(load_json("batch.json"))

    batch = [
        value(r, "batch_size", "batch")
        for r in data
    ]

    latency = [
        value(
            r,
            "per_query_ms",
            "per_query",
            "latency_ms",
            "p50_ms",
        )
        for r in data
    ]

    qps = [
        value(
            r,
            "qps",
            "QPS",
        )
        for r in data
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(batch, latency, marker="o")

    plt.xlabel("Batch size")
    plt.ylabel("Per-query latency (ms)")
    plt.title("Batch Size vs Per-Query Latency")
    plt.grid(True, alpha=0.3)

    save_plot("batch_latency.png")

    plt.figure(figsize=(8, 5))
    plt.plot(batch, qps, marker="o")

    plt.xlabel("Batch size")
    plt.ylabel("QPS")
    plt.title("Batch Size vs Throughput")
    plt.grid(True, alpha=0.3)

    save_plot("batch_qps.png")


def plot_batch_breakdown() -> None:
    """
    Uses profile_batch.py output if a JSON result containing
    distance/selection measurements exists.

    Otherwise this plot is skipped.
    """

    candidates = [
        "profile_batch.json",
        "batch_profile.json",
    ]

    filename = None

    for candidate in candidates:
        if (RESULTS / candidate).exists():
            filename = candidate
            break

    if filename is None:
        print(
            "⚠ batch_breakdown.png skipped: "
            "no batch profiling JSON found"
        )
        return

    data = rows(load_json(filename))

    batch = [
        value(r, "batch_size", "batch")
        for r in data
    ]

    distance = [
        value(r, "distance_ms", "distance")
        for r in data
    ]

    selection = [
        value(r, "selection_ms", "selection")
        for r in data
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        batch,
        distance,
        marker="o",
        label="Distance",
    )

    plt.plot(
        batch,
        selection,
        marker="o",
        label="Top-k selection",
    )

    plt.xlabel("Batch size")
    plt.ylabel("Time (ms)")
    plt.title("Batch Search Cost Breakdown")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_plot("batch_breakdown.png")


# ============================================================
# Dtype
# ============================================================

def plot_dtype_comparison() -> None:
    data = rows(load_json("dtype.json"))

    dtypes = [
        value(r, "dtype")
        for r in data
    ]

    p50 = [
        value(r, "p50_ms", "median_p50_ms")
        for r in data
    ]

    memory = [
        value(
            r,
            "memory_mb",
            "memory",
        )
        for r in data
    ]

    x = np.arange(len(dtypes))

    plt.figure(figsize=(7, 5))

    plt.bar(x, p50)
    plt.xticks(x, dtypes)
    plt.ylabel("p50 latency (ms)")
    plt.title("float32 vs float64 Latency")
    plt.grid(axis="y", alpha=0.3)

    save_plot("dtype_latency.png")

    plt.figure(figsize=(7, 5))

    plt.bar(x, memory)
    plt.xticks(x, dtypes)
    plt.ylabel("Memory (MB)")
    plt.title("float32 vs float64 Memory")
    plt.grid(axis="y", alpha=0.3)

    save_plot("dtype_memory.png")


# ============================================================
# Layout
# ============================================================

def plot_layout_comparison() -> None:
    data = rows(load_json("layout.json"))

    layouts = [
        value(r, "layout")
        for r in data
    ]

    p50 = [
        value(
            r,
            "median_p50_ms",
            "p50_ms",
        )
        for r in data
    ]

    qps = [
        value(
            r,
            "median_qps",
            "qps",
        )
        for r in data
    ]

    x = np.arange(len(layouts))

    plt.figure(figsize=(9, 5))

    plt.bar(x, p50)
    plt.xticks(x, layouts, rotation=20)
    plt.ylabel("p50 latency (ms)")
    plt.title("Memory Layout vs Latency")
    plt.grid(axis="y", alpha=0.3)

    save_plot("layout_latency.png")

    plt.figure(figsize=(9, 5))

    plt.bar(x, qps)
    plt.xticks(x, layouts, rotation=20)
    plt.ylabel("QPS")
    plt.title("Memory Layout vs Throughput")
    plt.grid(axis="y", alpha=0.3)

    save_plot("layout_qps.png")


# ============================================================
# Strided access
# ============================================================

def plot_strided_comparison() -> None:
    data = rows(load_json("strided.json"))

    layouts = [
        value(r, "layout")
        for r in data
    ]

    p50 = [
        value(
            r,
            "median_p50_ms",
            "p50_ms",
        )
        for r in data
    ]

    qps = [
        value(
            r,
            "median_qps",
            "qps",
        )
        for r in data
    ]

    x = np.arange(len(layouts))

    plt.figure(figsize=(9, 5))

    plt.bar(x, p50)
    plt.xticks(x, layouts, rotation=20)
    plt.ylabel("p50 latency (ms)")
    plt.title("Strided Memory Access vs Latency")
    plt.grid(axis="y", alpha=0.3)

    save_plot("strided_latency.png")

    plt.figure(figsize=(9, 5))

    plt.bar(x, qps)
    plt.xticks(x, layouts, rotation=20)
    plt.ylabel("QPS")
    plt.title("Strided Memory Access vs Throughput")
    plt.grid(axis="y", alpha=0.3)

    save_plot("strided_qps.png")


# ============================================================
# Distance methods
# ============================================================

def plot_distance_methods() -> None:
    """
    Try to read distance-method comparison results.

    Expected JSON examples:

    {
        "results": [
            {
                "method": "subtract + einsum",
                "p50_ms": 7.170
            },
            {
                "method": "dot-product identity",
                "p50_ms": 0.934
            }
        ]
    }

    or equivalent keys.
    """

    candidates = [
        "distance_methods.json",
        "compare_distance_methods.json",
    ]

    filename = None

    for candidate in candidates:
        if (RESULTS / candidate).exists():
            filename = candidate
            break

    if filename is None:
        print(
            "⚠ distance_methods.png skipped: "
            "no distance-method JSON found"
        )
        return

    data = rows(load_json(filename))

    methods = [
        value(r, "method", "name")
        for r in data
    ]

    latency = [
        value(
            r,
            "p50_ms",
            "median_p50_ms",
        )
        for r in data
    ]

    x = np.arange(len(methods))

    plt.figure(figsize=(9, 5))

    plt.bar(x, latency)
    plt.xticks(x, methods)
    plt.ylabel("p50 latency (ms)")
    plt.title("Distance Computation Methods")
    plt.grid(axis="y", alpha=0.3)

    save_plot("distance_methods.png")


# ============================================================
# Search breakdown
# ============================================================

def plot_search_breakdown() -> None:
    """
    Use profile_distance.json if available.
    """

    candidates = [
        "profile_distance.json",
        "distance_profile.json",
    ]

    filename = None

    for candidate in candidates:
        if (RESULTS / candidate).exists():
            filename = candidate
            break

    if filename is None:
        print(
            "⚠ search_breakdown.png skipped: "
            "no distance profiling JSON found"
        )
        return

    data = load_json(filename)

    # Handle a possible structure:
    #
    # {
    #   "distance": {...},
    #   "selection": {...}
    # }
    #
    # or rows.

    if isinstance(data, dict) and {
        "distance",
        "selection",
    }.issubset(data.keys()):

        distance_data = data["distance"]
        selection_data = data["selection"]

        distance = value(
            distance_data,
            "p50_ms",
            "median_p50_ms",
        )

        selection = value(
            selection_data,
            "p50_ms",
            "median_p50_ms",
        )

    else:
        data_rows = rows(data)

        distance_values = []
        selection_values = []

        for r in data_rows:
            if "distance_ms" in r:
                distance_values.append(r["distance_ms"])

            if "selection_ms" in r:
                selection_values.append(r["selection_ms"])

        if not distance_values or not selection_values:
            print(
                "⚠ search_breakdown.png skipped: "
                "could not find distance/selection data"
            )
            return

        distance = float(np.median(distance_values))
        selection = float(np.median(selection_values))

    components = [
        "Distance computation",
        "Top-k selection",
    ]

    values = [
        distance,
        selection,
    ]

    x = np.arange(len(components))

    plt.figure(figsize=(9, 5))

    plt.bar(x, values)
    plt.xticks(x, components, rotation=10)
    plt.ylabel("p50 time (ms)")
    plt.title("Exact Search Cost Breakdown")
    plt.grid(axis="y", alpha=0.3)

    save_plot("search_breakdown.png")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("Generating benchmark plots...\n")

    plot_exact_scaling()
    print("✓ exact_scaling.png")

    plot_exact_qps()
    print("✓ exact_qps.png")

    plot_batch_scaling()
    print("✓ batch_latency.png")
    print("✓ batch_qps.png")

    plot_batch_breakdown()

    plot_dtype_comparison()
    print("✓ dtype_latency.png")
    print("✓ dtype_memory.png")

    plot_layout_comparison()
    print("✓ layout_latency.png")
    print("✓ layout_qps.png")

    plot_strided_comparison()
    print("✓ strided_latency.png")
    print("✓ strided_qps.png")

    plot_distance_methods()

    plot_search_breakdown()

    print("\nAll available plots written to:")
    print(f"  {PLOTS}")


if __name__ == "__main__":
    main()