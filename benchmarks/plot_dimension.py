import json
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_PATH = Path("benchmarks/results/dimension.json")
OUTPUT_DIR = Path("plots")


def main():
    with RESULTS_PATH.open() as f:
        data = json.load(f)

    results = data["results"]

    dimensions = [r["d"] for r in results]
    latency_p50 = [r["p50_ms"] for r in results]
    latency_p95 = [r["p95_ms"] for r in results]
    qps = [r["qps"] for r in results]
    memory_mb = [r["memory_mb"] for r in results]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Plot 1: Dimension vs Latency
    # ---------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        dimensions,
        latency_p50,
        marker="o",
        label="p50 latency",
    )

    plt.plot(
        dimensions,
        latency_p95,
        marker="o",
        label="p95 latency",
    )

    plt.xlabel("Vector dimension (D)")
    plt.ylabel("Latency (ms)")
    plt.title("Vector Dimension vs Query Latency")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    latency_path = OUTPUT_DIR / "dimension_latency.png"
    plt.savefig(latency_path, dpi=200)
    plt.close()

    # ---------------------------------------------------------
    # Plot 2: Dimension vs QPS
    # ---------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        dimensions,
        qps,
        marker="o",
    )

    plt.xlabel("Vector dimension (D)")
    plt.ylabel("Queries per second (QPS)")
    plt.title("Vector Dimension vs Throughput")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    qps_path = OUTPUT_DIR / "dimension_qps.png"
    plt.savefig(qps_path, dpi=200)
    plt.close()

    # ---------------------------------------------------------
    # Plot 3: Dimension vs Memory
    # ---------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        dimensions,
        memory_mb,
        marker="o",
    )

    plt.xlabel("Vector dimension (D)")
    plt.ylabel("Vector memory footprint (MB)")
    plt.title("Vector Dimension vs Memory Footprint")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    memory_path = OUTPUT_DIR / "dimension_memory.png"
    plt.savefig(memory_path, dpi=200)
    plt.close()

    print(f"Saved: {latency_path}")
    print(f"Saved: {qps_path}")
    print(f"Saved: {memory_path}")


if __name__ == "__main__":
    main()