import json
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_PATH = Path("benchmarks/results/cache.json")
OUTPUT_DIR = Path("plots")


def main():
    with RESULTS_PATH.open() as f:
        data = json.load(f)
        
    results = data["results"]
    OUTPUT_DIR.mkdir(exist_ok=True)

    memory_mb = []
    p50_ms = []
    p95_ms = []
    qps = []

    for result in results:
        memory = result["memory_mb"]
        memory_mb.append(memory)
        p50_ms.append(result["p50_ms"])
        p95_ms.append(result["p95_ms"])
        qps.append(result["qps"])

    # ---------------------------------------------------------
    # Plot 1: Memory footprint vs latency
    # ---------------------------------------------------------
    plt.figure(figsize=(9, 6))

    plt.plot(memory_mb, p50_ms, marker="o", label="p50 latency")
    plt.plot(memory_mb, p95_ms, marker="o", label="p95 latency")

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Vector memory footprint (MB)")
    plt.ylabel("Latency (ms)")
    plt.title("Cache / Memory Experiment: Memory vs Query Latency")

    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    latency_path = OUTPUT_DIR / "cache_latency.png"
    plt.savefig(latency_path, dpi=200)
    plt.close()

    # ---------------------------------------------------------
    # Plot 2: Memory footprint vs QPS
    # ---------------------------------------------------------
    plt.figure(figsize=(9, 6))

    plt.plot(memory_mb, qps, marker="o")

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Vector memory footprint (MB)")
    plt.ylabel("Queries per second (QPS)")
    plt.title("Cache / Memory Experiment: Memory vs Throughput")

    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    qps_path = OUTPUT_DIR / "cache_qps.png"
    plt.savefig(qps_path, dpi=200)
    plt.close()

    print(f"Saved: {latency_path}")
    print(f"Saved: {qps_path}")


if __name__ == "__main__":
    main()