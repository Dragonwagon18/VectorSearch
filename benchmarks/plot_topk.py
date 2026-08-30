import json
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_PATH = Path("benchmarks/results/topk.json")
OUTPUT_DIR = Path("plots")


def main():
    with RESULTS_PATH.open() as f:
        data = json.load(f)

    results = data["results"]

    n = [r["n"] for r in results]

    argsort_p50 = [
        r["argsort"]["p50_ms"]
        for r in results
    ]

    argpartition_p50 = [
        r["argpartition"]["p50_ms"]
        for r in results
    ]

    argsort_qps = [
        r["argsort"]["qps"]
        for r in results
    ]

    argpartition_qps = [
        r["argpartition"]["qps"]
        for r in results
    ]

    speedup = [
        r["p50_speedup"]
        for r in results
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Plot 1: Latency
    # ---------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        n,
        argsort_p50,
        marker="o",
        label="np.argsort",
    )

    plt.plot(
        n,
        argpartition_p50,
        marker="o",
        label="np.argpartition",
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Dataset size (N)")
    plt.ylabel("p50 latency (ms)")
    plt.title("Top-k Selection: Argsort vs Argpartition")

    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    latency_path = OUTPUT_DIR / "topk_latency.png"
    plt.savefig(latency_path, dpi=200)
    plt.close()

    # ---------------------------------------------------------
    # Plot 2: QPS
    # ---------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        n,
        argsort_qps,
        marker="o",
        label="np.argsort",
    )

    plt.plot(
        n,
        argpartition_qps,
        marker="o",
        label="np.argpartition",
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Dataset size (N)")
    plt.ylabel("Queries per second (QPS)")
    plt.title("Top-k Selection: Throughput Comparison")

    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    qps_path = OUTPUT_DIR / "topk_qps.png"
    plt.savefig(qps_path, dpi=200)
    plt.close()

    # ---------------------------------------------------------
    # Plot 3: Speedup
    # ---------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        n,
        speedup,
        marker="o",
    )

    plt.xscale("log")

    plt.xlabel("Dataset size (N)")
    plt.ylabel("Speedup (×)")
    plt.title("Top-k Selection: Argpartition Speedup")

    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    speedup_path = OUTPUT_DIR / "topk_speedup.png"
    plt.savefig(speedup_path, dpi=200)
    plt.close()

    print(f"Saved: {latency_path}")
    print(f"Saved: {qps_path}")
    print(f"Saved: {speedup_path}")


if __name__ == "__main__":
    main()