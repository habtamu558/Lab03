
import time
import csv
from pathlib import Path

import torch
import numpy as np

from src.models import MLP, CNN, AttentionModel


def measure_latency(model, batch_size, num_runs):
    model.eval()

    dummy_input = torch.randn(batch_size, 1, 28, 28)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)

    times = []

    # Measure inference time
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            model(dummy_input)
            end_time = time.perf_counter()

            times.append(end_time - start_time)

    median_latency = np.median(times) * 1000

    return median_latency


def run_latency_experiment():
    models = {
        "MLP": MLP(),
        "CNN": CNN(),
        "Attention": AttentionModel()
    }

    batch_sizes = [1, 128]
    results = []

    for model_name, model in models.items():
        print("\n" + "=" * 50)
        print(f"{model_name} LATENCY RESULTS")
        print("=" * 50)

        for batch_size in batch_sizes:
            num_runs = 100 if batch_size == 1 else 20

            latency = measure_latency(
                model,
                batch_size,
                num_runs
            )

            print(
                f"Batch size {batch_size}: "
                f"{latency:.4f} ms"
            )

            results.append({
                "model": model_name,
                "batch_size": batch_size,
                "num_runs": num_runs,
                "median_latency_ms": round(latency, 4)
            })

    # Save results to CSV.
    output_dir = Path("results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "latency_results.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "batch_size",
                "num_runs",
                "median_latency_ms"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nLatency results saved to: {output_file}")


if __name__ == "__main__":
    run_latency_experiment()
