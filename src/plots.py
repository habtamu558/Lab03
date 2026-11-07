
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_results():
    accuracy_file = Path("results/metrics/model_accuracy.csv")
    latency_file = Path("results/metrics/latency_results.csv")

    accuracy_df = pd.read_csv(accuracy_file)
    latency_df = pd.read_csv(latency_file)

    # Keep batch size 1 latency.
    latency_batch1 = latency_df[
        latency_df["batch_size"] == 1
    ]

    # Combine accuracy and latency results.
    results = accuracy_df.merge(
        latency_batch1[["model", "median_latency_ms"]],
        on="model"
    )

    # Add actual parameter counts.
    parameter_counts = {
        "MLP": 235146,
        "CNN": 20490,
        "Attention": 70794
    }

    results["parameters"] = results["model"].map(parameter_counts)

    return results


def plot_accuracy_vs_parameters(results, output_dir):
    plt.figure(figsize=(8, 5))

    plt.scatter(
        results["parameters"],
        results["test_accuracy"]
    )

    # Add model names beside each point.
    for _, row in results.iterrows():
        plt.annotate(
            row["model"],
            (row["parameters"], row["test_accuracy"])
        )

    plt.xlabel("Number of Parameters")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy vs. Model Parameters")

    plt.grid(True)
    plt.tight_layout()

    # Save the plot.
    plt.savefig(
        output_dir / "accuracy_vs_parameters.png"
    )

    plt.close()


def plot_accuracy_vs_latency(results, output_dir):
    plt.figure(figsize=(8, 5))

    plt.scatter(
        results["median_latency_ms"],
        results["test_accuracy"]
    )

    # Add model names beside each point.
    for _, row in results.iterrows():
        plt.annotate(
            row["model"],
            (row["median_latency_ms"], row["test_accuracy"])
        )

    plt.xlabel("Inference Latency (ms) - Batch Size 1")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy vs. Inference Latency")

    plt.grid(True)
    plt.tight_layout()

    # Save the plot.
    plt.savefig(
        output_dir / "accuracy_vs_latency.png"
    )

    plt.close()

    
def main():
    # Create the plots directory.
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the results.
    results = load_results()

    # Generate both graphs.
    plot_accuracy_vs_parameters(results, output_dir)
    plot_accuracy_vs_latency(results, output_dir)

    print("\nBoth plots generated successfully!")
    print(f"Plots saved in: {output_dir}")


if __name__ == "__main__":
    main()

