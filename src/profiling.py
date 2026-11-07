
from pathlib import Path

import torch
from torchinfo import summary
from thop import profile

from src.models import MLP, CNN, AttentionModel


def profile_models():
    # Input shape: batch size, channels, height, width
    input_size = (128, 1, 28, 28)

    # Create the three models.
    models = {
        "MLP": MLP(),
        "CNN": CNN(),
        "Attention": AttentionModel()
    }

    # Create the output directory.
    output_dir = Path("results/model_summaries")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Inspect each model.
    for model_name, model in models.items():
        print("\n" + "=" * 60)
        print(f"{model_name} MODEL SUMMARY")
        print("=" * 60)

        model_summary = summary(
            model,
            input_size=input_size,
            device="cpu",
            verbose=1
        )

        # Save the summary to a text file.
        output_file = output_dir / f"{model_name.lower()}_summary.txt"

        with open(output_file, "w", encoding="utf-8") as file:
            file.write(str(model_summary))

        print(f"\nSummary saved to: {output_file}")


def profile_flops():
    # Create the three models.
    models = {
        "MLP": MLP(),
        "CNN": CNN(),
        "Attention": AttentionModel()
    }

    # Create one sample image.
    dummy_input = torch.randn(1, 1, 28, 28)

    print("\n" + "=" * 60)
    print("THOP FLOPs PROFILING")
    print("=" * 60)

    for model_name, model in models.items():
        model.eval()

        # Calculate MACs and THOP parameters.
        macs, thop_params = profile(
            model,
            inputs=(dummy_input,),
            verbose=False
        )

        # Count all model parameters directly using PyTorch.
        actual_params = sum(
            p.numel() for p in model.parameters()
        )

        print(f"\n{model_name}")
        print(f"MACs: {macs:,.0f}")
        print(f"Parameters (PyTorch): {actual_params:,}")
        print(f"Parameters (THOP): {thop_params:,.0f}")
        print(f"Estimated FLOPs (2 x MACs): {2 * macs:,.0f}")


if __name__ == "__main__":
    profile_models()
    profile_flops()
