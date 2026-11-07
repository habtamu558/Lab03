
def calculate_mlp_flops():
    """
    Calculate MLP FLOPs for one image.
    Linear FLOPs = 2 * input_features * output_features
    """

    layer1 = 2 * 784 * 256
    layer2 = 2 * 256 * 128
    layer3 = 2 * 128 * 10

    total = layer1 + layer2 + layer3

    return {
        "Layer 1 (784 -> 256)": layer1,
        "Layer 2 (256 -> 128)": layer2,
        "Layer 3 (128 -> 10)": layer3,
        "Total": total
    }


def calculate_cnn_flops():
    """
    Calculate CNN FLOPs for one image.
    Conv FLOPs = 2 * K*K * Cin * Cout * Hout * Wout
    """

    conv1 = 2 * 3 * 3 * 1 * 16 * 28 * 28
    conv2 = 2 * 3 * 3 * 16 * 32 * 14 * 14
    linear = 2 * (32 * 7 * 7) * 10

    total = conv1 + conv2 + linear

    return {
        "Conv 1": conv1,
        "Conv 2": conv2,
        "Final Linear": linear,
        "Total": total
    }


def calculate_attention_flops():
    """
    Estimate Transformer FLOPs for one image.

    Sequence length = 16 patches
    Embedding dimension = 64
    Feedforward dimension = 128
    Transformer layers = 2
    """

    sequence_length = 16
    d_model = 64
    feedforward_dim = 128
    num_layers = 2

    # Patch projection: 1 input channel to 64 output channels.
    patch_projection = 2 * 7 * 7 * 1 * 64 * 4 * 4

    # Q, K, V projections.
    qkv = (
        2 * sequence_length * d_model * 3 * d_model
    )

    # Attention scores and weighted values.
    attention_scores = (
        2 * sequence_length * sequence_length * d_model
    )

    attention_values = (
        2 * sequence_length * sequence_length * d_model
    )

    # Output projection.
    output_projection = (
        2 * sequence_length * d_model * d_model
    )

    # Feedforward network.
    feedforward = (
        2 * sequence_length * d_model * feedforward_dim
        + 2 * sequence_length * feedforward_dim * d_model
    )

    # Two Transformer encoder layers.
    transformer_total = num_layers * (
        qkv
        + attention_scores
        + attention_values
        + output_projection
        + feedforward
    )

    # Final classification layer.
    classifier = 2 * d_model * 10

    total = (
        patch_projection
        + transformer_total
        + classifier
    )

    return {
        "Patch projection": patch_projection,
        "QKV projections (per layer)": qkv,
        "Attention scores (per layer)": attention_scores,
        "Attention values (per layer)": attention_values,
        "Output projection (per layer)": output_projection,
        "Feedforward (per layer)": feedforward,
        "Transformer total (2 layers)": transformer_total,
        "Final classifier": classifier,
        "Total": total
    }


def print_results(model_name, results):
    print("\n" + "=" * 55)
    print(model_name)
    print("=" * 55)

    for name, value in results.items():
        print(f"{name}: {value:,} FLOPs")

    print(f"Total FLOPs per image: {results['Total']:,}")


if __name__ == "__main__":
    print_results("MLP", calculate_mlp_flops())
    print_results("CNN", calculate_cnn_flops())
    print_results("Attention", calculate_attention_flops())
