"""
Test script for weight loader - demonstrates usage with the ACT decode model.
"""

from pathlib import Path

import torch

from weight_loader import WeightLoader


def test_weight_loader_basic():
    """Test basic weight loading and conversion."""
    print("=" * 80)
    print("Test 1: Basic Weight Loading and Conversion")
    print("=" * 80)

    source_path = "/x2robot_v2/share/liangyuxin/share_ckpts/multi_task_moe_1030/40/model.safetensors"
    output_path = "/x2robot_v2/vincent/workspace/inference/tests/converted_weights_lucy.safetensors"

    # Load and convert weights
    loader = WeightLoader(
        source_path=source_path,
        expert_index=1,  # Use expert 1 as specified
        verbose=True,
    )

    converted_weights = loader.convert()
    loader.save(output_path)

    print("\n✓ Conversion successful!")
    print(f"✓ Saved to: {output_path}")

    return converted_weights


def test_load_into_model():
    """Test loading converted weights into the ACT decode model."""
    print("\n" + "=" * 80)
    print("Test 2: Loading Converted Weights into Model")
    print("=" * 80)

    try:
        from wall_x.infer.qwen2_5_based.modeling_qwen2_5_vl_act import (
            Qwen2_5_VL_ACT_Decode,
        )
        from transformers import AutoConfig

        # Load config
        config_path = "/x2robot_v2/vincent/workspace/inference/wall-x/workspace/models_config/moe_flash.json"
        config = AutoConfig.from_pretrained(config_path)

        # Create model
        print("\nCreating model...")
        model = Qwen2_5_VL_ACT_Decode(config)

        converted_path = "/x2robot_v2/vincent/workspace/inference/tests/converted_weights_lucy.safetensors"

        if Path(converted_path).exists():
            from safetensors.torch import load_file

            state_dict = load_file(converted_path)

            # Load state dict
            print(f"\nLoading weights from: {converted_path}")
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )

            print("\n✓ Weights loaded successfully!")

            if missing_keys:
                print(f"\nMissing keys ({len(missing_keys)}):")
                for key in missing_keys[:10]:  # Show first 10
                    print(f"  - {key}")
                if len(missing_keys) > 10:
                    print(f"  ... and {len(missing_keys) - 10} more")

            if unexpected_keys:
                print(f"\nUnexpected keys ({len(unexpected_keys)}):")
                for key in unexpected_keys[:10]:  # Show first 10
                    print(f"  - {key}")
                if len(unexpected_keys) > 10:
                    print(f"  ... and {len(unexpected_keys) - 10} more")

            return model, state_dict
        else:
            print(f"\n✗ Converted weights not found at: {converted_path}")
            print("  Run test_weight_loader_basic() first to create converted weights")
            return None, None

    except Exception as e:
        print(f"\n✗ Error loading into model: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def inspect_weight_shapes():
    """Inspect and compare weight shapes between source and target."""
    print("\n" + "=" * 80)
    print("Test 3: Weight Shape Inspection")
    print("=" * 80)

    source_path = "/x2robot_v2/share/liangyuxin/share_ckpts/multi_task_moe_1030/40/model.safetensors"

    if not Path(source_path).exists():
        print(f"✗ Source file not found: {source_path}")
        return

    from safetensors.torch import load_file

    source_weights = load_file(source_path)

    print("\nSource weight shapes (first layer):")
    print("-" * 80)

    layer_idx = 0

    # QKV weights
    q_key = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
    k_key = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
    v_key = f"model.layers.{layer_idx}.self_attn.v_proj.weight"

    if q_key in source_weights:
        q_shape = source_weights[q_key].shape
        k_shape = source_weights[k_key].shape
        v_shape = source_weights[v_key].shape

        print(f"Q proj: {q_shape}")
        print(f"K proj: {k_shape}")
        print(f"V proj: {v_shape}")
        print(f"→ Merged QKV: ({q_shape[0] + k_shape[0] + v_shape[0]}, {q_shape[1]})")

    # Gate/Up weights
    gate_key = f"model.layers.{layer_idx}.moe.experts.1.gate_proj.weight"
    up_key = f"model.layers.{layer_idx}.moe.experts.1.up_proj.weight"

    if gate_key in source_weights:
        gate_shape = source_weights[gate_key].shape
        up_shape = source_weights[up_key].shape

        print(f"\nGate proj (expert 1): {gate_shape}")
        print(f"Up proj (expert 1): {up_shape}")
        print(f"→ Merged gate_up: ({gate_shape[0] + up_shape[0]}, {gate_shape[1]})")

    converted_path = "/x2robot_v2/vincent/workspace/inference/tests/converted_weights_lucy.safetensors"
    if Path(converted_path).exists():
        converted_weights = load_file(converted_path)

        print("\n" + "-" * 80)
        print("Converted weight shapes (first layer):")
        print("-" * 80)

        qkv_key = f"layers.{layer_idx}.self_attn.qkv_proj.weight"
        if qkv_key in converted_weights:
            print(f"QKV merged: {converted_weights[qkv_key].shape}")

        gate_up_key = f"layers.{layer_idx}.act_expert_mlp.gate_up_proj.weight"
        if gate_up_key in converted_weights:
            print(f"gate_up merged: {converted_weights[gate_up_key].shape}")


def compare_forward_pass():
    """Compare forward pass results (if possible)."""
    print("\n" + "=" * 80)
    print("Test 4: Forward Pass Verification")
    print("=" * 80)

    print("\nCreating dummy input...")

    # Create a small dummy input
    batch_size = 1
    seq_len = 10
    hidden_size = 4096  # Adjust based on your model

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    print(f"Input shape: {hidden_states.shape}")
    print(
        "\nNote: Full forward pass test requires proper model initialization and KV cache."
    )
    print("This is a placeholder for more detailed testing.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test weight loader functionality")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "convert", "load", "inspect", "forward"],
        default="all",
        help="Which test to run",
    )

    args = parser.parse_args()

    if args.mode == "all" or args.mode == "convert":
        converted_weights = test_weight_loader_basic()

    if args.mode == "all" or args.mode == "load":
        test_load_into_model()

    if args.mode == "all" or args.mode == "inspect":
        inspect_weight_shapes()

    if args.mode == "all" or args.mode == "forward":
        compare_forward_pass()

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
