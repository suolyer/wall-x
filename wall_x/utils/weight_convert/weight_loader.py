"""
Weight loader for converting HuggingFace/standard format weights to merged QKV and gate_up format.

This loader supports:
1. Merging Q, K, V projections into a single qkv_proj weight
2. Merging gate and up projections into a single gate_up_proj weight
3. Remapping layer names from model.layers.X to layers.X
4. Using expert 1 for MoE models (configurable)
"""

import torch
from safetensors.torch import save_file
from typing import Dict, Optional
from pathlib import Path


class WeightLoader:
    def __init__(
        self,
        config: Dict,
        source_weights: Dict[str, torch.Tensor],
        expert_index: int = 1,
        num_layers: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Initialize the weight loader.

        Args:
            source_path: Path to source weights (safetensors or pytorch checkpoint)
            expert_index: Which expert to use for MoE models (default: 1)
            num_layers: Number of layers in the model (auto-detected if None)
            verbose: Print conversion progress
        """
        self.config = config
        self.source_weights = source_weights
        self.expert_index = expert_index
        self.verbose = verbose

        # Auto-detect number of layers if not provided
        if num_layers is None:
            self.num_layers = self._detect_num_layers()
        else:
            self.num_layers = num_layers

        if self.verbose:
            print(f"Number of layers detected: {self.num_layers}")
            print(f"Using expert index: {expert_index}")

    def _detect_num_layers(self) -> int:
        """Auto-detect number of layers from weight keys."""
        layer_indices = set()
        for key in self.source_weights.keys():
            if "layers." in key:
                # Extract layer index
                parts = key.split(".")
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts):
                        try:
                            layer_indices.add(int(parts[i + 1]))
                        except ValueError:
                            continue
        return max(layer_indices) + 1 if layer_indices else 0

    def _get_weight(self, key: str) -> Optional[torch.Tensor]:
        """Safely get a weight tensor, returning None if not found."""
        return self.source_weights.get(key)

    def _merge_qkv(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        Merge Q, K, V projection weights and biases for a layer.

        Expected source format:
            model.layers.{layer_idx}.self_attn.q_proj.weight
            model.layers.{layer_idx}.self_attn.k_proj.weight
            model.layers.{layer_idx}.self_attn.v_proj.weight
            (and corresponding .bias if present)

        Target format:
            layers.{layer_idx}.self_attn.qkv_proj.qkv_proj.weight
            layers.{layer_idx}.self_attn.qkv_proj.qkv_proj.bias

        Returns:
            Dictionary with merged weights
        """
        result = {}

        # Get Q, K, V weights
        if self.config.attention_moe:
            q_weight = self._get_weight(
                f"model.layers.{layer_idx}.self_attn.q_proj_experts.{self.expert_index}.weight"
            )
            k_weight = self._get_weight(
                f"model.layers.{layer_idx}.self_attn.k_proj_experts.{self.expert_index}.weight"
            )
            v_weight = self._get_weight(
                f"model.layers.{layer_idx}.self_attn.v_proj_experts.{self.expert_index}.weight"
            )
        else:
            q_weight = self._get_weight(
                f"model.layers.{layer_idx}.self_attn.q_proj.weight"
            )
            k_weight = self._get_weight(
                f"model.layers.{layer_idx}.self_attn.k_proj.weight"
            )
            v_weight = self._get_weight(
                f"model.layers.{layer_idx}.self_attn.v_proj.weight"
            )

        if q_weight is None or k_weight is None or v_weight is None:
            merged_weight = self._get_weight(
                f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"
            )
            if merged_weight is None:
                merged_weight = self._get_weight(
                    f"model.layers.{layer_idx}.self_attn.qkv_proj_experts.{self.expert_index}.weight"
                )

        # Merge along output dimension (dim 0)
        if q_weight is not None and k_weight is not None and v_weight is not None:
            merged_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        if merged_weight is None:
            raise ValueError(
                f"Missing QKV weights for layer {layer_idx} (expert {self.expert_index})"
            )
        result[f"layers.{layer_idx}.self_attn.qkv_proj.qkv_proj.weight"] = merged_weight

        # Get Q, K, V biases (if they exist)
        if self.config.attention_moe:
            q_bias = self._get_weight(
                f"model.layers.{layer_idx}.self_attn.q_proj_experts.{self.expert_index}.bias"
            )
            k_bias = self._get_weight(
                f"model.layers.{layer_idx}.self_attn.k_proj_experts.{self.expert_index}.bias"
            )
            v_bias = self._get_weight(
                f"model.layers.{layer_idx}.self_attn.v_proj_experts.{self.expert_index}.bias"
            )
        else:
            q_bias = self._get_weight(f"model.layers.{layer_idx}.self_attn.q_proj.bias")
            k_bias = self._get_weight(f"model.layers.{layer_idx}.self_attn.k_proj.bias")
            v_bias = self._get_weight(f"model.layers.{layer_idx}.self_attn.v_proj.bias")

        if q_bias is None or k_bias is None or v_bias is None:
            merged_bias = self._get_weight(
                f"model.layers.{layer_idx}.self_attn.qkv_proj.bias"
            )
            if merged_bias is None:
                merged_bias = self._get_weight(
                    f"model.layers.{layer_idx}.self_attn.qkv_proj_experts.{self.expert_index}.bias"
                )
        else:
            merged_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
        result[f"layers.{layer_idx}.self_attn.qkv_proj.qkv_proj.bias"] = merged_bias

        if self.verbose:
            print(f"Layer {layer_idx}: Merged QKV - shape {merged_weight.shape}")

        return result

    def _merge_gate_up(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        Merge gate and up projection weights for a layer.

        Expected source format (MoE):
            model.layers.{layer_idx}.moe.experts.{expert_idx}.gate_proj.weight
            model.layers.{layer_idx}.moe.experts.{expert_idx}.up_proj.weight

        Alternative source format (standard MLP):
            model.layers.{layer_idx}.mlp.gate_proj.weight
            model.layers.{layer_idx}.mlp.up_proj.weight

        Target format:
            layers.{layer_idx}.act_expert_mlp.gate_up_proj.weight

        Returns:
            Dictionary with merged weights
        """
        result = {}

        # Try MoE format first
        gate_weight = self._get_weight(
            f"model.layers.{layer_idx}.moe.experts.{self.expert_index}.gate_proj.weight"
        )
        up_weight = self._get_weight(
            f"model.layers.{layer_idx}.moe.experts.{self.expert_index}.up_proj.weight"
        )

        # If not found, try standard MLP format
        if gate_weight is None or up_weight is None:
            gate_weight = self._get_weight(
                f"model.layers.{layer_idx}.mlp.gate_proj.weight"
            )
            up_weight = self._get_weight(f"model.layers.{layer_idx}.mlp.up_proj.weight")

        if gate_weight is None or up_weight is None:
            merged_weight = self._get_weight(
                f"model.layers.{layer_idx}.moe.experts.{self.expert_index}.gate_up_proj.weight"
            )
        else:
            merged_weight = torch.cat([gate_weight, up_weight], dim=0)

        if merged_weight is None:
            raise ValueError(
                f"Missing gate/up weights for layer {layer_idx} (expert {self.expert_index})"
            )
        result[f"layers.{layer_idx}.act_expert_mlp.gate_up_proj.weight"] = merged_weight

        if self.verbose:
            print(f"Layer {layer_idx}: Merged gate_up - shape {merged_weight.shape}")

        return result

    def _copy_other_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        Copy other weights that don't need merging (o_proj, down_proj, layer norms, etc.)

        Mapping:
            model.layers.X.self_attn.o_proj.weight -> layers.X.self_attn.o_proj.o_proj.weight
            model.layers.X.moe.experts.{expert}.down_proj.weight -> layers.X.act_expert_mlp.down_proj.weight
            model.layers.X.input_layernorm.weight -> layers.X.input_layernorm.weight
            model.layers.X.post_attention_layernorm.weight -> layers.X.post_attention_layernorm.weight
        """
        result = {}

        # O projection
        if self.config.attention_moe:
            o_weight = self._get_weight(
                f"model.layers.{layer_idx}.self_attn.o_proj_experts.{self.expert_index}.weight"
            )
        else:
            o_weight = self._get_weight(
                f"model.layers.{layer_idx}.self_attn.o_proj.weight"
            )
        if o_weight is not None:
            result[f"layers.{layer_idx}.self_attn.o_proj.o_proj.weight"] = o_weight
        else:
            raise ValueError(
                f"Missing o_proj weight for layer {layer_idx} (expert {self.expert_index})"
            )

        # Down projection (MoE format)
        down_weight = self._get_weight(
            f"model.layers.{layer_idx}.moe.experts.{self.expert_index}.down_proj.weight"
        )

        # If not found, try standard MLP format
        if down_weight is None:
            down_weight = self._get_weight(
                f"model.layers.{layer_idx}.mlp.down_proj.weight"
            )
            if down_weight is None:
                raise ValueError(
                    f"Missing down_proj weight for layer {layer_idx} (expert {self.expert_index})"
                )

        result[f"layers.{layer_idx}.act_expert_mlp.down_proj.weight"] = down_weight

        # Layer norms
        if self.config.norm_moe:
            input_ln = self._get_weight(
                f"model.layers.{layer_idx}.input_layernorms.{self.expert_index}.weight"
            )
        else:
            input_ln = self._get_weight(
                f"model.layers.{layer_idx}.input_layernorm.weight"
            )
        if input_ln is not None:
            result[f"layers.{layer_idx}.input_layernorm.weight"] = input_ln

        if self.config.norm_moe:
            post_attn_ln = self._get_weight(
                f"model.layers.{layer_idx}.post_attention_layernorms.{self.expert_index}.weight"
            )
        else:
            post_attn_ln = self._get_weight(
                f"model.layers.{layer_idx}.post_attention_layernorm.weight"
            )
        if post_attn_ln is not None:
            result[f"layers.{layer_idx}.post_attention_layernorm.weight"] = post_attn_ln

        return result

    def _copy_embeddings_and_norms(self) -> Dict[str, torch.Tensor]:
        """Copy embedding and final norm weights."""
        result = {}

        # Embeddings
        embed = self._get_weight("model.embed_tokens.weight")
        print("embed.shape: ", embed.shape)
        if embed is not None:
            result["embed_tokens.weight"] = embed

        # Final norm
        if self.config.norm_moe:
            norm = self._get_weight(f"model.norms.{self.expert_index}.weight")
        else:
            norm = self._get_weight("model.norm.weight")
        print("norm.shape: ", norm.shape)
        if norm is not None:
            result["norm.weight"] = norm

        # LM head (if present)
        lm_head = self._get_weight("lm_head.weight")
        if lm_head is not None:
            result["lm_head.weight"] = lm_head

        return result

    def convert(self) -> Dict[str, torch.Tensor]:
        """
        Convert all weights to the target format.

        Returns:
            Dictionary with converted weights in target format
        """
        converted_weights = {}
        if self.verbose:
            print("\n" + "=" * 60)
            print("Starting weight conversion...")
            print("=" * 60)

        # Convert each layer
        for layer_idx in range(self.num_layers):
            if self.verbose:
                print(f"\nProcessing layer {layer_idx}...")

            # Merge QKV
            qkv_weights = self._merge_qkv(layer_idx)
            converted_weights.update(qkv_weights)

            # Merge gate_up
            gate_up_weights = self._merge_gate_up(layer_idx)
            converted_weights.update(gate_up_weights)

            # Copy other weights
            other_weights = self._copy_other_weights(layer_idx)
            converted_weights.update(other_weights)

        # Copy embeddings and final norms
        if self.verbose:
            print("\nProcessing embeddings and final norms...")
        embed_weights = self._copy_embeddings_and_norms()
        converted_weights.update(embed_weights)

        if self.verbose:
            print("\n" + "=" * 60)
            print(f"Conversion complete! Total weights: {len(converted_weights)}")
            print("=" * 60)

        return converted_weights

    def save(
        self,
        output_path: str,
        converted_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Save converted weights to file.

        Args:
            output_path: Path to save converted weights
            converted_weights: Pre-converted weights (if None, will run conversion)
        """
        if converted_weights is None:
            converted_weights = self.convert()

        output_path = Path(output_path)

        if output_path.suffix == ".safetensors":
            save_file(converted_weights, str(output_path))
        else:
            torch.save(converted_weights, str(output_path))

        if self.verbose:
            print(f"\nSaved converted weights to: {output_path}")
            print(f"File size: {output_path.stat().st_size / (1024**3):.2f} GB")
