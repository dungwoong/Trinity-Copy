"""Kernel signature and autotune helpers for Triton emission."""

from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..state import CodeGenState
    from ...TritonGen import TritonCodeGen


class KernelGenerator:
    def __init__(self, state: CodeGenState, gen: TritonCodeGen) -> None:
        self.state = state
        self.gen = gen

    def generate_header(self, kernel_name: str = "kernel") -> str:
        # Generate Triton kernel function header with only stride parameters (no shape)
        params = []
        for name in sorted(self.state.tensors_used):
            # Skip intermediate tensors that are NOT cross-kernel tensors and NOT cross-sloop memory tensors
            if name in self.state.intermediate_tensors and name not in self.state.cross_kernel_tensors and name not in self.state.cross_sloop_memory_tensors:
                continue

            # pointer
            params.append(f"{name}_ptr")

            # Determine number of dimensions based on tensor_shapes or parallel_dims
            num_dims = 2  # default
            if self.state.tensor_shapes and name in self.state.tensor_shapes:
                shape = self.state.tensor_shapes[name]
                # Count the actual dimensions (could be symbolic)
                num_dims = len(shape)
            else:
                # Infer from parallel dimensions
                num_dims = max(len(self.state.parallel_dims), 1)

            # NO shape parameters - will use constants instead

            # Stride parameters (still needed as they're runtime values)
            for dim in range(num_dims):
                params.append(f"{name}_stride{dim}: tl.constexpr")

        # NO shape parameters for local intermediate tensors either

        # block sizes for all loops (parallel and sequential)
        for loop_var, start, end, tile_size in self.state.all_loops:
            params.append(f"BLOCK_{loop_var.upper()}: tl.constexpr")

        # Add constants as parameters
        for const_name in sorted(self.state.constants.keys()):
            params.append(f"{const_name}: tl.constexpr")

        params_str = ",\n    ".join(params)
        header = f"""import triton
import triton.language as tl
import torch

@triton.jit
def {kernel_name}(
    {params_str}
):
"""
        return header

    def generate_autotune_decorator(self, block_params: list) -> str:
        """Generate autotune decorator with configurations based on actual block parameters"""
        if not block_params:
            return ""

        # Generate power-of-2 values for each parameter
        values = [32, 64, 128]

        # Generate configurations
        configs = []

        if len(block_params) == 1:
            # Single parameter
            for val in values:
                config_dict = {block_params[0]: val}
                config_str = "{" + ", ".join([f"'{k}': {v}" for k, v in config_dict.items()]) + "}"
                configs.append(f"        triton.Config({config_str})")
        elif len(block_params) == 2:
            # Two parameters - generate combinations
            for val1 in values:
                for val2 in values:
                    config_dict = {block_params[0]: val1, block_params[1]: val2}
                    # Format as string with quotes around keys
                    config_str = "{" + ", ".join([f"'{k}': {v}" for k, v in config_dict.items()]) + "}"
                    configs.append(f"        triton.Config({config_str})")
        else:
            # More than 2 parameters - generate a reasonable subset of combinations
            # Start with all parameters having the same value
            for val in values:
                config_dict = {param: val for param in block_params}
                config_str = "{" + ", ".join([f"'{k}': {v}" for k, v in config_dict.items()]) + "}"
                configs.append(f"        triton.Config({config_str})")

            # Add some mixed configurations
            # Generate a few more strategic combinations
            if len(block_params) >= 3:
                # Add configurations where first param is different
                for val1 in [32, 64]:
                    for val2 in [64, 128]:
                        config_dict = {block_params[0]: val1}
                        for i in range(1, len(block_params)):
                            config_dict[block_params[i]] = val2
                        config_str = "{" + ", ".join([f"'{k}': {v}" for k, v in config_dict.items()]) + "}"
                        config = f"        triton.Config({config_str})"
                        if config not in configs:
                            configs.append(config)

        # Limit to reasonable number of configs
        if len(configs) > 16:
            configs = configs[:16]

        decorator = "@triton.autotune(\n    configs = [\n"
        decorator += ",\n".join(configs)
        decorator += "\n    ], key=[]\n)"

        return decorator
