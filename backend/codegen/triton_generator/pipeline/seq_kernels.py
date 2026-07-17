"""Sequential-kernel orchestration helpers."""

from __future__ import annotations

from ...AstNode import ASTNode
from ...NodeType import NodeType


class SequentialKernelPipeline:
    def has_only_dummy(self, node: ASTNode) -> bool:
        """Check if a node contains only dummy operations."""
        if node.node_type == NodeType.DUMMY:
            return True
        if node.node_type == NodeType.SEQ:
            return all(self.has_only_dummy(child) for child in node.children)
        if node.node_type in [NodeType.PLOOP, NodeType.SLOOP]:
            if len(node.children) > 4:
                return self.has_only_dummy(node.children[4])
            return False
        return False

    def flatten_seq(self, node: ASTNode):
        """Recursively flatten nested seq nodes into a list of operations."""
        if node.node_type == NodeType.SEQ:
            operations = []
            for child in node.children:
                operations.extend(self.flatten_seq(child))
            return operations
        if node.node_type == NodeType.DUMMY:
            return []
        if self.has_only_dummy(node):
            return []
        return [node]

    def generate_seq_kernels(self, seq_node: ASTNode) -> str:
        """Generate separate kernels for each operation in a (possibly nested) seq."""
        all_code = ""

        all_code += """import triton
import triton.language as tl
import torch
"""

        operations = self.flatten_seq(seq_node)
        all_local_intermediate_tensors = set()

        for i, op in enumerate(operations):
            kernel_name = f"kernel_{i}"

            self.state.intermediate_tensors = set()
            self.state.tensors_used = set()
            self.state.kernel_accumulators = set()
            self.state.stored_accumulators = set()

            self.gen.analyzer.collect_intermediate_tensors(op, in_ploop=False, ploop_var=None)

            cross_sloop_memory_tensors = self.gen.analyzer.identify_cross_sloop_memory_tensors(op)
            accumulators = self.gen.analyzer.identify_accumulators(op)
            cross_sloop_tensors = self.gen.analyzer.identify_cross_sloop_tensors(op)

            non_cross_sloop_accumulators = accumulators - cross_sloop_tensors
            cross_sloop_memory_tensors -= non_cross_sloop_accumulators

            self.state.intermediate_tensors -= cross_sloop_memory_tensors
            self.state.tensors_used.update(cross_sloop_memory_tensors)
            self.state.cross_sloop_memory_tensors = cross_sloop_memory_tensors

            local_intermediate_tensors = (
                self.state.intermediate_tensors
                - self.state.cross_kernel_tensors
                - cross_sloop_memory_tensors
            )
            all_local_intermediate_tensors.update(local_intermediate_tensors)

            kernel_code = self.generate_single_kernel(op, kernel_name=kernel_name)

            self.state.generated_kernels.append(
                (
                    kernel_name,
                    kernel_code,
                    self.state.tensors_used.copy(),
                    self.state.parallel_dims.copy(),
                    self.state.all_loops.copy(),
                    self.state.stored_tensor_dims.copy(),
                    self.state.intermediate_tensors.copy(),
                    cross_sloop_memory_tensors.copy(),
                )
            )

            kernel_block_params = []
            for loop_var, _, _, tile_size in self.state.all_loops:
                if isinstance(tile_size, str) and not tile_size.isdigit():
                    kernel_block_params.append(f"BLOCK_{loop_var.upper()}")

            if kernel_block_params:
                all_code += "\n" + self.gen.kernel.generate_autotune_decorator(kernel_block_params) + "\n"

            kernel_code = kernel_code.replace("import triton\nimport triton.language as tl\nimport torch\n\n", "")
            all_code += kernel_code + "\n\n"

        all_code += self.generate_wrapper_function(all_local_intermediate_tensors)

        return all_code
