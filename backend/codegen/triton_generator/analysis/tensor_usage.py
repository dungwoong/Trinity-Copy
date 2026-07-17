"""Tensor collection and tensor-usage analysis helpers."""

from __future__ import annotations

from typing import List

from ...AstNode import ASTNode
from ...NodeType import NodeType


class TensorUsageAnalysis:
    def collect_tensors(self, node: ASTNode):
        """Collect all tensor names used in the AST."""
        if node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
            # Handle multiple tensor names (comma-separated).
            for child in node.children:
                if child.node_type == NodeType.VAR:
                    tensor_name = child.value
                    self.state.tensors_used.add(tensor_name)
                    if node.node_type == NodeType.INPUT:
                        self.state.input_tensors.add(tensor_name)
                    elif node.node_type == NodeType.OUTPUT:
                        self.state.output_tensors.add(tensor_name)

        for child in node.children:
            if isinstance(child, ASTNode):
                self.collect_tensors(child)

    def collect_intermediate_tensors(self, node: ASTNode, in_ploop=False, ploop_var=None):
        """Collect intermediate tensors (tensor operators used in store operations)."""
        if node.node_type == NodeType.LOAD:
            tensor_node = node.children[0] if node.children else None
            if tensor_node and tensor_node.node_type == NodeType.TENSOR:
                for child in tensor_node.children:
                    if child.node_type == NodeType.VAR:
                        tensor_name = child.value
                        if len(node.children) >= 2:
                            index_node = node.children[1]
                            if index_node.node_type == NodeType.INDEX:
                                if tensor_name not in self.state.intermediate_tensor_indices:
                                    self.state.intermediate_tensor_indices[tensor_name] = index_node
        elif node.node_type == NodeType.STORE:
            tensor_node = node.children[0]
            if tensor_node.node_type == NodeType.TENSOR:
                for child in tensor_node.children:
                    if child.node_type == NodeType.VAR:
                        tensor_name = child.value
                        self.state.intermediate_tensors.add(tensor_name)

                        if tensor_name not in self.state.tensor_shapes and len(node.children) >= 3:
                            index_node = node.children[2]
                            shape = self.gen.indexer.infer_tensor_shape_from_index(index_node)
                            if shape:
                                self.state.tensor_shapes[tensor_name] = shape

                        if len(node.children) >= 3:
                            index_node = node.children[2]
                            if index_node.node_type == NodeType.INDEX:
                                if not hasattr(self.state, "intermediate_tensor_indices"):
                                    self.state.intermediate_tensor_indices = {}
                                self.state.intermediate_tensor_indices[tensor_name] = index_node
            elif tensor_node.node_type == NodeType.OUTPUT:
                for child in tensor_node.children:
                    if child.node_type == NodeType.VAR:
                        tensor_name = child.value
                        if len(node.children) >= 3:
                            index_node = node.children[2]
                            if index_node.node_type == NodeType.INDEX:
                                if not hasattr(self.state, "intermediate_tensor_indices"):
                                    self.state.intermediate_tensor_indices = {}
                                self.state.intermediate_tensor_indices[tensor_name] = index_node

        if node.node_type == NodeType.PLOOP:
            loop_var = node.children[3].value if len(node.children) > 3 else None
            for child in node.children:
                if isinstance(child, ASTNode):
                    self.collect_intermediate_tensors(child, in_ploop=True, ploop_var=loop_var)
        else:
            for child in node.children:
                if isinstance(child, ASTNode):
                    self.collect_intermediate_tensors(child, in_ploop, ploop_var)

    def identify_cross_kernel_tensors(self, seq_node: ASTNode):
        """Identify tensors that cross kernel boundaries in seq operations."""
        if seq_node.node_type != NodeType.SEQ:
            return

        operations = self.gen.pipeline.flatten_seq(seq_node)

        tensor_writes = {}
        tensor_reads = {}

        for kernel_id, op in enumerate(operations):
            writes_in_kernel = set()
            reads_in_kernel = set()
            self.collect_tensor_usage(op, writes_in_kernel, reads_in_kernel)

            for tensor in writes_in_kernel:
                tensor_writes[tensor] = kernel_id

            for tensor in reads_in_kernel:
                if tensor not in tensor_reads:
                    tensor_reads[tensor] = set()
                tensor_reads[tensor].add(kernel_id)

        for tensor, write_kernel in tensor_writes.items():
            if tensor in tensor_reads:
                for read_kernel in tensor_reads[tensor]:
                    if write_kernel != read_kernel:
                        self.state.cross_kernel_tensors.add(tensor)
                        break

    def collect_tensor_usage(self, node: ASTNode, writes: set, reads: set):
        """Collect tensor reads and writes in a node."""
        if node.node_type == NodeType.STORE:
            tensor_node = node.children[0]
            if tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
                for child in tensor_node.children:
                    if child.node_type == NodeType.VAR:
                        writes.add(child.value)
            if len(node.children) > 1:
                self.collect_tensor_usage(node.children[1], writes, reads)
        elif node.node_type == NodeType.LOAD:
            tensor_node = node.children[0]
            if tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
                for child in tensor_node.children:
                    if child.node_type == NodeType.VAR:
                        reads.add(child.value)
        else:
            for child in node.children:
                if isinstance(child, ASTNode):
                    self.collect_tensor_usage(child, writes, reads)

    def analyze_parallel_store_dims(self, node: ASTNode, current_ploop_var: str = None):
        """Analyze which tensors are stored in parallel loops and track their dimensions."""
        if node.node_type == NodeType.PLOOP:
            loop_var = node.children[3].value
            body = node.children[4]
            self.analyze_parallel_store_dims(body, loop_var)
        elif node.node_type == NodeType.STORE and current_ploop_var:
            tensor_node = node.children[0]
            if tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
                for child in tensor_node.children:
                    if child.node_type == NodeType.VAR:
                        tensor_name = child.value
                        if len(node.children) >= 3:
                            index_node = node.children[2]
                            dims_using_ploop = self.find_dims_using_loop_var(index_node, current_ploop_var)

                            if current_ploop_var not in self.state.stored_tensor_dims:
                                self.state.stored_tensor_dims[current_ploop_var] = []

                            for dim in dims_using_ploop:
                                self.state.stored_tensor_dims[current_ploop_var].append((tensor_name, dim))
        else:
            for child in node.children:
                if isinstance(child, ASTNode):
                    self.analyze_parallel_store_dims(child, current_ploop_var)

    def find_dims_using_loop_var(
        self,
        index_node: ASTNode,
        loop_var: str,
        current_dim: int = 0,
    ) -> List[int]:
        """Find which dimensions in an index expression use the given loop variable."""
        dims = []

        if index_node.node_type == NodeType.INDEX:
            for i, child in enumerate(index_node.children):
                if self.index_uses_loop_var(child, loop_var):
                    dims.append(i)

        return dims

    def index_uses_loop_var(self, node: ASTNode, loop_var: str) -> bool:
        """Check if an index expression uses the given loop variable."""
        if node.node_type == NodeType.TILE:
            if node.children and node.children[0].value == loop_var:
                return True
        elif node.node_type == NodeType.ELEM:
            if node.children and node.children[0].value == loop_var:
                return True

        for child in node.children:
            if isinstance(child, ASTNode) and self.index_uses_loop_var(child, loop_var):
                return True

        return False
