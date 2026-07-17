"""
Index and shape inference helpers.

Extracted from control_flow.py to keep responsibilities smaller and isolated.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple, TYPE_CHECKING

from ...AstNode import ASTNode
from ...NodeType import NodeType

if TYPE_CHECKING:
    from ...state import CodeGenState
    from ....TritonGen import TritonCodeGen


class Indexer:
    def __init__(self, state: CodeGenState, gen: TritonCodeGen) -> None:
        self.state = state
        self.gen = gen

    def generate_index(self, node: ASTNode, tensor_name: str) -> str:
        """Generate index calculation for memory access"""
        # (index tile1 tile2 ...)
        offsets = []

        for i, child in enumerate(node.children):
            if child.node_type == NodeType.TILE:
                loop_var = child.children[0].value
                if loop_var in self.state.loop_vars:
                    start, end, tile_size, is_parallel = self.state.loop_vars[loop_var]

                    # Use BLOCK_SIZE parameter instead of hardcoded tile_size
                    block_param = f"BLOCK_{loop_var.upper()}"

                    # For both parallel and sequential loops, create tile range
                    offset_expr = f"{loop_var} + tl.arange(0, {block_param})"

                    offsets.append(offset_expr)
                else:
                    # Not a loop variable, use a single index (0)
                    offsets.append("0")
            elif child.node_type == NodeType.FULLTILE:
                # For fulltile, we need to determine the appropriate dimension
                # This represents the full dimension being accessed
                # Get the dimension from tensor_shapes
                if tensor_name in self.state.tensor_shapes:
                    shape = self.state.tensor_shapes[tensor_name]
                    if i < len(shape):
                        dim_value = shape[i]
                        if isinstance(dim_value, str):
                            # It's a symbolic dimension, use it directly
                            fulltile_dim = dim_value
                        else:
                            # It's a literal number
                            fulltile_dim = str(dim_value)
                    else:
                        # Fallback to parameter name
                        fulltile_dim = f"{tensor_name}_dim{i}"
                else:
                    # Fallback to parameter name
                    fulltile_dim = f"{tensor_name}_dim{i}"
                # Use padded size for fulltile dimensions
                padded_dim, needs_padding = self.gen.shape_utils.get_padded_block_size(fulltile_dim)
                offsets.append(f"tl.arange(0, {padded_dim})")
            elif child.node_type == NodeType.ELEM:
                # For elem indexing
                # (elem n) means use n as a scalar index
                if child.children:
                    elem_var = child.children[0].value
                    # Always check the actual dimension size in tensor shape
                    dim_size = None
                    if tensor_name in self.state.tensor_shapes:
                        shape = self.state.tensor_shapes[tensor_name]
                        if i < len(shape):
                            dim_value = shape[i]
                            # Resolve the dimension value
                            dim_size = self.gen.shape_utils.resolve_value(dim_value)
                            if isinstance(dim_size, str) and dim_size.isdigit():
                                dim_size = int(dim_size)

                    # ELEM always needs tl.arange(0, 1) to maintain tensor dimensionality
                    # Each kernel instance processes only one element in this dimension
                    block_param = f"BLOCK_{elem_var.upper()}"
                    offsets.append(f"(({elem_var} // {block_param})+tl.arange(0, 1))")
            elif child.node_type == NodeType.CONST_TILE:
                # For const_tile, create a range from start to start+size
                # (const_tile start size) means [start:start+size]
                if len(child.children) >= 2:
                    start = self.gen.dispatch.generate_node(child.children[0])
                    size = self.gen.dispatch.generate_node(child.children[1])
                    # Use padded size for range dimensions
                    padded_size, needs_padding = self.gen.shape_utils.get_padded_block_size(size)
                    offsets.append(f"{start} + tl.arange(0, {padded_size})")

        # Generate multi-dimensional indexing
        if len(offsets) == 1:
            # For 1D tensors, still need to multiply by stride
            offset_str = f"({offsets[0]}) * {tensor_name}_stride0"
        else:
            # Multi-dimensional indexing: i[:,None]*tensor_stride0 + j[None,:]*tensor_stride1
            offset_parts = []

            # Check which dimensions are scalar (elem) vs array (tile/fulltile)
            is_scalar = []
            for off in offsets:
                # Check if offset contains tl.arange (array) or is just a scalar expression
                is_scalar.append("tl.arange" not in off)

            for dim, off in enumerate(offsets):
                stride_term = f" * {tensor_name}_stride{dim}"

                if is_scalar[dim]:
                    # Scalar index - no broadcasting needed
                    offset_parts.append(f"({off}){stride_term}")
                else:
                    # Array index - need broadcasting
                    # Calculate how many None dimensions to add
                    nones = []
                    for i in range(len(offsets)):
                        if i == dim:
                            nones.append(":")
                        else:
                            nones.append("None")

                    # Count non-scalar dimensions
                    array_dims = sum(1 for i, s in enumerate(is_scalar) if not s)

                    if array_dims == 1:
                        # Only one array dimension, no broadcasting needed
                        offset_parts.append(f"({off}){stride_term}")
                    else:
                        # Need broadcasting for multiple array dimensions
                        broadcast_expr = "[" + ", ".join(nones) + "]"
                        offset_parts.append(f"({off}){broadcast_expr}{stride_term}")

            offset_str = " + ".join(offset_parts)

        return offset_str

    def analyze_loop_contexts(self, node: ASTNode, current_loops=None, depth=0):
        """Analyze loop contexts to determine which tensor dimensions each loop variable corresponds to"""
        if current_loops is None:
            current_loops = []

        if node.node_type in [NodeType.PLOOP, NodeType.SLOOP]:
            # Extract loop variable from 4th argument
            loop_var = node.children[3].value

            # Add current loop to the context
            new_loops = current_loops + [(loop_var, depth)]

            # Analyze body with updated context
            if len(node.children) > 4:
                self.analyze_loop_contexts(node.children[4], new_loops, depth + 1)

        elif node.node_type == NodeType.LOAD or node.node_type == NodeType.STORE:
            # Found a memory access - analyze its index pattern
            tensor_node = node.children[0]
            index_node = node.children[1] if node.node_type == NodeType.LOAD else node.children[2]

            if tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
                tensor_name = tensor_node.children[0].value

                # Analyze index to see which loop variables access which dimensions
                self._analyze_index_pattern(index_node, tensor_name, current_loops)

        # Continue analyzing children
        for child in node.children:
            if isinstance(child, ASTNode):
                self.analyze_loop_contexts(child, current_loops, depth)

    def _analyze_index_pattern(self, index_node: ASTNode, tensor_name: str, current_loops):
        """Analyze index pattern to map loop variables to tensor dimensions"""
        if index_node.node_type == NodeType.INDEX:
            # Go through each tile in the index
            for dim, tile in enumerate(index_node.children):
                if tile.node_type == NodeType.TILE and tile.children:
                    # Get the loop variable used in this tile
                    loop_var = tile.children[0].value

                    # Map this loop variable to the tensor dimension
                    if loop_var not in self.state.loop_var_to_tensor_dim:
                        self.state.loop_var_to_tensor_dim[loop_var] = (tensor_name, dim)

    def infer_tensor_shape_from_index(self, index_node: ASTNode) -> tuple:
        """Infer the shape of a tensor from its index pattern"""
        if index_node.node_type != NodeType.INDEX:
            return None

        shape = []
        for i, child in enumerate(index_node.children):
            if child.node_type == NodeType.FULLTILE:
                # fulltile represents a full dimension
                # Look for common patterns or use symbolic names
                if i == 0:
                    # First dimension often M (batch/sequence)
                    shape.append('M')
                elif i == 1:
                    # Second dimension could be various things
                    # Check if we're in a context that suggests R (rank)
                    shape.append('R')
                else:
                    shape.append(f'DIM_{i}')
            elif child.node_type == NodeType.TILE:
                # tile with a loop variable
                if child.children:
                    loop_var = child.children[0].value
                    if loop_var in self.state.loop_vars:
                        _, _, tile_size, _ = self.state.loop_vars[loop_var]
                        # Use resolved tile size
                        shape.append(self.gen.shape_utils.resolve_value(tile_size))
                    else:
                        # Not a loop variable, use 1 (single tile)
                        shape.append(1)
            elif child.node_type == NodeType.ELEM:
                # elem represents a scalar index
                shape.append(1)

        return tuple(shape) if shape else None

    def infer_block_shape_from_index(self, index_node: ASTNode, tensor_name: str) -> tuple:
        """Infer padded block shape from index pattern for block-tensor ops."""
        if index_node.node_type != NodeType.INDEX:
            return None

        shape = []
        for i, child in enumerate(index_node.children):
            if child.node_type == NodeType.FULLTILE:
                dim_value = None
                if tensor_name in self.state.tensor_shapes and i < len(self.state.tensor_shapes[tensor_name]):
                    dim_value = self.state.tensor_shapes[tensor_name][i]
                else:
                    dim_value = f"{tensor_name}_dim{i}"
                padded_dim, _ = self.gen.shape_utils.get_padded_block_size(dim_value)
                shape.append(padded_dim)
            elif child.node_type == NodeType.TILE:
                if child.children:
                    loop_var = child.children[0].value
                    block_param = f"BLOCK_{loop_var.upper()}"
                    padded_dim, _ = self.gen.shape_utils.get_padded_block_size(block_param)
                    shape.append(padded_dim)
                else:
                    shape.append(1)
            elif child.node_type == NodeType.ELEM:
                shape.append(1)
            elif child.node_type == NodeType.CONST_TILE:
                if len(child.children) >= 2:
                    size = self.gen.dispatch.generate_node(child.children[1])
                    padded_dim, _ = self.gen.shape_utils.get_padded_block_size(size)
                    shape.append(padded_dim)
        return tuple(shape) if shape else None
