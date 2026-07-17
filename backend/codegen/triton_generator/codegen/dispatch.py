"""
AST dispatch helpers.

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


class Dispatch:
    def __init__(self, state: CodeGenState, gen: TritonCodeGen) -> None:
        self.state = state
        self.gen = gen

    def generate_node(self, node: ASTNode) -> str:
        """Generate code for a single AST node"""
        if node.node_type == NodeType.PLOOP:
            return self.gen.loops.generate_ploop(node)
        elif node.node_type == NodeType.SLOOP:
            return self.gen.loops.generate_sloop(node)
        elif node.node_type == NodeType.SEQ:
            # If we encounter a nested seq, just generate its children inline
            # (This shouldn't happen at the top level since we handle that separately)
            return self.gen.loops.generate_seq(node)
        elif node.node_type == NodeType.LOAD:
            return self.gen.memory.generate_load(node)
        elif node.node_type == NodeType.STORE:
            return self.gen.memory.generate_store(node)
        elif node.node_type == NodeType.ADD:
            return self.gen.scalar_ops.generate_binary_op(node, "+")
        elif node.node_type == NodeType.SUB:
            return self.gen.scalar_ops.generate_binary_op(node, "-")
        elif node.node_type == NodeType.MUL:
            return self.gen.scalar_ops.generate_binary_op(node, "*")
        elif node.node_type == NodeType.DIV:
            return self.gen.scalar_ops.generate_binary_op(node, "/")
        elif node.node_type == NodeType.LE:
            return self.gen.scalar_ops.generate_binary_op(node, "<=")
        elif node.node_type == NodeType.MAX:
            return self.gen.scalar_ops.generate_binary_func(node, "tl.maximum")
        elif node.node_type == NodeType.MIN:
            return self.gen.scalar_ops.generate_binary_func(node, "tl.minimum")
        elif node.node_type == NodeType.MATMUL:
            return self.gen.matmul.generate_matmul(node)
        elif node.node_type == NodeType.EXP:
            return self.gen.scalar_ops.generate_unary_op(node, "tl.exp")
        elif node.node_type == NodeType.SQR:
            return self.gen.scalar_ops.generate_binary_op(node, "*")
        elif node.node_type == NodeType.SQRT:
            return self.gen.scalar_ops.generate_unary_op(node, "tl.sqrt")
        elif node.node_type == NodeType.SIGMOID:
            return self.gen.scalar_ops.generate_unary_op(node, "tl.sigmoid")
        elif node.node_type == NodeType.ERF:
            return self.gen.scalar_ops.generate_unary_op(node, "tl.math.erf")
        elif node.node_type == NodeType.CAST:
            return self.gen.scalar_ops.generate_cast(node)
        elif node.node_type == NodeType.ABS:
            return self.gen.scalar_ops.generate_unary_op(node, "tl.abs")
        elif node.node_type == NodeType.RSUM:
            return self.gen.scalar_ops.generate_reduce_sum(node)
        elif node.node_type == NodeType.RMAX:
            return self.gen.scalar_ops.generate_reduce_max(node)
        elif node.node_type == NodeType.RMIN:
            return self.gen.scalar_ops.generate_reduce_min(node)
        elif node.node_type == NodeType.BCAST:
            return self.gen.scalar_ops.generate_broadcast(node)
        elif node.node_type == NodeType.CONCAT:
            return self.generate_concat(node)
        elif node.node_type == NodeType.INPUT:
            return self.generate_input(node)
        elif node.node_type == NodeType.OUTPUT:
            return self.generate_output(node)
        elif node.node_type == NodeType.TENSOR:
            return self.generate_tensor(node)
        elif node.node_type == NodeType.NUM:
            return str(node.value)
        elif node.node_type == NodeType.VAR:
            return node.value
        elif node.node_type == NodeType.DUMMY:
            # Dummy node is a no-op placeholder
            indent = '    ' * self.state.indent_level
            return f"{indent}pass  # dummy node"
        elif node.node_type == NodeType.PERMUTE3:
            return self.gen.transforms.generate_permute3(node)
        elif node.node_type == NodeType.TRANSPOSE:
            return self.gen.transforms.generate_transpose(node)
        elif node.node_type == NodeType.SQUEEZE:
            return self.gen.transforms.generate_squeeze(node)
        elif node.node_type == NodeType.UNSQUEEZE:
            return self.gen.transforms.generate_unsqueeze(node)
        else:
            return f"# TODO: Implement {node.node_type.value}"

    def generate_concat(self, node: ASTNode) -> str:
        """Generate concatenation operation

        The concat node has at least 3 children:
        1. First tensor to concatenate
        2. Second tensor to concatenate
        3. Axis along which to concatenate

        In the IR expressions:
        - (concat (load X ...) (load X ...) 1) means concatenate along axis 1
        - (concat (* A B) (load W ...) 0) means concatenate along axis 0

        Since Triton doesn't have a direct concat operation, we handle this
        by recognizing the pattern where concat is used in matmul operations
        and transforming it into separate computations.
        """
        if len(node.children) < 3:
            raise ValueError("concat requires at least 3 arguments: tensor1, tensor2, axis")

        # Get the axis
        axis = int(self.gen.dispatch.generate_node(node.children[-1]))

        # For the LoRA patterns, concat is used in a specific way:
        # 1. Two concat operations are used in a matmul
        # 2. This represents a block matrix multiplication

        # Store concat information in the node for parent operations to handle
        node.is_concat = True
        node.concat_axis = axis
        node.concat_children = node.children[:-1]  # All children except the axis

        # Return a special marker that _generate_matmul can recognize
        # This is a workaround since concat needs special handling in matmul context
        return "__CONCAT_NODE__"

    def generate_input(self, node: ASTNode) -> str:
        """Generate code for input tensor reference"""
        # (input X) -> just return the tensor name
        if node.children:
            return node.children[0].value
        return "input"

    def generate_output(self, node: ASTNode) -> str:
        """Generate code for output tensor reference"""
        # (output O) -> just return the tensor name
        if node.children:
            return node.children[0].value
        return "output"

    def generate_tensor(self, node: ASTNode) -> str:
        """Generate code for intermediate tensor reference"""
        # (tensor C) -> just return the tensor name
        if node.children:
            tensor_name = node.children[0].value
            # Mark this as an intermediate tensor
            self.state.intermediate_tensors.add(tensor_name)
            return tensor_name
        return "tensor"
