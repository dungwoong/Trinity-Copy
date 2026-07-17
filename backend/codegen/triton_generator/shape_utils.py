"""Shape and constant resolution helpers for the Triton code generator."""

from __future__ import annotations

from typing import Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..state import CodeGenState
    from ...TritonGen import TritonCodeGen


class ShapeUtils:
    def __init__(self, state: CodeGenState, gen: TritonCodeGen) -> None:
        self.state = state
        self.gen = gen

    def next_power_of_2(self, n):
        """Round up to the next power of 2"""
        if n <= 0:
            return 1
        # Check if already power of 2
        if (n & (n - 1)) == 0:
            return n
        # Find the next power of 2
        power = 1
        while power < n:
            power <<= 1
        return power

    def get_padded_block_size(self, size_expr):
        """Get padded block size and whether padding was applied"""
        # Try to evaluate if it's a constant
        try:
            if isinstance(size_expr, str) and size_expr in self.state.constants:
                size_val = self.state.constants[size_expr]
            elif isinstance(size_expr, str) and size_expr.isdigit():
                size_val = int(size_expr)
            elif isinstance(size_expr, (int, float)):
                size_val = int(size_expr)
            else:
                # If it's a dynamic expression, return as is
                return size_expr, False

            padded_size = self.next_power_of_2(size_val)
            return padded_size, padded_size != size_val
        except:
            # If evaluation fails, return original
            return size_expr, False

    def get_padded_shape(self, shape_values):
        padded_parts = []
        padded_values = []
        for dim in shape_values:
            padded_dim, _ = self.get_padded_block_size(dim)
            padded_parts.append(str(padded_dim))
            padded_values.append(padded_dim)
        return padded_parts, padded_values

    def resolve_value(self, value) -> str:
        """Resolve a value that might be a variable name or expression to its constant value.

        Args:
            value: Could be a string, int, or other type

        Returns:
            The resolved value as a string
        """
        # If it's already a number (int or float), return as string
        if isinstance(value, (int, float)):
            return str(value)

        # Convert to string if not already
        value = str(value)

        # If it's already a number string, return as is
        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            return value

        # Check if it's a simple variable in our constants dictionary
        if value in self.state.constants:
            return str(self.state.constants[value])

        # Check if it contains an expression (e.g., P+M, N-1, etc.)
        if any(op in value for op in ['+', '-', '*', '//']):
            # Try to evaluate the expression by replacing variables with their values
            expr = value
            for var_name, var_value in self.state.constants.items():
                # Replace whole words only (to avoid replacing 'M' in 'MAX')
                import re
                expr = re.sub(r'\b' + var_name + r'\b', str(var_value), expr)

            result = eval(expr, {"__builtins__": {}}, {})
            return str(result)

        # Otherwise return the original value
        return value

    def resolve_tensor_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Resolve symbolic tensor shapes to concrete values using constants.

        Returns:
            Dictionary mapping tensor names to resolved shape tuples
        """
        resolved_shapes = {}

        for tensor_name, shape in self.state.tensor_shapes.items():
            resolved_shape = []
            for dim in shape:
                if isinstance(dim, str):
                    # It's a symbolic dimension or expression, resolve it
                    resolved_dim = self.resolve_value(dim)
                    try:
                        # Convert to integer
                        resolved_shape.append(int(resolved_dim))
                    except ValueError:
                        raise ValueError(f"Could not resolve dimension '{dim}' to an integer value")
                else:
                    # It's already a number
                    resolved_shape.append(dim)
            resolved_shapes[tensor_name] = tuple(resolved_shape)

        return resolved_shapes
