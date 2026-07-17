"""Public entrypoints for Triton code generation."""

from .TritonGen import TritonCodeGen
from .convert_module import convert_ir_to_triton

__all__ = ["TritonCodeGen", "convert_ir_to_triton"]
