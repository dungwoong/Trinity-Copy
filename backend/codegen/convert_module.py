from typing import Dict, Tuple

from .IrParser import IRParser
from .TritonGen import TritonCodeGen


def convert_ir_to_triton(
    ir_code: str,
    tensor_shapes: Dict[str, Tuple[int, ...]] = None,
    constants: Dict[str, int] = None,
) -> str:
    """Convert IR code to Triton kernel."""
    parser = IRParser()
    ast = parser.parse(ir_code)

    codegen = TritonCodeGen()
    triton_code = codegen.generate(ast, tensor_shapes, constants)

    return triton_code
