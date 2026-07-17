"""Canonical Triton code generator entrypoint (Mediator pattern)."""

from .triton_generator.state import CodeGenState
from .triton_generator.shape_utils import ShapeUtils
from .triton_generator.analysis import Analyzer
from .triton_generator.codegen.dispatch import Dispatch
from .triton_generator.codegen.loops import LoopEmitter
from .triton_generator.codegen.indexing import Indexer
from .triton_generator.codegen.expressions import ExpressionLowerer
from .triton_generator.codegen.memory_ops import MemoryOps
from .triton_generator.codegen.math_ops import ScalarOps
from .triton_generator.codegen.matmul_ops import MatmulOps
from .triton_generator.codegen.transforms import Transforms
from .triton_generator.codegen.masking import MaskGenerator
from .triton_generator.kernel import KernelGenerator
from .triton_generator.pipeline import Pipeline


class TritonCodeGen:
    """Triton code generator assembled via composition (Mediator pattern).

    Each component receives a shared ``CodeGenState`` and a back-reference
    to this mediator so that cross-component calls are explicit::

        self.gen.dispatch.generate_node(node)
        self.state.indent_level += 1
    """

    def __init__(self):
        self.state = CodeGenState()
        self.state.gen = self  # back-reference for state helpers

        self.shape_utils = ShapeUtils(self.state, self)
        self.analyzer = Analyzer(self.state, self)
        self.dispatch = Dispatch(self.state, self)
        self.loops = LoopEmitter(self.state, self)
        self.indexer = Indexer(self.state, self)
        self.expressions = ExpressionLowerer(self.state, self)
        self.memory = MemoryOps(self.state, self)
        self.scalar_ops = ScalarOps(self.state, self)
        self.matmul = MatmulOps(self.state, self)
        self.transforms = Transforms(self.state, self)
        self.masking = MaskGenerator(self.state, self)
        self.kernel = KernelGenerator(self.state, self)
        self.pipeline = Pipeline(self.state, self)

    def generate(self, ast, tensor_shapes=None, constants=None):
        """Generate Triton kernel code from an IR AST.

        Args:
            ast: The AST to generate code from.
            tensor_shapes: Optional tensor shape information.
            constants: Optional constant values.

        Returns:
            Generated Triton kernel source code as a string.
        """
        return self.pipeline.generate(ast, tensor_shapes, constants)
