"""Focused pipeline helpers assembled into the public Pipeline component."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .entrypoint import PipelineEntryPoint
from .seq_kernels import SequentialKernelPipeline
from .single_kernel import SingleKernelPipeline
from .wrapper import WrapperPipeline

if TYPE_CHECKING:
    from ..state import CodeGenState
    from ...TritonGen import TritonCodeGen


class Pipeline(
    PipelineEntryPoint,
    SequentialKernelPipeline,
    SingleKernelPipeline,
    WrapperPipeline,
):
    """Public pipeline component used by the composition-based generator."""

    def __init__(self, state: CodeGenState, gen: TritonCodeGen):
        self.state = state
        self.gen = gen


__all__ = ["Pipeline"]
