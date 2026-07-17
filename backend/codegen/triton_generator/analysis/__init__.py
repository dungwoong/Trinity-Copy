"""Focused analysis helpers assembled into the public Analyzer component."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .accumulators import AccumulatorAnalysis
from .allocations import AllocationPlanner
from .dependencies import DependencyAnalysis
from .tensor_usage import TensorUsageAnalysis

if TYPE_CHECKING:
    from ..state import CodeGenState
    from ...TritonGen import TritonCodeGen


class Analyzer(
    TensorUsageAnalysis,
    DependencyAnalysis,
    AccumulatorAnalysis,
    AllocationPlanner,
):
    """Public analysis component used by the composition-based generator."""

    def __init__(self, state: CodeGenState, gen: TritonCodeGen):
        self.state = state
        self.gen = gen


__all__ = ["Analyzer"]
