"""Artifact and result dataclasses used by the Trinity package facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple


@dataclass
class FrontendArtifacts:
    """Materialized outputs of the frontend stage."""

    ir_path: Path
    shapes_path: Path
    source_ir_path: Path
    source_shapes_path: Path
    tensor_shapes: Dict[str, Tuple[int, ...]]
    errors: list[str] = field(default_factory=list)


@dataclass
class OptimizerArtifacts:
    """Materialized outputs of the optimizer stage."""

    expressions_path: Path
    source_expressions_path: Path
    semi_expressions_path: Optional[Path] = None
    source_semi_expressions_path: Optional[Path] = None
    expression_count: int = 0
    stdout_log_path: Optional[Path] = None
    stderr_log_path: Optional[Path] = None


@dataclass
class BackendArtifacts:
    """Materialized outputs of the backend stage."""

    benchmark_path: Optional[Path]
    source_benchmark_path: Optional[Path]
    all_results: list[dict[str, Any]] = field(default_factory=list)
    best_kernel_path: Optional[Path] = None
    best_kernel: Optional[Callable] = None
    best_execution_time_ms: float = float("inf")
    best_ir_id: int = -1
    best_ir_expression: str = ""


@dataclass
class OptimizeResult:
    """Result returned by trinity.optimize()."""

    kernel: Optional[Callable]
    kernel_path: Optional[Path]
    execution_time_ms: float
    ir_id: int
    ir_expression: str
    all_results: list[dict[str, Any]] = field(default_factory=list)
    workspace: Optional[Path] = None
    frontend: Optional[FrontendArtifacts] = None
    optimizer: Optional[OptimizerArtifacts] = None
    backend: Optional[BackendArtifacts] = None

    def print_summary(self) -> None:
        """Print a summary of the optimization result."""
        print()
        print("=" * 50)
        print("Trinity Optimization Result")
        print("=" * 50)
        print(f"Workspace: {self.workspace}")
        print()

        if self.frontend is not None:
            print("[Frontend]")
            print(f"  IR:     {self.frontend.ir_path}")
            print(f"  Shapes: {self.frontend.shapes_path}")
            print(f"  Errors: {len(self.frontend.errors)}")
            print()

        if self.optimizer is not None:
            print("[Optimizer]")
            print(f"  Expressions: {self.optimizer.expressions_path}")
            print(f"  Count:       {self.optimizer.expression_count}")
            print()

        if self.backend is not None:
            valid = len([r for r in self.all_results if r.get("error") is None])
            print("[Backend]")
            print(f"  Benchmark: {self.backend.benchmark_path}")
            print(f"  Kernels:   {valid}/{len(self.all_results)} valid")
            print()

        print("[Best Kernel]")
        print(f"  Path:      {self.kernel_path}")
        print(f"  Time:      {self.execution_time_ms:.4f} ms")
        print(f"  IR ID:     {self.ir_id}")
        print("=" * 50)
