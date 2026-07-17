"""Internal stage wrappers used by the Trinity package facade."""

from .backend import BackendStage
from .frontend import FrontendStage
from .optimizer import OptimizerStage

__all__ = ["BackendStage", "FrontendStage", "OptimizerStage"]
