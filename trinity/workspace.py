"""Workspace management for Trinity optimization runs."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from .config import OptimizeConfig


class PipelineWorkspace:
    """Owns package-facing artifacts copied from the internal engines."""

    def __init__(self, root: Path, basename: str):
        self.root = root
        self.basename = basename
        self.frontend_dir = self.root / "frontend"
        self.optimizer_dir = self.root / "optimizer"
        self.backend_dir = self.root / "backend"
        self.kernels_dir = self.root / "kernels"
        self.logs_dir = self.root / "logs"

        for directory in (
            self.root,
            self.frontend_dir,
            self.optimizer_dir,
            self.backend_dir,
            self.kernels_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, output_dir: Path, basename: str) -> "PipelineWorkspace":
        root = (output_dir / basename).expanduser().resolve()
        return cls(root=root, basename=basename)

    def mirror_file(self, source: Path, stage_dir: Path, target_name: Optional[str] = None) -> Path:
        destination = stage_dir / (target_name or source.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination

    def write_text(self, stage_dir: Path, name: str, text: str) -> Path:
        destination = stage_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
        return destination

    def write_json(self, stage_dir: Path, name: str, data: Any) -> Path:
        destination = stage_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(data, indent=2, sort_keys=True))
        return destination

    def write_config(self, config: OptimizeConfig) -> Path:
        payload = asdict(config)
        payload["output_dir"] = str(config.output_path)
        return self.write_json(self.root, "config.json", payload)
