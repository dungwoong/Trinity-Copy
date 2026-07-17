"""Optimizer wrapper that keeps the original Rust optimizer untouched."""

from __future__ import annotations

import os
import subprocess

from ..artifacts import FrontendArtifacts, OptimizerArtifacts
from ..config import OptimizeConfig
from ..runtime import OPTIMIZER_DIR, count_expressions, log
from ..workspace import PipelineWorkspace


class OptimizerStage:
    def run(
        self,
        basename: str,
        frontend: FrontendArtifacts,
        config: OptimizeConfig,
        workspace: PipelineWorkspace,
    ) -> OptimizerArtifacts:
        log("Optimizer: running equality saturation ...", config.verbose)

        source_ir_path = frontend.ir_path.resolve()
        source_shapes_path = frontend.shapes_path.resolve()
        source_expressions_path = (
            OPTIMIZER_DIR / "expressions" / f"{basename}_cost{config.cost}_kern{config.kern}.txt"
        )
        source_semi_path = (
            OPTIMIZER_DIR
            / "expressions"
            / "semi"
            / f"{basename}_cost{config.cost}_kern{config.kern}.json"
        )

        if not source_ir_path.exists() or not source_shapes_path.exists():
            raise FileNotFoundError(
                f"Expected frontend artifacts at {source_ir_path.parent}, but they were not found."
            )

        build_tmp_dir = OPTIMIZER_DIR / ".tmp"
        build_tmp_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["TMPDIR"] = str(build_tmp_dir)

        result = subprocess.run(
            [
                "cargo",
                "run",
                "--release",
                "--bin",
                "trinity_opt",
                "--",
                "--ir",
                str(source_ir_path),
                "--shapes",
                str(source_shapes_path),
                "--semi-output",
                str(source_semi_path),
                "--output",
                str(source_expressions_path),
                "--cost",
                str(config.cost),
                "--kern",
                str(config.kern),
                "--iter",
                str(config.optimizer_iter_limit),
            ],
            cwd=str(OPTIMIZER_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=config.optimizer_timeout_s,
        )

        stdout_log = workspace.write_text(workspace.logs_dir, "optimizer.stdout.log", result.stdout)
        stderr_log = workspace.write_text(workspace.logs_dir, "optimizer.stderr.log", result.stderr)

        if result.returncode != 0:
            err_msg = result.stderr or result.stdout
            raise RuntimeError(f"Optimizer failed (exit {result.returncode}):\n{err_msg}")

        artifacts = self.load_existing(basename, config, workspace)
        artifacts.stdout_log_path = stdout_log
        artifacts.stderr_log_path = stderr_log
        return artifacts

    def load_existing(
        self,
        basename: str,
        config: OptimizeConfig,
        workspace: PipelineWorkspace,
    ) -> OptimizerArtifacts:
        source_expressions_path = (
            OPTIMIZER_DIR / "expressions" / f"{basename}_cost{config.cost}_kern{config.kern}.txt"
        )
        source_semi_path = (
            OPTIMIZER_DIR
            / "expressions"
            / "semi"
            / f"{basename}_cost{config.cost}_kern{config.kern}.json"
        )

        if not source_expressions_path.exists():
            raise FileNotFoundError(
                f"Expected optimizer expressions at {source_expressions_path}, but they were not found."
            )

        expressions_path = workspace.mirror_file(source_expressions_path, workspace.optimizer_dir)
        semi_path = None
        if source_semi_path.exists():
            semi_path = workspace.mirror_file(source_semi_path, workspace.optimizer_dir)

        expression_count = count_expressions(expressions_path)
        if expression_count == 0:
            raise RuntimeError(
                f"Optimizer generated 0 expressions. Try increasing cost (current: {config.cost})."
            )

        return OptimizerArtifacts(
            expressions_path=expressions_path,
            source_expressions_path=source_expressions_path,
            semi_expressions_path=semi_path,
            source_semi_expressions_path=source_semi_path if source_semi_path.exists() else None,
            expression_count=expression_count,
        )
