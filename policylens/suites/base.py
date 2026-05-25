"""Suite — an ordered bundle of Diagnostics with a uniform run interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from policylens.core.results import DiagnosticResult
from policylens.core.types import Scene, Trajectory
from policylens.diagnostics.base import Diagnostic
from policylens.diagnostics.trajectory import (
    TrajectoryDiagnostic,
    TrajectoryDiagnosticResult,
)
from policylens.models.protocol import VLAModel


@dataclass
class SuiteResult:
    suite_name: str
    model_id: str
    scene_id: str
    results: dict[str, DiagnosticResult] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "suite": self.suite_name,
            "model": self.model_id,
            "scene": self.scene_id,
            "diagnostics": {k: r.to_summary() for k, r in self.results.items()},
        }


@dataclass
class TrajectorySuiteResult:
    """Suite result for a full trajectory run — one TrajectoryDiagnosticResult per diagnostic."""

    suite_name: str
    model_id: str
    trajectory_source: str
    n_frames: int
    results: dict[str, TrajectoryDiagnosticResult] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "suite": self.suite_name,
            "model": self.model_id,
            "trajectory_source": self.trajectory_source,
            "n_frames": self.n_frames,
            "diagnostics": {k: r.to_summary() for k, r in self.results.items()},
        }


@dataclass
class Suite:
    """A named ordered list of Diagnostics."""

    name: str
    diagnostics: list[Diagnostic] = field(default_factory=list)
    description: str = ""

    def run(self, model: VLAModel, scene: Scene) -> SuiteResult:
        out = SuiteResult(suite_name=self.name, model_id=model.model_id, scene_id=scene.scene_id)
        for d in self.diagnostics:
            out.results[d.name] = d.run(model, scene)
        return out

    def run_trajectory(
        self,
        model: VLAModel,
        trajectory: Trajectory,
        stride: int = 1,
    ) -> TrajectorySuiteResult:
        """Run every diagnostic across every frame of the trajectory.

        `stride` lets you subsample frames (e.g., stride=4 for fast prototyping).
        Each diagnostic in this suite gets wrapped with TrajectoryDiagnostic
        and run frame-by-frame.
        """
        traj = trajectory.subsample(stride) if stride > 1 else trajectory
        out = TrajectorySuiteResult(
            suite_name=self.name,
            model_id=model.model_id,
            trajectory_source=traj.source,
            n_frames=len(traj),
        )
        for d in self.diagnostics:
            td = TrajectoryDiagnostic(d)
            out.results[d.name] = td.run(model, traj)
        return out

    def applicable_diagnostics(self, model: VLAModel) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.applicable_to(model)]
