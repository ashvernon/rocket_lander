"""
Best-run recorder + replay playback + ghost trajectory overlay.

Responsibilities:
- Record per-step state history during an episode
- Maintain best run (by success + stability score)
- Slow-mo replay renderer
- Ghost overlay of best path
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import config as C


@dataclass
class ReplayFrame:
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    angular_v: float
    fuel: float
    action: str
    pad_x1: float
    pad_x2: float
    pad_y: float


@dataclass
class RecordedRun:
    outcome: int
    stability: float
    steps: int
    frames: List[ReplayFrame] = field(default_factory=list)
    terrain_points: List[tuple] = field(default_factory=list)


class RunRecorder:
    """Collects step-by-step state for the current episode and keeps the best run."""

    def __init__(self):
        self.current_frames: List[ReplayFrame] = []
        self.current_terrain: List[tuple] = []
        self.best_run: Optional[RecordedRun] = None

    def start_episode(self, terrain) -> None:
        self.current_frames = []
        self.current_terrain = list(terrain.points)

    def record_step(self, lander, terrain, action: str) -> None:
        frame = ReplayFrame(
            x=lander.x,
            y=lander.y,
            vx=lander.vx,
            vy=lander.vy,
            angle=lander.angle,
            angular_v=lander.angular_v,
            fuel=lander.fuel,
            action=action,
            pad_x1=terrain.pad_x1,
            pad_x2=terrain.pad_x2,
            pad_y=terrain.pad_y,
        )
        self.current_frames.append(frame)

    def end_episode(self, lander, steps: int) -> None:
        if not self.current_frames:
            return

        stability = self._stability_score(lander)
        run = RecordedRun(
            outcome=lander.outcome,
            stability=stability,
            steps=steps,
            frames=list(self.current_frames),
            terrain_points=list(self.current_terrain),
        )

        if self._is_better(run, self.best_run):
            self.best_run = run

    def _stability_score(self, lander) -> float:
        """Lower is better. Combine touchdown stability with fuel left."""
        fuel_term = 1 - (lander.fuel / C.MAX_FUEL)
        return abs(lander.vx) + abs(lander.vy) + abs(lander.angle) + fuel_term

    def _is_better(self, candidate: RecordedRun, incumbent: Optional[RecordedRun]) -> bool:
        if incumbent is None:
            return True

        if candidate.outcome != incumbent.outcome:
            return candidate.outcome > incumbent.outcome

        if candidate.stability != incumbent.stability:
            return candidate.stability < incumbent.stability

        return candidate.steps < incumbent.steps


class ReplayPlayer:
    """Plays back the best recorded run in slow motion."""

    def __init__(self, slowmo_factor: int = 4):
        self.slowmo_factor = max(1, slowmo_factor)
        self.active = False
        self._frames: List[ReplayFrame] = []
        self._terrain_points: List[tuple] = []
        self._frame_idx = 0
        self._slowmo_counter = 0

    def start(self, run: Optional[RecordedRun]) -> bool:
        if run is None:
            return False

        self._frames = list(run.frames)
        self._terrain_points = list(run.terrain_points)
        self._frame_idx = 0
        self._slowmo_counter = 0
        self.active = bool(self._frames)
        return self.active

    def stop(self) -> None:
        self.active = False

    def step(self, lander, terrain) -> Optional[ReplayFrame]:
        if not self.active:
            return None

        if self._frame_idx >= len(self._frames):
            self.active = False
            return None

        frame = self._frames[self._frame_idx]
        self._apply_frame(lander, terrain, frame)

        self._slowmo_counter += 1
        if self._slowmo_counter >= self.slowmo_factor:
            self._slowmo_counter = 0
            self._frame_idx += 1

        if self._frame_idx >= len(self._frames):
            # allow last frame to be displayed once more before stopping next tick
            self.active = False

        return frame

    def _apply_frame(self, lander, terrain, frame: ReplayFrame) -> None:
        terrain.points = list(self._terrain_points)
        terrain.pad_x1 = frame.pad_x1
        terrain.pad_x2 = frame.pad_x2
        terrain.pad_y = frame.pad_y

        lander.x = frame.x
        lander.y = frame.y
        lander.vx = frame.vx
        lander.vy = frame.vy
        lander.angle = frame.angle
        lander.angular_v = frame.angular_v
        lander.fuel = frame.fuel
        lander.last_action = frame.action
