"""
Metrics and reporting helpers.

Responsibilities:
- Per-episode metrics aggregation
- Rolling windows (success rates, averages)
- Stability score computation
- Export to CSV/JSON at exit
"""
from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional

from . import config as C


@dataclass
class EpisodeRecord:
    episode: int
    steps: int
    total_reward: float
    epsilon: float
    outcome: str
    fail_reason: Optional[str]
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    angular_v: float
    fuel: float
    dist_to_pad_center: float
    wall_time_sec_episode: float
    wall_time_sec_cumulative: float
    rolling_success_rate: float
    rolling_avg_reward: float
    rolling_avg_steps: float
    rolling_perfect_rate: float
    rolling_ok_rate: float
    stability_score: float


class RunMetrics:
    """Capture per-episode metrics and export them as CSV/JSON."""

    def __init__(self, run_name: Optional[str] = None, run_tag: str = "") -> None:
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_tag = run_tag
        self.records: List[EpisodeRecord] = []
        self._success_window: Deque[bool] = deque(maxlen=C.EP_N)
        self._reward_window: Deque[float] = deque(maxlen=C.EP_N)
        self._steps_window: Deque[int] = deque(maxlen=C.EP_N)
        self._perfect_window: Deque[bool] = deque(maxlen=C.EP_N)
        self._ok_window: Deque[bool] = deque(maxlen=C.EP_N)
        self._cumulative_wall_time = 0.0

    def record_episode(
        self,
        episode: int,
        *,
        lander,
        terrain,
        total_reward: float,
        steps: int,
        epsilon: float,
        wall_time_sec_episode: float,
    ) -> None:
        """Record metrics for a completed episode."""
        self._cumulative_wall_time += wall_time_sec_episode
        pad_center = (terrain.pad_x1 + terrain.pad_x2) / 2
        dist_to_pad_center = lander.x - pad_center

        stability_score = self._compute_stability_score(
            vx=lander.vx,
            vy=lander.vy,
            angle=lander.angle,
            angular_v=lander.angular_v,
            dist_to_pad_center=dist_to_pad_center,
        )

        is_ok = lander.outcome == 1
        is_perfect = lander.outcome == 2
        is_success = is_ok or is_perfect

        self._success_window.append(is_success)
        self._perfect_window.append(is_perfect)
        self._ok_window.append(is_ok)
        self._reward_window.append(total_reward)
        self._steps_window.append(steps)

        rolling_success_rate = self._mean_bool(self._success_window)
        rolling_perfect_rate = self._mean_bool(self._perfect_window)
        rolling_ok_rate = self._mean_bool(self._ok_window)
        rolling_avg_reward = self._mean_float(self._reward_window)
        rolling_avg_steps = self._mean_float(self._steps_window)

        record = EpisodeRecord(
            episode=episode,
            steps=steps,
            total_reward=total_reward,
            epsilon=epsilon,
            outcome=self._outcome_label(lander.outcome),
            fail_reason=getattr(lander, "fail_reason", None),
            x=lander.x,
            y=lander.y,
            vx=lander.vx,
            vy=lander.vy,
            angle=lander.angle,
            angular_v=lander.angular_v,
            fuel=lander.fuel,
            dist_to_pad_center=dist_to_pad_center,
            wall_time_sec_episode=wall_time_sec_episode,
            wall_time_sec_cumulative=self._cumulative_wall_time,
            rolling_success_rate=rolling_success_rate,
            rolling_avg_reward=rolling_avg_reward,
            rolling_avg_steps=rolling_avg_steps,
            rolling_perfect_rate=rolling_perfect_rate,
            rolling_ok_rate=rolling_ok_rate,
            stability_score=stability_score,
        )
        self.records.append(record)

    def export_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.records[0]).keys()))
            writer.writeheader()
            for rec in self.records:
                writer.writerow(asdict(rec))

    def export_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, object] = {
            "run_name": self.run_name,
            "run_tag": self.run_tag,
            "episodes": [asdict(r) for r in self.records],
            "episode_count": len(self.records),
            "created_at": datetime.now().isoformat(),
        }
        with path.open("w") as f:
            json.dump(payload, f, indent=2)

    def finalize_and_export(
        self,
        *,
        out_dir: Path = C.REPORTS_DIR,
        export_csv: bool = True,
        export_json: bool = True,
    ) -> None:
        if not self.records:
            return

        out_dir = Path(out_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.run_name}_{timestamp}"
        if export_csv:
            self.export_csv(out_dir / f"{base_name}.csv")
        if export_json:
            self.export_json(out_dir / f"{base_name}.json")

    @staticmethod
    def _outcome_label(outcome: int) -> str:
        if outcome == 2:
            return "perfect"
        if outcome == 1:
            return "ok"
        return "fail"

    @staticmethod
    def _mean_bool(values: Deque[bool]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _mean_float(values: Deque[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    @staticmethod
    def _normalize(value: float, scale: float) -> float:
        return min(1.0, abs(value) / scale)

    def _compute_stability_score(
        self,
        *,
        vx: float,
        vy: float,
        angle: float,
        angular_v: float,
        dist_to_pad_center: float,
    ) -> float:
        """Compute a simple stability score in the range [0, 100]."""
        vx_term = self._normalize(vx, 2.0)
        vy_term = self._normalize(vy, 2.5)
        angle_term = self._normalize(angle, 1.0)
        ang_v_term = self._normalize(angular_v, 4.0)
        dist_term = self._normalize(dist_to_pad_center, C.WIDTH / 2)

        penalty = (
            vx_term * 20
            + vy_term * 25
            + angle_term * 25
            + ang_v_term * 15
            + dist_term * 15
        )
        return max(0.0, 100.0 - penalty)
