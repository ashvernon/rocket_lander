"""
Rocket (lander) physics, state representation, reward shaping, and drawing.

Responsibilities:
- Maintain rocket state (x, y, vx, vy, angle, angular_v, fuel)
- Apply action -> physics step
- Detect collisions and outcomes (FAIL / OK / PERFECT)
- Provide state vector for the RL agent
- Render rocket (and optional debug gizmos)
"""
import math
import random
import pygame
import numpy as np

from . import config as C
from .terrain import Terrain


class Lander:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = C.WIDTH / 2
        self.y = 80
        self.vx = random.uniform(-1, 1)
        self.vy = 0.0
        self.angle = 0.0
        self.angular_v = 0.0
        self.fuel = C.MAX_FUEL

        self.alive = True
        self.landed = False
        self.outcome = 0  # 0 fail, 1 ok, 2 perfect
        self.last_action = "NONE"

    def state(self) -> np.ndarray:
        return np.array(
            [
                self.x / C.WIDTH,
                self.y / C.HEIGHT,
                self.vx / 5,
                self.vy / 5,
                self.angle / math.pi,
                self.angular_v,
            ],
            dtype=np.float32,
        )

    def step(self, action: str, terrain: Terrain):
        self.last_action = action
        if self.fuel <= 0:
            action = "NONE"

        if action == "MAIN":
            self.vx -= math.sin(self.angle) * C.MAIN_THRUST
            self.vy -= math.cos(self.angle) * C.MAIN_THRUST
            self.fuel -= 1
        elif action == "LEFT":
            self.angular_v -= C.SIDE_THRUST
            self.fuel -= 1
        elif action == "RIGHT":
            self.angular_v += C.SIDE_THRUST
            self.fuel -= 1

        self.vy += C.GRAVITY
        self.x += self.vx
        self.y += self.vy
        self.angle += self.angular_v

        if (
            self.x < -C.OUT_OF_BOUNDS_MARGIN
            or self.x > C.WIDTH + C.OUT_OF_BOUNDS_MARGIN
            or self.y < -C.OUT_OF_BOUNDS_MARGIN
        ):
            self.alive = False
            self.outcome = 0
            return

        ground_y = terrain.height_at(self.x)
        if self.y >= ground_y:
            self.y = ground_y

            if terrain.on_pad(self.x):
                avx, avy, aang = abs(self.vx), abs(self.vy), abs(self.angle)

                if avy < C.PERFECT_VY and avx < C.PERFECT_VX and aang < C.PERFECT_ANGLE:
                    self.landed = True
                    self.outcome = 2
                elif avy < C.OK_VY and avx < C.OK_VX and aang < C.OK_ANGLE:
                    self.landed = True
                    self.outcome = 1
                else:
                    self.alive = False
                    self.outcome = 0
            else:
                self.alive = False
                self.outcome = 0

    def reward(self) -> float:
        if not self.alive:
            return C.OUT_OF_BOUNDS_PENALTY
        if self.landed:
            return 200.0 if self.outcome == 2 else 150.0

        return (
            -abs(self.vy) * 0.5
            -abs(self.vx) * 0.2
            -abs(self.angle) * 0.2
        )

    def draw(self, screen):
        rocket = [(0, -14), (8, 10), (4, 12), (-4, 12), (-8, 10)]

        def rotate(p):
            x, y = p
            ca, sa = math.cos(self.angle), math.sin(self.angle)
            return (x * ca - y * sa, x * sa + y * ca)

        pts = [(self.x + rx, self.y + ry) for rx, ry in map(rotate, rocket)]
        pygame.draw.polygon(screen, (220, 220, 220), pts)
        pygame.draw.polygon(screen, (80, 80, 80), pts, 2)

        if self.last_action == "MAIN":
            flame = [(0, 14), (3, 26), (-3, 26)]
            fpts = [(self.x + rx, self.y + ry) for rx, ry in map(rotate, flame)]
            pygame.draw.polygon(screen, (255, 140, 0), fpts)

        if self.last_action == "LEFT":
            side = [(-10, 0), (-22, 4), (-22, -4)]
            spts = [(self.x + rx, self.y + ry) for rx, ry in map(rotate, side)]
            pygame.draw.polygon(screen, (0, 200, 255), spts)

        if self.last_action == "RIGHT":
            side = [(10, 0), (22, 4), (22, -4)]
            spts = [(self.x + rx, self.y + ry) for rx, ry in map(rotate, side)]
            pygame.draw.polygon(screen, (0, 200, 255), spts)
