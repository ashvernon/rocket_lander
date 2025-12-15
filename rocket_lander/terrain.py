"""
Terrain generation and landing pad logic.

Responsibilities:
- Generate piecewise-linear terrain
- Define landing pad bounds + height
- Provide height_at(x), on_pad(x)
- Render terrain/pad
"""
import random
import pygame

from . import config as C


class Terrain:
    def __init__(self):
        self.points = []
        self.pad_x1 = 0
        self.pad_x2 = 0
        self.pad_y = C.HEIGHT - 80
        self.reset()

    def reset(self):
        pad_w = random.randint(120, 200)
        pad_center = random.randint(200, C.WIDTH - 200)
        self.pad_x1 = pad_center - pad_w // 2
        self.pad_x2 = pad_center + pad_w // 2

        self.points = []
        x = 0
        last_y = C.HEIGHT - random.randint(70, 140)

        while x < self.pad_x1:
            y = C.HEIGHT - random.randint(70, 160)
            self.points.append((x, y))
            x += random.randint(60, 120)
            last_y = y

        self.pad_y = max(min(last_y, C.HEIGHT - 70), C.HEIGHT - 220)

        self.points.append((self.pad_x1, self.pad_y))
        self.points.append((self.pad_x2, self.pad_y))

        x = self.pad_x2
        while x < C.WIDTH:
            y = C.HEIGHT - random.randint(70, 160)
            self.points.append((x, y))
            x += random.randint(60, 120)

        if self.points[-1][0] != C.WIDTH:
            self.points.append((C.WIDTH, C.HEIGHT - random.randint(70, 160)))

        self.points = sorted(self.points, key=lambda p: p[0])

    def height_at(self, x: float) -> float:
        for i in range(len(self.points) - 1):
            x1, y1 = self.points[i]
            x2, y2 = self.points[i + 1]
            if x1 <= x <= x2:
                if x2 == x1:
                    return y1
                t = (x - x1) / (x2 - x1)
                return y1 * (1 - t) + y2 * t
        return C.HEIGHT

    def on_pad(self, x: float) -> bool:
        return self.pad_x1 <= x <= self.pad_x2

    def draw(self, screen):
        pygame.draw.lines(screen, (120, 120, 120), False, self.points, 3)
        pygame.draw.line(
            screen,
            (255, 255, 255),
            (self.pad_x1, self.pad_y),
            (self.pad_x2, self.pad_y),
            6,
        )
