import random
from typing import List, Tuple

import pygame


class Starfield:
    def __init__(
        self,
        width: int,
        height: int,
        count: int,
        layers: int,
        size_range: Tuple[int, int],
        parallax: bool,
        parallax_scale: float,
        reseed_each_episode: bool,
    ) -> None:
        self.width = width
        self.height = height
        self.count = count
        self.layers = max(1, layers)
        self.size_range = size_range
        self.parallax = parallax
        self.parallax_scale = parallax_scale
        self.reseed_each_episode = reseed_each_episode

        self.seed = random.randint(0, 999_999)
        self.stars: List[Tuple[float, float, int, Tuple[int, int, int], int]] = []
        self.layer_surfaces: List[pygame.Surface] = []
        self.cached_surface: pygame.Surface | None = None

        self._generate()

    def _generate(self) -> None:
        rng = random.Random(self.seed)
        self.stars.clear()

        min_size, max_size = self.size_range
        min_size = max(1, min_size)
        max_size = max(min_size, max_size)

        for _ in range(self.count):
            layer = rng.randrange(self.layers)
            brightness = 150 + int(100 * ((layer + 1) / self.layers))
            brightness = max(120, min(255, brightness + rng.randint(-20, 20)))
            size = rng.randint(min_size, max_size)
            self.stars.append(
                (
                    rng.uniform(0, self.width),
                    rng.uniform(0, self.height),
                    size,
                    (brightness, brightness, brightness),
                    layer,
                )
            )

        self.layer_surfaces = [pygame.Surface((self.width, self.height), pygame.SRCALPHA) for _ in range(self.layers)]
        for x, y, size, color, layer in self.stars:
            pygame.draw.circle(self.layer_surfaces[layer], color, (int(x), int(y)), size)

        self.cached_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for surf in self.layer_surfaces:
            self.cached_surface.blit(surf, (0, 0))

    def reset_episode(self) -> None:
        if self.reseed_each_episode:
            self.seed = random.randint(0, 999_999)
            self._generate()

    def _blit_with_wrap(self, target: pygame.Surface, layer_surface: pygame.Surface, offset_x: float, offset_y: float) -> None:
        if layer_surface.get_width() != self.width or layer_surface.get_height() != self.height:
            return

        ox = (-int(round(offset_x))) % self.width
        oy = (-int(round(offset_y))) % self.height

        for dx in (-self.width, 0):
            for dy in (-self.height, 0):
                target.blit(layer_surface, (ox + dx, oy + dy))

    def draw(self, target: pygame.Surface, vx: float = 0.0, vy: float = 0.0) -> None:
        if not self.parallax:
            if self.cached_surface:
                target.blit(self.cached_surface, (0, 0))
            return

        base_offset_x = -vx * self.parallax_scale
        base_offset_y = -vy * self.parallax_scale

        for idx, surf in enumerate(self.layer_surfaces):
            factor = (idx + 1) / self.layers
            self._blit_with_wrap(target, surf, base_offset_x * factor, base_offset_y * factor)
