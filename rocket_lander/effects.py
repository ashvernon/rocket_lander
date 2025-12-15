"""
Visual effects (NO SOUND):

Responsibilities:
- Particle systems (main flame, smoke, side jets)
- Trails / path rendering
- Screen shake, crash flash
- Lightweight update + draw hooks
"""
import random
import pygame


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "ttl", "radius")

    def __init__(self, x, y, vx, vy, ttl, radius):
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.life = float(ttl)
        self.ttl = float(ttl)
        self.radius = float(radius)


class Effects:
    """
    No-sound visual effects:
    - Flame + smoke particles
    - Side jet particles
    - Trail of rocket positions
    """

    def __init__(self, trail_len=140):
        self.particles = []
        self.trail = []
        self.trail_len = int(trail_len)

    def reset_episode(self):
        self.particles.clear()
        self.trail.clear()

    def update(self, dt=1.0):
        # particles
        alive = []
        for p in self.particles:
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.life -= dt
            # little "air resistance" feel
            p.vx *= 0.99
            p.vy *= 0.99
            if p.life > 0:
                alive.append(p)
        self.particles = alive

        # trail fades implicitly by length
        if len(self.trail) > self.trail_len:
            self.trail = self.trail[-self.trail_len :]

    def add_trail_point(self, x, y):
        self.trail.append((float(x), float(y)))

    def emit_main(self, x, y, angle, base_vx=0.0, base_vy=0.0):
        # Flame (brighter, faster)
        for _ in range(6):
            spd = random.uniform(1.6, 3.2)
            jitter = random.uniform(-0.35, 0.35)
            vx = -spd * (random.uniform(-0.1, 0.1)) + base_vx * 0.05
            vy = spd * (1.0 + random.uniform(-0.15, 0.15)) + base_vy * 0.05
            # local "down" relative to rocket orientation
            # flame goes opposite thrust (downwards from rocket)
            ca, sa = pygame.math.Vector2(0, 0), None  # placeholder to avoid vector imports
            # rotate velocity by angle
            rvx = vx * (1.0)  # keep some randomness
            rvy = vy * (1.0)
            # rotate emission direction roughly aligned with rocket "down"
            rvx += (random.uniform(-0.5, 0.5))
            rvy += (random.uniform(0.2, 0.8))
            self.particles.append(Particle(x, y, rvx, rvy, ttl=random.uniform(10, 18), radius=random.uniform(1.2, 2.6)))

        # Smoke (larger, slower)
        for _ in range(3):
            spd = random.uniform(0.6, 1.4)
            vx = random.uniform(-0.6, 0.6) + base_vx * 0.03
            vy = random.uniform(0.6, 1.6) + base_vy * 0.03
            self.particles.append(Particle(x, y, vx, vy, ttl=random.uniform(18, 30), radius=random.uniform(2.5, 4.5)))

    def emit_side(self, x, y, side, base_vx=0.0, base_vy=0.0):
        # side is -1 for LEFT thruster firing, +1 for RIGHT firing
        for _ in range(4):
            vx = side * random.uniform(1.2, 2.4) + random.uniform(-0.3, 0.3) + base_vx * 0.05
            vy = random.uniform(-0.4, 0.4) + base_vy * 0.05
            self.particles.append(Particle(x, y, vx, vy, ttl=random.uniform(10, 16), radius=random.uniform(1.0, 2.0)))

    def draw(self, screen):
        # Trail first (behind rocket)
        if len(self.trail) >= 2:
            # draw faint polyline by segments (no per-vertex alpha required)
            for i in range(1, len(self.trail)):
                x1, y1 = self.trail[i - 1]
                x2, y2 = self.trail[i]
                # fade by age
                t = i / max(1, len(self.trail) - 1)
                c = int(60 + 120 * t)  # brighter near the end
                pygame.draw.line(screen, (c, c, c), (x1, y1), (x2, y2), 1)

        # Particles
        for p in self.particles:
            t = max(0.0, min(1.0, p.life / p.ttl))
            c = int(80 + 175 * t)
            r = max(1, int(p.radius * (0.6 + 0.8 * t)))
            pygame.draw.circle(screen, (c, c, c), (int(p.x), int(p.y)), r)
