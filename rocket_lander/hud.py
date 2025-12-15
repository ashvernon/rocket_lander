"""
HUD and explainability overlays.

Responsibilities:
- Panels + line charts (speed, reward, fuel, distance-to-pad, etc.)
- Outcome history chart
- Action indicator icons / text
- Q-value bar visualizer
- Vector overlays (velocity arrow, target arrow)
- Landing funnel overlay
- Badge / streak visuals
"""
import math
import pygame

from . import config as C


def panel_rect(y, height, width=None, padding=None):
    """Helper to create a panel-relative rect respecting padding."""

    pad = C.PANEL_PADDING if padding is None else padding
    w = (C.PANEL_WIDTH if width is None else width) - 2 * pad
    return pygame.Rect(pad, y, w, height)


def draw_panel(screen, rect, title, font):
    pygame.draw.rect(screen, (20, 20, 20), rect)
    pygame.draw.rect(screen, (80, 80, 80), rect, 1)
    t = font.render(title, True, (255, 255, 255))
    screen.blit(t, (rect.x + 6, rect.y + 4))


def draw_line_chart(screen, rect, data, vmin, vmax, label, font):
    draw_panel(screen, rect, label, font)
    if len(data) < 2:
        return

    pad_top, pad = 22, 6
    x0, y0 = rect.x + pad, rect.y + pad_top
    w, h = rect.w - 2 * pad, rect.h - pad_top - pad

    def norm(v):
        if vmax == vmin:
            return 0.5
        v = max(min(v, vmax), vmin)
        return (v - vmin) / (vmax - vmin)

    pts = []
    for i, v in enumerate(data):
        x = x0 + (i / (len(data) - 1)) * w
        y = y0 + (1 - norm(v)) * h
        pts.append((x, y))

    pygame.draw.lines(screen, (255, 255, 255), False, pts, 2)

    txt = font.render(f"{data[-1]:.2f}", True, (200, 200, 200))
    screen.blit(txt, (rect.right - txt.get_width() - 6, rect.y + 4))


def draw_outcome_chart(screen, rect, outcomes, font):
    draw_panel(screen, rect, "Outcomes (last 100)", font)
    if not outcomes:
        return

    pad_top, pad = 22, 6
    x0, y0 = rect.x + pad, rect.y + pad_top
    w, h = rect.w - 2 * pad, rect.h - pad_top - pad
    n = len(outcomes)
    bar_w = max(1, int(w / n))

    for i, o in enumerate(outcomes):
        x = x0 + i * bar_w
        if o == 2:
            y = y0
            bh = h
        elif o == 1:
            y = y0 + int(h * 0.35)
            bh = int(h * 0.65)
        else:
            y = y0 + int(h * 0.80)
            bh = int(h * 0.20)
        pygame.draw.rect(screen, (255, 255, 255), (x, y, bar_w - 1, bh), 0)

    perfect = sum(1 for o in outcomes if o == 2)
    ok = sum(1 for o in outcomes if o == 1)
    succ = perfect + ok
    total = len(outcomes)
    succ_rate = succ / total
    perf_rate = perfect / total

    txt = font.render(f"S:{succ_rate*100:.1f}%  P:{perf_rate*100:.1f}%", True, (255, 255, 255))
    screen.blit(txt, (rect.right - txt.get_width() - 6, rect.y + 4))


def draw_action_badge(screen, rect, action_name, font):
    # a bold “current action” badge
    pygame.draw.rect(screen, (20, 20, 20), rect)
    pygame.draw.rect(screen, (80, 80, 80), rect, 1)
    label = font.render("Action", True, (200, 200, 200))
    screen.blit(label, (rect.x + 6, rect.y + 4))

    big = pygame.font.SysFont(None, 28)
    txt = big.render(action_name, True, (255, 255, 255))
    screen.blit(txt, (rect.centerx - txt.get_width() // 2, rect.y + 20))


def draw_q_bars(screen, rect, q_values, chosen_idx, font):
    """
    q_values: list/tuple of floats length 4
    chosen_idx: index of chosen action
    """
    pygame.draw.rect(screen, (20, 20, 20), rect)
    pygame.draw.rect(screen, (80, 80, 80), rect, 1)

    title = font.render("Q-values", True, (255, 255, 255))
    screen.blit(title, (rect.x + 6, rect.y + 4))

    if q_values is None:
        return

    pad_top = 22
    pad = 6
    x0, y0 = rect.x + pad, rect.y + pad_top
    w, h = rect.w - 2 * pad, rect.h - pad_top - pad

    qmin = min(q_values)
    qmax = max(q_values)
    span = (qmax - qmin) if (qmax != qmin) else 1.0

    bar_h = (h - 6) // 4
    for i, q in enumerate(q_values):
        t = (q - qmin) / span
        bw = int(max(2, t * w))

        y = y0 + i * bar_h
        # highlight chosen bar by drawing a thicker outline
        pygame.draw.rect(screen, (255, 255, 255), (x0, y + 2, bw, bar_h - 6), 0)
        if i == chosen_idx:
            pygame.draw.rect(screen, (255, 255, 255), (x0, y + 1, w, bar_h - 4), 1)

        qtxt = font.render(f"{q:+.2f}", True, (200, 200, 200))
        screen.blit(qtxt, (rect.right - qtxt.get_width() - 6, y + 2))


def _scaled_vector(vec, scale, max_len):
    vx, vy = vec
    mag = math.hypot(vx, vy)
    if mag < 1e-5:
        return None

    length = min(mag * scale, max_len)
    ux, uy = vx / mag, vy / mag
    return ux * length, uy * length


def _draw_arrow(screen, start, vec, color=(255, 255, 255), width=2):
    sx, sy = vec
    mag = math.hypot(sx, sy)
    if mag < 1e-5:
        return

    end = (start[0] + sx, start[1] + sy)
    pygame.draw.line(screen, color, start, end, width)

    ux, uy = sx / mag, sy / mag
    head_len = 12
    head_w = 8
    base_x = end[0] - ux * head_len
    base_y = end[1] - uy * head_len
    perp_x, perp_y = -uy, ux
    left = (base_x + perp_x * head_w / 2, base_y + perp_y * head_w / 2)
    right = (base_x - perp_x * head_w / 2, base_y - perp_y * head_w / 2)

    pygame.draw.polygon(screen, color, [end, left, right])


def draw_velocity_vector(screen, position, velocity, scale, max_len):
    scaled = _scaled_vector(velocity, scale, max_len)
    if scaled is None:
        return

    _draw_arrow(screen, position, scaled, color=(0, 200, 255), width=3)


def draw_target_vector(screen, position, target, scale, max_len):
    dx = target[0] - position[0]
    dy = target[1] - position[1]
    scaled = _scaled_vector((dx, dy), scale, max_len)
    if scaled is None:
        return

    _draw_arrow(screen, position, scaled, color=(120, 255, 120), width=3)


def draw_ghost_path(screen, points):
    """Draw the best-run path as a faint polyline."""

    if not points or len(points) < 2:
        return

    surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    color = (150, 200, 255, 120)
    for i in range(1, len(points)):
        pygame.draw.line(surface, color, points[i - 1], points[i], 2)

    screen.blit(surface, (0, 0))


def draw_ghost_rocket(screen, frame):
    """Draw a simple translucent rocket at the ghost frame's position."""

    if frame is None:
        return

    surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    body = [(0, -12), (7, 9), (3, 11), (-3, 11), (-7, 9)]

    ca, sa = math.cos(frame.angle), math.sin(frame.angle)

    def rotate(pt):
        x, y = pt
        return (x * ca - y * sa, x * sa + y * ca)

    pts = [(frame.x + rx, frame.y + ry) for rx, ry in map(rotate, body)]
    pygame.draw.polygon(surface, (180, 220, 255, 160), pts)
    pygame.draw.polygon(surface, (80, 120, 160, 200), pts, 1)

    screen.blit(surface, (0, 0))

