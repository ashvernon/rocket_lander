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
import pygame


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

