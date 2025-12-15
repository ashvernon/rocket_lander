import math
import random
from collections import deque

import pygame
import torch
import torch.optim as optim

from . import config as C
from .agent import DQN, set_torch_stability
from .terrain import Terrain
from .lander import Lander
from .replay_buffer import ReplayBuffer
from .trainer import Trainer
from .checkpoint import load_checkpoint, save_checkpoint
from .hud import draw_line_chart, draw_outcome_chart, draw_q_bars, draw_action_badge
from .effects import Effects


def run():
    set_torch_stability()

    pygame.init()
    screen = pygame.display.set_mode((C.WIDTH, C.HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)

    policy_net = DQN()
    target_net = DQN()
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=C.LR)

    buffer = ReplayBuffer(C.MEMORY_SIZE)
    trainer = Trainer(policy_net, target_net, optimizer, buffer)

    epsilon = C.EPSILON_START
    best_success_streak = 0
    success_streak = 0

    ckpt = load_checkpoint(policy_net, target_net, optimizer)
    if ckpt:
        epsilon = min(epsilon, float(ckpt.get("epsilon", epsilon)))
        best_success_streak = int(ckpt.get("best_success_streak", 0))
        print(f"Loaded model from {C.MODEL_PATH} | ep={ckpt.get('episode')} | best_streak={best_success_streak}")
        note = ckpt.get("note")
        if note:
            print(f"  note: {note}")

    speed_hist = deque(maxlen=C.HUD_N)
    reward_hist = deque(maxlen=C.HUD_N)
    fuel_hist = deque(maxlen=C.HUD_N)
    outcome_hist = deque(maxlen=C.EP_N)

    terrain = Terrain()
    lander = Lander()
    effects = Effects(trail_len=160)

    for episode in range(C.EPISODES):
        lander.reset()
        terrain.reset()
        effects.reset_episode()

        speed_hist.clear()
        reward_hist.clear()
        fuel_hist.clear()

        # Showcase episode: every N episodes, run deterministic (ε = 0)
        is_showcase = (C.SHOWCASE_EVERY > 0 and episode > 0 and (episode % C.SHOWCASE_EVERY == 0))
        eps_for_ep = 0.0 if is_showcase else epsilon

        steps = 0
        while lander.alive and not lander.landed:
            steps += 1
            if is_showcase and steps > C.SHOWCASE_STEPS_CAP:
                # safety cap
                lander.alive = False
                lander.outcome = 0
                break

            clock.tick(C.FPS)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

            state = lander.state()

            # ε-greedy (or deterministic in showcase)
            if random.random() < eps_for_ep:
                a = random.randrange(C.ACTION_SIZE)
                q_vals = None
            else:
                with torch.no_grad():
                    qs = policy_net(torch.from_numpy(state))
                    a = int(qs.argmax().item())
                    q_vals = qs.detach().cpu().numpy().tolist()

            action = C.ACTIONS[a]
            lander.step(action, terrain)

            # emit effects (based on action + rocket position)
            if C.SHOW_TRAIL:
                effects.add_trail_point(lander.x, lander.y)

            if C.SHOW_PARTICLES:
                if action == "MAIN":
                    effects.emit_main(lander.x, lander.y + 16, lander.angle, lander.vx, lander.vy)
                elif action == "LEFT":
                    effects.emit_side(lander.x - 10, lander.y, side=-1, base_vx=lander.vx, base_vy=lander.vy)
                elif action == "RIGHT":
                    effects.emit_side(lander.x + 10, lander.y, side=+1, base_vx=lander.vx, base_vy=lander.vy)

            r = lander.reward()

            # pad proximity shaping
            pad_center = (terrain.pad_x1 + terrain.pad_x2) / 2
            dist_to_pad = abs(lander.x - pad_center) / (C.WIDTH / 2)
            r += 0.25 * (1 - min(1.0, dist_to_pad))

            ns = lander.state()
            done = int(lander.landed or (not lander.alive))

            # train only if not showcase (keeps showcase “clean”)
            if not is_showcase:
                buffer.push(state, a, r, ns, done)
                trainer.train_step()

            speed_hist.append(math.hypot(lander.vx, lander.vy))
            reward_hist.append(r)
            fuel_hist.append(lander.fuel / C.MAX_FUEL)

            effects.update(dt=1.0)

            # render
            screen.fill((0, 0, 0))
            terrain.draw(screen)
            effects.draw(screen)
            lander.draw(screen)

            if not lander.alive and not lander.landed:
                warn = font.render("OUT OF BOUNDS", True, (255, 80, 80))
                screen.blit(warn, (C.WIDTH // 2 - warn.get_width() // 2, C.HEIGHT // 2))

            base_x = C.WIDTH - C.HUD_PANEL_X_OFFSET
            draw_line_chart(screen, pygame.Rect(base_x, 40, C.HUD_PANEL_W, 80), list(speed_hist), 0, 5, "Speed", font)
            draw_line_chart(screen, pygame.Rect(base_x, 130, C.HUD_PANEL_W, 80), list(reward_hist), -3, 3, "Reward", font)
            draw_line_chart(screen, pygame.Rect(base_x, 220, C.HUD_PANEL_W, 80), list(fuel_hist), 0, 1, "Fuel", font)
            draw_outcome_chart(screen, pygame.Rect(base_x, 310, C.HUD_PANEL_W, 90), list(outcome_hist), font)

            # fun HUD extras
            if C.SHOW_ACTION_BADGE:
                draw_action_badge(
                    screen,
                    pygame.Rect(base_x, 410, C.HUD_PANEL_W, 60),
                    action,
                    font,
                )

            if C.SHOW_Q_BARS:
                draw_q_bars(
                    screen,
                    pygame.Rect(base_x, 475, C.HUD_PANEL_W, 100),
                    q_vals,
                    chosen_idx=a,
                    font=font,
                )

            # status + showcase banner
            status = f"Ep {episode}  ε {eps_for_ep:.2f}  vx {lander.vx:.2f}  vy {lander.vy:.2f}  ang {lander.angle:.2f}"
            txt = font.render(status, True, (255, 255, 255))
            screen.blit(txt, (10, 10))

            if is_showcase:
                badge = font.render("SHOWCASE (ε=0)", True, (255, 255, 255))
                screen.blit(badge, (10, 32))
                # gold-ish border
                pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(3, 3, C.WIDTH - 6, C.HEIGHT - 6), 2)

            pygame.display.flip()

        # episode end
        outcome_hist.append(lander.outcome)

        # epsilon decay + target sync (skip for showcase)
        if not is_showcase:
            epsilon = max(C.EPSILON_MIN, epsilon * C.EPSILON_DECAY)
            if episode % 10 == 0:
                trainer.sync_target()

        is_ok = (lander.outcome == 1)
        is_perfect = (lander.outcome == 2)
        is_success = (is_ok and C.SAVE_ON_OK) or (is_perfect and C.SAVE_ON_PERFECT)

        if is_success:
            success_streak += 1
        else:
            success_streak = 0

        if success_streak > best_success_streak and success_streak >= C.MIN_SUCCESSES_TO_SAVE:
            best_success_streak = success_streak
            note = f"Saved on streak={best_success_streak} (outcome={lander.outcome})"
            try:
                save_checkpoint(
                    policy_net, target_net, optimizer,
                    episode=episode,
                    epsilon=epsilon,
                    best_success_streak=best_success_streak,
                    note=note
                )
                print(f"✅ Saved model to {C.MODEL_PATH} | best_success_streak={best_success_streak} | ep={episode}")
            except Exception as ex:
                print(f"⚠️ Failed to save checkpoint to {C.MODEL_PATH}: {ex}")

    pygame.quit()
