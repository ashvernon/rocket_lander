import math
import random
from collections import deque

import pygame
import torch
import torch.optim as optim

from . import config as C
from .agent import DQN, set_torch_stability
from .checkpoint import load_checkpoint, save_checkpoint
from .effects import Effects
from .hud import (
    draw_action_badge,
    draw_ghost_path,
    draw_ghost_rocket,
    draw_line_chart,
    draw_outcome_chart,
    draw_q_bars,
    panel_rect,
    draw_target_vector,
    draw_velocity_vector,
)
from .lander import Lander
from .replay import ReplayPlayer, RunRecorder
from .replay_buffer import ReplayBuffer
from .starfield import Starfield
from .terrain import Terrain
from .trainer import Trainer


def run():
    set_torch_stability()

    pygame.init()
    screen = pygame.display.set_mode((C.WINDOW_WIDTH, C.HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)
    world_surf = pygame.Surface((C.WIDTH, C.HEIGHT))
    panel_surf = pygame.Surface((C.PANEL_WIDTH, C.HEIGHT))
    starfield = Starfield(
        width=C.WIDTH,
        height=C.HEIGHT,
        count=C.STAR_COUNT,
        layers=C.STAR_LAYERS,
        size_range=C.STAR_SIZES,
        parallax=C.STAR_PARALLAX,
        parallax_scale=C.STAR_PARALLAX_SCALE,
        reseed_each_episode=C.STAR_RESEED_EACH_EPISODE,
    )

    policy_net = DQN()
    target_net = DQN()
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=C.LR)

    buffer = ReplayBuffer(C.MEMORY_SIZE)
    trainer = Trainer(policy_net, target_net, optimizer, buffer)

    epsilon = C.EPSILON_START
    episodes_since_success = 0
    reheat_cooldown_left = 0
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
    recorder = RunRecorder()
    replay_player = ReplayPlayer(C.REPLAY_SLOWMO)
    replay_notice_timer = 0

    def render_scene(action_label, q_vals, is_showcase=False, replay_mode=False, notice_time=0, step_idx=0):
        world_surf.fill((0, 0, 0))
        if C.SHOW_STARFIELD:
            starfield.draw(world_surf, vx=lander.vx, vy=lander.vy)
        panel_surf.fill((8, 8, 8))

        terrain.draw(world_surf)

        ghost_run = None
        if not replay_mode and C.SHOW_GHOST:
            ghost_run = recorder.best_run

        if ghost_run:
            if not ghost_run.ghost_points and ghost_run.frames:
                stride = max(1, len(ghost_run.frames) // C.GHOST_MAX_POINTS)
                ghost_run.ghost_points = [
                    (f.x, f.y) for f in ghost_run.frames[::stride][: C.GHOST_MAX_POINTS]
                ]

            if ghost_run.ghost_points:
                draw_ghost_path(world_surf, ghost_run.ghost_points)
                if C.SHOW_GHOST_ROCKET:
                    idx = max(0, min(len(ghost_run.frames) - 1, step_idx))
                    draw_ghost_rocket(world_surf, ghost_run.frames[idx])

        if not replay_mode:
            effects.draw(world_surf)
        lander.draw(world_surf)

        pad_center_x = (terrain.pad_x1 + terrain.pad_x2) / 2
        if C.SHOW_VELOCITY_VECTOR:
            draw_velocity_vector(
                world_surf,
                (lander.x, lander.y),
                (lander.vx, lander.vy),
                scale=C.VECTOR_SCALE,
                max_len=C.VECTOR_MAX_LEN,
            )

        if C.SHOW_TARGET_VECTOR:
            draw_target_vector(
                world_surf,
                (lander.x, lander.y),
                (pad_center_x, terrain.pad_y),
                scale=C.VECTOR_SCALE,
                max_len=C.VECTOR_MAX_LEN,
            )

        if not replay_mode and not lander.alive and not lander.landed:
            warn = font.render("OUT OF BOUNDS", True, (255, 80, 80))
            world_surf.blit(warn, (C.WIDTH // 2 - warn.get_width() // 2, C.HEIGHT // 2))

        status = f"Ep {episode}  ε {eps_for_ep:.2f}  vx {lander.vx:.2f}  vy {lander.vy:.2f}  ang {lander.angle:.2f}"
        txt = font.render(status, True, (255, 255, 255))
        panel_y = C.PANEL_PADDING
        panel_surf.blit(txt, (C.PANEL_PADDING, panel_y))
        panel_y += txt.get_height() + 10

        draw_line_chart(panel_surf, panel_rect(panel_y, 80), list(speed_hist), 0, 5, "Speed", font)
        panel_y += 90
        draw_line_chart(panel_surf, panel_rect(panel_y, 80), list(reward_hist), -3, 3, "Reward", font)
        panel_y += 90
        draw_line_chart(panel_surf, panel_rect(panel_y, 80), list(fuel_hist), 0, 1, "Fuel", font)
        panel_y += 90
        draw_outcome_chart(panel_surf, panel_rect(panel_y, 90), list(outcome_hist), font)
        panel_y += 100

        if C.SHOW_ACTION_BADGE:
            draw_action_badge(
                panel_surf,
                panel_rect(panel_y, 60),
                action_label,
                font,
            )
            panel_y += 70

        if C.SHOW_Q_BARS:
            draw_q_bars(
                panel_surf,
                panel_rect(panel_y, 100),
                q_vals,
                chosen_idx=C.ACTIONS.index(action_label) if action_label in C.ACTIONS else 0,
                font=font,
            )

        if is_showcase:
            badge = font.render("SHOWCASE (ε=0)", True, (255, 255, 255))
            world_surf.blit(badge, (10, 32))
            pygame.draw.rect(world_surf, (255, 255, 255), pygame.Rect(3, 3, C.WIDTH - 6, C.HEIGHT - 6), 2)

        if replay_mode:
            replay_badge = font.render("REPLAY (slow-mo)", True, (255, 220, 0))
            world_surf.blit(replay_badge, (10, 32))
            pygame.draw.rect(world_surf, (255, 220, 0), pygame.Rect(3, 3, C.WIDTH - 6, C.HEIGHT - 6), 2)

        if notice_time > 0:
            msg = font.render("No best run yet", True, (255, 120, 120))
            world_surf.blit(msg, (C.WIDTH // 2 - msg.get_width() // 2, 60))

        screen.blit(world_surf, (0, 0))
        screen.blit(panel_surf, (C.WIDTH, 0))
        pygame.draw.line(screen, (90, 90, 90), (C.WIDTH, 0), (C.WIDTH, C.HEIGHT), 2)

        pygame.display.flip()

    for episode in range(C.EPISODES):
        lander.reset()
        terrain.reset()
        effects.reset_episode()
        recorder.start_episode(terrain)
        if C.SHOW_STARFIELD:
            starfield.reset_episode()

        speed_hist.clear()
        reward_hist.clear()
        fuel_hist.clear()

        # Showcase episode: every N episodes, run deterministic (ε = 0)
        is_showcase = (C.SHOWCASE_EVERY > 0 and episode > 0 and (episode % C.SHOWCASE_EVERY == 0))
        eps_for_ep = 0.0 if is_showcase else epsilon

        steps = 0
        while lander.alive and not lander.landed:
            clock.tick(C.FPS)
            if replay_notice_timer > 0:
                replay_notice_timer -= 1

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE and replay_player.active:
                        replay_player.stop()
                    if e.key == pygame.K_r and C.ENABLE_REPLAY:
                        if replay_player.active:
                            replay_player.stop()
                        else:
                            started = replay_player.start(recorder.best_run)
                            if not started:
                                replay_notice_timer = 90

            if replay_player.active:
                replay_player.step(lander, terrain)
                render_scene(lander.last_action, None, replay_mode=True, notice_time=replay_notice_timer)
                continue

            steps += 1
            if is_showcase and steps > C.SHOWCASE_STEPS_CAP:
                # safety cap
                lander.alive = False
                lander.outcome = 0
                break

            state = lander.state(terrain)

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

            r = lander.reward(terrain)

            ns = lander.state(terrain)
            done = int(lander.landed or (not lander.alive))

            recorder.record_step(lander, terrain, action)

            # train only if not showcase (keeps showcase “clean”)
            if not is_showcase:
                buffer.push(state, a, r, ns, done)
                trainer.train_step()

            speed_hist.append(math.hypot(lander.vx, lander.vy))
            reward_hist.append(r)
            fuel_hist.append(lander.fuel / C.MAX_FUEL)

            effects.update(dt=1.0)

            render_scene(
                action,
                q_vals,
                is_showcase=is_showcase,
                notice_time=replay_notice_timer,
                step_idx=max(0, steps - 1),
            )

        # episode end
        recorder.end_episode(lander, steps)
        outcome_hist.append(lander.outcome)

        # epsilon decay + target sync (skip for showcase)
        if not is_showcase:
            epsilon = max(C.EPSILON_MIN, epsilon * C.EPSILON_DECAY)
            if episode % 10 == 0:
                trainer.sync_target()

        is_ok = (lander.outcome == 1)
        is_perfect = (lander.outcome == 2)
        is_success = (is_ok and C.SAVE_ON_OK) or (is_perfect and C.SAVE_ON_PERFECT)

        if reheat_cooldown_left > 0:
            reheat_cooldown_left -= 1

        if is_success:
            episodes_since_success = 0
        else:
            episodes_since_success += 1

        if (
            C.ENABLE_EPS_REHEAT
            and not is_showcase
            and epsilon <= C.EPSILON_MIN + 1e-9
            and episodes_since_success >= C.STALL_EPISODES
            and reheat_cooldown_left == 0
        ):
            before_eps = epsilon
            epsilon = max(epsilon, C.REHEAT_EPS)
            episodes_since_success = 0
            reheat_cooldown_left = C.REHEAT_COOLDOWN
            print(
                f"🔥 Reheat: eps {before_eps:.2f} -> {epsilon:.2f} "
                f"(stalled {C.STALL_EPISODES} eps without success)"
            )

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
