"""
Headless training loop for the Rocket Lander environment.

This script avoids Pygame rendering during training for faster wall-clock
performance, then optionally spins up a lightweight visualization pass using the
learned policy.
"""
import argparse
import math
import random
import time
from typing import Optional, Tuple

import torch
import torch.optim as optim

from rocket_lander import config as C
from rocket_lander.agent import DQN, set_torch_stability
from rocket_lander.checkpoint import load_checkpoint, save_checkpoint
from rocket_lander.lander import Lander
from rocket_lander.metrics import RunMetrics
from rocket_lander.replay_buffer import ReplayBuffer
from rocket_lander.terrain import Terrain
from rocket_lander.trainer import Trainer


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def init_system(device: torch.device, compile_model: bool = False) -> Tuple[DQN, DQN, optim.Optimizer, float, int]:
    """Initialize networks, optimizer, and load any saved checkpoint."""
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=C.LR)

    epsilon = C.EPSILON_START
    best_success_streak = 0

    ckpt = load_checkpoint(policy_net, target_net, optimizer)
    if ckpt:
        epsilon = min(epsilon, float(ckpt.get("epsilon", epsilon)))
        best_success_streak = int(ckpt.get("best_success_streak", 0))
        print(
            f"Loaded checkpoint from {C.MODEL_PATH} | ep={ckpt.get('episode')} | "
            f"best_streak={best_success_streak} | eps={epsilon:.3f}"
        )
        note = ckpt.get("note")
        if note:
            print(f"  note: {note}")

    if compile_model and hasattr(torch, "compile"):
        try:
            policy_net = torch.compile(policy_net)
            target_net.load_state_dict(policy_net.state_dict())
            print("Using torch.compile for policy_net")
        except Exception as ex:  # pragma: no cover - compile not always available
            print(f"⚠️ torch.compile failed, continuing without it: {ex}")
    elif compile_model:
        print("⚠️ torch.compile requested but not available on this torch build")

    return policy_net, target_net, optimizer, epsilon, best_success_streak


def train_headless(policy_net: DQN, target_net: DQN, optimizer: optim.Optimizer,
                   epsilon: float, best_success_streak: int,
                   episodes: int, target_sync_every: int,
                   device: torch.device, log_every: int,
                   metrics: Optional[RunMetrics] = None) -> Tuple[DQN, float, int]:
    """Run a headless training loop using the same physics/rewards as game_loop."""
    buffer = ReplayBuffer(C.MEMORY_SIZE)
    trainer = Trainer(policy_net, target_net, optimizer, buffer, device=device)

    terrain = Terrain()
    lander = Lander()

    metrics = metrics or RunMetrics(run_tag=C.RUN_TAG)
    success_streak = 0
    state_tensor = torch.empty(C.STATE_SIZE, device=device)

    for episode in range(episodes):
        episode_start = time.perf_counter()
        lander.reset()
        terrain.reset()

        total_reward = 0.0
        steps = 0

        while lander.alive and not lander.landed:
            state = lander.state(terrain)

            if random.random() < epsilon:
                action_idx = random.randrange(C.ACTION_SIZE)
            else:
                state_tensor.copy_(torch.from_numpy(state), non_blocking=True)
                with torch.inference_mode():
                    qs = policy_net(state_tensor)
                    action_idx = int(qs.argmax().item())

            action = C.ACTIONS[action_idx]
            lander.step(action, terrain)

            reward = lander.reward(terrain)

            next_state = lander.state(terrain)
            done = int(lander.landed or (not lander.alive))

            buffer.push(state, action_idx, reward, next_state, done)
            trainer.train_step()

            total_reward += reward
            steps += 1

        epsilon = max(C.EPSILON_MIN, epsilon * C.EPSILON_DECAY)
        if episode % target_sync_every == 0:
            trainer.sync_target()

        is_ok = lander.outcome == 1
        is_perfect = lander.outcome == 2
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
                    policy_net,
                    target_net,
                    optimizer,
                    episode=episode,
                    epsilon=epsilon,
                    best_success_streak=best_success_streak,
                    note=note,
                )
                print(
                    f"✅ Saved model to {C.MODEL_PATH} | best_success_streak={best_success_streak} "
                    f"| ep={episode}"
                )
            except Exception as ex:
                print(f"⚠️ Failed to save checkpoint to {C.MODEL_PATH}: {ex}")

        wall_time = time.perf_counter() - episode_start
        metrics.record_episode(
            episode=episode + 1,
            lander=lander,
            terrain=terrain,
            total_reward=total_reward,
            steps=steps,
            epsilon=epsilon,
            wall_time_sec_episode=wall_time,
        )

        if (episode + 1) % log_every == 0 or (episode + 1) == episodes:
            print(
                f"Ep {episode+1:04d}/{episodes} | steps={steps:<4d} | "
                f"reward={total_reward:7.2f} | eps={epsilon:.3f} | outcome={lander.outcome} "
                f"| stability={metrics.records[-1].stability_score:5.1f}"
            )

    return policy_net, epsilon, best_success_streak


def visualize_policy(policy_net: DQN, episodes: int, fps: int, device: torch.device) -> None:
    """Run one or more purely-inference episodes with rendering enabled."""
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((C.WIDTH, C.HEIGHT))
    clock = pygame.time.Clock()

    policy_net.eval()
    state_tensor = torch.empty(C.STATE_SIZE, device=device)

    terrain = Terrain()
    lander = Lander()

    for ep in range(episodes):
        lander.reset()
        terrain.reset()
        running = True

        while running and lander.alive and not lander.landed:
            clock.tick(fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

            with torch.inference_mode():
                state_tensor.copy_(torch.from_numpy(lander.state(terrain)), non_blocking=True)
                action_idx = int(policy_net(state_tensor).argmax().item())
            lander.step(C.ACTIONS[action_idx], terrain)

            screen.fill((0, 0, 0))
            terrain.draw(screen)
            lander.draw(screen)

            pygame.display.flip()

        # brief pause to see the end state
        if running:
            pygame.time.delay(600)
        else:
            break

    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Headless training for Rocket Lander")
    parser.add_argument("--episodes", type=int, default=C.EPISODES, help="Episodes to train for")
    parser.add_argument("--target-sync", type=int, default=10, dest="target_sync", help="How often to sync the target network")
    parser.add_argument("--visualize", action="store_true", help="Run a post-training visualization using the trained policy")
    parser.add_argument("--visualize-episodes", type=int, default=1, dest="visualize_episodes", help="Number of visualization episodes to run")
    parser.add_argument("--fps", type=int, default=C.FPS, help="FPS cap during visualization")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and just visualize the latest checkpoint")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to run policy/training on")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for the policy network (if available)")
    parser.add_argument("--log-every", type=int, default=1, dest="log_every", help="Print metrics every N episodes")
    args = parser.parse_args()

    set_torch_stability()
    device = resolve_device(args.device)
    policy_net, target_net, optimizer, epsilon, best_success_streak = init_system(device, compile_model=args.compile)

    metrics: Optional[RunMetrics] = None

    if not args.skip_train:
        metrics = RunMetrics(run_tag=C.RUN_TAG)
        policy_net.train()
        interrupted = False
        try:
            policy_net, epsilon, best_success_streak = train_headless(
                policy_net,
                target_net,
                optimizer,
                epsilon,
                best_success_streak,
                episodes=args.episodes,
                target_sync_every=args.target_sync,
                device=device,
                log_every=args.log_every,
                metrics=metrics,
            )
        except KeyboardInterrupt:
            interrupted = True
            print("Training interrupted by user; exporting metrics...")
        finally:
            metrics.finalize_and_export(
                out_dir=C.REPORTS_DIR,
                export_csv=C.EXPORT_CSV,
                export_json=C.EXPORT_JSON,
            )

        if interrupted:
            return
    else:
        print("Skipping training; using weights from the loaded checkpoint.")

    if args.visualize:
        visualize_policy(policy_net, episodes=args.visualize_episodes, fps=args.fps, device=device)
    else:
        print("Visualization skipped. Use --visualize to watch the policy fly.")


if __name__ == "__main__":
    main()
