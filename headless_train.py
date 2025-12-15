"""
Headless training loop for the Rocket Lander environment.

This script avoids Pygame rendering during training for faster wall-clock
performance, then optionally spins up a lightweight visualization pass using the
learned policy.
"""
import argparse
import math
import random
from typing import Tuple

import torch
import torch.optim as optim

from rocket_lander import config as C
from rocket_lander.agent import DQN, set_torch_stability
from rocket_lander.checkpoint import load_checkpoint, save_checkpoint
from rocket_lander.lander import Lander
from rocket_lander.replay_buffer import ReplayBuffer
from rocket_lander.terrain import Terrain
from rocket_lander.trainer import Trainer


def init_system() -> Tuple[DQN, DQN, optim.Optimizer, float, int]:
    """Initialize networks, optimizer, and load any saved checkpoint."""
    policy_net = DQN()
    target_net = DQN()
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

    return policy_net, target_net, optimizer, epsilon, best_success_streak


def train_headless(policy_net: DQN, target_net: DQN, optimizer: optim.Optimizer,
                   epsilon: float, best_success_streak: int,
                   episodes: int, target_sync_every: int) -> Tuple[DQN, float, int]:
    """Run a headless training loop using the same physics/rewards as game_loop."""
    buffer = ReplayBuffer(C.MEMORY_SIZE)
    trainer = Trainer(policy_net, target_net, optimizer, buffer)

    terrain = Terrain()
    lander = Lander()

    success_streak = 0

    for episode in range(episodes):
        lander.reset()
        terrain.reset()

        total_reward = 0.0
        steps = 0

        while lander.alive and not lander.landed:
            state = lander.state()

            if random.random() < epsilon:
                action_idx = random.randrange(C.ACTION_SIZE)
            else:
                with torch.no_grad():
                    qs = policy_net(torch.from_numpy(state))
                    action_idx = int(qs.argmax().item())

            action = C.ACTIONS[action_idx]
            lander.step(action, terrain)

            reward = lander.reward()
            pad_center_x = (terrain.pad_x1 + terrain.pad_x2) / 2
            dist_to_pad = abs(lander.x - pad_center_x) / (C.WIDTH / 2)
            reward += 0.25 * (1 - min(1.0, dist_to_pad))

            next_state = lander.state()
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

        print(
            f"Ep {episode+1:04d}/{episodes} | steps={steps:<4d} | "
            f"reward={total_reward:7.2f} | eps={epsilon:.3f} | outcome={lander.outcome}"
        )

    return policy_net, epsilon, best_success_streak


def visualize_policy(policy_net: DQN, episodes: int, fps: int) -> None:
    """Run one or more purely-inference episodes with rendering enabled."""
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((C.WIDTH, C.HEIGHT))
    clock = pygame.time.Clock()

    policy_net.eval()

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

            with torch.no_grad():
                state = lander.state()
                action_idx = int(policy_net(torch.from_numpy(state)).argmax().item())
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
    args = parser.parse_args()

    set_torch_stability()

    policy_net, target_net, optimizer, epsilon, best_success_streak = init_system()

    if not args.skip_train:
        policy_net.train()
        policy_net, epsilon, best_success_streak = train_headless(
            policy_net,
            target_net,
            optimizer,
            epsilon,
            best_success_streak,
            episodes=args.episodes,
            target_sync_every=args.target_sync,
        )
    else:
        print("Skipping training; using weights from the loaded checkpoint.")

    if args.visualize:
        visualize_policy(policy_net, episodes=args.visualize_episodes, fps=args.fps)
    else:
        print("Visualization skipped. Use --visualize to watch the policy fly.")


if __name__ == "__main__":
    main()
