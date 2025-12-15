# Agent Notes for Rocket Lander RL Visual Lab

This repository hosts a modular DQN-based rocket-landing project meant to separate environment pieces (terrain, lander, agent, trainer, game loop) from configuration constants. Use these notes as quick guidance when developing or reviewing changes.

## Development principles
- Keep the modular layout described in `README.md`: core logic lives under `rocket_lander/` with per-component modules, and shared settings belong in `rocket_lander/config.py`.
- Prefer clear, single-responsibility functions and classes; avoid mixing rendering, training, and environment updates in the same module.
- When adding new features, gate any visualization-specific code so `headless_train.py` remains runnable without graphics.

## Training + evaluation tips
- Use `python headless_train.py --episodes <n>` for default training. To visualize a checkpoint without retraining, add `--skip-train --visualize` and optionally `--visualize-episodes <k>`.
- Track metrics such as reward, stability, and epsilon decay in logs; surface notable changes in commit messages to document training health.
- If exploration stalls (low epsilon with no successful landings), consider adjusting the epsilon schedule or reward shaping before major refactors.

## Testing + quality
- Add or extend tests under `tests/` for any new environment behaviors or trainer utilities. Prefer fast, deterministic tests when possible.
- Keep imports free of try/except wrappers, and organize them logically (standard library, third-party, local modules).
- Document any new CLI flags or configuration knobs in the README or inline docstrings to keep training reproducible.
