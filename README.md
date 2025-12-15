# Rocket Lander RL Visual Lab

Scaffold created: 2025-12-15 21:11:25

A modular reinforcement-learning + visualization project where a DQN agent learns to land a rocket on a pad.
Focus: watchable learning (Q-values, intent vectors, replays, curriculum), **no sound**.

## Structure

- `rocket_lander/` core package
- `main.py` entrypoint (to be implemented)
- `outputs/` checkpoints + run exports
- `tests/` future test suite

## Next steps

1. Move your current single-file lander into:
   - `rocket_lander/terrain.py`
   - `rocket_lander/lander.py`
   - `rocket_lander/agent.py`
   - `rocket_lander/trainer.py`
   - `rocket_lander/game_loop.py`
2. Keep constants in `rocket_lander/config.py`.
3. Add features incrementally: particles, Q-bars, showcase, replay, ghost.

