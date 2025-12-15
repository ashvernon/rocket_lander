"""
Entrypoint for Rocket Lander RL Visual Lab.

Planned responsibilities:
- Initialize config
- Construct Terrain, Lander, Agent, Trainer, HUD, Effects, Replay, Curriculum
- Run the game loop
"""
from rocket_lander.game_loop import run

if __name__ == "__main__":
    run()
