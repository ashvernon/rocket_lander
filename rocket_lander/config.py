"""
Central configuration for simulation, training, HUD layout, and curriculum.

Keep ALL constants here so tuning doesn't require hunting through code.
"""
from pathlib import Path

# Window
WIDTH, HEIGHT = 900, 600
FPS = 60

# Physics
GRAVITY = 0.05
MAIN_THRUST = 0.18
SIDE_THRUST = 0.07
MAX_FUEL = 500

# RL
ACTIONS = ["NONE", "MAIN", "LEFT", "RIGHT"]
ACTION_SIZE = len(ACTIONS)
STATE_SIZE = 6

EPISODES = 500
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
MEMORY_SIZE = 50_000

# Landing thresholds
OK_VY = 0.50
OK_VX = 0.50
OK_ANGLE = 0.25

PERFECT_VY = 0.25
PERFECT_VX = 0.25
PERFECT_ANGLE = 0.12

# Bounds
OUT_OF_BOUNDS_MARGIN = 40
OUT_OF_BOUNDS_PENALTY = -120

# Model persistence
MODEL_PATH = Path("lander_dqn.pt")
SAVE_ON_OK = True
SAVE_ON_PERFECT = True
MIN_SUCCESSES_TO_SAVE = 3

# HUD
HUD_N = 240
EP_N = 100
HUD_PANEL_W = 260
HUD_PANEL_X_OFFSET = 280  # base_x = WIDTH - HUD_PANEL_X_OFFSET

# Vector overlays
SHOW_VELOCITY_VECTOR = True
SHOW_TARGET_VECTOR = True
VECTOR_SCALE = 35
VECTOR_MAX_LEN = 90

# Fun layer toggles
SHOW_Q_BARS = True
SHOW_ACTION_BADGE = True
SHOW_TRAIL = True
SHOW_PARTICLES = True
SHOW_GHOST = True
GHOST_MAX_POINTS = 500
SHOW_GHOST_ROCKET = False

# Replay
ENABLE_REPLAY = True
REPLAY_SLOWMO = 4  # frames to repeat per recorded frame

# Showcase episodes
SHOWCASE_EVERY = 20          # every N episodes run a showcase
SHOWCASE_STEPS_CAP = 2500    # safety cap so showcase never hangs

