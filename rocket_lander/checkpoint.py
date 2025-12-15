"""
Checkpoint load/save.

Responsibilities:
- Save policy/target/optimizer state
- Save epsilon, episode, best scores/streaks, curriculum level
- Version metadata to allow safe evolution of the project
"""
import torch
from . import config as C


def load_checkpoint(policy_net, target_net, optimizer):
    if not C.MODEL_PATH.exists():
        return None

    try:
        ckpt = torch.load(C.MODEL_PATH, map_location="cpu")
        policy_net.load_state_dict(ckpt["policy_state"])
        target_net.load_state_dict(ckpt.get("target_state", ckpt["policy_state"]))
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        return ckpt
    except Exception as ex:
        print(f"⚠️ Failed to load checkpoint from {C.MODEL_PATH}: {ex}")
        return None


def save_checkpoint(policy_net, target_net, optimizer, episode, epsilon, best_success_streak, note=""):
    ckpt = {
        "policy_state": policy_net.state_dict(),
        "target_state": target_net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "episode": int(episode),
        "epsilon": float(epsilon),
        "best_success_streak": int(best_success_streak),
        "note": note,
    }
    torch.save(ckpt, C.MODEL_PATH)
