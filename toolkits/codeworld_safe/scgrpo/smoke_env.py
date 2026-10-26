"""Standalone smoke test of SafeWorldModelEnv: reset from keyframes, 8 chunks of small random actions."""
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.sim.world_model.safe.safe_env import SafeWorldModelEnv

CFG = OmegaConf.create({
    "seed": 0, "group_size": 8, "max_chunks": 8, "success_threshold": 0.2, "cost_top_m": 16,
    "cost_budget": 0.2, "eta_lr": 0.05, "eta_init": 0.0, "pose_scale": 8.0, "pose_clip": 4.0,
    "main_image_size": 256, "wrist_image_size": 224, "num_inference_steps": 5, "auto_reset": False,
    "keyframe_dir": "/data_cf/cws_sam/v1/scgrpo/keyframes_v1",
    "channel_mask": "/data_cf/cws_sam/v1/scgrpo/channel_mask.json",
    "model_path": "/data_cf/wan_ac_sft/runs/s3_dual_b10/ema-step-41000.safetensors",
    "VAE_path": "/mnt/nfs/fjj/wan_ac_sft/ckpt/RLinf-Wan-LIBERO-Spatial/Wan2.2_VAE.pth",
    "sam_path": "/data_cf/cws_sam/v1/checkpoints/cost_all_d32/best.pt",
    "state_reader_path": "/data_cf/cws_sam/v1/scgrpo/state_reg_v1/state_reg.pt",
    "reward_model": {"type": "CWSRewardModel", "from_pretrained": "/data_cf/cws_rm/runs/rm_v3_agent_w41k/best.pt"},
})


def main():
    torch.cuda.set_device(0)
    from rlinf.scheduler import Worker
    Worker.torch_device_type = "cuda"
    t = time.time()
    env = SafeWorldModelEnv(CFG, num_envs=16, seed_offset=0, total_num_processes=1)
    print(f"init {time.time() - t:.1f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
    obs, _ = env.reset()
    print({k: (tuple(v.shape), v.dtype) if hasattr(v, "shape") else len(v) for k, v in obs.items()})
    print("states0", obs["states"][0].cpu().numpy().round(3), "task", obs["task_descriptions"][0][:50])
    rng = np.random.default_rng(0)
    for c in range(8):
        a = np.zeros((16, 8, 7), np.float32)
        a[..., :6] = rng.normal(0, 0.3, (16, 8, 6)) * 0.02 / 0.3
        a[..., 6] = env.mem_actions[:, -1:, 6]
        t = time.time()
        obs, r, term, trunc, info = env.chunk_step(torch.from_numpy(a))
        ep = info[0]["episode"]
        print(f"chunk {c + 1}: {time.time() - t:.1f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G "
              f"r_end {r[:, -1].cpu().numpy().round(2).tolist()[:8]} term {int(term.sum())} trunc {int(trunc.sum())} "
              f"cost_d{c + 1} {ep[f'cost_d{c + 1}'].mean().item():.3f} succ {int(ep['success_once'].sum())} "
              f"state0 {obs[0]['states'][0].cpu().numpy().round(3)[:3]}", flush=True)
    env.update_reset_state_ids()
    print("eta after update", env.eta)


if __name__ == "__main__":
    main()
