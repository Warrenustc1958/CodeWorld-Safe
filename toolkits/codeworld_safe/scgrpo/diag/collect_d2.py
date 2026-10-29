"""D2: frozen-policy affordance diagnosis. For each policy (base, R0b8 step 5/10, R1 step 10) roll out the same
48 affordance keyframes x G=8 in Wan with the synchronised real simulator, a fixed 64-step window (no early end
at success or violation, so RM firing does not cut the real outcome). Per step it records the real goal,
wrong-grasp-part (aux) and official violation, RM v3 probability, and agent+wrist Wan frames (JPEG).
Output: OUT/<policy>/batch_XX.pkl. Same keyframes and seeds for every policy.
  python collect_d2.py OUT"""
import io
import json
import pathlib
import pickle
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

S = pathlib.Path("/data_cf/cws_sam/v1/scgrpo")
POLICIES = {
    "base": None,
    "r0b8_5": S / "causal/r0b8/scgrpo_causal_r0b8/checkpoints/global_step_5",
    "r0b8_10": S / "causal/r0b8/scgrpo_causal_r0b8/checkpoints/global_step_10",
    "r1_10": S / "causal/r1/scgrpo_causal_r1/checkpoints/global_step_10",
}
B, G, N_ORIGINS = 32, 8, 48


def jpg(a):
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def pick_origins(meta):
    rng = np.random.default_rng(20261029)
    aff = np.flatnonzero(meta["suite"] == "affordance")
    by = {}
    for k in aff:
        by.setdefault(str(meta["tcell"][k]), []).append(int(k))
    cells = sorted(by)
    out = []
    while len(out) < N_ORIGINS:  # round-robin over task cells
        for c in cells:
            if len(out) < N_ORIGINS:
                out.append(int(rng.choice(by[c])))
    return out


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    from rlinf.envs.action_utils import prepare_actions
    from rlinf.envs.sim.world_model.safe.sync_real_env import SyncRealWorldModelEnv
    from rlinf.models import get_model
    from rlinf.models.embodiment.openpi_rlinf.checkpoint import load_full_weights
    from rlinf.scheduler import Worker
    Worker.torch_device_type = "cuda"

    cfg = OmegaConf.load(S / "diagnostics/rootcause_20261027_v1/original_config.yaml")
    ec = cfg.env.train
    ec.group_size, ec.eta_init = 1, 0
    ec.sync_real, ec.reward_source = True, "real"
    ec.end_on_violation = False          # fixed window: violations recorded, never end the episode
    ec.success_tail_chunks = 8           # keep stepping after the first real success until 64 steps
    ec.record_steps = True
    model = get_model(cfg.actor.model).cuda().eval()
    model.requires_grad_(False)
    env = SyncRealWorldModelEnv(ec, num_envs=B, seed_offset=7070, total_num_processes=1)
    meta = np.load(S / "keyframes_v1/meta.npz")
    origins = pick_origins(meta)
    (out / "origins.json").write_text(json.dumps({"origins": origins, "tcell": [str(meta["tcell"][k]) for k in origins]}))
    log = open(out / "collect.log", "a")
    per_b = B // G
    for name, path in POLICIES.items():
        if path is not None:
            load_full_weights(model, path / "actor/model_state_dict/full_weights.pt", expect_rlt=False)
            model.eval()
        pdir = out / name
        pdir.mkdir(exist_ok=True)
        env._batch_counter = 0  # identical Wan seeds for every policy
        for b in range(0, N_ORIGINS, per_b):
            f = pdir / f"batch_{b // per_b:02d}.pkl"
            if f.exists():
                continue
            kf = np.repeat(origins[b:b + per_b], G)
            torch.manual_seed(1000 + b)
            np.random.seed(1000 + b)
            t = time.time()
            obs, _ = env.reset(episode_indices=kf)
            at_reset = env.real_succ.copy()
            agent, wrist, prob, acts = [[] for _ in range(B)], [[] for _ in range(B)], [], []
            with torch.no_grad():
                for c in range(8):
                    a, _ = model.predict_action_batch(obs, mode="train", compute_values=False)
                    a = np.asarray(prepare_actions(a, "world_model", cfg.actor.model.model_type, 8, 7,
                                                   wm_env_type="libero"), np.float32)
                    obs_list, _, _, _, _ = env.chunk_step(torch.from_numpy(a))
                    obs = obs_list[0]
                    prob.append(env.last_p_succ.float().cpu().numpy())
                    acts.append(a)
                    for i in range(B):
                        for k in range(8):
                            agent[i].append(jpg(env.last_frames[i, k, :256]))
                            wrist[i].append(jpg(env.last_frames[i, k, 256:]))
            rec = {"keyframes": kf, "origin_index": np.repeat(np.arange(b, b + per_b), G),
                   "tcell": [str(meta["tcell"][k]) for k in kf], "at_reset_success": at_reset,
                   "step_rec": env.step_rec.copy(),                  # [B, 64, 3] goal, wrong part, violation
                   "rm_v3_p": np.concatenate(prob, 1),               # [B, 64]
                   "actions": np.stack(acts, 1), "agent_jpg": agent, "wrist_jpg": wrist}
            pickle.dump(rec, open(f, "wb"))
            g = rec["step_rec"]
            print(f"{name} batch {b // per_b + 1}/{N_ORIGINS // per_b} {time.time() - t:.0f}s goal_any "
                  f"{g[:, :, 0].any(1).mean():.2f} wrongpart_any {g[:, :, 1].any(1).mean():.2f}", file=log, flush=True)
    print("D2_COLLECT_DONE", file=log, flush=True)


if __name__ == "__main__":
    main()
