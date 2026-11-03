#!/usr/bin/env python3
"""T1 evaluation on L1 episodes: dual-41k vs the pose-conditioned checkpoint (training-free, same noise).

A  rollout   all chunks from the keyframe with the recorded pi0.5 actions and the online noise of every chunk;
             T1 is given the pose of every frame from
               t1_int   the causal estimator (action integration fitted on L1, R4 int_l1)   <- usable at inference
               t1_real  the measured state                                                  (upper bound)
             "free" = the online dual-41k rollout (dumped frames; the replay reproduces them bit-exactly).
B  control   teacher-forced chunk at depths D (real history, fixed noise), only the chunk's inputs change:
             both models: x1 (recorded), pose0 (action pose dims zeroed), gripflip, seed (other noise)
             T1 only (real pose unless stated): shiftx (future pose x + 3 cm), rotz (future pose yawed 20 deg),
             hold (action pose dims zeroed and future pose held at the last history pose: a still robot)
Metrics per view: pixel MAE to the real chunk and to the x1 output; state reader at the chunk end (eef pos,
rotation); for shiftx / rotz the change of the reading relative to x1 (pose gain = read shift / commanded shift).

  python t1_eval.py run --ckpt EMA --dump DIR [...] --tracks r4_tracks.npz --out OUT --shard i --nshard n
  python t1_eval.py summary --out OUT
"""

from __future__ import annotations

import argparse
import collections
import copy
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p_replay import CHUNK, NULL_ACTION, R, Episode, build_pipe, jpeg_roundtrip, mae2, orig_noise, scale_pose, seed_noise  # noqa: E402

POSE_DIR = f"{R}/work/scgrpo/drift/pose_index"
POSE_COND = f"{R}/code/RLinf-Safety/toolkits/libero_safety/wan_ac_sft/pose_cond.py"


def _pose_cond():
    """pose_cond.py by path: CodeWorld-Safe's own ``toolkits`` package shadows RLinf-Safety's."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("pose_cond", POSE_COND)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
DEPTHS = (4, 8, 16, 24)
SHIFT_M, ROT_DEG = 0.03, 20.0


def to9(s8):
    """[T, 8] pos, axis-angle, gripper -> [T, 9] pos, quat xyzw, gripper (pose_index layout)."""
    s8 = np.atleast_2d(s8)
    return np.concatenate([s8[:, :3], Rot.from_rotvec(s8[:, 3:6]).as_quat(), s8[:, 6:8]], 1).astype(np.float32)


def load_t1(ckpt, device):
    from safetensors.torch import load_file

    enable_pose = _pose_cond().enable_pose

    p2 = build_pipe(device)
    dit = p2.dit
    enable_pose(dit, views=2)
    sd = load_file(ckpt)
    res = dit.load_state_dict(sd, strict=False)
    assert not res.missing_keys and not res.unexpected_keys, (res.missing_keys[:5], res.unexpected_keys[:5])
    del p2.vae
    return dit


def main_run(args):
    pose_features = _pose_cond().pose_features
    stats = json.load(open(f"{POSE_DIR}/stats.json"))
    files = sorted(f for d in args.dump for f in glob.glob(os.path.join(d, "arrays", "*.npz")))
    by_kf = collections.defaultdict(list)
    for f in files:
        by_kf[int(np.load(f)["kf"])].append(f)
    kfd = f"{R}/data/scgrpo_bundle/keyframes_v1"
    meta = np.load(f"{kfd}/meta.npz")
    kf_anchor = np.load(f"{kfd}/anchor.npy", mmap_mode="r")
    kf_memory = np.load(f"{kfd}/memory.npy", mmap_mode="r")
    kf_inst = [str(s) for s in meta["instruction"]]
    mem_a, states = meta["mem_actions"].astype(np.float32), meta["states"].astype(np.float32)
    tr = np.load(args.tracks, allow_pickle=True)
    tracks = {str(n): np.asarray(t, np.float64).astype(np.float32) for n, t in zip(tr["names"], tr["int_l1"])}
    kfs = sorted(k for k in by_kf if os.path.exists(f"{POSE_DIR}/{str(meta['uid'][k]).split('/')[-1]}.npy"))
    mine = [k for i, k in enumerate(kfs) if i % args.nshard == args.shard]

    device = "cuda:0"
    pipe = build_pipe(device)
    dit_base = pipe.dit
    dit_t1 = load_t1(args.ckpt, device)
    from rlinf.envs.sim.world_model.safe.safe_models import StateReader
    from rlinf.models.embodiment.reward.cws_reward_model import CWSRewardModel

    reader = StateReader(f"{R}/data/scgrpo_bundle/scoring/state_reg_v1/state_reg.pt").to(device)
    rm = CWSRewardModel.from_checkpoint(args.rm).eval().to(device)
    rows = 256 * int(getattr(rm, "views", 1))

    @torch.no_grad()
    def read(frames, ep):
        off = ep.state0 - reader(torch.from_numpy(np.ascontiguousarray(ep.kf_mem[-1:])).to(device)).float().cpu().numpy()[0]
        return reader(torch.from_numpy(np.ascontiguousarray(frames)).to(device)).float().cpu().numpy() + off

    @torch.no_grad()
    def rm_max(fr, ep):
        x = torch.from_numpy(fr[[3, 7], :rows]).to(device).permute(0, 3, 1, 2).float().div(127.5).sub(1)
        return float(rm.predict_rew(x, [ep.inst] * 2).float().max())

    def gen(model, items):
        """items: dicts with anchor, mem [4 frames], mem_a [4,7], act [8,7], pose [13, 9] or None, noise."""
        pipe.dit = dit_base if model == "base" else dit_t1
        win = scale_pose(np.stack([np.concatenate([NULL_ACTION[None], it["mem_a"], it["act"]], 0) for it in items]))
        if model == "t1":
            win = np.concatenate([win, np.stack([pose_features(it["pose"], stats) for it in items])], -1)
        pipe._fixed_noise = torch.stack([it["noise"] for it in items])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = pipe(seed=0, tiled=False, input_image=[Image.fromarray(it["anchor"]) for it in items],
                       input_image4=[[Image.fromarray(np.ascontiguousarray(m)) for m in it["mem"]] for it in items],
                       action=torch.from_numpy(win).to(device), height=512, width=256, num_frames=13,
                       num_inference_steps=5, cfg_scale=1.0, progress_bar_cmd=lambda x: x, batch_size=len(items))
        pipe._fixed_noise = None
        return [np.stack([np.asarray(im, np.uint8) for im in o[5:13]]) for o in out]

    out = open(f"{args.out}/rec.jsonl.{args.shard}", "w")
    for kf in mine:
        sel = [by_kf[kf][i] for i in sorted(np.random.default_rng(kf).permutation(len(by_kf[kf]))[: args.per_kf])]
        src = np.load(f"{POSE_DIR}/{str(meta['uid'][kf]).split('/')[-1]}.npy")
        t0 = int(meta["t0"][kf])
        eps = []
        for f in sel:
            ep = Episode(f, kf_anchor, kf_memory, mem_a, kf_inst, states)
            ep.rm_real = np.load(f)["rm_real"][: ep.D]
            ep.real9 = to9(ep.real_state)
            ep.int9 = tracks[ep.name]
            ep.anchor9, ep.mem9 = src[0], src[t0 - 3: t0 + 1]
            eps.append(ep)

        def pose13(ep, d, src_name):
            """[13, 9]: anchor, 4 history frames, 8 chunk frames (L1 step indices 8d-3 .. 8d+8)."""
            p = ep.real9 if src_name == "real" else ep.int9
            hist = ep.mem9 if d == 0 else p[CHUNK * d - 3: CHUNK * d + 1]
            return np.concatenate([ep.anchor9[None], hist, p[CHUNK * d + 1: CHUNK * (d + 1) + 1]]).astype(np.float32)

        # ---------------- A: rollouts
        for ep in eps:
            for d in range(ep.D):
                fr = ep.wan[d * CHUNK:(d + 1) * CHUNK]
                st = read(fr, ep)
                rs = ep.real_state[1 + d * CHUNK: 1 + (d + 1) * CHUNK]
                a, w = mae2(fr, ep.real_chunk(d))
                out.write(json.dumps({"part": "A", "ep": ep.name, "kf": kf, "mode": "free", "chunk": d + 1,
                                      "mae_agent": a, "mae_wrist": w,
                                      "eef_cm": float(np.linalg.norm(st[:, :3] - rs[:, :3], axis=1).mean() * 100),
                                      "rot_deg": float(np.degrees((Rot.from_rotvec(st[:, 3:6]) * Rot.from_rotvec(rs[:, 3:6]).inv()).magnitude()).mean()),
                                      "rm_max": rm_max(fr, ep), "rm_real": float(np.max(ep.rm_real[d]))}) + "\n")
        jobs = [{"ep": ep, "mode": m, "d": 0, "mem": ep.kf_mem, "mem_a": ep.kf_mem_a} for ep in eps
                for m in ("t1_int", "t1_real")]
        while jobs:
            batch, jobs = jobs[: args.batch], jobs[args.batch:]
            items = [{"anchor": j["ep"].anchor, "mem": j["mem"], "mem_a": j["mem_a"], "act": j["ep"].actions[j["d"]],
                      "pose": pose13(j["ep"], j["d"], "int" if j["mode"] == "t1_int" else "real"),
                      "noise": orig_noise(j["ep"], j["d"])} for j in batch]
            for j, fr in zip(batch, gen("t1", items)):
                ep, d = j["ep"], j["d"]
                fj = np.stack([jpeg_roundtrip(x) for x in fr])
                st = read(fj, ep)
                rs = ep.real_state[1 + d * CHUNK: 1 + (d + 1) * CHUNK]
                a, w = mae2(fj, ep.real_chunk(d))
                out.write(json.dumps({"part": "A", "ep": ep.name, "kf": kf, "mode": j["mode"], "chunk": d + 1,
                                      "mae_agent": a, "mae_wrist": w,
                                      "eef_cm": float(np.linalg.norm(st[:, :3] - rs[:, :3], axis=1).mean() * 100),
                                      "rot_deg": float(np.degrees((Rot.from_rotvec(st[:, 3:6]) * Rot.from_rotvec(rs[:, 3:6]).inv()).magnitude()).mean()),
                                      "rm_max": rm_max(fj, ep), "rm_real": float(np.max(ep.rm_real[d]))}) + "\n")
                j["d"] += 1
                if j["d"] < ep.D:
                    j["mem"], j["mem_a"] = fr[-4:], ep.actions[d, -4:]
                    jobs.append(j)
        out.flush()

        # ---------------- B: teacher-forced controllability
        items, keys = [], []
        for ep in eps:
            for d in DEPTHS:
                if d >= ep.D:
                    continue
                base = {"anchor": ep.anchor, "mem": ep.history(d, "real"), "mem_a": ep.mem_actions(d),
                        "act": ep.actions[d].copy(), "pose": pose13(ep, d, "real"), "noise": orig_noise(ep, d)}
                for model in ("base", "t1"):
                    for v in ("x1", "pose0", "gripflip", "seed") + (("shiftx", "shiftx10", "rotz", "hold") if model == "t1" else ()):
                        it = copy.deepcopy(base)
                        if v in ("pose0", "hold"):
                            it["act"][:, :6] = 0
                        if v == "gripflip":
                            it["act"][:, 6] = -it["act"][:, 6]
                        if v == "seed":
                            it["noise"] = seed_noise(ep, d, 7)
                        if v == "shiftx":
                            it["pose"][5:, 0] += SHIFT_M
                        if v == "shiftx10":
                            it["pose"][5:, 0] += 0.10
                        if v == "rotz":
                            rz = Rot.from_euler("z", ROT_DEG, degrees=True)
                            it["pose"][5:, 3:7] = (rz * Rot.from_quat(it["pose"][5:, 3:7])).as_quat()
                        if v == "hold":
                            it["pose"][5:] = it["pose"][4]
                        items.append(it)
                        keys.append((ep, d, model, v))
        res = {}
        for model in ("base", "t1"):
            idx = [i for i, k in enumerate(keys) if k[2] == model]
            for s in range(0, len(idx), args.batch):
                sub = idx[s:s + args.batch]
                for i, fr in zip(sub, gen(model, [items[i] for i in sub])):
                    res[i] = fr
        ref = {(k[0].name, k[1], k[2]): res[i] for i, k in enumerate(keys) if k[3] == "x1"}
        for i, (ep, d, model, v) in enumerate(keys):
            fr = res[i]
            r0 = ref[(ep.name, d, model)]
            s_end, r_end = read(fr[-1:], ep)[0], read(r0[-1:], ep)[0]
            real_end = ep.real_state[(d + 1) * CHUNK]
            a, w = mae2(fr, ep.real_chunk(d))
            ax, wx = mae2(fr, r0)
            rec = {"part": "B", "ep": ep.name, "kf": kf, "model": model, "v": v, "d": d,
                   "real_agent": a, "real_wrist": w, "x1_agent": ax, "x1_wrist": wx,
                   "eef_err_cm": float(np.linalg.norm(s_end[:3] - real_end[:3]) * 100),
                   "rot_err_deg": float(np.degrees((Rot.from_rotvec(s_end[3:6]) * Rot.from_rotvec(real_end[3:6]).inv()).magnitude())),
                   "dx_cm": float((s_end[0] - r_end[0]) * 100),
                   "drot_deg": float(np.degrees((Rot.from_rotvec(s_end[3:6]) * Rot.from_rotvec(r_end[3:6]).inv()).magnitude())),
                   "disp_cm": float(np.linalg.norm(s_end[:3] - ep.real_state[d * CHUNK][:3]) * 100),
                   "disp_real_cm": float(np.linalg.norm(real_end[:3] - ep.real_state[d * CHUNK][:3]) * 100)}
            out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"shard {args.shard}: kf {kf} done", flush=True)
    out.close()


def main_summary(args):
    recs = [json.loads(l) for f in sorted(glob.glob(f"{args.out}/rec.jsonl.*")) for l in open(f)]
    A = [r for r in recs if r["part"] == "A"]
    B = [r for r in recs if r["part"] == "B"]
    lines = [f"episodes {len({r['ep'] for r in A})}  keyframes {len({r['kf'] for r in A})}", "",
             "A  rollout (recorded pi0.5 actions, online noise), median over episodes",
             "mode      | wrist MAE @ 64/128/192/256 | agent MAE @ 256 | eef cm / rot deg @ 256 | RM agree chunks 17-32 (f01)"]
    res = {"A": {}, "B": {}}
    for m in ("free", "t1_int", "t1_real"):
        v = [r for r in A if r["mode"] == m]
        if not v:
            continue
        row = {c: {k: float(np.median([r[k] for r in v if r["chunk"] == c])) for k in
                   ("mae_agent", "mae_wrist", "eef_cm", "rot_deg")} for c in (8, 16, 24, 32)}
        late = [r for r in v if r["chunk"] >= 17]
        agree = float(np.mean([(r["rm_max"] >= .877) == (r["rm_real"] >= .877) for r in late]))
        f01 = float(np.mean([(r["rm_max"] >= .877) and (r["rm_real"] < .877) for r in late]))
        res["A"][m] = {"by_chunk": row, "agree": agree, "f01": f01}
        lines.append(f"{m:9s} | {row[8]['mae_wrist']:5.1f} {row[16]['mae_wrist']:5.1f} {row[24]['mae_wrist']:5.1f} "
                     f"{row[32]['mae_wrist']:5.1f}  | {row[32]['mae_agent']:5.1f}           | {row[32]['eef_cm']:5.1f} / "
                     f"{row[32]['rot_deg']:5.1f}          | {agree:.3f} ({f01:.3f})")
    lines += ["", "B  teacher-forced chunk (real history), median over episodes x depths 4/8/16/24",
              "model v        | MAE to x1 agent/wrist | MAE to real agent/wrist | eef err cm / rot err deg | "
              "reading change vs x1: dx cm, drot deg"]
    for model in ("base", "t1"):
        for v in ("x1", "seed", "pose0", "gripflip", "shiftx", "shiftx10", "rotz", "hold"):
            x = [r for r in B if r["model"] == model and r["v"] == v]
            if not x:
                continue
            md = {k: float(np.median([r[k] for r in x])) for k in
                  ("x1_agent", "x1_wrist", "real_agent", "real_wrist", "eef_err_cm", "rot_err_deg", "dx_cm", "drot_deg")}
            res["B"][f"{model}_{v}"] = md
            lines.append(f"{model:4s} {v:9s} | {md['x1_agent']:5.2f} / {md['x1_wrist']:5.2f}         | "
                         f"{md['real_agent']:5.2f} / {md['real_wrist']:5.2f}           | {md['eef_err_cm']:5.2f} / "
                         f"{md['rot_err_deg']:5.2f}            | {md['dx_cm']:+5.2f}, {md['drot_deg']:5.2f}")
    sx = [r["dx_cm"] for r in B if r["model"] == "t1" and r["v"] == "shiftx"]
    rz = [r["drot_deg"] for r in B if r["model"] == "t1" and r["v"] == "rotz"]
    s10 = [r["dx_cm"] for r in B if r["model"] == "t1" and r["v"] == "shiftx10"]
    if sx:
        res["pose_gain_x"] = float(np.median(sx) / (SHIFT_M * 100))
        res["pose_gain_rot"] = float(np.median(rz) / ROT_DEG)
        lines += ["", f"pose gain: commanded +{SHIFT_M * 100:.0f} cm x -> read {np.median(sx):+.2f} cm "
                      f"(gain {res['pose_gain_x']:.2f}); commanded {ROT_DEG:.0f} deg yaw -> read {np.median(rz):.2f} deg "
                      f"(gain {res['pose_gain_rot']:.2f})"]
        if s10:
            res["pose_gain_x10"] = float(np.median(s10) / 10)
            lines.append(f"           commanded +10 cm x -> read {np.median(s10):+.2f} cm (gain {res['pose_gain_x10']:.2f})")
    json.dump(res, open(f"{args.out}/summary.json", "w"), indent=1)
    open(f"{args.out}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--ckpt", required=True); r.add_argument("--dump", nargs="+", required=True)
    r.add_argument("--tracks", required=True); r.add_argument("--out", required=True)
    r.add_argument("--shard", type=int, default=0); r.add_argument("--nshard", type=int, default=1)
    r.add_argument("--per-kf", type=int, default=2); r.add_argument("--batch", type=int, default=12)
    r.add_argument("--rm", default=f"{R}/data/scgrpo_bundle/scgrpo_rm2_20261102/rm_models/R-A_resnet_rm2/best.pt")
    s = sub.add_parser("summary")
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    main_run(args) if args.cmd == "run" else main_summary(args)


if __name__ == "__main__":
    main()
