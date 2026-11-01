"""Restore every keyframe in a real LIBERO simulator (same code path as SyncRealWorldModelEnv) and grade it.

  python scan_keyframes.py OUT_DIR [N_PROC]
Writes OUT_DIR/kf_reset.npz: viol [N, 3] (c_obj, c_robot, c_grasp at the keyframe), success [N], error [N].
"""
import json
import multiprocessing as mp
import os
import sys

import numpy as np

KF = "/data_cf/cws_sam/v1/scgrpo/keyframes_v1"


def work(args):
    wid, idxs = args
    from rlinf.envs.sim.world_model.safe import sync_real_env as sre

    stub = sre.SyncRealWorldModelEnv.__new__(sre.SyncRealWorldModelEnv)
    stub._records = {}
    for line in open(sre.EPISODES):
        r = json.loads(line)
        stub._records[r["uid"]] = r
    meta = np.load(f"{KF}/meta.npz")
    stub._kf_uid = [str(u) for u in meta["uid"]]
    stub._kf_t0 = meta["t0"].astype(np.int64)
    stub._raw_snaps, stub._pool, stub._suites, stub.seed = {}, {}, {}, 0
    stub.obs_source, stub.render_egl_remap = "wan", {}
    stub._real, stub._real_key = [None], [None]
    out = []
    for n, i in enumerate(idxs):
        try:
            stub._restore(0, int(i))
            v = stub._grader_channels(0)
            s = bool(stub._real[0].check_success())
            out.append((int(i), v.astype(np.int8), s, 0))
        except Exception as exc:  # recorded, not fatal
            out.append((int(i), np.zeros(3, np.int8), False, 1))
            if n < 3:
                print(f"[w{wid}] kf {i}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if stub._real[0] is not None:
                key = stub._real_key[0]
                for k in [k for k in stub._pool if k != key]:  # keep only the current task's simulator
                    for e in stub._pool.pop(k):
                        e.close()
                stub._pool[key].append(stub._real[0])
                stub._real[0] = None
            stub._raw_snaps.clear()
        if n % 200 == 0:
            print(f"[w{wid}] {n}/{len(idxs)}", flush=True)
    return out


def main(out_dir, nproc):
    meta = np.load(f"{KF}/meta.npz")
    N = len(meta["uid"])
    order = np.argsort([f"{s}|{t}" for s, t in zip(meta["suite"], meta["task_id"])], kind="stable")
    chunks = [(w, order[w::nproc]) for w in range(nproc)]  # every worker sees every task (env pool reuse)
    viol, succ, err = np.zeros((N, 3), np.int8), np.zeros(N, bool), np.zeros(N, np.int8)
    with mp.get_context("spawn").Pool(nproc) as pool:
        for res in pool.imap_unordered(work, chunks):
            for i, v, s, e in res:
                viol[i], succ[i], err[i] = v, s, e
    os.makedirs(out_dir, exist_ok=True)
    np.savez(f"{out_dir}/kf_reset.npz", viol=viol, success=succ, error=err)
    print(f"done: {N} keyframes, errors {int(err.sum())}, any violation {int(viol.any(1).sum())} "
          f"(obj {int(viol[:, 0].sum())} robot {int(viol[:, 1].sum())} grasp {int(viol[:, 2].sum())}), "
          f"success at reset {int(succ.sum())}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 32)
