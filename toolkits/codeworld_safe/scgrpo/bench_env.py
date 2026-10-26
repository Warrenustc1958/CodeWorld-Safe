"""SafeWorldModelEnv throughput vs envs per GPU (B = 16, 32). Shares the GPU with training, so read
the per-env-chunk ratio, not absolute seconds."""
import gc
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/data_cf/cws_sam/v1/scgrpo")
from smoke_env import CFG  # noqa: E402

from rlinf.envs.sim.world_model.safe.safe_env import SafeWorldModelEnv  # noqa: E402


def main():
    torch.cuda.set_device(0)
    from rlinf.scheduler import Worker
    Worker.torch_device_type = "cuda"
    rng = np.random.default_rng(0)
    for B in (16, 32, 16, 32):  # interleaved to average out training load
        env = SafeWorldModelEnv(CFG, num_envs=B, seed_offset=0, total_num_processes=1)
        env.reset()
        ts = []
        for c in range(5):
            a = np.zeros((B, 8, 7), np.float32)
            a[..., :6] = rng.normal(0, 0.02, (B, 8, 6))
            a[..., 6] = env.mem_actions[:, -1:, 6]
            torch.cuda.synchronize()
            t = time.time()
            env.chunk_step(torch.from_numpy(a))
            torch.cuda.synchronize()
            if c:  # skip warm-up chunk
                ts.append(time.time() - t)
        print(f"B {B}: {np.mean(ts):.2f} s/chunk, {np.mean(ts) / B * 1000:.0f} ms per env-chunk, "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        del env
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
