"""R2: D2-style on-policy affordance collection (48 origins x G=8, fixed 64 steps, per-step real labels, agent +
wrist frames) for ONE policy checkpoint, reusing collect_d2.main. Finished batches are skipped on restart.
  R2_NAME=e7_20 R2_CKPT=<run>/checkpoints/global_step_20 R2_OUT=<dir> python collect_r2.py"""
import os
import pathlib
import sys

sys.path.insert(0, "/data_cf/cws_sam/v1/scgrpo/diag")
import collect_d2 as C  # noqa: E402

C.POLICIES = {os.environ["R2_NAME"]: pathlib.Path(os.environ["R2_CKPT"])}
sys.argv = [sys.argv[0], os.environ["R2_OUT"]]
C.main()
