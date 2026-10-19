# CodeWorld-Safe: OpenVLA-OFT SFT on LIBERO-Safety

Step 1 of CodeWorld-Safe: full-parameter SFT of OpenVLA-OFT on the LIBERO-Safety
expert demonstrations, as the shared starting policy for all world-model RL
methods (WoVR*, SafeDojo*, HaWMPO-style, ours).

## Files

| File | Purpose |
|---|---|
| `make_50demo_subset.py` | Builds the 50-demos-per-task-instance subset (2000 episodes). Samples 100 episodes for instructions that occur in 2 (suite, level) instances and 200 for those in 4; keeps the full-set `meta/stats.json`; symlinks `data/` and `videos/`. |
| `sft_oft_env.sh` | Shared environment (FFmpeg libs for torchcodec, Ray tmp dirs, Ray token auth, per-process thread caps). Source it before `ray start`. |
| `start_ray_head.sh` | Starts one shared Ray head (port 6390, `--num-cpus=32`). |
| `sft_oft_launch.sh` | Launches `examples/sft/run_safelibero_openvlaoft.sh` with full-parameter overrides. |
| `sft_ckpt_janitor.sh` | Keeps only the newest complete checkpoint of each run. |

## Usage

```bash
python toolkits/codeworld_safe/sft/make_50demo_subset.py <LIBERO-Safety> <LIBERO-Safety-50demo>
bash toolkits/codeworld_safe/sft/start_ray_head.sh
# <run_name> <dataset> <gpu_ranks> <max_steps> <save_interval> [hydra overrides...]
TOTAL=27570 MICRO=8 bash toolkits/codeworld_safe/sft/sft_oft_launch.sh \
  oft_full_50demo_ep5 <LIBERO-Safety-50demo> 0-7 27570 5514 \
  actor.fsdp_config.gradient_checkpointing=false
bash toolkits/codeworld_safe/sft/sft_ckpt_janitor.sh
```

Paths at the top of each script point to the gpu005 layout; adjust them for another machine.

## Settings and deviations from LIBERO-Safety (Table A.3)

| Item | Paper | Here |
|---|---|---|
| Fine-tuning | LoRA r=32 | Full parameters, starting from `RLinf/RLinf-OpenVLAOFT-LIBERO-130` |
| Learning rate | 5e-4, 10x decay at 100k | 2e-5, cosine with 500 warmup steps, min 10% (RLinf has no step decay) |
| Batch | 8 GPUs x 8 = 64 | Same |
| Steps | 150k | 50-demo: 5 epochs (27,570 steps) first, then decide whether to continue |
| Image augmentation | Yes | Not implemented in RLinf's OFT SFT path |

## Notes

- `actor.model.lora_path=null` must be passed explicitly: `run_safelibero_openvlaoft.sh` defaults to the Base-Lora adapter.
- The yaml key `warmup_style` is ignored; the scheduler is set with `+actor.optim.lr_scheduler=cosine`.
- Hydra needs comma-separated GPU placements quoted (`'0-1,3-4'`).
- Each save is about 60 GB (15 GB `full_weights.pt` + 45 GB FSDP DCP for resuming).
- With 8 H200s, global batch 64 and gradient checkpointing off, steady state is about 2.6 s/step and about 40 GB per GPU.
