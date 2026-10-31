# SC-GRPO E10-O 复现指南：真实奖励 80 步长程训练与 LIBERO-Safety 评测

目的：在冻结的 Wan 世界模型里，用**真实仿真器的目标判定**作奖励（η=0，不加安全约束）训练 π0.5，训 80 步，看去掉学习出来的 RM 之后是否还会出现长程漂移。评测 step 10/20/40/60/80。

背景（为什么做这个实验）：
- RM 奖励（RM-all / E7）训久了会退化：E7 在 step 40/60 的 affordance SR 降到 .400/.363，夹爪开合次数明显上升。
- 用真实奖励的 E8-O 只验证了 10 步（总体 SR .703 → .732），R1-all 的 affordance 在 step 10 → 20 也出现过回落（.600 → .540）。
- 所以需要同配置的真实奖励长程对照，判断退化是不是 RM 特有的。

完整的排查过程见仓库里的 `toolkits/codeworld_safe/scgrpo/DIAGNOSIS.md`。

> 标注说明：**【已核实】** 表示在 gpu006 上确认过；**【需确认】** 表示我没有核实，需要向原作者（fjj）确认。

---

## 1. 代码

| 仓库 | 地址 | 分支 / 版本 | 用途 |
|---|---|---|---|
| CodeWorld-Safe | `https://github.com/Warrenustc1958/CodeWorld-Safe` | `main`，commit `b3d0ec2c` 或之后 | RLinf 训练框架 + SC-GRPO 世界模型环境 + 评测脚本 |
| LIBERO-Safety | `https://github.com/LIBERO-SAFETY/LIBERO-Safety` | — | LIBERO-Safety 任务、BDDL、grader、robosuite 1.4（`third_party/robosuite-1.4`）；服务器上在 `/mnt/nfs/fjj/Code/LIBERO-Safety` |

```bash
git clone https://github.com/Warrenustc1958/CodeWorld-Safe.git
cd CodeWorld-Safe && git log --oneline -1    # 应为 b3d0ec2c 或之后
```

关键代码位置【已核实】：

| 文件 | 内容 |
|---|---|
| `rlinf/envs/sim/world_model/safe/safe_env.py` | Wan 世界模型环境（关键帧采样、SAM 代价、η） |
| `rlinf/envs/sim/world_model/safe/sync_real_env.py` | 同步真实仿真器：同一动作在真实 LIBERO 里执行，取真实目标 / 违规标签；`reward_source`、`termination_source`、`outcome_order`、`success_reward_rule` |
| `rlinf/models/embodiment/openpi_rlinf/pi0.py` | π0.5；`train()` 时 SigLIP 保持 eval（修过的 bug） |
| `tests/unit_tests/test_world_model_safe.py` | 环境语义单测（14 个） |
| `toolkits/codeworld_safe/scgrpo/e1/` | 启动、续训、评测脚本（见第 5、6 节） |

---

## 2. 数据与模型权重

E10-O 训练需要以下 8 项。配置文件里都是 gpu006 上的绝对路径，换机器后要逐项改路径（见第 4 节）。

| # | 内容 | gpu006 上的路径 | 下载来源 | 大小 |
|---|---|---|---|---|
| 1 | π0.5 SFT base（策略初始化） | `/mnt/nfs/fjj/Code/rlinf-safety-runtime/checkpoints/pi05_libero_safety_rlinf/`（`model.safetensors`、`config.json`、`assets/lerobot/norm_stats.json`） | HuggingFace `LIBERO-Safety/pi05_libero_safety`（JAX/orbax 格式），需转换成 OpenPI_RLinf 格式，见第 3.5 节 | 约 7 GB |
| 2 | Wan 双视角世界模型 DiT ema-step-41000 | `/data_cf/wan_ac_sft/runs/s3_dual_b10/ema-step-41000.safetensors` | ModelScope `Warrenustc1958/Wan-CodeWorld-Safe-pi05-dual-41000` | 9.4 GB |
| 3 | Wan2.2 VAE | `/mnt/nfs/fjj/wan_ac_sft/ckpt/RLinf-Wan-LIBERO-Spatial/Wan2.2_VAE.pth` | HuggingFace `RLinf/RLinf-Wan-LIBERO-Spatial` | — |
| 4 | 训练起点关键帧 `keyframes_v1`（47,658 个，含 `meta.npz`、`anchor.npy`、`memory.npy`） | `/data_cf/cws_sam/v1/scgrpo/keyframes_v1/` | ModelScope 数据集 `Warrenustc1958/CodeWorld-Safe-LIBERO-Safety-pi05-GRPO-rollout` 的 `keyframes_v1/`（大文件切成 4 GB，`cat memory.npy.part_* > memory.npy`，再 `sha256sum -c SHA256SUMS`） | 88 GB |
| 5 | SAM 安全代价模型 | `/data_cf/cws_sam/v1/checkpoints/cost_all_d32/best.pt` | 同上数据集 `scoring/sam_cost_all_d32/best.pt` | — |
| 6 | 代价通道掩码 | `/data_cf/cws_sam/v1/scgrpo/channel_mask.json` | 同上 `scoring/channel_mask.json` | — |
| 7 | 状态读出模型（画面 → π0.5 proprio） | `/data_cf/cws_sam/v1/scgrpo/state_reg_v1/state_reg.pt` | 同上 `scoring/state_reg_v1/state_reg.pt` | — |
| 8 | RM v3（E10-O 中只记录、不参与奖励） | `/data_cf/cws_rm/runs/rm_v3_agent_w41k/best.pt` | 同上 `scoring/rm_v3_agent_w41k/best.pt` | 87 MB |

同步真实仿真器还需要：

| # | 内容 | gpu006 上的路径 | 来源 |
|---|---|---|---|
| 9 | 关键帧对应的仿真器快照 | `/data_cf/cws_sam/v1/labels/episodes.jsonl` + 它指向的 rollout 审计文件 | 见第 2.1 节 |

评测还需要：

| # | 内容 | gpu006 上的路径 | 来源 |
|---|---|---|---|
| 10 | 评测初始状态（L0/L1，策略和 Wan 都没见过） | `/data_cf/cws/fresh_init/v1/` | 同上数据集 `eval/fresh_init_v1/` |
| 11 | base 策略的评测结果（对比用） | `/data_cf/cws_sam/v1/scgrpo/eval_real/off_base/` | 同上数据集 `scgrpo_e1_e8_20261030/eval_real/off_base/` |

ModelScope 下载（SDK 需要 token；数据集是私有的话需要原作者授权）：

```python
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download("Warrenustc1958/CodeWorld-Safe-LIBERO-Safety-pi05-GRPO-rollout", repo_type="dataset",
                  allow_patterns=["keyframes_v1/*", "scoring/*", "eval/*", "scgrpo_e1_e8_20261030/eval_real/off_base/*"],
                  local_dir="/your/data/scgrpo_bundle")
```

磁盘规划：数据约 110 GB；每个训练 checkpoint 15 GB（`model_state_dict` 6.5 GB + 续训用的 `dcp_checkpoint` 8.6 GB）。保留 step 10/20/40/60/80 + 最新 1 个，最多约 90 GB。

---

### 2.1 仿真器快照（同步真实仿真器用）

【已核实】训练时 `sync_real_env.py` 要把真实 LIBERO 恢复到每个关键帧的仿真器状态，查找链路是：

1. `keyframes_v1/meta.npz` 里每个关键帧有 `uid` 和 `t0`；
2. `labels/episodes.jsonl`（11,186 条，4,429 root + 6,757 反事实分支）按 `uid` 给出该回合审计文件 `json`（分支还有 `root_json` 和 `branch_timestep`）；
3. 审计 `.json` 的 `sim_state_trajectory.path` 指向同目录下 `state/<...>.npz`，存每 8 步一个快照（`mujoco_state`、`controller_state`、`gripper_state`、`motion_generators` 等）。

`episodes.jsonl` 里共 22,372 个路径，全部存在于 gpu006：

| 来源 | 本地前缀 | 数量 | ModelScope 位置 |
|---|---|---|---|
| Step E rollout | `/data_cf/cws/step_e/runs/` | 20,238 | 数据集 `Warrenustc1958/CodeWorld-Safe-pi05-counterfactual-rollout` 的 `step_e/runs/`（整个数据集 570.8 GB，已核对） |
| 早期反事实 rollout | `/mnt/nfs/fjj/Code/rlinf-safety-runtime/rollouts/` | 2,134 | 同一数据集的 `legacy/rollouts/` |

只需要下载审计 `.json` 和 `state/*.npz`，不需要 `visual/*.npz`（训练图像由 Wan 生成）：

```python
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download("Warrenustc1958/CodeWorld-Safe-pi05-counterfactual-rollout", repo_type="dataset",
                  allow_patterns=["step_e/runs/**/*.json", "step_e/runs/**/state/*.npz",
                                  "legacy/rollouts/**/*.json", "legacy/rollouts/**/state/*.npz"],
                  local_dir="/your/data/cws_rollout")
```

然后把 `episodes.jsonl` 里的绝对路径换成本机路径，并在配置 `env.train` 下用 `episodes_jsonl:` 指向改好的文件：

```bash
sed -e 's|/data_cf/cws/step_e/runs/|/your/data/cws_rollout/step_e/runs/|g' \
    -e 's|/mnt/nfs/fjj/Code/rlinf-safety-runtime/rollouts/|/your/data/cws_rollout/legacy/rollouts/|g' \
    /data_cf/cws_sam/v1/labels/episodes.jsonl > /your/data/labels/episodes.jsonl
```

`labels/episodes.jsonl`（187 MB 的 `labels/` 目录）不在上面的数据集里，直接从 gpu006 拷贝：`/data_cf/cws_sam/v1/labels/`（见第 2.3 节）。

另外，`episodes.jsonl` 引用的 11,682 个审计 `.json` 中，Step E 的 10,479 个都有 `state/*.npz` 快照；早期 rollout 的 1,203 个没有快照。`keyframes_v1` 来自 Step E 回合，训练用到的是 Step E 的快照，早期的 `.json` 一并拷过去即可。

### 2.2 存储预算【已核实，gpu006 实测】

| 类别 | 内容 | 大小 |
|---|---|---|
| 代码 | LIBERO-Safety（其中仿真器资产 `assets/` 17 GB） | 18 GB |
| | diffsynth-studio | 0.14 GB |
| 环境 | `.venv-openpi-libero-safety` | 9.5 GB |
| 策略 | π0.5 OpenPI_RLinf（转换后） | 13 GB |
| | π0.5 JAX 原件（只在自己转换时需要） | 12 GB |
| 世界模型 | Wan DiT ema-step-41000 | 9.4 GB |
| | Wan2.2 VAE | 2.7 GB |
| 训练起点 | `keyframes_v1` | 88 GB |
| 仿真器快照 | 审计 `.json` 11,682 个 + `state/*.npz` 10,479 个 | 29.3 + 1.0 GB |
| | `labels/` | 0.19 GB |
| 评分模型 | SAM、通道掩码、状态读出、RM v3 | 0.11 GB |
| 评测 | `fresh_init/v1`、base 评测结果 `off_base` | 0.62 GB |
| **数据和模型小计** | | **约 175 GB**（自己转换 π0.5 再加 12 GB） |
| E10-O 训练 | checkpoint：step 10/20/40/60/80 + 最新 1 个，每个 15 GB | 90 GB |
| E10-O 评测 | 5 个 checkpoint 的评测结果 | 约 3 GB |
| **合计** | | **约 270 GB** |

另外：
- **根分区要留空间**：Ray 的临时目录在 `/tmp`，根分区可用空间低于 5% 时 Ray 会报警，需要溢写对象时会失败。建议 `/tmp` 所在分区至少留 60 GB。
- **数据盘**：checkpoint 写在配置的 `log_path` 下，建议和数据放在同一块本地 NVMe 盘上，不要放 NFS。

### 2.3 从 gpu006 直接拷贝【已核实路径】

最省事的做法是在新机器上**保持和 gpu006 完全相同的绝对路径**，这样配置文件和 `episodes.jsonl` 都不用改。做不到的话，按第 4 节和第 2.1 节修改路径。

| 内容 | gpu006 路径 | 大小 |
|---|---|---|
| π0.5（已转换） | `/mnt/nfs/fjj/Code/rlinf-safety-runtime/checkpoints/pi05_libero_safety_rlinf/` | 13 GB |
| π0.5 JAX 原件（可选） | `/mnt/nfs/fjj/Models/pi05_libero_safety/` | 12 GB |
| Wan DiT | `/data_cf/wan_ac_sft/runs/s3_dual_b10/ema-step-41000.safetensors` | 9.4 GB |
| Wan2.2 VAE | `/mnt/nfs/fjj/wan_ac_sft/ckpt/RLinf-Wan-LIBERO-Spatial/Wan2.2_VAE.pth` | 2.7 GB |
| 关键帧 | `/data_cf/cws_sam/v1/scgrpo/keyframes_v1/` | 88 GB |
| SAM | `/data_cf/cws_sam/v1/checkpoints/cost_all_d32/` | 16 MB |
| 通道掩码 | `/data_cf/cws_sam/v1/scgrpo/channel_mask.json` | 4 KB |
| 状态读出 | `/data_cf/cws_sam/v1/scgrpo/state_reg_v1/` | 9 MB |
| RM v3 | `/data_cf/cws_rm/runs/rm_v3_agent_w41k/` | 87 MB |
| 快照索引 | `/data_cf/cws_sam/v1/labels/` | 187 MB |
| 快照文件 | `/data_cf/cws/step_e/runs/` 和 `/mnt/nfs/fjj/Code/rlinf-safety-runtime/rollouts/` 下被引用的部分（清单见下） | 30.3 GB |
| 评测初始状态 | `/data_cf/cws/fresh_init/v1/` | 0.5 MB |
| base 评测结果 | `/data_cf/cws_sam/v1/scgrpo/eval_real/off_base/` | 0.6 GB |
| LIBERO-Safety（含资产） | `/mnt/nfs/fjj/Code/LIBERO-Safety/` | 18 GB |
| diffsynth-studio | `/mnt/nfs/fjj/Code/related_works/diffsynth-studio/` | 0.14 GB |
| 评测脚本和链 | `/data_cf/cws_sam/v1/scgrpo/{run_causal.sh,run_multi.sh,launch_train.sh,eval_real_p2*.sh}`、`diag/{matched_summ.py,level_summ.py}` | 很小 |

在**新机器**上执行（需要能 ssh 到 gpu006；`D=/` 表示保持原路径，也可以换成别的根目录）：

```bash
G=gpu006-root      # ssh 主机名；gpu006 的 ~/.ssh/config 有 RemoteForward 时要加 -o ClearAllForwardings=yes
D=/                # 目标根目录
R="rsync -aP --mkpath -e 'ssh -o ClearAllForwardings=yes'"

# 1) 模型、关键帧、评分模型、评测数据（整目录 / 单文件）
for p in /mnt/nfs/fjj/Code/rlinf-safety-runtime/checkpoints/pi05_libero_safety_rlinf/ \
         /data_cf/wan_ac_sft/runs/s3_dual_b10/ema-step-41000.safetensors \
         /mnt/nfs/fjj/wan_ac_sft/ckpt/RLinf-Wan-LIBERO-Spatial/Wan2.2_VAE.pth \
         /data_cf/cws_sam/v1/scgrpo/keyframes_v1/ \
         /data_cf/cws_sam/v1/checkpoints/cost_all_d32/ \
         /data_cf/cws_sam/v1/scgrpo/channel_mask.json \
         /data_cf/cws_sam/v1/scgrpo/state_reg_v1/ \
         /data_cf/cws_rm/runs/rm_v3_agent_w41k/ \
         /data_cf/cws_sam/v1/labels/ \
         /data_cf/cws/fresh_init/v1/ \
         /data_cf/cws_sam/v1/scgrpo/eval_real/off_base/ \
         /mnt/nfs/fjj/Code/LIBERO-Safety/ \
         /mnt/nfs/fjj/Code/related_works/diffsynth-studio/; do
  eval $R "$G:$p" "$D${p#/}"
done

# 2) 仿真器快照：只拷 episodes.jsonl 引用的审计 .json 和 state/*.npz（30.3 GB）
ssh -o ClearAllForwardings=yes $G 'python3 - <<EOF
import json, pathlib
out = set()
for l in open("/data_cf/cws_sam/v1/labels/episodes.jsonl"):
    r = json.loads(l)
    for k in ("json", "root_json"):
        p = r.get(k)
        if not p: continue
        out.add(p)
        st = (json.load(open(p)).get("sim_state_trajectory") or {}).get("path")
        if st: out.add(str(pathlib.Path(p).parent / st))
open("/tmp/cws_snap_files.txt", "w").write("".join(x.lstrip("/") + "\n" for x in sorted(out)))
print(len(out), "files")
EOF'
scp -o ClearAllForwardings=yes $G:/tmp/cws_snap_files.txt .
rsync -aP --files-from=cws_snap_files.txt -e 'ssh -o ClearAllForwardings=yes' $G:/ $D
```

拷完后做两项检查：
- 快照文件数：`wc -l cws_snap_files.txt` 应约为 22,161（11,682 个 `.json` 加 10,479 个 `.npz`）；
- 关键帧大文件：`keyframes_v1/` 里的 `anchor.npy`、`memory.npy` 是完整文件，gpu006 上没有切片，不需要合并。

---

## 3. 运行环境

### 3.1 gpu006 上的现有环境【已核实】

| 项 | 值 |
|---|---|
| GPU | 8 × 143 GB；训练每卡约 55 GB，评测每卡约 20 GB |
| venv | `/mnt/nfs/fjj/Code/rlinf-safety-runtime/.venv-openpi-libero-safety`，由 uv 0.11.8 建，CPython 3.11.14 |
| 主要包 | torch 2.6.0 / torchvision 0.21.0（cu124）、jax 0.5.3、orbax-checkpoint 0.11.13、ray 2.59.0、transformers 4.57.6、numpy 1.26.4、flash-attn 2.7.4.post1、hydra-core 1.4.0.dev6、omegaconf 2.4.0、safetensors 0.8.0、rlinf-openpi 0.1.1、modelscope 1.20.0 |
| 仿真器 | LIBERO-Safety（`libero` 0.1.0，editable）+ robosuite 1.4.0（LIBERO-Safety 自带）+ mujoco 3.8.1 + bddl 1.0.1 |
| Wan 依赖 | diffsynth-studio 不是 pip 安装，而是通过 `PYTHONPATH=/mnt/nfs/fjj/Code/related_works/diffsynth-studio` 引入 |
| 渲染 | 训练：`MUJOCO_GL=osmesa`（系统包 `libosmesa6`）；评测：EGL |
| ImageMagick | 本地编译在 `rlinf-safety-runtime/local-packages/imagemagick/prefix` |

### 3.2 建环境步骤

CodeWorld-Safe 的 `requirements/install.sh` 已经支持 `--model openpi --env liberosafety`，它会建 uv venv、装 torch、openpi 依赖（`requirements/embodied/models/openpi.txt` 固定 `jax[cuda12]==0.5.3`、`orbax-checkpoint==0.11.13`、`pyarrow==20.0.0`），并用 `install_liberosafety_env` 安装 LIBERO-Safety。

```bash
# 0) 系统依赖（Ubuntu 22.04）
sudo apt-get install -y libosmesa6 libosmesa6-dev libegl1 libgl1 ffmpeg git
curl -LsSf https://astral.sh/uv/install.sh | sh          # uv

# 1) 代码
mkdir -p ~/cws && cd ~/cws
git clone https://github.com/Warrenustc1958/CodeWorld-Safe.git
git clone https://github.com/LIBERO-SAFETY/LIBERO-Safety.git
git clone https://github.com/RLinf/diffsynth-studio.git related_works/diffsynth-studio
git -C related_works/diffsynth-studio checkout 2a2e05f        # gpu006 上使用的 commit，无本地修改【已核实】

# 2) venv：openpi 模型 + LIBERO-Safety 环境，torch 2.6.0 / cu124，Python 3.11
cd CodeWorld-Safe
export LIBERO_SAFETY_PATH=~/cws/LIBERO-Safety
UV_TORCH_BACKEND=cu124 bash requirements/install.sh embodied --model openpi --env liberosafety \
    --venv ~/cws/.venv-openpi-libero-safety --torch 2.6.0
source ~/cws/.venv-openpi-libero-safety/bin/activate      # activate 里已加 LIBERO_CONFIG_PATH / LIBERO_TYPE=safety
```

`install_liberosafety_env` 会：
- 卸载标准 `libero` / `robosuite`；
- 装 `mujoco==3.3.7`、`third_party/robosuite-1.4`（editable）和 LIBERO-Safety（editable，另写 `libero_safety_source.pth`）；
- 装 `bddl==1.0.1 future easydict imageio[ffmpeg] scikit-image usd-core wand matplotlib termcolor`；
- 生成 `$VIRTUAL_ENV/share/libero-safety/config.yaml`（`assets` / `bddl_files` / `benchmark_root` / `init_states` 指向 LIBERO-Safety 源码）。

gpu006 上实际是 mujoco 3.8.1，而脚本固定 3.3.7；两者我都没在新机器上测过，建议先用脚本的 3.3.7。

```bash
# 3) LIBERO-Safety 仿真器资产：脚本提示缺 assets 时，按 LIBERO-Safety 的 SAFELIBERO_INTEGRATION.md 下载 assets.zip 并解压到
#    ~/cws/LIBERO-Safety/libero/libero/assets

# 4) 补充包（gpu006 上有，脚本未必装）
uv pip install "ray==2.59.0" "flash-attn==2.7.4.post1" --no-build-isolation "modelscope==1.20.0"
```

### 3.3 每次运行前的环境变量

和 `run_causal.sh` 一致：

```bash
V=~/cws/.venv-openpi-libero-safety
source $V/bin/activate
export PYTHONPATH=~/cws/related_works/diffsynth-studio:~/cws/CodeWorld-Safe:$PYTHONPATH
export LIBERO_TYPE=safety RLINF_LIBERO_SAFETY_AUX=1 LIBERO_CONFIG_PATH=$V/share/libero-safety
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYOPENGL_OSMESA_LIBRARY=/lib/x86_64-linux-gnu/libOSMesa.so.8
export RAY_AUTH_MODE=disabled OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
# 如果用到 ImageMagick（wand），再设 MAGICK_HOME / MAGICK_CONFIGURE_PATH / MAGICK_CODER_MODULE_PATH
```

`RLINF_LIBERO_SAFETY_AUX=1` 打开辅助安全标签（抓错部位等），必须设。

### 3.4 验收

```bash
python -c 'import libero.libero as l; print(l.get_libero_path("benchmark_root"))'   # 指向 LIBERO-Safety
python -c 'import torch, jax, ray, diffsynth; print(torch.__version__, torch.cuda.is_available(), jax.__version__, ray.__version__)'
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/unit_tests/test_world_model_safe.py   # 应 14 passed
```

### 3.5 π0.5 权重转换（JAX → OpenPI_RLinf）

HuggingFace `LIBERO-Safety/pi05_libero_safety` 是 JAX/orbax 格式（`params/` + `assets/lerobot/norm_stats.json`），gpu006 上的原件在 `/mnt/nfs/fjj/Models/pi05_libero_safety/`。RLinf 要的是 OpenPI_RLinf 格式（`model.safetensors` + `config.json`），用仓库自带的转换器：

```bash
huggingface-cli download LIBERO-Safety/pi05_libero_safety --local-dir ~/cws/models/pi05_libero_safety
python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi_rlinf \
    --input-model       ~/cws/models/pi05_libero_safety \
    --input-norm-stats  ~/cws/models/pi05_libero_safety/assets/lerobot/norm_stats.json \
    --output-model      ~/cws/checkpoints/pi05_libero_safety_rlinf \
    --output-norm-stats ~/cws/checkpoints/pi05_libero_safety_rlinf/assets/lerobot/norm_stats.json \
    --action-horizon 32
```

- 转换器要 jax 和 orbax（venv 里已有）。输出权重是 fp32，`config.json` 记 `"dtype": "bfloat16"`。
- **norm_stats 必须放在 `assets/lerobot/norm_stats.json`**：配置里 `norm_stats_path: ${..model_path}/assets/lerobot/norm_stats.json`。转换器 README 的示例路径是 `physical-intelligence/behavior/`，不要照抄。
- gpu006 上已转换好的 `config.json` 是：`action_dim 32, action_horizon 32, max_token_len 200, paligemma_variant gemma_2b, action_expert_variant gemma_300m, pi05 true, pcd false, dtype bfloat16`。`--action-horizon 32` 是按这个推的，当初实际用的命令没有记录。**建议直接从 gpu006 拷贝已转换好的目录** `/mnt/nfs/fjj/Code/rlinf-safety-runtime/checkpoints/pi05_libero_safety_rlinf/`（13 GB，第 2.3 节），这样能保证和所有已有实验完全同一份权重；自己转换时，转完对照这份 `config.json`，并核对 `model.safetensors` 的键与 gpu006 上的一致。训练时 `num_action_chunks: 8`，实际只执行前 8 个动作。

---

## 4. 训练配置

E10-O 配置文件 `libero_safety_scgrpo_causal_e10o.yaml`：
- 在 gpu006 的 `/mnt/nfs/fjj/Code/CodeWorld-Safe/examples/embodiment/config/`，本地副本在 `E:\WM Safety\scgrpo_oracle_20261031\`；
- **还没提交到 GitHub**，需要原作者提交或单独发给你。

它和 E7（RM 奖励的长程实验）的唯一区别是奖励换成真实目标【已核实】：

| 项 | 值 |
|---|---|
| 并行 | `actor,env,rollout: 0-7`（8 卡），`total_num_envs: 128`，`rollout_epoch: 4`，每步 512 条轨迹，`group_size: 8` |
| 回合 | `max_episode_steps: 64`，`termination_source: horizon`（固定 64 步） |
| 奖励 | `reward_source: real`，`success_reward_rule: once`，`outcome_order: step` |
| 违规 | `end_on_violation: False`，`terminate_on_real_violation: False`（只记录） |
| 约束 | `eta_init: 0.0`，`eta_lr: 0.0`（SAM 代价只记录，不影响奖励） |
| 优化 | `lr: 5.0e-6`，`update_epoch: 1`，`adv_std_floor: 0.1`，`noise_level: 0.5` |
| 采样 | uniform（按关键帧均匀） |
| 起点 | `resume_dir: null`，从 SFT base 开始 |

**80 步要改的两项：**
```yaml
runner:
  max_steps: 80
  save_interval: 2      # 每 2 步存一次；保留哪些由 launch_train.sh 的 KEEP 控制
```

**换机器要改的路径**（配置里的绝对路径）：`runner.logger.log_path`、`env.train.model_path`、`VAE_path`、`keyframe_dir`、`channel_mask`、`sam_path`、`state_reader_path`、`reward_model.from_pretrained`、`rollout.model.model_path`、`actor.model.model_path`。`sync_real_env.py` 里还有默认的 `EPISODES=/data_cf/cws_sam/v1/labels/episodes.jsonl`，可以在配置的 `env.train` 下加 `episodes_jsonl:` 覆盖。

---

## 5. 训练

脚本都在 `toolkits/codeworld_safe/scgrpo/e1/`，原本放在 `/data_cf/cws_sam/v1/scgrpo/` 下运行（脚本里写死了这个目录，换机器要改 `S=`）：

| 脚本 | 作用 |
|---|---|
| `run_causal.sh` | 起独立的 Ray head（端口 `PORT`，临时目录 `/tmp/rp_c$TAG`），用 `libero_safety_scgrpo_causal_$TAG.yaml` 训练 |
| `launch_train.sh` | 外层循环：被杀后从最新的完整 checkpoint 续训；按 `KEEP` 清理旧 checkpoint |

启动（`KEEP` 是要保留的 step，`KEEP_LATEST=1` 表示另外保留最新 1 个）：

```bash
cd /data_cf/cws_sam/v1/scgrpo
TAG=e10o PORT=6691 WR=56000-57999 KEEP="10 20 40 60 80" KEEP_LATEST=1 \
  nohup setsid bash launch_train.sh e10o >> causal/e10o_driver.log 2>&1 < /dev/null &
```

进度：
```bash
grep -aoE "Global Step: *[0-9]+/[0-9]+" causal/train_e10o.log | tail -1
ls causal/e10o/scgrpo_causal_e10o/checkpoints/          # global_step_*
tail causal/e10o_driver.log                            # start / died / pruned / TRAIN_DONE e10o
```

预计耗时【按 E7 实测】：启动约 4 分钟，每步约 4.4 分钟，80 步约 6 小时。

---

## 6. LIBERO-Safety 评测

**协议**【已核实】：官方协议，60 个 cell（4 suite × L0/L1/L2 × 5 任务）× 20 次。L0/L1 用 `fresh_init/v1` 的初始状态，L2 用 LIBERO-Safety 自带的初始状态；最长 512 步，首次违反约束即结束；成功 = 达成目标且未违规。

**脚本**：
- `run_multi.sh`：把多个 checkpoint 的未完成 cell 分给 N 条评测链，每条链一张卡，完成后再重试一轮失败的 cell。
- 每条链 `eval_real_pN.sh`：每条链独立的 Ray 端口、worker 端口段、LIBERO 锁，每个 cell 限时 20 分钟。模板在 `eval_real_chain_template.sh`，服务器上已有 `eval_real_p20.sh` ~ `p31.sh`。

训完后评测 5 个 checkpoint（300 个 cell，6 条链）：

```bash
CK=/data_cf/cws_sam/v1/scgrpo/causal/e10o/scgrpo_causal_e10o/checkpoints
W=actor/model_state_dict/full_weights.pt
JOBS=""; for s in 10 20 40 60 80; do JOBS="$JOBS e10o_step_$s|$CK/global_step_$s/$W"; done
JOBS="$JOBS" GPUS_LIST="0 1 2 5 6 7" CHAINS="20 21 22 23 24 25" nohup setsid bash run_multi.sh e10o > causal/e10o_eval.log 2>&1 &
```

预计耗时：每个 cell 约 2~3 分钟，300 个 cell 分 6 条链，约 2 小时。

**汇总**（和 base 在相同的 cell、相同的初始状态上配对比较）：
```bash
cd diag
python matched_summ.py e10o_step_10 e10o_step_20 e10o_step_40 e10o_step_60 e10o_step_80   # 各 suite SR、unsafe、夹爪开合
python level_summ.py   e10o_step_10 e10o_step_20 e10o_step_40 e10o_step_60 e10o_step_80   # suite × level
```

**对照组**（已有结果，同为 uniform 采样）：

| 组 | 奖励 | step 10 | step 20 | step 40 | step 60 | step 80 |
|---|---|---|---|---|---|---|
| base | — | .703 | | | | |
| E8-O（4 卡，64 × 8） | 真实，固定 64 步 | .732 | | | | |
| E7（8 卡，128 × 4） | RM v3，`final` | | .672 | .652 | .647 | .679 |
| E7 affordance | | | .440 | .400 | .363 | .423 |

E10-O 和 E7 只差奖励来源（另有 `once` / `final` 的差别）。判断：
- **E10-O 在 40/60/80 不退化**：退化主要来自 RM；
- **E10-O 也退化**：说明还有优化漂移、Wan 观测误差等原因，下一步才考虑 E10-L（学习率 1e-6）或 SFT 回放。

除了官方 SR，还要看官方违规、辅助 unsafe（抓错部位等）、affordance 各 level、夹爪开合次数。

---

## 7. 注意事项（在 gpu006 上踩过的坑）

1. **GPU 3/4 不要跑评测**：评测默认 EGL 渲染，GPU 3 有 Xid 109 故障。训练用 osmesa，不受影响。
2. **每张卡只放 1 条评测链**，链之间错开 90 秒启动。12 条链同时 reset 会卡在 NVIDIA 驱动锁上，reset 从 4 秒变成 460 秒。
3. **每个 Ray head 单独指定端口**：`--min/max-worker-port`、`--ray-client-server-port` 互不重叠；worker 端口段不能包含 dashboard agent 的默认端口 52365；CPU 限 32。**永远不要执行 `ray stop`**，它会杀掉整台机器上所有的 Ray。
4. **停 Ray 残留进程时按环境变量 `RAY_TMPDIR` 找**。Ray client server 的命令行里不带临时目录，按命令行匹配会漏掉，残留进程占着端口会让新 head 卡死。
5. **不要用 `pkill -f` 匹配自己 ssh 命令里出现的字符串**，会把自己的 shell 杀掉。
6. **根分区 `/tmp` 要留空间**：Ray 的临时目录在 `/tmp`，可用空间低于 5% 时，对象溢写会失败。
7. **gpu006 上训练曾被其他账号反复用 sudo 杀掉**（10-28 到 10-31）。可以用 systemd 托管（`Restart=on-failure`、`KillMode=process`），再加每 5 分钟的定时检查，示例见 `night_pipeline.sh`、`cws-night.service`、`cws-e10-watch.*`。但对方执行 `systemctl stop` 就挡不住，**先和管理员确认机器归属**。

---

## 8. 产出清单

- 训练：`causal/e10o/scgrpo_causal_e10o/checkpoints/global_step_{10,20,40,60,80}` 和 TensorBoard（`causal/e10o/tensorboard/`）；
- 评测：`eval_real/e10o_step_{10,20,40,60,80}/`（每个 60 个 cell）；
- 汇总：上面两条命令的输出；
- 建议上传 ModelScope：评测结果全部，权重只传 `model_state_dict`。
