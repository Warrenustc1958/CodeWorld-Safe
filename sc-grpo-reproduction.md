# SC-GRPO / CoWM-PO 在 BAAI PPU 机器上的部署复现指南

更新：2026-11-02。用途：交给目标机器上的 Claude，按本文从零建环境、下载资源、跑通现有训练。第 9 节是可以直接粘贴的完整 prompt。

标注：**【已核实】** = 在 gpu006 上确认过；**【需确认】** = 未核实，部署时必须先查证；**【缺口】** = 目前不在任何公开或 ModelScope 位置，需要从 gpu006 / Song2 补传。

---

## 0. 目标机器

| 项         | 值                                                           |
| ---------- | ------------------------------------------------------------ |
| 登录       | 堡垒机 `ssh -p 2224 huangxu@bastion.aiops.baai.ac.cn` → 菜单 `p` → `4`（资产 247-pm-aliyun-wlcb-zoned-d-810e-96G）→ 账号 `1`（root）。密码不写入本文 |
| 主机       | `dsw-896086-588f6f777f-9qswv`（阿里云 PAI-DSW 容器，乌兰察布） |
| 加速卡     | 16 × 平头哥 **PPU-ZW810E**（96 GB），`nvidia-smi` 可用，**不是 NVIDIA GPU** |
| CPU / 内存 | 176 核 / 1.8 TB，无 swap                                     |
| 系统       | Ubuntu 24.04.2，系统 Python 3.12.3（`/usr/local/bin/python`） |
| 预装       | **厂商适配 torch 2.9.0**、torchvision 0.24.0、torchaudio 2.9.0、torchao 0.11.0、vllm 0.13.1（+vllm-plugin-FL） |
| 项目目录   | `/mnt/cpfs/jieduan`（CPFS 共享卷，20 T，**已用 95%，剩 1.2 T，是全卷余量**） |
| 网络       | GitHub 200、ModelScope 302（可达）、hf-mirror 200            |

与 gpu006 的关键差异（决定部署策略）：

1. **torch 不能重装。** gpu006 用 torch 2.6.0+cu124；这里必须用厂商 torch 2.9.0，否则 PPU 不可用。所有 venv 用 `--system-site-packages`，所有安装带 `-c torch_pin.txt` 约束。
2. **没有 flash-attn**（CUDA 专用内核），π0.5 / Wan 注意力改走 SDPA。
3. **EGL 渲染大概率不可用**，训练和评测都用 `MUJOCO_GL=osmesa`（纯 CPU，176 核足够）。
4. **jax[cuda12] 不可用**，jax 只用于 π0.5 权重转换，装 CPU 版即可（或直接下载已转换权重，见 2.3）。
5. Python 3.12（gpu006 为 3.11），部分固定版本可能需要放宽。
6. NCCL：PPU 的集合通信库是否以 `nccl` 后端暴露给 torch.distributed **【需确认】**（FSDP 和 Ray 多卡依赖它）。

---

## 1. 代码

| 仓库                      | 地址                                             | 分支 / 版本                                  | 用途                                                         |
| ------------------------- | ------------------------------------------------ | -------------------------------------------- | ------------------------------------------------------------ |
| CodeWorld-Safe（SC-GRPO） | https://github.com/Warrenustc1958/CodeWorld-Safe | `main`，≥ `9e1b20f7`（11-02 推送）【已核实】 | RLinf fork + SC-GRPO 世界模型环境 + 评测/诊断脚本 + RM 重训工具（`toolkits/codeworld_safe/rm2/`） |
| CoWMPO                    | https://github.com/Warrenustc1958/CoWMPO         | `cowmpo`，≥ `8091fa91`（11-01 推送）         | Code WM（MuJoCo 代码世界）+ OFT 后训练                       |
| LIBERO-Safety             | https://github.com/LIBERO-SAFETY/LIBERO-Safety   | 默认分支                                     | 任务/BDDL/grader + `third_party/robosuite-1.4`               |
| diffsynth-studio          | https://github.com/RLinf/diffsynth-studio        | commit `2a2e05f`【已核实】                   | Wan 推理，用 `PYTHONPATH` 引入，不 pip 安装                  |

**GitHub 版落后于工作副本的部分【缺口】：**

- SC-GRPO：11-02 起 E10-O、E11–E16、M0–M3、M1R 和阶段 1 收集（col*）的配置都已提交到 `main`【已核实】。E7/E8 系列配置是否都已提交 **【需确认】**，部署后 `ls examples/embodiment/config/libero_safety_scgrpo_causal_*.yaml` 核对。
- CoWMPO：11-01 之后 stove v69–v78、G2 v2（`claude_g2v2_20261101`）在 gpu006 `/data_cf/cowmpo/staging/` 下，未合入 `toolkits/cowm`。

---

## 2. 模型与数据下载

### 2.1 公开资源

| 内容                                                | 来源                                                         | 用于               | 备注                                                         |
| --------------------------------------------------- | ------------------------------------------------------------ | ------------------ | ------------------------------------------------------------ |
| π0.5 LIBERO-Safety SFT（JAX/orbax）                 | HF `LIBERO-Safety/pi05_libero_safety`                        | SC-GRPO 策略初始化 | 需转换成 OpenPI_RLinf 格式（2.3）                            |
| Wan2.2 VAE `Wan2.2_VAE.pth`                         | HF `RLinf/RLinf-Wan-LIBERO-Spatial`                          | SC-GRPO 世界模型   | 只取 VAE 文件                                                |
| RLinf-Wan-LIBERO-Goal（含 `taskemb_resnet_rm.pth`） | HF `RLinf/RLinf-Wan-LIBERO-Goal`                             | CoWMPO 共享 RM     |                                                              |
| OpenVLA-OFT SFT libero-goal traj1                   | HF `Haozhan72/Openvla-oft-SFT-libero-goal-traj1` **【需确认仓库名】** | CoWMPO 策略        | gpu006 路径 `/mnt/nfs/fjj/Models/Openvla-oft-SFT-libero-goal-traj1` |
| GroundingDINO / SAM2 权重                           | HF（具体仓库 **【需确认】**，以 gpu006 `/data_cf/cowmpo/builder_assets/models/` 下的目录名为准） | CoWMPO 感知（S5）  | 只在跑感知时需要                                             |
| LIBERO-Safety 仿真资产 `assets.zip`                 | 按 LIBERO-Safety 仓库 `SAFELIBERO_INTEGRATION.md` 的链接     | 两个项目的仿真器   | 解压到 `LIBERO-Safety/libero/libero/assets`（约 17 GB）      |

HF 统一走镜像：`export HF_ENDPOINT=https://hf-mirror.com`，大文件 `huggingface-cli download --resume-download`；遇 429 降并发。

### 2.2 我们自己上传的 ModelScope 资源

| ModelScope 仓库                                              | 类型   | 内容                                                         | 大小                                           |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ | ---------------------------------------------- |
| `Warrenustc1958/Wan-CodeWorld-Safe-pi05-dual-41000`          | 模型   | 冻结 Wan 双视角 DiT ema-step-41000（SC-GRPO 世界模型）；`rm/v1`、`rm/v2`、`rm/v3`（`rm_v3_agent_w41k` = SC-GRPO 训练用 RM，含 best.pt + args） | DiT 9.4 GB                                     |
| `Warrenustc1958/Wan-CodeWorld-Safe-pi05-single-40000`        | 模型   | 单视角 Wan ema-40000（不含 VAE），对照用                     | —                                              |
| `Warrenustc1958/CodeWorld-Safe-LIBERO-Safety-pi05-GRPO-rollout` | 数据集 | `keyframes_v1/`（训练起点 47,658 帧，大文件切 4 GB：`cat memory.npy.part_* > memory.npy` 后 `sha256sum -c SHA256SUMS`）；`scoring/`（`sam_cost_all_d32/best.pt`、`channel_mask.json`、`state_reg_v1/state_reg.pt`、`rm_v3_agent_w41k/best.pt`）；`eval/fresh_init_v1/`（L0/L1 评测初始状态）；`scgrpo_e1_e8_20261030/`（R1-all/E2/E8-O/E8-a/E8-b 的 `model_state_dict`、RM v6、`frames_d2`、配置、真实评测结果含 `eval_real/off_base/`、TensorBoard、gate 结果） | keyframes 88 GB                                |
| 同上 GRPO-rollout 数据集，`scgrpo_rm2_20261102/`             | 数据集 | 11-02 新增。**已核实上传（大小逐文件核对）**：`rm_models/`（旧 RM v1–v3c、v7、v7a、v7w、v8；**R-A** = 重训的双视角 ResNet RM，`best.pt` 内含阈值；**R-C** = Qwen3-VL-4B-Instruct 的 LoRA r16 适配器 `best/`）、`rm_eval/`（所有 RM 在 ds_v1 上的统一离线评估）、`rm_data/ds_v1/`（RM 数据集）、`keyframe_scan/`（47,658 个关键帧 reset 时的 grader 扫描、剔除表与采样权重）。**上传中**：`checkpoints/`（E10-O s10/20/40、E11-SC/SAM s20、E12-R、E13、E15-A、E15-Q 的 actor 权重，>4 GB 分片）、`collection/`、`dumps/`、`eval_real/`、`reports/`；以 `scgrpo_rm2_20261102/README.md` 出现为准 | RM 部分约 2.5 GB；全部 116.8 GB                |
| `Qwen/Qwen3-VL-4B-Instruct`（公开）                          | 模型   | R-C 的基座模型                                               | 8.3 GB                                         |
| `Warrenustc1958/CodeWorld-Safe-pi05-counterfactual-rollout`  | 数据集 | `step_e/runs/`（Step E 反事实 rollout，审计 json + `state/*.npz` 仿真器快照 + visual）；`legacy/rollouts/`；另有 `rm/`、`sam/`（RM 帧与 SAM 缓存，10-25 上传）**【需确认 rm/ sam/ 是否在此仓库】** | 全量 570.8 GB，**只下 json + state，约 30 GB** |
| `Warrenustc1958/Pi05_LIBERO_Safety_counterfactual_v1` **【需确认 owner】** | 数据集 | 早期 8216 条 CF 数据                                         | 本次不需要                                     |

私有仓库需要 token：`modelscope login --token <TOKEN>`（token 由用户提供，不写入任何文件或日志）。

下载示例：

```python
from modelscope.hub.snapshot_download import snapshot_download
D = "/mnt/cpfs/jieduan/data"
snapshot_download("Warrenustc1958/Wan-CodeWorld-Safe-pi05-dual-41000", local_dir=f"{D}/wan_dual_41000")
snapshot_download("Warrenustc1958/CodeWorld-Safe-LIBERO-Safety-pi05-GRPO-rollout", repo_type="dataset",
                  allow_patterns=["keyframes_v1/*", "scoring/*", "eval/*", "scgrpo_e1_e8_20261030/eval_real/off_base/*"],
                  local_dir=f"{D}/scgrpo_bundle")
snapshot_download("Warrenustc1958/CodeWorld-Safe-pi05-counterfactual-rollout", repo_type="dataset",
                  allow_patterns=["step_e/runs/**/*.json", "step_e/runs/**/state/*.npz",
                                  "legacy/rollouts/**/*.json", "legacy/rollouts/**/state/*.npz"],
                  local_dir=f"{D}/cws_rollout")
# 11-02 新 RM（R-A / R-C）、RM 数据集、离线评估、关键帧扫描
snapshot_download("Warrenustc1958/CodeWorld-Safe-LIBERO-Safety-pi05-GRPO-rollout", repo_type="dataset",
                  allow_patterns=["scgrpo_rm2_20261102/rm_models/R-A_resnet_rm2/*",
                                  "scgrpo_rm2_20261102/rm_models/R-C_qwen3vl4b_lora/*",
                                  "scgrpo_rm2_20261102/rm_eval/*", "scgrpo_rm2_20261102/keyframe_scan/*",
                                  "scgrpo_rm2_20261102/rm_data/ds_v1/*"],
                  local_dir=f"{D}/scgrpo_bundle")
snapshot_download("Qwen/Qwen3-VL-4B-Instruct", local_dir=f"{R}/models/Qwen3-VL-4B-Instruct")  # R-C 基座
```

R-C 的使用：Qwen3-VL 需要 transformers ≥ 4.57，而 openpi 环境是 4.53，所以 R-C 在单独的 venv 里作为推理服务运行（`toolkits/codeworld_safe/rm2/qwen_rm_server.py`，每卡一个进程，端口 29700+卡号），env 里用 `reward_model.type: QwenRMClient` 调用。gpu006 的 Qwen venv：torch 2.6.0、transformers 4.57.1、peft 0.17.1、qwen-vl-utils 0.0.14（`rm2/../scgrpo/night2/setup_qwen.sh`）；PPU 上同样要继承厂商 torch。

### 2.3 不在 ModelScope 上、需要补传的【缺口】

| 内容                                                         | gpu006 路径                                                  | 大小                   | 为什么需要                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------- | ------------------------------------------------------------ |
| π0.5 已转换权重（OpenPI_RLinf）                              | `/mnt/nfs/fjj/Code/rlinf-safety-runtime/checkpoints/pi05_libero_safety_rlinf/` | 13 GB                  | 保证与所有已有实验同一份权重；自己转换时 `--action-horizon 32` 是推测值 |
| 快照索引 `labels/episodes.jsonl`                             | `/data_cf/cws_sam/v1/labels/`                                | 187 MB                 | `sync_real_env` 用它把关键帧映射到仿真器快照（所有 causal / E 系列配置都需要） |
| CoWMPO 数据 `/data_cf/cowmpo/{evidence,worlds,init,s0-s3,staging,target_eval,priors}` | gpu006；Song2 副本 `/NAS/fujiajun/data_cf/cowmpo`            | 数十 GB **【需确认】** | CoWMPO 的冻结世界、证据、初始状态                            |

建议由用户在 gpu006 上把这几项打包上传到 ModelScope（例如新建数据集 `Warrenustc1958/CoWMPO-assets` 和在 GRPO-rollout 数据集里加 `labels/`、`pi05_rlinf/`），目标机器再下载。目标机器无法直连 gpu006。

### 2.4 π0.5 自己转换（拿不到已转换权重时）【已核实，11-02】

**用的脚本：** CodeWorld-Safe 仓库自带的统一转换入口 `rlinf/utils/ckpt_convertor/openpi/convert.py`，模式 `jax_to_openpi_rlinf`（实现在同目录 `jax_to_openpi_rlinf.py`，说明见同目录 `README.md`）。不需要另装 openpi 官方仓库。

**输入（HF `LIBERO-Safety/pi05_libero_safety`，约 12 GB）的目录结构**，gpu006 `/mnt/nfs/fjj/Models/pi05_libero_safety` 实测：

```
pi05_libero_safety/
├── params/                         # orbax / ocdbt 格式：_METADATA、manifest.ocdbt、ocdbt.process_0/、d/ …
└── assets/lerobot/norm_stats.json
```

**输出**（gpu006 `/mnt/nfs/fjj/Code/rlinf-safety-runtime/checkpoints/pi05_libero_safety_rlinf`，所有实验共用的这份）：

```
pi05_libero_safety_rlinf/
├── model.safetensors               # 13.4 GB（权重按 fp32 写出）
├── config.json
└── assets/lerobot/norm_stats.json  # 配置里 norm_stats_path: ${..model_path}/assets/lerobot/norm_stats.json
```

**命令：**

```bash
cd $R/code/CodeWorld-Safe
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download LIBERO-Safety/pi05_libero_safety --local-dir $R/models/pi05_libero_safety --resume-download

# jax / orbax 只在这一步用到；gpu006 上是 jax 0.5.3 + orbax-checkpoint 0.11.13。PPU 上装 CPU 版 jax，转换在 CPU 上做
pip install -c $R/envs/torch_pin.txt "jax==0.5.3" "orbax-checkpoint==0.11.13"

python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi_rlinf \
  --input-model       $R/models/pi05_libero_safety \
  --input-norm-stats  $R/models/pi05_libero_safety/assets/lerobot/norm_stats.json \
  --output-model      $R/models/pi05_libero_safety_rlinf \
  --output-norm-stats $R/models/pi05_libero_safety_rlinf/assets/lerobot/norm_stats.json
```

- **形状参数：** `--action-dim 32`、`--action-horizon 32`、`--max-token-len 200`、`--paligemma-variant gemma_2b`、`--action-expert-variant gemma_300m` 都是脚本默认值，和我们这份权重的 `config.json` 一致，所以不用另外指定；pi05 默认开启，转非 pi05 模型才加 `--no-pi05`。
- **norm_stats 路径要改：** 仓库 README 示例里的 `--output-norm-stats` 是 `physical-intelligence/behavior/norm_stats.json`，那是 BEHAVIOR 任务的布局。**我们的配置读的是 `assets/lerobot/norm_stats.json`，必须按上面的命令写。**
- **转完核对 `config.json`：** `action_dim 32, action_horizon 32, max_token_len 200, paligemma_variant gemma_2b, action_expert_variant gemma_300m, pi05 true, pcd false, dtype bfloat16`（gpu006 这份就是这些值）。`dtype bfloat16` 只是给加载器的提示，文件里的权重是 fp32，所以 `model.safetensors` 约 13.4 GB。
- **其他模式：** 同一个入口还支持 `openpi_pytorch_to_openpi_rlinf`（已有 OpenPI PyTorch 版权重时用）、`openpi_rlinf_to_openpi_pytorch`、`sft2deploy`，用法见 `README.md`；`--mode <模式> --help` 可查看各自参数。
- **一致性：** 我们这份权重生成于 10-13。自己转换出来的结果理论上应该一致，但没有逐位比对过。如果最终拿到了 gpu006 的原件（2.3 节缺口），优先用原件。

### 2.5 存储预算

| 项                                                | 大小                                                         |
| ------------------------------------------------- | ------------------------------------------------------------ |
| LIBERO-Safety + assets                            | 18 GB                                                        |
| π0.5（转换后，自转再 +12 GB JAX 原件）            | 13 GB                                                        |
| Wan DiT + VAE                                     | 12 GB                                                        |
| keyframes_v1                                      | 88 GB（合并分片时峰值约 176 GB，合并后删分片）               |
| 快照 json + state                                 | 30 GB                                                        |
| 评分模型 / 评测初始状态 / base 评测               | 1 GB                                                         |
| venv × 2–3                                        | 30 GB                                                        |
| CoWMPO 模型 + 数据                                | 约 50 GB **【需确认】**                                      |
| SC-GRPO checkpoint（每个 15 GB，保留 5 + 最新 1） | 90 GB                                                        |
| **合计**                                          | **约 330 GB**，低于 CPFS 余量 1.2 T，但卷是共享的，**任何时候不要让 `/mnt/cpfs` 可用空间低于 500 GB** |

Ray 临时目录放 `/mnt/cpfs/jieduan/tmp/ray_*`（或本地盘，先 `df -h /tmp` 确认），不要写满根分区。

---

## 3. 环境配置

### 3.1 原则

- 不动系统 Python 的已装包；不 `pip install torch*`；不 `apt upgrade`。

- 先锁 torch：

  ```bash
  pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton|torchao)[=@ ]' > $R/envs/torch_pin.txt
  ```

- venv 继承系统包：`python -m venv --system-site-packages $R/envs/cws`，之后所有安装 `pip install -c $R/envs/torch_pin.txt ...`；装完每次都跑一次第 3.4 节的 torch 检查，确认没被替换。

- 不跑 `requirements/install.sh` 的 torch 安装段（它会装 cu124 torch）；参照它的包列表手工装。

### 3.2 gpu006 参考版本【已核实】

| 组件                   | gpu006                             | 本机建议                                           |
| ---------------------- | ---------------------------------- | -------------------------------------------------- |
| Python                 | 3.11.14（uv）                      | 系统 3.12.3                                        |
| torch / torchvision    | 2.6.0 / 0.21.0 cu124               | **保留厂商 2.9.0 / 0.24.0**                        |
| jax / orbax-checkpoint | 0.5.3 / 0.11.13                    | `jax==0.5.3`（CPU 版，不带 `[cuda12]`），仅转换用  |
| ray                    | 2.59.0                             | 2.59.0（需确认有 py3.12 wheel，否则取最近的 2.5x） |
| transformers           | 4.57.6                             | 4.57.6                                             |
| numpy                  | 1.26.4                             | 1.26.4                                             |
| flash-attn             | 2.7.4.post1                        | **不装**，确认代码回退 SDPA                        |
| hydra-core / omegaconf | 1.4.0.dev6 / 2.4.0                 | 同                                                 |
| safetensors            | 0.8.0                              | 同                                                 |
| rlinf-openpi           | 0.1.1                              | 同（若强依赖 jax cuda 则 `--no-deps`）             |
| modelscope             | 1.20.0                             | 同                                                 |
| mujoco                 | 3.8.1（install.sh 固定 3.3.7）     | 3.3.7                                              |
| robosuite / bddl       | 1.4.0（LIBERO-Safety 自带）/ 1.0.1 | 同                                                 |
| 渲染                   | 训练 osmesa，评测 EGL              | **全部 osmesa**                                    |

openpi 依赖列表：`CodeWorld-Safe/requirements/embodied/models/openpi.txt`；LIBERO-Safety 安装：`requirements/install.sh` 中 `install_liberosafety_env` 函数（卸载标准 libero/robosuite → 装 mujoco、`third_party/robosuite-1.4` editable、LIBERO-Safety editable、`bddl==1.0.1 future easydict imageio[ffmpeg] scikit-image usd-core wand matplotlib termcolor`，生成 `$VIRTUAL_ENV/share/libero-safety/config.yaml`）。

### 3.3 系统依赖

```bash
apt-get update && apt-get install -y libosmesa6 libosmesa6-dev libgl1 libegl1 ffmpeg git imagemagick libmagickwand-dev
```

Ubuntu 24.04 上 OSMesa 库路径用 `ldconfig -p | grep -i osmesa` 确认后填 `PYOPENGL_OSMESA_LIBRARY`。

### 3.4 运行前环境变量（SC-GRPO）

```bash
R=/mnt/cpfs/jieduan; V=$R/envs/cws; source $V/bin/activate
export PYTHONPATH=$R/code/diffsynth-studio:$R/code/CodeWorld-Safe:$PYTHONPATH
export LIBERO_TYPE=safety RLINF_LIBERO_SAFETY_AUX=1 LIBERO_CONFIG_PATH=$V/share/libero-safety
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYOPENGL_OSMESA_LIBRARY=<ldconfig 查到的 libOSMesa.so 路径>
export RAY_AUTH_MODE=disabled OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com
```

验收：

```bash
python -c "import torch;print(torch.__version__,torch.cuda.is_available(),torch.cuda.device_count())"   # 必须仍是厂商 2.9.0、16 卡
python -c "import torch.distributed as d;print(d.is_nccl_available())"
python -c 'import libero.libero as l; print(l.get_libero_path("benchmark_root"))'
python -c 'import ray, diffsynth, mujoco; print(ray.__version__, mujoco.__version__)'
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/unit_tests/test_world_model_safe.py tests/unit_tests/test_safety_cost_contact.py   # gpu006 上 63 passed（11-02）；第二个文件是真实 MuJoCo 接触的集成测试，需要 LIBERO-Safety 运行环境
```

### 3.5 CoWMPO 环境

gpu006 有三个 venv（均在 CoWMPO 仓库旁）：

- `.venv-oft-wan`：策略 / RM / LIBERO（RLinf + OpenVLA-OFT），Target 侧；
- `.venv-builder`：代码世界编译与仿真（无 libero），只能经沙箱 `toolkits/cowm/isolation/run_builder.sh` 运行，Builder 不可读 Target GT；
- `.venv-perception`：GroundingDINO + SAM2。

Song2 的重建脚本可作参考：`/NAS/fujiajun/tmp/build_oft_wan_song2.sh`（uv `--no-deps` 按 gpu006 freeze 安装；freeze 里 swanlab/wrapt 有冲突）。本机同样用 `--system-site-packages` + torch 约束，不装 cu128 torch。

运行约定【已核实】：

- Builder：`COWM_PYTHONPATH=$C/toolkits MUJOCO_GL=osmesa run_builder.sh python <script>`
- Target：`.venv-oft-wan` 的 python，`PYTHONPATH=$C/toolkits:staging`，`flock /tmp/rlinf-libero-run.lock`
- Target 侧 stove 导出必须 OSMesa（EGL 破坏 prefix 逐位一致检查）。

---

## 4. SC-GRPO：方法与当前结论

### 4.1 设置

- 策略：π0.5（openpi_rlinf，3.35B），`train_expert_only=True`，只训 action expert。
- 环境：冻结 Wan 双视角 41k 生成想象画面；RM v3（`rm_v3_agent_w41k`）判成功；SAM（`cost_all_d32`）给代价；`state_reg_v1` 从画面读 proprio。
- 算法：RLinf（Ray + FSDP），flow_sde，GRPO G=8，`adv_std_floor 0.1`，训练 noise 0.5、评测 0.3。
- 诊断环境 `SyncRealWorldModelEnv`（`rlinf/envs/sim/world_model/safe/sync_real_env.py`）：Wan 推演的同时让真实 LIBERO 用快照恢复并同步执行同一动作，给出真实目标 / 违规标签。开关：`reward_source`（real|rm）、`termination_source`（rm|real|horizon）、`outcome_order`（legacy|step）、`success_reward_rule`（once|final）、`end_on_violation`、`terminate_on_real_violation`。
- 真实评测：LIBERO-Safety 官方协议，4 suite（affordance / human_safety / obstacle_avoidance / obstacle_avoidance_human）× L0/L1/L2 × 5 任务 = 60 cell × 20 次；L0/L1 用 `fresh_init/v1`，L2 用官方初始状态；最长 512 步，首次违规即结束。

### 4.2 越训越差：现象

主线 `scgrpo_pi05_wan41k_main`（RM 奖励 + RM 判成功即结束 + SAM 代价约束），真实总体 SR：

| step | base | 20   | 80   | 120  | 160  | 200  | 240  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| SR   | .703 | .713 | .623 | .597 | .543 | .529 | .547 |

想象里成功率平在 .40–.45、代价平在 .19–.22；eta 0.2 → 5.5（budget 0.13 不可达）；真实失败全是超时，夹爪开合次数翻倍（affordance 1.7 → 5.8 次/回合）。

### 4.3 排查结论（截至 10-31）

| 假设                                                         | 证据                                                         | 判定                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------- |
| **RM 瞬时误报被结算为成功奖励**（RM 判成功即结束，或固定回合下 `once`） | RM-all step 10/20/40 = .696/.642/.609；E8-a（固定 64 步 + `once`）.664；E8-b（`final`）.692 ≈ base；有真实信息的 E2 / E8-O 不退化 | **主因**                                |
| RM 精度不足（约 90% 误报 = 抓错部位）                        | 首次决策 precision v3 .60；RM v6 也没过 .93 门槛；E8-b 比 E8-O 低约 4 个点 | 主因的前提                              |
| 优势归一化放大噪声                                           | 消融 a 修了仍退化                                            | 次要，已修（`adv_std_floor`）           |
| eta 不可达 / 拉格朗日爆炸                                    | 消融 b/c 修了仍退化                                          | 次要                                    |
| 想象环境奖励松手                                             | 强制张开夹爪 RM 成功率 .59                                   | 属于 RM 漏洞                            |
| SigLIP train/eval 不一致                                     | 伪 KL ≈ approx_kl，未超 clip                                 | 次要，已修（`pi0.py`，commit 8d3179d1） |
| Wan 噪声                                                     | 拷贝回放 RM 成功翻转 2/32 组                                 | 对代价影响大，对成功影响小              |

关键实验（全部 60 cell × 20，base .703）：

| 实验                        | 奖励         | 终止             | step 10 | step 20 | step 40 | step 60 | step 80 |
| --------------------------- | ------------ | ---------------- | ------- | ------- | ------- | ------- | ------- |
| R1-all                      | 真实目标     | 真实             | .724    | .712    |         |         |         |
| RM-all                      | RM v3 0.8×4  | RM 判成功        | .696    | .642    | .609    |         |         |
| E2                          | RM v3        | 真实（不可部署） | .722    | .711    |         |         |         |
| E8-O                        | 真实，`once` | 固定 64 步       | .732    |         |         |         |         |
| E8-a                        | RM，`once`   | 固定 64 步       | .664    |         |         |         |         |
| E8-b                        | RM，`final`  | 固定 64 步       | .692    |         |         |         |         |
| E7（E8-b 配置，8 卡 80 步） | RM，`final`  | 固定 64 步       |         | .672    | .652    | .647    | .679    |

E7 affordance：step 20/40/60/80 = .440/.400/.363/.423（base .530），说明 `final` 规则在 10 步内止住退化，但长程仍漂移。

### 4.4 11-01 / 11-02 进展【已核实】

**实验结果**（60 cell × 20，step 10，单种子；base .703 / unsafe .198）：

| 实验            | 设置                                       | SR                                         | unsafe      | 结论                                   |
| --------------- | ------------------------------------------ | ------------------------------------------ | ----------- | -------------------------------------- |
| E10-O s20 / s40 | 真实奖励，Wan 观测                         | **.769** / .738                            | .143 / .146 | 真实奖励有效；s20 后回落               |
| E12-R           | 真实奖励，真实渲染观测                     | 与 E10-O 相当（57 cell 配对 .730 对 .750） | .161        | Wan 观测作为策略输入不是瓶颈           |
| E13             | E10-O + grader 代价，固定 η 0.5            | .744                                       | .189        | 修好标签后 10 步内安全约束仍无收益     |
| E14             | 真实奖励 + 随机翻转（误报 .05 / 漏报 .14） | .693                                       | .197        | **几个百分点的随机误判就抹掉全部收益** |
| E15-A           | 新 RM R-A 作奖励                           | .699                                       | .209        | 与 base 相同                           |
| E15-Q           | 新 RM R-C 作奖励                           | .694                                       | .206        | 与 base 相同                           |

**主要发现：**

1. **LIBERO-Safety 的 robot 接触谓词永不触发**（geom 名字和 int id 比较），E11-SC 的代价里从来没有 robot 违规。修复：`sync_real_env.py` 新增 `cost_label: grader`，训练代价改用评测 grader 的二值判定，不改 LIBERO。
2. **RM 离线精度不能预测 GRPO 效果。** 统一标准下，旧 RM v5/v6 测试 AP .83–.86，R-A .911，R-C .860，但在 GRPO 里全部无效。GRPO 的组内标准差归一化会把"组内结果全相同的组"里的一次误判放大成和真实成功一样大的更新。
3. **组内信号由起点决定。** 同一关键帧前后两次的组成功率相关系数 .976；训练关键帧上只有约 16% 的组有组内差异。探索噪声从 0.5 调到 1.0 几乎没用；从初始状态起跑 256 步能把它提到 .47，但 RM 的误报随回合深度从 1.3% 升到 11.7%。
4. **用 RM 组内平均挑起点能把组内差异从 .19 提到 .47，但挑出来的正是 RM 最不准的边界状态**（误报 13%、漏判 47%），优势余弦最高 .56，没过预注册门槛 .60。

详见仓库 `toolkits/codeworld_safe/scgrpo/{E12_E14_RM_PROGRESS,VARIANCE_PREREG,VARIANCE_MEASUREMENT,RM2_PLAN,RM_GRPO_THEORY}.md`。

**新增环境选项**（`sync_real_env.py` / `safe_env.py`）：`cost_label`、`cost_window`、`obs_source: real` + `render_egl_remap`、`reward_noise`、`dump_dir` + `dump_arrays`、`keyframe_exclude`、`keyframe_sampling: weighted`、`rm_frames`、`reward_model.type: QwenRMClient`。

**进行中 / 下一步：**

1. **M1R**（真实渲染观测 + 初始状态 + 256 步，策略冻结，只测 R-A）：区分长回合下 RM 变差是 Wan 漂移还是 RM 分布外。11-02 15:53 重新启动。
2. **E16-L**（初始状态 + 256 步 + 真实奖励，训 10 步并评测）：端到端检验长程 Wan 观测能否支撑学习。15:06 被其他用户停掉，尚未重跑。
3. 待验证的方向：在边界状态上用 on-policy 难样本迭代训练 RM；成败判定加入状态信息；直接测量 Wan 漂移（Wan 观测运行中每个 chunk 末同时渲染真实帧）。
4. 待办：col11m（E11-SAM s20 策略）的收集因时间被砍掉，补收集步骤见 `RM2_PLAN.md` 5.1 节。

**gpu006 资源注意：** 11-02 两次有其他用户用 sudo `systemctl stop/kill` 停掉我们的服务（cws-ms2、cws-m1r、cws-e16l）。PPU 机器上长任务开始前先确认可用的卡、时长和磁盘上限。

完整排查记录：仓库 `toolkits/codeworld_safe/scgrpo/DIAGNOSIS.md`（= `E:\WM Safety\SC-GRPO_越训越差根因排查与实验规划.md`）；E10-O 细节：`E:\WM Safety\SC-GRPO_E10-O复现指南.md`。

### 4.5 E7 / E10-O 配置要点【已核实】

| 项   | 值                                                           |
| ---- | ------------------------------------------------------------ |
| 并行 | `actor,env,rollout: 0-7`，`total_num_envs 128`，`rollout_epoch 4`，每步 512 条轨迹，`group_size 8` |
| 回合 | `max_episode_steps 64`，`termination_source: horizon`        |
| 奖励 | E7：`reward_source: rm`，`success_reward_rule: final`；E10-O：`reward_source: real`，`once`；均 `outcome_order: step` |
| 违规 | `end_on_violation: False`，`terminate_on_real_violation: False` |
| 约束 | `eta_init 0.0`，`eta_lr 0.0`                                 |
| 优化 | `lr 5e-6`，`update_epoch 1`，`adv_std_floor 0.1`，`noise_level 0.5` |
| 保存 | `max_steps 80`，`save_interval 2`                            |

换机器要改的绝对路径：`runner.logger.log_path`、`env.train.model_path`、`VAE_path`、`keyframe_dir`、`channel_mask`、`sam_path`、`state_reader_path`、`reward_model.from_pretrained`、`rollout.model.model_path`、`actor.model.model_path`，以及 `env.train.episodes_jsonl`（覆盖 `sync_real_env.py` 默认的 `/data_cf/cws_sam/v1/labels/episodes.jsonl`）。`episodes.jsonl` 内的快照路径用 sed 改为本机：

```bash
sed -e "s|/data_cf/cws/step_e/runs/|$D/cws_rollout/step_e/runs/|g" \
    -e "s|/mnt/nfs/fjj/Code/rlinf-safety-runtime/rollouts/|$D/cws_rollout/legacy/rollouts/|g" \
    episodes.jsonl.orig > $D/labels/episodes.jsonl
```

脚本 `toolkits/codeworld_safe/scgrpo/e1/{run_causal.sh,launch_train.sh,run_multi.sh,eval_real_chain_template.sh}` 里写死了 `S=/data_cf/cws_sam/v1/scgrpo`，要改成本机工作目录。

---

## 5. CoWM-PO：方法与当前状态

- 思路：用 Code Agent 从公开 RGB 证据构建 MuJoCo 代码世界（Code WM），作为 OFT（OpenVLA-OFT libero-goal）后训练的模拟器。Builder（只看公开证据）与 Target（真实 LIBERO，持有 GT）严格隔离。
- 阶段：S0–S4 编译/执行/L0 后端可用；S5 状态估计（GDINO+SAM2，rack 检测/关联仍弱，S5 静态 3.7 cm 为已知限制）；S6 物理辨识；S7 RM。
- 11-01 状态：rack 已接受（`rack_hfield_v48/iter5`，v49/v50 6/6）；G2 v2 通过（1.512 px，100 条完整）；stove 止损（4 次独立验收 REJECT，最好候选 `stove_combo_v89 r3_000`）；long-hold 偏紧握（下一步：从公开 hold 视频辨识夹爪-物体摩擦）。
- **L1 GRPO 仍阻塞**：能跑但奖励为 0（共享 RM 特征 target vs L1 cos 0.72），不要扩大训练。
- 本机可跑的"现有训练" = **s0 冒烟**：`PLACE=3 NENVS=10 EPOCHS=1`，在 gpu006 全量 123/250 = .49，Song2 冒烟 4/10 = .40。具体启动脚本名 **【需确认】**（在仓库里 `grep -rn "s0" toolkits/cowm examples/embodiment/config | head`，或看计划书 §9.7）。
- 必读：`E:/WM as Simulator/CoWMPO推进进展与问题.md`（顶部为最新）、`CodeWM_PO_Research_Plan_20261027.md`（§9.6.1 进度表、§9.7）、`sessions/00_共同规则.md`。

---

## 6. 冒烟与验收顺序

| 阶段 | 内容                                                         | 通过标准                                                     |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| P0   | 硬件 / 通信检查：torch 2.9.0 bf16 matmul、16 卡可见、`torch.distributed` 2 卡 all_reduce | 全部成功                                                     |
| P1   | 建 venv，装依赖，torch 未被替换                              | 3.4 节验收全过；单测 14 passed                               |
| P2   | 下载 2.1/2.2 资源，补 2.3 缺口；keyframes 校验 sha256        | 文件数/大小与本文一致                                        |
| P3   | π0.5 单卡前向：加载权重，对一帧 LIBERO 观测出 action chunk   | 输出形状正确、无 NaN                                         |
| P4   | Wan 单卡前向：从 1 个关键帧生成 1 个 chunk                   | 画面非噪声，耗时记录                                         |
| P5   | SC-GRPO 冒烟：E8-b 配置，`actor,env,rollout: 0-1`，`total_num_envs 16`，`rollout_epoch 1`，`max_steps 2`，`save_interval 1` | 2 步训完、存出 checkpoint、TensorBoard 有 `real_success` / `rm_success` |
| P6   | 真实评测冒烟：base 策略 1 个 cell（affordance L0 task 0）× 20 次，osmesa | SR 与 `off_base` 同 cell 相差在 ±.15 内                      |
| P7   | 全量：E7 或 E10-O（8 卡 80 步），完成后 5 个 checkpoint × 60 cell 评测 | 与 4.3 节表可比                                              |
| P8   | CoWMPO s0 冒烟                                               | success_once 在 .3–.5                                        |

---

## 7. 已知坑（gpu006 / Song2 经验）

1. 每个任务独立 Ray head：`RAY_AUTH_MODE=disabled`，独立端口、`--temp-dir`、互不重叠的 `--min/max-worker-port`、`--num-cpus 32`；worker 端口段避开 52365。**永远不要 `ray stop`**。
2. 停 Ray 残留按环境变量 `RAY_TMPDIR` 找进程；不要 `pkill -f` 匹配自己命令行里的字符串。
3. RLinf 忽略 `CUDA_VISIBLE_DEVICES`，用配置里的 placement 指定卡。
4. 长任务 `setsid nohup ... > log 2>&1 < /dev/null &`；被杀后 `launch_train.sh` 从最新完整 checkpoint 续训。
5. 每张卡只放 1 条评测链，链间错开 90 秒启动。
6. LIBERO 任务持主机级锁 `/tmp/rlinf-libero-run.lock`，不要绕过。
7. `RLINF_LIBERO_SAFETY_AUX=1` 必须设（辅助安全标签）。
8. LIBERO-Safety 已知 bug：`check_robot_contact` 拿 geom 名字和 int id 比，官方机器人接触谓词永不触发；我们用 geom-id 的 `safety_grader`。
9. 删除不可再下载的数据前必须问用户；所有删除先 dry-run。

---

## 8. 需要用户先做的事

1. 把 2.3 节缺口（π0.5 已转换权重、`labels/`、E10-O 配置、CoWMPO 数据）从 gpu006 上传到 ModelScope 或推到 GitHub。
2. 在目标机器上提供 ModelScope token（`modelscope login`），不要贴进对话。
3. 确认这台机器的 16 张卡本项目可用多少张、能占用多久，CPFS 上允许使用的空间上限。

---

## 9. 给目标机器上 Claude 的完整 prompt

```text
你在阿里云 PAI-DSW 容器 dsw-896086-588f6f777f-9qswv 上，以 root 运行。项目根目录 R=/mnt/cpfs/jieduan（CPFS 共享卷，已用 95%，只在 $R 下写文件，不碰 /mnt/cpfs 其他目录）。
先完整阅读 $R/SC-GRPO_CoWMPO_BAAI-PPU部署复现指南.md（就是本文），它是唯一的事实来源；标注【需确认】的内容必须先查证再用，【缺口】的内容没有就停下来告诉我。

目标：在这台机器上复现并跑通 SC-GRPO（CodeWorld-Safe）现有训练，其次是 CoWMPO 的 s0 冒烟。不做方法改动，不调超参，只做让它在本机跑起来所必需的移植修改。

硬约束：
1. 加速卡是 16 张平头哥 PPU-ZW810E，不是 NVIDIA。系统预装的厂商 torch 2.9.0 / torchvision 0.24.0 绝对不能被替换或升级。先执行
   pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton|torchao)[=@ ]' > $R/envs/torch_pin.txt
   之后所有 venv 用 --system-site-packages，所有 pip 安装带 -c $R/envs/torch_pin.txt；每装完一批包都重新检查 torch.__version__ 和 torch.cuda.device_count()==16。
2. 不装 flash-attn、jax[cuda12] 等 CUDA 专用包；注意力回退 SDPA，jax 只装 CPU 版（仅权重转换需要）。
3. MuJoCo 渲染一律 MUJOCO_GL=osmesa（训练和评测都是），不用 EGL。
4. 任何时候 df -h /mnt/cpfs 的可用空间不得低于 500 GB；下载前先估算大小。不删除你没有创建的文件；删除自己的大文件前先列出来给我确认。
5. 每个 Ray 任务独立 head（RAY_AUTH_MODE=disabled、独立端口和 temp-dir、不重叠的 worker 端口段、--num-cpus 32），永远不执行 ray stop，不用 pkill -f 匹配宽泛字符串。
6. 长任务用 setsid nohup 后台运行并写日志到 $R/logs/；不要在前台阻塞等待。
7. 不把 token、密码写入任何文件、日志或 git 提交；ModelScope token 由我在终端用 modelscope login 设置。
8. 不 git push、不改远端仓库；本机的移植改动放在独立分支 ppu-port 上本地提交，并在 $R/PORTING_NOTES.md 记录每一处改动和原因。

执行顺序（每一阶段结束时给我一段简短汇报：做了什么、结果数字、遇到的问题，然后继续下一阶段；只有遇到【缺口】、硬约束冲突或同一问题两次修复失败时才停下来问我）：
P0 硬件检查：torch 版本、16 卡、bf16 4096x4096 matmul、torch.distributed 是否支持 nccl 后端，用 torchrun 在 2 卡上跑 all_reduce。若 nccl 不可用，查厂商文档给出的后端名，并记录到 PORTING_NOTES。
P1 代码与环境：在 $R/code 下 clone CodeWorld-Safe(main)、LIBERO-Safety、RLinf/diffsynth-studio(checkout 2a2e05f)、CoWMPO(cowmpo)；apt 装 libosmesa6 等系统依赖；按指南第 3 节建 $R/envs/cws，参照 requirements/embodied/models/openpi.txt 和 requirements/install.sh 的 install_liberosafety_env 手工安装（跳过其中的 torch 安装）；下载 LIBERO-Safety assets；通过第 3.4 节全部验收，包括 tests/unit_tests/test_world_model_safe.py。
P2 资源下载：按指南第 2.1/2.2 节下载到 $R/data 和 $R/models（HF 用 HF_ENDPOINT=https://hf-mirror.com；ModelScope 用 snapshot_download，counterfactual-rollout 只下 json 和 state/*.npz）；keyframes 合并分片并 sha256 校验，校验通过后删分片；对照第 2.3 节列出仍缺的项并告诉我。π0.5 如果拿不到已转换权重，按 2.4 节自己转换并核对 config.json。
P3/P4 单卡前向：π0.5 对一帧观测出 action chunk；Wan 从一个关键帧生成一个 chunk。记录显存和耗时。这里最可能遇到 PPU 不兼容的算子，逐个定位，优先通过配置或 SDPA 回退解决，记录到 PORTING_NOTES。
P5 SC-GRPO 冒烟：复制 libero_safety_scgrpo_causal_e8b.yaml（若不存在，用 ls examples/embodiment/config/libero_safety_scgrpo_causal_*.yaml 找最接近的 E7/E8 配置并告诉我选了哪个）为 _ppu_smoke.yaml，改所有绝对路径为本机路径、episodes_jsonl 指向改写后的 $R/data/labels/episodes.jsonl，placement 0-1、total_num_envs 16、rollout_epoch 1、max_steps 2、save_interval 1；修改 toolkits/codeworld_safe/scgrpo/e1/ 脚本里写死的 S= 目录；跑通 2 步并确认 checkpoint 和 TensorBoard 指标。
P6 评测冒烟：用 base 策略在 affordance L0 task 0 跑 1 个 cell×20 次（osmesa），与 $R/data/scgrpo_bundle/scgrpo_e1_e8_20261030/eval_real/off_base 的同 cell 结果比较。
P7 全量训练：先把 P5–P6 结果和预计耗时、显存、所需卡数报给我，等我确认再启动 E7 配置（8 卡、80 步、save_interval 2，KEEP="10 20 40 60 80" KEEP_LATEST=1，用 launch_train.sh 自动续训）。
P8 CoWMPO：按指南第 3.5、5 节建 .venv-oft-wan（同样继承厂商 torch），找到 s0 启动脚本，跑 PLACE=<一张空闲卡> NENVS=10 EPOCHS=1 冒烟，对比 gpu006 .49 / Song2 .40。CoWMPO 的 L1 GRPO 当前奖励为 0，不要启动 L1 训练。

背景（不要基于它改方法，只用于判断结果是否正常）：SC-GRPO 越训越差的主因是 RM 误判在 GRPO 组内归一化下被放大；E8-b（RM + success_reward_rule final + 固定 64 步）10 步内与 base 持平（.692 vs .703），E7（同配置 80 步）step 20/40/60/80 = .672/.652/.647/.679。真实奖励的 E10-O（libero_safety_scgrpo_causal_e10o.yaml，已在 main）step 10/20/40 = .733/.769/.738，可作为 P7 全量训练的参照；用 RM 奖励的 E15-A / E15-Q 在 step 10 为 .699 / .694。

完成后在 $R/DEPLOY_STATUS.md 写一页状态：各阶段是否通过、关键数字、所有本机特有的修改、未解决的问题。
```
