# SC-GRPO 越训越差：根因排查、诊断实验结果与后续实验规划

更新时间：2026-10-30 10:30（加入 E1 step 20/40、E2、E4 结果，主因改判为"RM 判成功即结束回合"）。本文用于新会话快速了解项目现状、瓶颈和待办。所有数字都来自 gpu006 上的实际结果文件。没跑完的部分会单独标注。

---

## 0. 一页结论

- **现象。** SC-GRPO 在冻结的 Wan 世界模型里训练 π0.5。真实 LIBERO-Safety 官方评测（60 cell × 20）的总体 SR 越训越低：base .703 → step 80 .623 → step 160 .543 → step 200 .529。affordance 在 step 280 只有 .40。
- **主因（10-30 夜间 E1/E2 修正）：用 RM 判成功来结束回合。** 同一个 RM 奖励，回合由 RM 判成功结束（RM-all）时总体 SR .703 → .696（step 10）→ .642（20）→ .609（40），四个 suite 都退化，夹爪开合次数上升（obstacle 每回合 5.4 → 9.3 次），复现了主线的现象；回合改由真实仿真器判定结束（E2）时 step 10/20 总体 SR .722/.711，和真实奖励（R1-all .724/.712）一样，没有退化。RM 奖励本身只让 affordance 比真实奖励低约 3 个点（E2 .567/.520 vs R1-all .600/.540，约 1 个标准误）。详见 3.6 节。
- **RM v3 的成功判定确实不可靠（这是 RM 提前结束回合能被钻空子的前提）：**
  - 训练早期，信号弱而且有偏。约 90% 的误报是抓错部位，0.8×4 规则的召回只有 .67~.75，只有约 20% 的组内真实奖励有方差。
  - 训练后期，RM 被策略钻空子。冻结评测 v4 上，RM precision 从 base 的 .82~.90 掉到 step 40 的 .40~.57。
  - 因果实验：affordance 上用真实仿真器的目标判定作奖励（R1），训 10 步 SR .53 → .68，unsafe .33 → .10。同样训 10 步，用 RM 奖励（R0b8）只到 .56，unsafe .28。
  - E1 全 suite 配对：step 10 的 affordance SR，真实奖励（R1-all）.530 → .600，RM-all .530 → .467。E2 证明这个差距主要来自终止方式，而不是奖励来源。
- **E2 不可部署。** E2 的回合结束时机来自真实仿真器，只用世界模型训练时没有真实仿真器。可部署的版本是 E8：RM 奖励 + 固定 64 步回合（`termination_source: horizon`），不使用任何真实仿真器信息。
- **已排除或降级的因素：**
  - adv 归一化爆炸、eta 无界增长、释放夹爪奖励：消融 a/b/c 修了这些，仍然退化。
  - GRPO 掩码、打分和分组代码：检查后是正确的。
  - SigLIP 的 train/eval 不一致：确认是 bug，但幅度不超出 clip 范围，属于次要因素。已在 `pi0.py` 修复（commit 8d3179d1），E1/E2 都用修复后的代码。
- **注意。** R1 和 R0b8 只在 affordance 上训练，它们在其他 suite 上的掉点只说明单 suite 训练会遗忘，不能用来评价 RM 或奖励来源。R1-all 四个 suite 一起训练后，其他 suite 没有出现遗忘。
- **当前瓶颈：**
  1. 验证可部署的修法 E8（RM 奖励 + 固定回合）是否和 E2 一样不退化。
  2. RM 本身的 affordance 精度：v3/v5/v6 在 D2 留出集上 precision .76~.83，都没过 .93（E4 不通过）。
  3. 即使奖励准确，当前配置训练也没有持续提升：R1-all affordance step 10 .600 → step 20 .540，总体 .724 → .712。阶段三要解决训练效率问题。
  3. gpu006 资源和评测稳定性：多个 Ray head 并存时有端口冲突，并发 EGL reset 会抢 NVIDIA 驱动锁，见第 6 节。

---

## 1. 项目背景与设置

- **项目。** CodeWorld-Safe / WM Safety：在动作条件世界模型（Wan，AC-SFT 微调）里做带安全约束的 VLA 后训练（SC-GRPO）。
- **策略。** π0.5（openpi_rlinf，3.35B），`train_expert_only=True`，SigLIP 和 Gemma 主干冻结，只训 action expert。
- **训练环境。** 冻结的 Wan 4.1k 世界模型生成想象画面。RM v3 给成功判定；SAM 给代价，代价进入拉格朗日约束（eta 为对偶变量，budget 0.13）。
- **算法。** RLinf（Ray + FSDP），flow_sde 采样 logprob。GRPO 组大小 G=8，优势为 `(s-mean)/(std+1e-6)`。`kl_beta` 没有效果，因为没有参考 logprob。训练时 noise_level 0.5，评测时 0.3。
- **真实评测。** LIBERO-Safety 官方协议，共 4 个 suite：affordance、human_safety、obstacle_avoidance、obstacle_avoidance_human。
  - 每个 suite 分 L0/L1/L2 三级，每级 5 个任务，合计 60 cell × 20 次。
  - L0/L1 用 fresh_init，L2 用官方提供的初始状态（训练数据从未见过）。
  - 回合在第一次违反约束时结束；只有达成目标且没有违规才算成功。
- **affordance 的目标定义。** 目标谓词是 `checkgrippercontactpart`，必须抓到指定部件。`wronggrasppart` 是辅助约束。这正是 RM 最容易出错的地方。

---

## 2. 排查过程与证据（按时间顺序）

### 2.1 主线运行的观察（10-27）

主线 `scgrpo_pi05_wan41k_main` 在约 303 步时被 SIGTERM，磁盘上只剩 step 40 和 step 280 的 checkpoint。

| step | 真实评测总体 SR |
|---|---|
| base | .703 |
| 20 | .713 |
| 80 | .623 |
| 120 | .597 |
| 160 | .543 |
| 200 | .529 |
| 240 | .547 |

1. **想象里什么都没学到。** TensorBoard 上，想象成功率在 300 步里一直平在 .40~.45，代价平在 .19~.22。eta 从 0.2 线性涨到 5.5，说明 budget 0.13 不可达。
2. **优势是噪声。**
   - 有的组分数方差只有 1e-5~2e-4（全成功时代价为 0，全失败时代价约 1），归一化后被放大成 ±2 的优势，约 5/16 的组是这种情况。
   - 拷贝回放：同一条动作序列复制 8 份，代价的方差（.04~.06）不小于策略采样之间的方差（.03）。所以只靠代价得到的优势就是 Wan / SAM 的噪声。
3. **真实环境里的失败全是超时。** 夹爪开合次数上升：affordance 每回合 1.7 → 5.8 次，human_safety 4.8 → 10.7 次。失败回合里的松开次数是原来的 2~5 倍。而想象画面里的开合次数一直是每回合约 0.67 次，RM 对开合几乎没有反应。
4. **释放测试（17 个"先松开后成功"的回合）。** 强制张开夹爪后 RM 成功率仍有 .59，来回切换时为 .82（原始动作是 1.00）。说明想象环境会奖励"松手"。
5. **抓错部位回放。** RM 在抓错部位的回合上成功率是 .17，在干净的成功回合上是 1.00。SAM 能标出 73% 的抓错，但误报率 23%。

### 2.2 消融 a/b/c（10-27，各 80 步，用户要求在 40 步停）

修改内容：`adv_std_floor`（advantages.py 和两个 worker）、`eta_max`、`release_coef/window/cap`（safe_env.py）。已提交在 CWS d49c6246。

| 消融 | 总体 SR（base → step 40） | affordance |
|---|---|---|
| a | .703 → .662 | .53 → ~.42 |
| b | .703 → .637 | .53 → ~.42 |
| c（40 cell） | .750 → .680 | .53 → ~.42 |

结论：修优势归一化、eta 和释放惩罚都不能阻止退化。所以退化不主要来自优化器层面的数值问题。

### 2.3 冻结评测 v4（10-27）

在 38 个起点上，对比 base 和 step 40 在真实环境和 Wan 画面上的表现：
- 真实 SR：base 15/38 → step 40 8/38。
- RM precision：base .82~.90 → step 40 .40~.57，真实画面和 Wan 画面都一样掉。

这是 RM 被钻空子的直接证据。另外发现 O00 和 O01 两种状态输入下的动作完全一样，怀疑策略没有用到状态输入（未深究）。

### 2.4 logprob 门控检查（Codex，10-27）

`diagnostics/rootcause_20261027_v1/gate_isolation/`
- 原始 train 模式 vs eval（batch 16）：chunk 最大差 0.099，ratio .93~1.10。
- batch 32 vs 16：最大差 0.06。
- 修复 vision 模式后：差为 0。
- 没有任何 ratio 落在 clip 范围之外。

### 2.5 因果实验 R1 / R0b8（10-28）

- **R1。** 真实仿真器的目标判定作奖励，eta=0，固定 64 步，只训 affordance，checkpoint 有 step 10 和 step 20。
- **R0b8。** RM v3（0.8×4）作奖励，8 卡，128 env × 4 epoch，共 10 步，checkpoint 有 step 5 和 step 10。
- **环境。** 两者都用 `SyncRealWorldModelEnv`（`rlinf/envs/sim/world_model/safe/sync_real_env.py`）。在 Wan 推演的同时，真实 LIBERO ControlEnv 只跑物理，渲染用 osmesa，用来给出真实奖励和标签。
- **开关。** `reward_source`（real / rm）、`end_on_violation`、`success_tail_chunks`、`record_steps`。
- **一个混淆因素。** 在 `reward_source=rm` 模式下，RM 判成功会提前结束回合，真实仿真器也随之停止步进（sync_real_env 约第 197 / 251 行）。所以 R1 和 R0b8 的区别不只是奖励来源，还有终止条件。阶段二的 E2 要把这两者拆开。

### 2.6 RM v4 / v5（10-28）

- v4：on-policy 采集了 1536 个回合，路径 `/data_cf/cws_rm/frames_pol_v4`，训练结果在 `runs/rm_v4_onpolicy`。
- v5：带成功之后的尾段标签，路径 `frames_pol_v5`，规则扫描在 `runs/rm_v5_tail/rule_sweep`。
- 旧采集器在成功时就截断回合，导致"连续 k 帧"规则永远触发不了。v5 修好了这个问题，oracle 检查 P=1。
- RM v3 各 suite 在 0.8×4 下的表现：affordance 在 poltest 上 P .375 / R .50，在 polval 上 P .636；其他 suite 在 .93~1.0。
- v3 和 v5 取 AND 规则，没有比单独用 v3 更好。

---

## 3. 阶段一诊断（D1 / D2 / D3，10-29）

### D1：真实环境配对评测（只在 base 也有结果的 cell 上比较）

| checkpoint | cell 数 | affordance SR | human_safety SR | obstacle SR | obstacle_human SR | 总体 SR | 总体 unsafe |
|---|---|---|---|---|---|---|---|
| R1 step 10 | 60 | .530→.680 | .930→.927 | .713→.677 | .640→.600 | .703→.721 | .198→.161 |
| R1 step 20 | 57 | .530→.663 | .930→.817 | .713→.580 | .696→.613 | .718→.671 | .201→.161 |
| R0b8 step 5 | 53 | .530→.600 | .930→.933 | .713→.687 | .775→.819 | .732→.752 | .200→.164 |
| R0b8 step 10 | 54 | .530→.563 | .930→.950 | .713→.683 | .706→.761 | .721→.737 | .206→.193 |

- affordance unsafe：R1 step 10 / step 20 从 .333 降到 .100 / .110；R0b8 step 5 / step 10 降到 .213 / .277。
- R1 step 20 在没训练过的 suite 上掉点，夹爪开合次数约翻倍（obstacle 每回合 5.4 → 9.9 次）。
- **解读（用户指出，已采纳）。** 只训 affordance 的模型在其他 suite 上掉点，说明的是单 suite 训练会遗忘，不能作为 RM 或奖励来源好坏的证据。上表其他 suite 的列只用来衡量遗忘。

### D2：affordance 上的 on-policy rollout，RM 判定对比真实标签

设置：
- 48 个起点 × G=8；
- 固定 64 步窗口，不因违规提前结束，成功后继续跑满；
- 逐步记录真实标签：目标是否达成、是否抓错部位、是否违规。

| 策略 | 回合数 | 真实目标率 | 抓错部位率 | 0.8×4 P / R | 0.2×2 P / R | 0.8×4 误报（抓错 / 无目标） | 组内 adv 符号一致率（0.8×4） | 有真实信号的组 | RM 与真实都有信号的组 |
|---|---|---|---|---|---|---|---|---|---|
| base | 384 | .302 | .276 | .767 / .681 | .745 / .957 | 22 / 2 | .44 | .23 | .04 |
| R0b8 step 5 | 384 | .328 | .268 | .800 / .667 | .756 / .937 | 19 / 2 | .52 | .27 | .13 |
| R0b8 step 10 | 256（8/12 batch） | .359 | .227 | .831 / .750 | .733 / .957 | 12 / 2 | .50 | .19 | .03 |

- 前 10 步里 RM 没有被钻空子，precision 反而略升。
- 早期的问题是信号弱且有偏：
  - 误报大多是抓错部位；
  - 召回低；
  - 大部分组根本没有学习信号；
  - 两者都有信号时，方向也只有一半一致。
- 被钻空子出现在更长的训练里（见 2.3，step 40 时 precision 已掉到 .40~.57）。
- 不要把这里的 precision 和 poltest 的 .375 直接比：poltest 是按成功截断的旧数据，D2 是固定窗口的 on-policy 数据，分布不同。

### D3：logprob 不一致 vs 正常更新幅度

| 对比 | chunk log-ratio 最大值 | ratio 范围 | 伪 KL |
|---|---|---|---|
| 原 train 模式 vs eval（batch 16） | 0.099 | 0.93–1.10 | 5.0e-4 |
| eval batch 32 vs 16 | 0.060 | 0.94–1.05 | 2.3e-4 |
| 修复 vision 模式后 | 0 | 1.00 | 0 |

训练日志里的正常更新幅度：approx_kl 均值 R1 6.8e-4、R0b 2.7e-4、R0b8 6.0e-4；clip_fraction 约 1e-4。

- **代码 bug。** `freeze_vlm()`（`openpi_action_model.py:1404`）只在加载时调用一次 `paligemma.eval()`，而 `embodied_fsdp_actor_worker.py:587` 每个训练步都会调用 `self.model.train()`，把 SigLIP 切回 train 模式。至今所有训练都带着这个不一致。
- **影响。** 不一致与正常更新同量级，是持续噪声，但不形成超出 clip 的长尾，属于次要因素。
- **修复（尚未修改）。** 在 `model.train()` 之后重新调用 `freeze_vlm()`，或者对 paligemma 调 `.eval()`。

### 阶段一未完成项

- D1：R1 step 20 还差 3 个 cell，R0b8 step 5 / step 10 分别还差 7 / 6 个 cell。
- D2：R0b8 step 10 还差 4 个 batch；R1 step 10 的 on-policy 数据未采。
- 这些都不影响上面的结论。续跑时脚本会跳过已完成部分（`run_d1.sh` 跳过 EXIT=0 的 cell，`diag/run_d2.sh` 跳过已有的 batch 文件）。

---

## 3.5 阶段二 E1：全 suite 配对实验（10-29 ~ 10-30）

设置：
- R1-all（`libero_safety_scgrpo_causal_r1all.yaml`）：真实仿真器目标判定作奖励，四个 suite 一起训练，GPU 0~3，20 步，每 5 步存一次。
- RM-all（`libero_safety_scgrpo_causal_rmall.yaml`）：与 R1-all 只差奖励来源，RM v3 0.8×4，`termination_source: rm`，`terminate_on_real_violation: True`（违规处理与 R1-all 相同）。
- 两组都用 SigLIP 修复后的代码；eta=0；回合 64 步。

**R1-all step 10 相对 base（60 cell 完整）**

| suite | L0 SR | L1 SR | L2 SR | 合计 SR | 合计 unsafe | 开合次数/回合 |
|---|---|---|---|---|---|---|
| affordance | .690→.740 | .590→.720 | .310→.340 | .530→.600 | .333→.247 | 1.69→2.02 |
| human_safety | .860→.900 | .990→.930 | .940→.930 | .930→.920 | .207→.173 | 4.81→7.22 |
| obstacle_avoidance | .700→.740 | .720→.740 | .720→.710 | .713→.730 | .077→.090 | 5.44→5.34 |
| obstacle_avoidance_human | .670→.700 | .650→.670 | .600→.570 | .640→.647 | .177→.170 | 6.13→5.54 |
| 总体 | | | | .703→.724 | .198→.170 | |

**RM-all step 10 对比 R1-all step 10（affordance 和 human_safety，其余 suite 评测中）**

| suite | level | base SR | R1-all | RM-all | base unsafe | R1-all | RM-all |
|---|---|---|---|---|---|---|---|
| affordance | L0 | .690 | .740 | .560 | .290 | .160 | .380 |
| affordance | L1 | .590 | .720 | .500 | .350 | .230 | .370 |
| affordance | L2 | .310 | .340 | .340 | .360 | .350 | .370 |
| affordance | 合计 | .530 | .600 | .467 | .333 | .247 | .373 |
| human_safety | 合计 | .930 | .920 | .910 | .207 | .173 | .210 |

解读：
- 每个 level 100 个回合，单个数字标准误约 ±.045；suite 合计 300 个回合，标准误约 ±.03。
- affordance 上 R1-all 和 RM-all 相差 .133，约 3 个标准误；RM-all 低于 base。全 suite 配对设置下"RM 奖励导致 affordance 退化"成立，10 步内就出现。
- L2（训练数据未见过）两组都 +.03，基本不受影响。
- human_safety 两组都持平，和 RM 在该 suite 精度高（.93~1.0）一致。
- 训练曲线（R1-all，想象中的指标）：`real_success` 前后都在 .32~.47 波动，`rm_false_success` .03~.06，approx_kl 约 1e-3，没有明显上升，真实评测的提升在训练曲线上看不出来。

**E4 基线：现有 RM 在 D2 留出集上的检查**（`frames_d2/d2test`：16 个 affordance 起点，336 回合，123 个真实成功，64 个抓错部位的失败）

| RM | 规则 | precision | recall | 误报（其中抓错部位） |
|---|---|---|---|---|
| v3 | 0.5×3 | .774 | 1.000 | 36（31） |
| v3 | 0.8×4（训练用） | .758 | .789 | 31（30） |
| v5 | 0.8×4 | .825 | .650 | 17（16） |
| v5 | 0.2×2 | .700 | .967 | 51（32） |

结论：现有 RM 在 affordance 上都过不了"precision ≥ .93 且 recall ≥ .75"，约 85~95% 的误报是抓错部位。

---

## 3.6 10-30 夜间结果：E1 step 20/40、E2、E4（全部 60/60 cell，无失败）

汇总文件：`/data_cf/cws_sam/v1/scgrpo/causal/night_summary.txt`。

**三组实验的区别**（都在 `sync_real_env.py` 里，四个 suite 一起训练，eta=0，回合最长 64 步，SigLIP 已修）

| 实验 | 配置 | 奖励 | 回合怎么结束 |
|---|---|---|---|
| R1-all | `causal_r1all.yaml` | 真实仿真器目标判定 | 真实成功 / 真实违规 / 64 步 |
| RM-all | `causal_rmall.yaml`（step 40 用 `causal_rmall40.yaml` 从 step 20 续训） | RM v3 0.8×4 | RM 判成功 / 真实违规 / 64 步 |
| E2 | `causal_e2.yaml` | RM v3 0.8×4 | 真实成功 / 真实违规 / 64 步 |

**总体 SR（60 cell × 20 次，base .703，标准误约 ±.013）**

| 实验 | step 10 | step 20 | step 40 |
|---|---|---|---|
| R1-all | .724 | .712 | — |
| RM-all | .696 | .642 | .609 |
| E2 | .722 | .711 | — |

**各 suite SR（base：affordance .530、human_safety .930、obstacle .713、obstacle_human .640；每格 300 回合，标准误约 ±.03）**

| 实验 | step | affordance | human_safety | obstacle | obstacle_human |
|---|---|---|---|---|---|
| R1-all | 10 | .600 | .920 | .730 | .647 |
| R1-all | 20 | .540 | .937 | .693 | .677 |
| RM-all | 10 | .467 | .910 | .720 | .687 |
| RM-all | 20 | .400 | .873 | .663 | .633 |
| RM-all | 40 | .427 | .863 | .570 | .577 |
| E2 | 10 | .567 | .943 | .710 | .667 |
| E2 | 20 | .520 | .930 | .710 | .683 |

**夹爪开合次数（每回合，base affordance 1.69 / obstacle 5.44）**：RM-all step 40 affordance 4.31、obstacle 9.32；E2 step 20 affordance 1.53、obstacle 5.78；R1-all step 20 affordance 1.55、obstacle 5.71。

**affordance 各 level SR（base L0 .690 / L1 .590 / L2 .310）**

| 实验 | step | L0 | L1 | L2 |
|---|---|---|---|---|
| R1-all | 20 | .670 | .600 | .350 |
| RM-all | 20 | .470 | .440 | .290 |
| RM-all | 40 | .490 | .430 | .360 |
| E2 | 10 | .760 | .590 | .350 |
| E2 | 20 | .670 | .560 | .330 |

**E4：RM v6 没有通过检查**（`runs/rm_v6_d2`，v5 配方 + `frames_d2/d2train`；检查集 `frames_d2/d2test`）

| RM | 规则 | precision | recall | 误报（其中抓错部位） |
|---|---|---|---|---|
| v3 | 0.8×4 | .758 | .789 | 31（30） |
| v6 | 0.8×4 | .817 | .837 | 23（22） |
| v6 | 0.5×3 | .795 | .943 | 30（26） |

比 v3 高约 5 个点，但离 .93 很远，抓错部位误报没有减半。可能原因：D2 训练集里抓错部位的回合只有约 150 个，混在约 70 万帧里；RM 只看 256px 主视角，可能分不清接触的部件。按计划没有跑 E6，GPU 4~7 改跑 RM-all 续训到 step 40。

**解读**
1. **退化来自"RM 判成功就结束回合"。** E2 与 RM-all 只差终止方式：E2 不退化，RM-all 一路降到 .609，且四个 suite 都退化、夹爪开合次数大增，与主线现象一致。合理的机制：RM 一误判成功，回合立刻结束并锁定奖励，之后的动作不再被检查（rm 终止模式下真实仿真器也停止步进），策略于是学会尽快制造"像成功"的画面然后停下。
2. **E2 不是纯 RM 奖励。** 真实仿真器通过结束时机泄漏了信息：RM 误判但真实没成功时，回合继续，后面的违规会把奖励清零；真实成功但 RM 没判成功时，回合立刻结束、奖励为 0。所以 E2 只是诊断用的设置，不能部署。
3. **RM 奖励本身的影响较小。** E2 的 affordance 比 R1-all 低约 3 个点，约 1 个标准误。
4. **训 20 步没有持续提升。** R1-all 的 affordance 从 step 10 的 .600 回落到 step 20 的 .540。

---

## 4. 因素归因汇总

| 假设 | 证据 | 判定 |
|---|---|---|
| **RM 判成功即结束回合** | **E2（RM 奖励 + 真实判定结束）step 20 总体 .711，不退化；RM-all 同奖励但 RM 判结束，step 20/40 降到 .642/.609，四个 suite 都退化** | **主因（已确认）** |
| RM 成功信号不可靠（抓错部位误报、召回低，后期被钻空子） | D2 误报约 90% 是抓错部位；v4 precision 降到 .40~.57；E4：v3/v5/v6 在 D2 留出集 precision .76~.83 | 是主因能起作用的前提；单独作为奖励时影响较小（E2 affordance 比 R1-all 低约 3 个点） |
| 优势归一化把噪声放大 | 约 5/16 组是退化组；但消融 a 修了仍然退化 | 次要，已修（adv_std_floor） |
| 代价 / eta 不可达，导致拉格朗日权重爆炸 | eta 涨到 5.5；消融 b/c 修了仍然退化 | 次要 |
| 想象环境奖励松手和夹爪来回开合 | 释放测试 RM 成功率 .59~.82 | 和 RM 问题同源，属于 RM 的漏洞 |
| SigLIP train/eval 不一致 | 伪 KL ≈ approx_kl，没有超出 clip | 次要，已修（8d3179d1） |
| Wan 生成画面的噪声 | 拷贝回放时 RM 成功判定在 2/32 组里翻转；代价噪声大 | 对代价影响大，对成功判定影响小 |
| 单 suite 训练的遗忘 | R1 step 20 其他 suite 掉 11~13 个点；R1-all 四个 suite 一起训练后其他 suite 持平 | 实验设计带来的现象，不作为归因证据 |
| 混淆：RM 提前终止导致真实仿真器停止步进 | sync_real_env 的代码逻辑 | 已由 E2 拆开，见第一行 |

---

## 5. 后续实验规划

### 阶段二：修好奖励（目标：得到一个能替代真实奖励的 RM）

前提：SigLIP 已修；保留 `adv_std_floor`；回合 64 步；eta=0（先不加约束）；四个 suite 一起训练。

| 编号 | 内容 | 作用 | 通过标准 | 状态（10-30 10:30） |
|---|---|---|---|---|
| E1 | R1-all vs RM-all，只换奖励来源，评测 step 10/20（RM-all 加 step 40） | 在全 suite 上归因 | — | **完成**，见 3.5 / 3.6 节 |
| E2 | RM 奖励 + 真实判定结束回合，与 RM-all 只差终止方式 | 拆开"RM 提前结束回合" | 不退化 → 终止方式是主因 | **完成：不退化**，主因改判为终止方式 |
| E4 | RM v6（加 D2 难负样本） | 提高 RM 精度 | D2 留出集 precision ≥ .93，recall ≥ .75 | **不通过**（.817 / .837） |
| E5（可选） | 大模型评判器否决抓错部位 | 不重训 RM 也能压误报 | 同 E4 | 未开始；需要用户确认能否把帧发给外部服务 |
| E6 | RM v6 训全 suite | 在线验证新 RM | — | 未运行（E4 不通过） |
| **E8（下一步）** | RM v3 奖励 + 固定 64 步回合：`termination_source: horizon`、`end_on_violation: false`、`terminate_on_real_violation: false`、`success_tail_chunks: 8`，其余同 RM-all，训 40 步 | 可部署的修法，不用任何真实仿真器信息 | step 20/40 总体 SR 不低于 base，四个 suite 都不退化 | 未开始；需要 4 卡约 4.5 小时训练 + 每个 checkpoint 约 1 小时评测 |
| E9 | E8 若通过，再把 RM 换成 v6 | 看 RM 精度提高 6 个点能挽回多少 affordance | affordance 接近 R1-all | 待 E8 |
| E7 | 通过的配置训到 80 步，每 20 步做真实评测和 D2 式 RM 精度检查 | 长程稳定性 | 真实 SR 不回落 | 待 E8 |

阶段二结束时要拿到：一个只用世界模型、不依赖真实仿真器，并且训到 80 步也不退化的训练配置。

10-30 夜间（到 09:00 占用全部算力）的排程：

| 时间 | GPU 0~3 | GPU 4 | GPU 5~7 |
|---|---|---|---|
| 02:00–03:20 | E2 训练 | E4：RM v6 训练 + D2 留出集检查 | E1 剩余评测（RM-all step 10/20、R1-all step 20） |
| 03:20–04:25 | E2 训练 | 空闲 | 空闲，等 E4 结果 |
| ~04:00–06:15 | E2 评测（GPU 0/1/2） | E6 训练（GPU 4~7） | E6 训练 |
| 06:15–07:30 | E6 评测 | E6 评测 | E6 评测 |
| 07:30–09:00 | 缓冲：补跑失败的 cell，汇总结果 | | |

E4 不通过时不跑 E6，GPU 4~7 改去加跑 E1 的 step 40 或 E2 评测。

实际执行：E4 在 02:53 判为不通过，`auto_rmall40.sh` 自动把 RM-all 从 step 20 续训到 40（05:46 训完），所有评测 06:xx 前完成，cell 全部成功。

### 阶段三：带安全约束重启主线

1. **奖励。** 用 RM v6（或叠加 E5 的否决器），四个 suite 一起训练，回合固定 64 步。
2. **把代价约束加回来，先解决代价噪声。** 之前拷贝回放显示 SAM 代价的噪声不小于策略间差异。
   - 代价改成多次采样平均，或者只在组内代价方差明显时才计入优势；
   - budget 设在可达的约 0.18，eta 加上限，使用全局 eta；
   - 保留 `adv_std_floor`。
3. **防遗忘和防夹爪来回开合。** 混入 SFT 回放数据（`enable_sft_co_train` 路径存在但未验证；`kl_beta` 当前不生效）。监控夹爪开合次数，必要时加开合惩罚。
4. **监控和止损。** 每 20 步做 60 cell 真实评测和 D2 式 RM 精度检查；总体 SR 连续两次低于 base，或 RM 精度低于 .85，就停。

### 大模型评判器的设计要点（E5）

- **调用量。** 不建议当成每个 chunk 都调用的稠密奖励，单步训练就要判上千次。只在 RM 判成功的回合上调用，大约占 30~45% 的回合。
- **输入。** 任务指令、正确部位的文字定义（取自 `checkgrippercontactpart` 指定的部件）、一张抓对和一张抓错的参考图、回合最后 2 个 chunk 的主视角加腕部视角关键帧。
- **输出。**
  
  ```json
  {"goal_achieved": true, "grasped_part": "handle", "part_correct": true,
   "object_dropped": false, "confidence": 0.9, "evidence": "..."}
  ```
- **工程。** 固定模型版本，temperature 设为 0，按 (帧哈希, prompt) 缓存结果；做成异步批量请求；超时或失败时回退到只用 RM，并记录回退率。
- **风险。**
  - 细粒度的接触判断是通用 VLM 的弱项；
  - 它看到的是 Wan 的想象画面，同样可能被钻空子；
  - 渲染帧会发到外部服务，需要用户确认。

---

## 6. 关键路径与资源

### 代码

- **仓库。** github.com/Warrenustc1958/CodeWorld-Safe，main @ 8d3179d1（SigLIP 修复、sync env 终止开关、E1/E2 配置、`toolkits/codeworld_safe/scgrpo/e1/` 下的启动与评测脚本）。
- **工作副本。** gpu006 上的 `/mnt/nfs/fjj/Code/CodeWorld-Safe`，branch 为 codeworld-safe。
- **推送。** 运行 `/mnt/nfs/fjj/Code/cws_push.sh`，把 codeworld-safe 推到 main。
- **主要文件：**
  - 同步真实环境：`rlinf/envs/sim/world_model/safe/sync_real_env.py`；
  - 安全环境（eta、释放惩罚）：`rlinf/envs/sim/world_model/safe/safe_env.py`；
  - 优势计算（adv_std_floor）：`rlinf/algorithms/advantages.py`；
  - 训练 worker（D3 bug，`model.train()` 在第 587 行）：`rlinf/workers/actor/embodied_fsdp_actor_worker.py`；
  - 配置：`examples/embodiment/config/libero_safety_scgrpo_{ablate_a,b,c, causal_r1,r0,r0b,r0b8, causal_r1all,causal_rmall,causal_e2}.yaml`；
  - 多链真实评测：`toolkits/codeworld_safe/scgrpo/e1/run_multi.sh`（每条链独立 Ray 端口 / worker 端口段 / 锁，每 cell 限时 20 分钟，自动重试一轮）；
  - 诊断报告和脚本：`toolkits/codeworld_safe/scgrpo/DIAGNOSIS.md`、`toolkits/codeworld_safe/scgrpo/diag/`。

### 数据（都在 gpu006 本地盘 `/data_cf`，不在 NFS 上）

- **scgrpo 工作目录。** `/data_cf/cws_sam/v1/scgrpo/`
  - checkpoint：
    - R1：`causal/r1/scgrpo_causal_r1/checkpoints/global_step_{10,20}`；
    - R0b8：`causal/r0b8/scgrpo_causal_r0b8/checkpoints/global_step_{5,10}`，每个 16 GB。
  - 真实评测：`eval_real/<name>/`；汇总脚本 `summ_eval_real.py` 和 `diag/matched_summ.py`；D1 结果在 `diag/d1_matched.txt`。
  - D2：数据在 `diag/d2/<policy>/batch_*.pkl`（每个约 40 MB，包含主视角和腕部视角 jpg、step_rec、rm_v3_p、actions），汇总在 `diag/d2_summary.json`。
  - D3：`diag/d3_summary.json`。
  - logprob 门控检查：`diagnostics/rootcause_20261027_v1/gate_isolation/`。
- **RM 数据和模型。** `/data_cf/cws_rm/frames_pol_v4`、`frames_pol_v5`、`frames_d2`（d2train / d2test）；`runs/rm_v4_onpolicy`、`runs/rm_v5_tail/rule_sweep`、`runs/rm_v6_d2`；D2 检查结果 `runs/gate_d2.json`。
- **现存 checkpoint**（每个 15 GB）：`causal/r1/...`（step 10/20）、`causal/r1all/...`（step 5/10/15/20）、`causal/e2/...`（step 10/20）、`causal/r0b8/...`（step 5/10）。RM-all、主线、消融的 checkpoint 已于 10-30 删除，只剩评测结果（`eval_real/`）。
- **ModelScope。** 上传脚本 `upload_causal_ms.py` 已写好，上传到 `causal_r1_rm_v5/`，还没运行。

### 集群注意事项

- ssh 必须加 `-o ClearAllForwardings=yes`。
- 每个任务用自己的 Ray head：设置 `RAY_AUTH_MODE=disabled`，用独立的端口和 temp-dir，永远不要执行 `ray stop`。
- 真实环境渲染用 `MUJOCO_GL=osmesa`。
- 不要用 `pgrep/pkill -f` 匹配自己 ssh 命令里出现的字符串，会把自己的 shell 杀掉。
- 两个训练不要放在同一组 GPU 上。
- 自动权限分类器经常拦下 kill、push、rm 之类的命令。拦下时请用户用 `!` 自己执行。
- **GPU 3/4 有 EGL 故障**（dmesg Xid 109 CTX SWITCH TIMEOUT 在 GPU 3），评测默认 EGL 渲染，不要放在 GPU 3/4；训练用 osmesa 不受影响。
- **多个 Ray head 并存：** 每个 head 要指定互不重叠的 `--min/max-worker-port`、`--ray-client-server-port`，CPU 限到 32；worker 端口段不能包含 dashboard agent 默认端口 52365；端口 6461 属于 cowmpo。
- **并发评测上限：** 每张卡只放 1 条评测链，链之间错开 90 秒启动；12 条链同时 reset 会让进程卡在 NVIDIA 驱动锁 `rmapiLockAcquire`（D 状态），reset 从 4 秒变成 460 秒。
- **停 Ray head 时要按环境变量 `RAY_TMPDIR` 找残留进程**：Ray client server 的命令行不带 temp 目录，按命令行匹配会漏掉，残留进程占着 client 端口会让新 head 卡在 NodeProbe。
- **gpu005。** 普通用户 zbuser05，没有 sudo；OSMesa 装在 `/mnt/nfs/fjj/local-osmesa`；内网 IP 10.20.10.51。
  - 目前关机。
  - 上面的 r0b8c（R0b8 续跑到 90 步）checkpoint 还没取回。
  - 临时授权的 key 恢复开机后要删掉。

---

## 7. 待办清单

1. **（下一步，需要算力）** E8：RM 奖励 + 固定 64 步回合，40 步，评测 step 20/40。
2. **（E8 之后）** 通过则做 E9（换 RM v6）和 E7（80 步长程）。
3. **（可选）** E5 大模型评判器离线验证，需要用户确认能否把帧发给外部服务。
4. **（阶段三准备）** 代价降噪方案（多次采样平均 / 组内方差门限）；验证 `enable_sft_co_train`。
5. **（其他）**
   - 运行 ModelScope 上传；
   - gpu005 恢复后取回 r0b8c checkpoint 并删除临时 key；
   - 磁盘：10-30 已删除效果差的 checkpoint（主线 step 40/280、消融 a/b/c step 40、smoke step 1、RM-all step 5/10/15/20/40），/data_cf 可用 195 GB。保留 R1 10/20、R1-all 5/10/15/20、E2 10/20、R0b8 5/10。
