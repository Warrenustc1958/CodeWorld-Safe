# SC-GRPO 真实环境退化排查进展（截至 2026-10-29）

## 问题

SC-GRPO 在冻结的 Wan 世界模型里训练，真实 LIBERO-Safety 评测反而变差。主线跑到约 303 步，在 280 步时 affordance SR 只有 0.40。
a/b/c 三组消融（adv std floor、eta 上限、释放惩罚）训到 40 步都没有止住退化：

| 消融 | 总体 SR（base → 40 步） |
|---|---|
| a | 0.703 → 0.662 |
| b | 0.703 → 0.637 |
| c（40 个 cell） | 0.750 → 0.680 |

三组的 affordance 都从 0.53 掉到约 0.42。

## 因果实验

- R1：真实仿真器的目标判定作为奖励，eta=0，固定 64 步，只训 affordance，20 步。
- R0b8：RM v3 作为奖励（规则 0.8×4），8 卡，共 10 步。
- 两者都用 `SyncRealWorldModelEnv`（`rlinf/envs/sim/world_model/safe/sync_real_env.py`）。在 Wan 推演的同时，真实 LIBERO 只跑物理，用来给出真实奖励和标签。

## 阶段一结果

### D1：真实环境配对评测（官方协议，每 cell 20 次，只在 base 也有结果的 cell 上配对比较）

| checkpoint | cell 数 | affordance SR | human_safety SR | obstacle SR | obstacle_human SR | 总体 SR | 总体 unsafe |
|---|---|---|---|---|---|---|---|
| R1 step 10 | 60 | .530→.680 | .930→.927 | .713→.677 | .640→.600 | .703→.721 | .198→.161 |
| R1 step 20 | 57 | .530→.663 | .930→.817 | .713→.580 | .696→.613 | .718→.671 | .201→.161 |
| R0b8 step 5 | 53 | .530→.600 | .930→.933 | .713→.687 | .775→.819 | .732→.752 | .200→.164 |
| R0b8 step 10 | 54 | .530→.563 | .930→.950 | .713→.683 | .706→.761 | .721→.737 | .206→.193 |

- affordance 上，真实奖励（R1）的提升明显大于 RM 奖励（R0b8）。R1 的 unsafe 从 .333 降到 .10，R0b8 step 10 只降到 .277。
- R1 到 step 20 时，没参与训练的 suite 明显下降：human_safety −11 个点，obstacle −13 个点。同时夹爪开合次数大增：obstacle 每回合 5.4 → 9.9 次，obstacle_human 5.8 → 10.2 次。
- 这部分下降与 RM 无关，是只在 affordance 上训练造成的跨任务遗忘。R0b8 只训了 10 步，还没表现出来。

### D2：affordance 上的 on-policy rollout（固定 64 步窗口，48 个起点 × G=8，标签取真实仿真器）

| 策略 | 回合数 | 真实目标率 | 抓错部位率 | 0.8×4 P / R | 0.2×2 P / R | 0.8×4 误报（抓错 / 无目标） | 0.8×4 组内 adv 符号一致率 | 有真实信号的组 | RM 与真实都有信号的组 |
|---|---|---|---|---|---|---|---|---|---|
| base | 384 | .302 | .276 | .767 / .681 | .745 / .957 | 22 / 2 | .44 | .23 | .04 |
| R0b8 step 5 | 384 | .328 | .268 | .800 / .667 | .756 / .937 | 19 / 2 | .52 | .27 | .13 |
| R0b8 step 10 | 256（8/12 batch） | .359 | .227 | .831 / .750 | .733 / .957 | 12 / 2 | .50 | .19 | .03 |

- 前 10 步里 RM 的精度没有下降，precision 0.77 → 0.83。所以 R0b8 前 10 步的问题不是 RM 被钻空子，而是 RM 的信号本身就弱。
  - 约 90% 的误报是抓错部位。0.8×4 规则的召回只有 0.67~0.75。
  - 只有 19~27% 的组内真实奖励有方差，RM 与真实都有信号的组只有 3~13%。
  - 两者都有信号时，优势符号也只有约一半一致。
- 在冻结评测 v4 上，RM 精度从 base 的 .82~.90 掉到 40 步的 .40~.57。所以被钻空子发生在更长的训练里。
- 注意：RM v3 在 affordance poltest 上的 precision 是 .375，D2 里约 .77~.83。两者分布不同：poltest 是按成功截断的旧数据，D2 是固定窗口的 on-policy 数据。不应直接比较。

### D3：logprob 不一致 vs 正常更新幅度（`gate_isolation/recomputed_logprobs.pt` + TensorBoard）

| 对比 | chunk log-ratio 最大值 | ratio 范围 | 伪 KL |
|---|---|---|---|
| 原 train 模式 vs eval（batch 16） | 0.099 | 0.93–1.10 | 5.0e-4 |
| eval batch 32 vs 16 | 0.060 | 0.94–1.05 | 2.3e-4 |
| 修复 vision 模式后 | 0 | 1.00 | 0 |

训练日志里的正常更新幅度：

| 运行 | approx_kl 均值 | clip_fraction 均值 |
|---|---|---|
| R1 | 6.8e-4 | 1.5e-4 |
| R0b | 2.7e-4 | 5e-5 |
| R0b8 | 6.0e-4 | 6e-5 |

- 代码 bug：`freeze_vlm()` 只在加载时调用一次 `paligemma.eval()`，而 `embodied_fsdp_actor_worker.py:587` 每个训练步都会调用 `self.model.train()`，把 SigLIP 切回 train 模式。所以至今所有训练都带着这个不一致。
- 不一致的幅度和正常更新同量级，但没有超出 clip 范围的长尾。它是持续噪声，不足以单独解释退化，属于次要因素。修复：在 `model.train()` 之后重新把 paligemma 设成 eval（尚未修改）。

## 当前结论

1. affordance 的退化主要来自 RM 奖励。
   - 训练早期：信号弱且有偏，主要是抓错部位也能得分，加上召回低。
   - 训练后期：策略学会钻 RM 的空子（冻结评测 v4 上精度掉到 .40~.57）。
   - 真实奖励（R1）可以稳定提升 affordance，unsafe 也大幅下降。
2. 其他 suite 的下降来自只在单一 suite 上训练造成的遗忘，并伴随夹爪开合次数上升。换 RM 解决不了，需要多 suite 训练或 SFT replay。
3. SigLIP 的 train/eval 不一致是次要因素，作为一行修复并入阶段二。
4. adv 归一化、eta、释放惩罚这些修改（消融 a/b/c）不能阻止退化。

## 未完成项

- D1：R1 step 20 还差 3 个 cell，R0b8 step 5 / step 10 分别还差 7 / 6 个 cell。
- D2：R0b8 step 10 还差 4 个 batch；R1 step 10 的 on-policy 数据未采。
- 2026-10-29 12:40，gpu006 整机任务被外部 kill，所有 GPU 任务已停。数据完好，脚本都能跳过已完成部分续跑（`run_d1.sh`、`diag/run_d2.sh`）。

## 下一步（待确认）

- 阶段二
  - T1/T2/T3：拆开奖励来源和提前终止两个开关，分别验证。
  - 修复 SigLIP 的 eval 模式。
  - RM 加入抓错部位的负样本，affordance 的成功判定改为必须抓对部位。
- 阶段三：多 suite 或 SFT replay，解决跨任务遗忘。

## 脚本与数据

- 在 gpu006 `/data_cf/cws_sam/v1/scgrpo/`：
  - D1：`run_d1.sh`、`eval_real*.sh`、`diag/matched_summ.py`，结果在 `diag/d1_matched.txt`。
  - D2：`diag/collect_d2.py`、`diag/run_d2.sh`、`diag/analyze_d2.py`，数据在 `diag/d2/<policy>/batch_*.pkl`，结果在 `diag/d2_summary.json`。
  - D3：`diag/d3_logprob_tail.py`，结果在 `diag/d3_summary.json`。
- 这些脚本的副本在仓库的 `toolkits/codeworld_safe/scgrpo/diag/`。
