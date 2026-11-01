# E12–E14、D3 与 RM 重训进展（2026-11-01 ~ 11-02）

接 `E10_E11_PROGRESS.md`。评测均为官方 LIBERO-Safety 协议，60 cell × 20 回合，单种子。

## 1. 安全标签漏报的修复

- LIBERO-Safety 的 `check_robot_contact` 拿整数 geom ID 和字符串名称比较，robot 接触永远返回假。官方谓词和 aux 谓词都走这个函数，所以 E11-SC 的代价里不含 robot 违规。
- 修复：`sync_real_env.py` 新增 `cost_label: grader`，训练代价直接用 `safety_grader.violated_channels()`（与评测 grader 同一实现，三通道 obj / robot / grasp）。不修改 LIBERO，官方协议和终止条件不变，历史结果可以直接比较。E11 的配置显式锁定为 `cost_label: predicate`。
- 集成测试 `tests/unit_tests/test_safety_cost_contact.py`：把人手移到夹爪上制造真实 MuJoCo 接触，grader 判为违规，LIBERO 谓词判为未违规。

## 2. 实验结果（step 10）

| 实验 | 设置 | SR | unsafe | safe_success |
|---|---|---|---|---|
| base | — | .703 | .198 | .613 |
| E10-O | Wan 观测 + 真实奖励 | .733 | .172 | .648 |
| E12-R | 真实渲染观测 + 真实奖励 | .730* | .161* | .641* |
| E13 | E10-O + grader 代价，固定 η=0.5 | .744* | .189* | .654* |
| E14 | E10-O + 随机翻转奖励（误报 .05 / 漏报 .14） | .693 | .197 | .602 |
| E9 | RM v8 奖励 | .698 | .202 | .622 |
| E8-a | RM v3 奖励 | .664 | .210 | .590 |

\* 与 E10-O 在相同的 57 个 cell 上配对比较；该子集上 E10-O 为 .750 / .173 / .662。

- **观测：** Wan 生成的观测作为策略输入，与真实渲染效果相当；成功、违规和终止仍来自真实仿真器。E12-R 在 obstacle_avoidance 上低 8 个点，原因未明。
- **安全约束：** 修好标签后（E13），10 步内 robot unsafe 仍未下降。
- **RM 误报：** 随机误报本身就足以抹掉真实奖励的收益（E14）。GRPO 的组内标准差归一化会把全失败组里的一次误报放大成一次真实成功（理论文档命题 4）。

## 3. 诊断（D3：E13 续跑 2 步，1024 回合逐回合导出）

- **组内信号稀疏：** 只有 16% 的组在成功与否上有差异，12% 在 robot 违规上有差异。
- **reset 时已违规：** 约 9% 的回合在关键帧上就已违规。扫描全部 47,658 个关键帧，11.1% 已违规或已成功，已剔除（`keyframe_exclude`）。
- **SAM：** 回合级 AUROC .921，但组内两两排序准确率只有 .700；无违规回合上的均值 .128，高于 E11-SAM 的预算 .101，所以预算不可达，η 必然打满。
- **RM 在线：** AUROC .978，误报 4.9%。
- **关键帧难度：** 主要由来源轨迹（`kind` / `quadrant`）和关键帧在轨迹中的位置决定，而不是任务 cell。按这三项分箱加权采样（`keyframe_sampling: weighted`），预计组内有差异的比例从 16% 提到 22%。

## 4. 新增的环境选项（`sync_real_env.py` / `safe_env.py`）

| 选项 | 作用 |
|---|---|
| `cost_label: grader \| predicate`、`cost_window: goal \| episode` | 训练代价标签和统计窗口 |
| `obs_source: wan \| real`、`render_egl_remap` | 真实渲染观测，渲染可挪到别的卡（GPU 3/4 的 EGL 有故障） |
| `reward_noise: {fp, fn}` | 诊断用的随机标签噪声 |
| `dump_dir`、`dump_arrays` | 逐回合 JSON；chunk 中间帧和末帧 JPEG、Wan 潜变量、动作、每步目标和三通道违规 |
| `keyframe_exclude`、`keyframe_sampling: weighted` | 关键帧剔除和加权采样 |
| `rm_frames` | RM 只给 chunk 内指定帧打分 |
| `reward_model.type: QwenRMClient` | Qwen3-VL RM 推理服务的客户端 |

测试：`test_world_model_safe.py` + `test_safety_cost_contact.py`，共 63 项通过。

## 5. RM 重训（进行中）

工具在 `toolkits/codeworld_safe/rm2/`：
- `build_rm_dataset.py`：按配额自动筛选，按 uid 划分；
- `train_rm_ra.py`：ResNet RM；
- `rb_eval.py`、`train_rm_rc.py`：Qwen3-VL 零样本评估和 LoRA 训练；
- `qwen_rm_server.py`：推理服务；
- `rm_metrics.py`：离线指标。

流程在 `toolkits/codeworld_safe/scgrpo/night2/rm2_pipeline.sh`。规划和理论推导见同目录的 `RM2_PLAN.md`、`RM_GRPO_THEORY.md`。阶段 1 的收集砍掉了 col11m（待办见 `RM2_PLAN.md`）。
