# 阶段二 TODO（依次完成，状态随进度更新）

依据：`WAN_DRIFT_PLAN.md` 第 5 节；阶段一结论见 1.10。

| # | 任务 | 状态 | 备注 |
|---|---|---|---|
| 1a | T1：wrist 位姿条件。从 dual-41k 续训，训练用仿真器真值位姿，推理用 π0.5 拟合的动作积分 | 训练中（8–15 号卡） | `work/scgrpo/wan_t1/t1_e1`，5250 步（**约 0.13 epoch**，见进度记录 19:20），预计 23:30 结束 |
| 1b | T0 训练数据：step_e train split 开环生成历史，配对真实下一块和位姿 | 生成中（0–7 号卡） | `work/scgrpo/drift/t0`；val 212 条已完成，然后 test、train 2000 条，预计 20:00 结束 |
| 2 | T2（每块历史纠偏 adapter）与 T3（直接在生成历史上微调）并行，同数据同算力 | 已排队（0–7 号卡，T0 结束后自动开始） | `drift/run_stage2.sh`：T2 缓存 → T3（5250 步，和 T1 同预算）+ T2 adapter（与 T3 共用 0 号卡） |
| 3 | 评测：T0 评测集（π0.5 闭环，约 200–300 个起点）上的 P1 曲线、RM 一致率、R7 | 待做 | 评测集按起点与训练分开 |
| 4 | T5：冻结策略配对评估，然后 Wan_old/new × oracle/RM 的 2×2 GRPO，真实 LIBERO-Safety 评测 | 待做 | |

## 进度记录

- 11-03 16:20 位姿标签 `work/scgrpo/drift/pose_index`：9776/9934 条（158 条 legacy 缺 json），和 keyframes_v1 的状态逐帧核对误差为 0。
- 11-03 16:40 T1 代码：`RLinf-Safety/toolkits/libero_safety/wan_ac_sft/pose_cond.py`（零初始化、每个视角一个 pose 头；单测：初始等价于 dual-41k）；`train_wan_cws.py` 新增 `--pose_index --pose_lr --grad_ckpt --zero1 --save_resume`。
- 11-03 16:45 T1 冒烟测试通过：全量微调 + 梯度检查点 + ZeRO-1，每卡峰值 70 GB（另有别的容器 11 GB）。不开梯度检查点时 batch 5 就 OOM，OOM 发生在前向的激活上。
- 11-03 16:50 T1 正式训练启动：有效 batch 80，lr 5e-6 余弦衰减到 0.2 倍，pose 头 lr 1e-4，EMA 0.999，每 500 步验证一次（含"换成别的样本的 pose"对照），约 7–8.5 秒/步。
- 11-03 17:00 T3 代码：`RLinf-Safety/.../gen_hist.py`（只把 1–4 帧换成 T0 生成的历史，未来 8 帧仍是录制的，已核对）；`train_wan_cws.py` 新增 `--gen_hist --gen_hist_prob --gen_val --val_at_start`。验证集额外报告"在生成历史上的 loss"（T3 的目标），第 0 步也算一次，作为 dual-41k 的基线。
- 11-03 17:05 T2 代码：`CodeWorld-Safe/.../drift/t2_adapter.py`（缓存生成历史和真实历史的 latent 第 1 格；每个视角一个 FiLM 卷积 adapter，零初始化，输入是生成历史的 latent、anchor 和带估计噪声的位姿）。在 CPU 上用 1 个 episode 跑通；wrist 的 latent 误差是 agent 的 3.9 倍，和像素结果一致。
- 11-03 19:20 **更正：5250 步不是 1 epoch。** 用训练器自己的采样器模拟 5250 × 80 次抽样：训练器以 12 帧窗口为采样单位（stride 1），共 3,240,094 个窗口，过一遍要 40,501 步（dual-41k 的 41k 步就是约 1 遍）；按"每帧作为预测目标出现一次"算是 4,088 步。5250 步实际抽到 9.1% 的窗口；9134 条 episode 每条都被抽到（19–50 个窗口），但只有 35.4% 的帧当过预测目标，当过的平均 2.7 次（采样器有 37% 的事件窗口，还按运动幅度加权）。之前算的 5250 步是"不重叠 8 帧块（含 branch 前缀）÷ 80"，口径错了。
- 11-03 19:30 T0 完成：train 2000 / val 212 / test 309 条。T2 缓存已开始（0–7 号卡，每卡约 7 GB）。T3 等 T1 测试结束（`drift/t1eval/DONE`）后再启动。
- 11-03 19:33 T1 第 2000 步 EMA 的可控性和 256 步 rollout 测试（`drift/t1_eval.py`）在 0–7 号卡上运行。1 个 episode 的冒烟测试：未来 pose 平移 3 cm，输出 MAE 只变 0.21 / 0.15（只换噪声时约 1.0），读出的 EEF 位移 0.06 cm（增益 0.02），pose 头对像素几乎还没有作用。
