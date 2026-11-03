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
- 11-03 20:02 **T1 第 2000 步测试**（34 个起点 × 2 条 episode，`work/scgrpo/drift/t1eval/summary.txt`）：
  - 位姿可控性约等于 0。未来 pose 平移 3 cm / 10 cm / 偏航 20°，输出相对原输出的 MAE 只有 0.24–0.35（agent）/ 0.40–0.89（wrist）；只换噪声时是 1.23 / 2.93。读出的 EEF 位移 0.04 / 0.08 cm，增益 0.01–0.02。
  - teacher-forced 单块误差比 dual-41k 略差：agent 4.02 vs 3.65，wrist 9.29 vs 8.59。动作可控性还在：位姿置零后变化 1.86 / 3.31（dual-41k 是 2.57 / 4.61）。
  - 256 步 rollout：wrist MAE 50.4（t1_int）、50.1（t1_real），free 是 47.2；第 17–32 块 RM 一致率 .876 / .859，free .834。
  - 结论：到第 2000 步，pose 只在 teacher-forced loss 上有一点作用（换掉 pose 后 loss 变大），对生成画面几乎没有控制力。
- 11-03 20:00 代码已在本地提交：CodeWorld-Safe `wan-drift-stage2` @ 35f9d958，RLinf-Safety `wan-drift-t1t3` @ bc016226。**两个都没推上去**：本机 SSH key 认证的是 GitHub 账号 nomoretime-cell，对 Warrenustc1958 的两个仓库没有写权限；HTTPS 没有凭证。
- 11-03 20:05 ModelScope 上传被权限检查拦下（打包里含两个私有仓库的 git bundle）。已把 code bundle 从暂存目录删掉，`work/scgrpo/ms_stage` 现在只有 T1 位姿数据，等用户确认后再上传。
- 11-03 20:05 停掉了 stage-2 编排脚本，T2 缓存进程继续跑，T3 不会自动在 8 卡上启动。T2 adapter 训练改由 `drift/run_t2train.sh` 在缓存结束后于 0 号卡启动。
- 11-03 两个分支已推送到 GitHub：RLinf-Safety `wan-drift-t1t3` @ bc016226，CodeWorld-Safe `wan-drift-stage2` @ 35f9d958（用专门的 SSH key `~/.ssh/id_ed25519_warrenustc1958`，主机别名 `github-warren`），远端和本地提交一致。
- 11-03 20:34 T2 adapter 训练完成（`work/scgrpo/drift/t2_run/adapter.pt`，20k 步，26 分钟）：val 9,374 个生成历史上 latent κ = 0.454（agent）/ 0.310（wrist），达到 R1 要求的 ≤ 0.5。推理时的位姿是"真实位姿 + 3 cm / 3° 噪声"，用来模拟估计器误差；闭环效果还没测。缓存缺 93 条 legacy `o*` episode（没有位姿标签）。
- 11-03 20:50 host3 的 T1 在第 3000 步保存后停止（只停了它自己的 57 个进程）。`step-3000` 和 `ema-step-3000` 已用硬链接保存在 `wan_t1/pinned/`。第 2500 / 3000 步验证：`all` .1021 / .1014，`shuffled_pose` .1116 / .1126（两者的差从第 500 步的 .0023 增大到 .0112）。16 张卡全部空闲，等用户决定 T3 的配置。
- 11-03 20:55 `ms_stage` 共 29 GB 待上传：t1（位姿标签、`t1_step-3000.safetensors`）和 t2（T0 历史、T2 缓存、adapter），已生成 SHA256SUMS。`train_wan_cws.py` 新增 `--rewarmup`（50964930，已推送）。
