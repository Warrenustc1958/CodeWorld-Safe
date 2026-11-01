# 学习型奖励与代价下的 GRPO：误差如何进入策略梯度（对标 DRILL 的理论框架）

2026-11-01。DRILL（arXiv 2610.30814）分析的是"世界模型误差如何影响策略改进"，并据此按下游回报训练世界模型。我们的实验（E10-O、E8-a、E9、E11、E13）说明，主要损失来自奖励模型（RM）和代价模型（SAM），不是 Wan 的观测。本文给出对应的理论：学习型奖励和代价的误差怎样进入 GRPO 梯度，哪类误差无害、哪类有害，RM 和代价模型应该按什么目标训练。

所有结论都标明了假设。和已有工作重合的部分在第 9 节列出，写论文时需要引用。

---

## 0. 记号与设定

- 起始状态 $s_0 \sim \rho$（训练关键帧）。策略 $\pi_\theta$ 在世界模型 $W$ 中展开，得到轨迹 $\tau = (o_{0:T}, a_{0:T-1})$，
  $$p_\theta(\tau \mid s_0) = \prod_t \pi_\theta(a_t \mid o_{\le t})\, W(o_{t+1} \mid o_{\le t}, a_t).$$
  $W$ 与 $\theta$ 无关，因此轨迹得分函数为 $s_\theta(\tau) := \nabla_\theta \log p_\theta(\tau\mid s_0) = \sum_t \nabla_\theta \log \pi_\theta(a_t\mid o_{\le t})$。
- 真实结果 $R(\tau) \in \{0,1\}$：同步真实仿真器执行同一动作序列得到的结果。仿真器是确定性的，所以 $R$ 是 $(s_0, a_{0:T-1})$ 的函数。真实代价 $C(\tau) = \max_k C_k(\tau)$，通道 $k \in \{\text{obj}, \text{robot}, \text{grasp}\}$。
- 学习型奖励 $\hat R(\tau)$，条件均值 $\mu(\tau) := \mathbb E[\hat R \mid \tau]$。误差 $e(\tau) := \mu(\tau) - R(\tau)$。
- 目标 $J(\theta) = \mathbb E_{s_0}\mathbb E_{\tau\sim p_\theta(\cdot|s_0)}[R(\tau)]$，条件成功率 $J(s_0) := \mathbb E[R\mid s_0]$。
- Fisher 矩阵 $F := \mathbb E_{s_0}\mathbb E_\tau[s_\theta s_\theta^\top]$。范数 $\|u\|_F := \sqrt{u^\top F u}$；对偶范数 $\|b\|_{F^{-1}} := \sup_{u\neq 0} \langle u, b\rangle / \|u\|_F$（$F$ 奇异时用这个上确界定义，等价于伪逆）。函数的 $L_2$ 范数 $\|f\|_{L_2(\pi)} := \sqrt{\mathbb E_{s_0}\mathbb E_\tau[f(\tau)^2]}$。
- GRPO：每个 $s_0$ 采 $G$ 条轨迹，组内优势 $\hat A_i$，梯度估计 $\hat g = \frac1G \sum_i \hat A_i\, s_\theta(\tau_i)$。

**假设 A（单步、同策略）**：分析一次更新，重要性比为 1，裁剪不起作用。这对应 GRPO 每批数据的第一个梯度步；我们的配置 update_epoch=1。

第 1–4 节使用中心化优势（不除以组内标准差），第 5 节处理标准差归一化。

---

## 1. 定理 1：奖励误差进入梯度的精确形式

**定理 1。** 在假设 A 下，采用留一基线 $\hat A_i = \hat R_i - \frac{1}{G-1}\sum_{j\neq i}\hat R_j$，有
$$\mathbb E[\hat g] = \nabla_\theta\, \mathbb E[\mu(\tau)] = \nabla_\theta J(\theta) + \underbrace{\mathbb E\big[e(\tau)\, s_\theta(\tau)\big]}_{=:\, b}.$$
如果用组均值基线（含自身），右边整体乘以 $(G-1)/G$。

**证明。** 组内轨迹在给定 $s_0$ 时独立同分布，基线 $\bar R_{-i}$ 与 $\tau_i$ 独立，且 $\mathbb E[s_\theta(\tau_i)\mid s_0]=0$，所以
$$\mathbb E[\hat A_i s_\theta(\tau_i)] = \mathbb E[\hat R_i s_\theta(\tau_i)] - \mathbb E[\bar R_{-i}]\,\mathbb E[s_\theta(\tau_i)] = \mathbb E[\mu(\tau) s_\theta(\tau)].$$
由得分函数恒等式（$\mu$ 不依赖 $\theta$，RM 冻结），$\mathbb E[\mu\, s_\theta] = \nabla_\theta \mathbb E[\mu]$；再代入 $\mu = R + e$。组均值基线满足 $\hat R_i - \bar R = \frac{G-1}{G}(\hat R_i - \bar R_{-i})$。∎

**含义。** 用 RM 训练时，GRPO 无偏地上升"RM 认为的回报" $\mathbb E[\mu]$。偏差 $b$ 是期望误差 $\mathbb E[e]$ 的梯度，也就是策略能把 RM 误差往有利于自己的方向推多少。决定危害的不是误差有多大，而是误差和策略行为的相关性。这就是 reward hacking 的形式化。

---

## 2. 命题 2：与行为无关的随机标签噪声只缩放梯度

**命题 2。** 设 RM 误判只依赖真实标签、与轨迹的其他内容无关：
$$P(\hat R = 1 \mid \tau) = \alpha\,(1-R(\tau)) + (1-\beta)\,R(\tau),$$
其中 $\alpha$ 是误报率，$\beta$ 是漏报率，都是常数。则
$$\mathbb E[\hat g] = (1-\alpha-\beta)\, \nabla_\theta J(\theta).$$

**证明。** $\mu(\tau) = \alpha + (1-\alpha-\beta)R(\tau)$，再用定理 1 和 $\mathbb E[s_\theta]=0$。∎

**可证伪的预测。** 我们在 E10-O 上测得的 RM 在线误差约为：误报率 $\alpha \approx 0.03/0.60 = 0.05$（rm_false / 真实失败比例），漏报率 $\beta \approx 1-0.86 = 0.14$，所以 $1-\alpha-\beta \approx 0.81$。这种量级的随机噪声只相当于把学习率乘 0.81，而且组内标准差归一化会部分抵消这个缩放。它不足以解释 E8-a（.664）比 E8-O（.732）低 7 个点。

由此得到噪声注入实验的预测：在真实奖励上按 $(\alpha, \beta) = (0.05, 0.14)$ 随机翻转标签，结果应接近 E10-O。如果预测成立，RM 的危害就来自系统性误差（第 3 节）。如果不成立，说明假设 A 或组内归一化的非线性（第 5 节）起了主要作用。（命题 2 的思想与 Wang et al. 2020 的带扰动奖励 RL 一致，见第 9 节。）

---

## 3. 定理 3：系统性误差的 Fisher 对偶界与上升条件

把条件均值分解为
$$\mu(\tau) = \bar\alpha + (1-\bar\alpha-\bar\beta)\,R(\tau) + \delta(\tau),$$
其中 $\bar\alpha := \mathbb E_\pi[\mu \mid R=0]$，$1-\bar\beta := \mathbb E_\pi[\mu \mid R=1]$（在当前策略分布下取平均）。于是 $\mathbb E[\delta \mid R] = 0$：$\delta$ 是去掉"平均误报和漏报"之后、与行为相关的系统性误差。

另外，由于 $\mathbb E[s_\theta\mid s_0]=0$，只依赖起始状态的偏移 $c(s_0)$ 不会进入梯度。定义组内中心化误差 $\tilde\delta(\tau) := \delta(\tau) - \mathbb E[\delta\mid s_0]$。

**定理 3。**
$$\mathbb E[\hat g] = (1-\bar\alpha-\bar\beta)\,\nabla J + b, \qquad b = \mathbb E[\tilde\delta\, s_\theta],$$
$$\|b\|_{F^{-1}} \;\le\; \|\tilde\delta\|_{L_2(\pi)}.$$

**证明。** 第一式与命题 2 相同，常数项乘 $\mathbb E[s_\theta]=0$；把 $\delta$ 换成 $\tilde\delta$ 是因为 $\mathbb E[\mathbb E[\delta|s_0]\, s_\theta] = \mathbb E[\mathbb E[\delta|s_0]\,\mathbb E[s_\theta|s_0]] = 0$。对任意 $u$，由 Cauchy–Schwarz，
$$\langle u, b\rangle = \mathbb E[\tilde\delta\,\langle u, s_\theta\rangle] \le \|\tilde\delta\|_{L_2(\pi)}\sqrt{\mathbb E\langle u,s_\theta\rangle^2} = \|\tilde\delta\|_{L_2(\pi)}\,\|u\|_F. \quad\blacksquare$$

**推论 3.1（真实信号的上界）。** 由于 $\nabla J = \mathbb E[(R - J(s_0))\, s_\theta]$，同样的论证给出
$$\|\nabla J\|_{F^{-1}} \le \sigma_R, \qquad \sigma_R := \sqrt{\mathbb E_{s_0}\big[J(s_0)(1-J(s_0))\big]}.$$

**推论 3.2（上升条件）。** 对自然梯度方向 $d = F^{-1}\mathbb E[\hat g]$，
$$\langle \nabla J, d\rangle \ge (1-\bar\alpha-\bar\beta)\|\nabla J\|_{F^{-1}}^2 - \|\nabla J\|_{F^{-1}}\,\|\tilde\delta\|_{L_2(\pi)}.$$
因此只要
$$\boxed{\;\|\tilde\delta\|_{L_2(\pi)} < (1-\bar\alpha-\bar\beta)\,\|\nabla J\|_{F^{-1}}\;}$$
这一步就会提高真实目标（这是充分条件，不是必要条件）。

**含义。**
1. 衡量 RM 好坏的应该是当前策略分布下、组内中心化的系统性误差 $\|\tilde\delta\|_{L_2(\pi)}$，而不是离线测试集上的准确率。这解释了 RM v8 离线指标更好、在线结果却没改善（E9）的现象：离线集不是 $\pi_\theta$ 的分布。
2. **后期崩坏：** 随着训练推进，策略在每个起始状态上的成败变得确定，$J(s_0)$ 趋向 0 或 1，真实信号的上界 $\sigma_R \to 0$。系统性误差却不必随之减小。所以上升条件会在训练后期先失效。这与 rmall 训到 40 步后 SR 从 .696 降到 .609 的现象一致。真实奖励（$\tilde\delta = 0$）不受这个限制。
3. 只依赖任务或起始状态的整体偏移（比如 RM 在某个任务上一律偏高）是无害的；有害的是同一起始状态下、随行为变化的误判（比如抓错部位却判成功）。

---

## 4. 命题 4：组内标准差归一化会把单个误报放大成一次完整更新

我们使用的 GRPO 优势是 $\hat A_i = (\hat R_i - \bar R)/\max(\hat\sigma, \sigma_f)$，其中 $\hat\sigma$ 是组内标准差，$\sigma_f = 0.1$。

**命题 4。** 设一组 $G$ 条轨迹真实全部失败，其中一条被 RM 误判为成功。则它的优势为
$$\hat A_{\text{FP}} = \frac{1-1/G}{\sqrt{\frac1G(1-\frac1G)}} = \sqrt{G-1},$$
其他轨迹为 $-1/\sqrt{G-1}$。这和"一组里恰好有一次真实成功"时的优势完全相同。$G=8$ 时 $\hat A_{\text{FP}} = \sqrt 7 \approx 2.65$，此时 $\hat\sigma \approx 0.33 > \sigma_f$，下限不起作用。

**证明。** 直接计算：组均值 $1/G$，总体标准差 $\sqrt{\frac1G(1-\frac1G)}$。∎

**推论 4.1（脆弱性集中在难任务）。** 某起始状态下整组真实全失败的概率是 $(1-J(s_0))^G$。$J(s_0) = 0.1$ 时约为 0.43，$J(s_0) = 0.4$ 时约为 0.017。所以误报造成的伪更新主要发生在当前成功率低的起始状态上。这对应 affordance L2 一类的难 cell。

**推论 4.2（归一化下的定理 3）。** 把每个起始状态的组内标准差近似为总体值 $\sigma(s_0) = \sqrt{J(s_0)(1-J(s_0))}$，并当作常数（一阶近似），则定理 3 中的 $\tilde\delta$ 换成 $\tilde\delta(\tau)/\max(\sigma(s_0),\sigma_f)$：
$$\|b_{\text{norm}}\|_{F^{-1}} \lesssim \Big\|\frac{\tilde\delta}{\max(\sigma(s_0),\sigma_f)}\Big\|_{L_2(\pi)}.$$
也就是说，在组内结果方差小的起始状态（几乎全失败或几乎全成功）上，RM 误差的权重是 $1/\sigma(s_0)^2$。

---

## 5. 定理 5：RM 的一步元梯度有闭式解（与 DRILL-IMG 对比）

DRILL 的外层目标是 $\max_\phi J(\theta^+(\phi))$，$\theta^+ = \theta + \eta\nabla_\theta \hat J_{\text{GRPO}}(\pi_\theta; W_\phi)$。世界模型 $\phi$ 改变的是轨迹分布，所以外层梯度需要混合二阶导 $\nabla_\phi\nabla_\theta \hat J$，DRILL-IMG 用截断 BPTT 和 Hessian-向量积来估计（DRILL 式 7）。

把同样的问题用到 RM $\psi$ 上：$\theta^+(\psi) = \theta + \eta\, \bar g(\psi)$，其中 $\bar g(\psi) = \mathbb E[\mu_\psi(\tau)\, s_\theta(\tau)]$（定理 1，留一基线下的总体梯度）。

**定理 5。**
$$\nabla_\psi J(\theta^+(\psi)) = \eta\; \mathbb E_{\tau\sim p_\theta}\Big[\big\langle \nabla J(\theta^+),\, s_\theta(\tau)\big\rangle\; \nabla_\psi \mu_\psi(\tau)\Big].$$

**证明。** 链式法则：$\nabla_\psi J(\theta^+) = (\partial\theta^+/\partial\psi)^\top \nabla J(\theta^+)$，而 $\partial\theta^+/\partial\psi = \eta\,\mathbb E[s_\theta\, \nabla_\psi\mu_\psi^\top]$。采样分布 $p_\theta$ 不依赖 $\psi$。∎

**和 DRILL 的区别。** RM 只通过轨迹权重进入内层梯度，不改变采样分布，所以内层梯度对 $\mu_\psi$ 是线性的。外层梯度不需要 BPTT，也不需要 Hessian-向量积，只要对每条轨迹算一个标量 $\langle \nabla J, s_\theta(\tau)\rangle$。

**但纯元目标退化。** 外层目标对 $\mu$ 是线性的，最优解是 $\mu^*(\tau) = \mathbf 1\{\langle\nabla J, s_\theta(\tau)\rangle > 0\}$：它把"提升这一步"当成唯一目标，与真实成功无关，不能直接当训练目标。因此我们采用定理 3 和推论 4.2 给出的上界作为目标：上界在 $\mu = R$ 时为零，并且直接控制梯度偏差。

**训练目标（下游一致的 RM 学习）：**
$$\mathcal L(\psi) = \mathbb E_{s_0\sim\rho}\; w(s_0)\; \mathbb E_{\tau\sim\pi_\theta(\cdot|s_0)}\Big[\big(\mu_\psi(\tau) - R(\tau) - c_\psi(s_0)\big)^2\Big], \qquad w(s_0) = \frac{1}{\max(\sigma(s_0),\sigma_f)^2}.$$

- **数据：** 当前策略的 rollout，标签来自同步仿真器，按 DAgger 式迭代：训 RM → 用新策略收集数据 → 再训。
- **偏移项 $c_\psi(s_0)$：** 可选。允许 RM 在每个起始状态上有任意整体偏移，因为这部分不进入梯度（定理 3）。实际中可以把损失写成组内中心化的平方误差。
- **权重：** $\sigma(s_0)$ 用同一组的真实标签估计。
- **保证：** 由推论 4.2，$\|b_{\text{norm}}\|_{F^{-1}} \lesssim \sqrt{\mathcal L(\psi)}$，再代入推论 3.2 得到这一步提高真实目标的充分条件。

---

## 6. 代价侧：两个结构性结论

带约束的 GRPO 使用奖励 $r = \hat R - \eta\,\hat C$。由定理 1，代价梯度的偏差是 $-\eta\,\mathbb E[e_C\, s_\theta]$，定理 3 和推论 4.2 同样适用。另外有两个专属于约束问题的结论。

**命题 6（漏掉的通道不可约）。** 如果代价标签是 $\hat C = \max_{k \in K'} C_k$，漏掉了通道 $k^* \notin K'$，并且 $C_{k^*}$ 不影响 $\hat R$ 和 $\hat C$，那么 GRPO 的期望更新与 $C_{k^*}$ 完全无关。对任意 $\eta$ 和预算，拉格朗日问题的驻点都不依赖 $C_{k^*}$。$J_{C_{k^*}}$ 只会随其他目标的变化间接改变。

**证明。** 由定理 1，期望更新 $= \nabla\mathbb E[\mu_R] - \eta\nabla\mathbb E[\mu_C]$，两项都不含 $C_{k^*}$。∎

**和实验的对应。** E11-SC 的代价来自 LIBERO 谓词，robot 接触谓词恒为假（ID 和名称不匹配）。由命题 6，robot 通道不受约束；E11-SC 的 robot unsafe 停在 .089–.093，与 base 相同。修复后（E13，grader 标签）这个通道才第一次进入梯度。

**命题 7（预算不可达时乘子必然饱和）。** 投影对偶上升 $\eta_{t+1} = \mathrm{clip}(\eta_t + \beta(\hat J_C(\pi_t) - d), 0, \eta_{\max})$。如果对所有 $t$ 都有 $\hat J_C(\pi_t) \ge d + \epsilon$，那么 $\eta$ 最多 $\lceil \eta_{\max}/(\beta\epsilon)\rceil$ 步就到达 $\eta_{\max}$，此后保持不变，约束退化成固定惩罚。

另外，学习型代价有误报下限：即使策略完全安全，$\hat J_C \ge \bar\alpha_C$。所以 $d \le \bar\alpha_C$ 时预算一定不可达。

**和实验的对应。** E11-SAM：$d = 0.101$，训练中 $\hat J_C \in [0.176, 0.337]$，$\epsilon \gtrsim 0.075$；$\beta = 2$、$\eta_{\max} = 5$ 时，界给出至多约 34 步，按典型的 $\epsilon \approx 0.15$ 约为 17 步。实际在第 15 步饱和，与界一致。同时 E11-SAM 的真实 unsafe 并没有比 E10-O 低，说明 SAM 的误报下限远高于真实代价（SAM 约 .2–.3，真实约 .05）。

**实践规则。** 开训前，用 base 策略多轮估计 $\hat J_C$ 的误报下限，预算要设在下限之上。不能只用"首轮均值的一半"。

---

## 7. 定理 8：截断回合下的安全成功目标（价值自举）

训练回合从中间关键帧起跑，只有 $T=64$ 步；完整任务更长。第 0 步分析显示，robot 首次违规约一半发生在评测第 64 步之后。截断目标看不到这些违规。

定义安全成功事件 SS = "到达目标，且此前没有违规"。安全可完成价值
$$V_S(s) := P_\pi(\text{SS} \mid s),$$
$s$ 是仿真状态加上策略的观测历史（使过程满足 Markov 性）。在第 $T$ 步"仍未结束"记为 $\text{alive}_T$：还没成功，也还没违规。定义
$$G_T := \mathbf 1\{\text{SS 在 } T \text{ 步内发生}\} + \mathbf 1\{\text{alive}_T\}\, V_S(s_T).$$

**定理 8。** $\mathbb E[G_T \mid s_0] = V_S(s_0)$，即用 $G_T$ 做 GRPO 等价于优化完整时域的安全成功率。如果用估计值 $\hat V$ 代替 $V_S$，梯度偏差为 $\mathbb E[\mathbf 1\{\text{alive}_T\}(\hat V - V_S)(s_T)\, s_\theta]$，满足定理 3 的界，其中 $\delta = \mathbf 1\{\text{alive}_T\}(\hat V - V_S)(s_T)$。

**证明。** 按"SS 是否在 $T$ 步内发生"分解。若 $T$ 步内已结束，后续不影响结果。若 alive，由 Markov 性，$P(\text{SS} \mid \mathcal F_T) = V_S(s_T)$。取期望（塔性质）即得。偏差部分由定理 1 得到。∎

**对比当前目标。** 当前奖励是 $\mathbf 1\{T \text{ 步内成功}\}$，减去 $T$ 步内的代价，等于把 $V_S(s_T)$ 当成 0（未结束视为失败）、把 $T$ 之后的违规视为不存在。它的梯度和完整时域目标相差 $\mathbb E[\mathbf 1\{\text{alive}_T\} V_S(s_T)\, s_\theta]$ 加上 $T$ 之后违规的那部分。训练中约六成回合在 64 步内没有结束，这个差不可忽略。

**$V_S$ 的估计。**
- 蒙特卡洛：从保存的仿真状态出发，用当前策略做多条续写。这就是 Do VLAs Know They Will Fail 估计"不可挽回点"的方法；我们的 Step E 反事实分支数据可以直接用。
- 学习型：在 Wan 潜变量或 VLA 特征上训练 $\hat V$，用分支数据作标签。误差同样受定理 3 约束。

（定理 8 的思路与截断时域的价值自举相同，见 Pardo et al. 2018。我们的贡献是把它用到"安全成功"这个复合事件，并给出与第 3 节一致的误差界。）

---

## 8. 与 DRILL 的逐项对比

| 方面 | DRILL | 本文 |
|---|---|---|
| 学习对象 | 世界模型 $W_\phi$ | 奖励模型、代价模型、安全价值 $\hat V$ |
| 误差进入策略梯度的方式 | 改变轨迹分布，非线性 | 只改变轨迹权重，线性（定理 1） |
| 外层梯度 | 截断 BPTT + Hessian-向量积（DRILL-IMG）；一阶近似（DRILL-VWI） | 闭式（定理 5），不需要 BPTT |
| 训练目标 | 按访问分布、组内优势、置信度加权的像素误差 | 当前策略下、组内中心化、按 $1/\sigma(s_0)^2$ 加权的奖励误差（第 5 节） |
| 理论结论 | VWI 是 IMG 元梯度的一阶近似；截断 BPTT 的收敛率 | 偏差的精确表达式；随机噪声只缩放梯度；系统性误差的对偶界与上升条件；归一化放大；漏通道不可约；预算不可达必饱和；截断一致目标 |
| 对仿真器的依赖 | DRILL-IMG 每个外层步需要 30 条仿真 rollout | 训练 RM 时需要同步仿真标签；策略训练阶段可以不用 |
| 能解释的现象 | 像素 MSE 和策略成功率脱钩 | RM 离线精度与在线效果脱钩（E9）、后期崩坏（rmall）、robot 通道不下降（E11-SC）、乘子饱和（E11-SAM） |

两者互补：DRILL 处理观测或动力学误差，本文处理奖励和代价误差。我们的 oracle 拆分（E10-O、E12-R、E8-a/E9）显示后者是主要损失来源。

---

## 9. 与已有理论的关系（需要引用，避免夸大新颖性）

- 命题 2：Wang, Liu, Li, "Reinforcement Learning with Perturbed Rewards"（AAAI 2020）研究过类条件奖励噪声，并在已知混淆矩阵时给出无偏替代奖励。我们的新增是在 GRPO 组内基线下的精确形式，以及与系统性误差的分离（定理 3）。
- 定理 8：与 Pardo et al., "Time Limits in Reinforcement Learning"（ICML 2018）的截断时域价值自举是同一思路。我们用在安全成功事件上，并结合学习型估计的误差界。
- 定理 1 的得分函数恒等式和推论 3.1 的 Cauchy–Schwarz 论证是标准工具。新的部分是对 RM 误差的分解方式、命题 4 的归一化放大、定理 5 与 DRILL 的对比，以及第 6 节的约束结构性结论。

---

## 10. 可证伪的预测与验证计划

| 编号 | 预测 | 实验 | 不成立时说明什么 |
|---|---|---|---|
| P1 | 按 $(\alpha,\beta)=(0.05,0.14)$ 随机翻转真实奖励，结果接近 E10-O | 噪声注入，10 步 | 组内归一化的非线性或单步假设不成立 |
| P2 | RM 的有害误判集中在组内结果方差低的起始状态（推论 4.1） | 训练时逐条记录 RM 和真实标签及组号，统计全失败组里的误报 | 需要别的机制解释 RM 失败 |
| P3 | 用推论 4.2 加权、在当前策略数据上训练的 RM，在线结果比 E8-a/E9 更接近 E10-O | 迭代重训 RM 后跑 10–20 步 | 上界太松，或需要定理 5 的元梯度 |
| P4 | 组内 robot 代价方差接近 0 时，任何固定 η 都降不了 robot unsafe（E13 的现象） | 统计 E13 训练中 robot 代价组内不全相同的组占比 | 命题 6 之外还有别的瓶颈 |
| P5 | 预算设在误报下限之上时，η 不再饱和（命题 7） | 开训前估计下限并据此设预算 | 下限估计不准 |
| P6 | 用 $G_T$（定理 8）训练能降低第 64 步之后才出现的 robot 违规 | 用分支数据估计 $V_S$，跑短程对照 | $\hat V$ 误差过大，或 Markov 假设不成立 |

---

### 10.1 已有结果（2026-11-02）

- **P1 不成立。** E14 在真实奖励上按误报 0.05、漏报 0.14 随机翻转，10 步后 SR .693，E10-O 是 .733，base 是 .703；unsafe .197，E10-O 是 .172。按命题 2 的预测，结果应接近 E10-O，实际接近 base 和 E9（.698）。
- **命题 4 能解释这个结果。** D3 统计显示：54% 的组真实全失败，只有约 16% 的组有组内差异。误报率 0.05 时，全失败组里至少出现一次误报的概率是 1−0.95⁸≈34%，所以约 18% 的组产生伪更新，每次的优势都是 √7。伪信号组的数量超过了真实信号组，所以命题 2 的"只缩放梯度"在 GRPO 的组内归一化下不成立。
- **对 RM 的要求。** 全失败组中出现误报的比例要远低于真实信号组的比例，大约要求单回合误报率 ≲1%。RM 的阈值因此标定为"失败回合误报 ≤ 1%"。

## 11. 假设与局限

1. 假设 A 只覆盖每批数据的第一个梯度步；多轮更新中的重要性比和裁剪会引入额外偏差，这里没有分析。
2. 推论 4.2 把组内标准差当作总体值和常数，是一阶近似；组大小 $G=8$ 时样本标准差的波动不小。
3. 定理 8 要求状态包含策略的观测历史，才满足 Markov 性；用 Wan 观测训练时，真实仿真状态和策略看到的观测并不一致，这个差距由世界模型误差负责，不在本文范围内。
4. 推论 3.2 是充分条件。界可能很松，所以实验上要以 P1–P6 为准，不能只靠界的数值。
5. 本文没有处理 Wan 观测本身的误差。这部分可以用 DRILL 的方法，或者用 E12-R 的拆分来度量。
