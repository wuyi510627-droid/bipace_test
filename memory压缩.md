# memory表示空间软分组信用分配

## 一. 目的

我做了三条线的文献核实、并基于 3060（12G）小算力、Qwen2.5-7B-Instruct 模型判断其可行性。**结论**：真正的瓶颈不是"压缩空间难分组"，而是"**memory 表示的保真度**"。

---

## 1. 研究问题定位

- 场景：agentic RL，单智能体多轮交互，稀疏轨迹奖励 → 需要把功劳/罪责摊到每一步（credit assignment, CA）。
- 主流范式：**分组式 CA**（把"处于相同处境"的多次决策凑成组，组内比不同动作的回报 → 得到每步优势）。
- 切入点：当 agent 把历史**压缩进 memory 模块**时，如何在这个**压缩表示空间**里做软分组信用分配，并且要判断是否值得压缩进memory。

---

## 2. 文献调研（逐篇核实）

### 2.1 CA 主线
| 方法 | 核心 | 自己论文里承认的未解问题 |
|---|---|---|
| **GiGPO** (NeurIPS'25, 2505.10978) | anchor-state 分组：把**完全相同的环境状态**下的动作凑组（ε=0） | future work 明写：需要"通过 embedding 或近似匹配引入**状态相似性**" |
| **HGPO** (ICLR'26, 2602.22817) | 层级分组，修 GiGPO 的 context 不一致；用原始 K-window | 承认：一旦历史被 **memory 模块压缩**，层级分组变得 intractable |
| **GraphGPO** (2605.26684) | 状态转移图 + 反向 Dijkstra 距离 `10·γ^d` | 假设**确定性环境 + 目标可达**，开放/随机环境不适用 |

### 2.2 两个正面竞品（都把我的方向列为 future work）

**BiPACE** (2606.25556, *Bisimulation-Guided Policy Optimization with Action Counterfactual Estimation for LLM Agents*)。它指出 step-level group RL 有 **state-action credit mismatch** 两侧毛病：状态侧（"观测完全相同"当价值等价太稀疏→单点组）、动作侧（组内均值给所有动作同一 baseline）。对应两招：
- **BiGPO（互模拟分组）**：不做精确哈希，用 **actor 冻结隐状态**（7B 第 −8 层、**final prompt token**、L2 归一化 φ_θ(s)）+ **朴素 cosine 距离** `d_cos=1−uᵀv` + 贪心聚类（ε=0.10）。**注意：是朴素 cosine、隐状态未用 bisimulation loss 训练**（只当 bisimulation 的经验代理）。
- **PACE（动作反事实）**：簇内按执行动作拆分算 `Q̂(s,a)−V̂(s)`（簇均值=V̂，同动作同伴=Q̂）。

只在 ALFWorld/WebShop/TextCraft（状态会重复的环境）验证。**结论里 future work 逐字原文**：
> *"Finally, extending the bisimulation-guided grouping to agents that compress history into a memory module (where direct observation hashing is intractable) is an interesting avenue for future work."*

（把互模拟引导的分组推广到"把历史压缩进 memory 模块"的智能体——此时直接哈希观测行不通——是值得探索的未来方向。）**这就是本课题=BiPACE 自己点名的 future work。**

- **Memory-R2** (2605.21768)：LoGo-GRPO，credit 的是 **memory 操作**（不是任务动作），符号化 memory 空间，只在对话/QA（LoCoMo）。future work：扩展到 web 导航等交互环境。

→ **gap 明确**：软相似度 × 任务动作 credit × **压缩 memory 空间** × 开放环境 —— 两个竞品各自缺一块，且都写进 future work。

### 2.3 隐空间/bisimulation 线（度量的理论支撑）
DBC (ICLR'21, 2006.10742)、DeepMDP (ICML'19)、Markov State Abstractions (NeurIPS'21)、MICo/bisimulation 度量 (Castro'21)、CBM (2302.12003)。

### 2.4 研究历程与两个创新点

**一路是怎么走到这的：**

- **最初的想法①——软分组**：verl-agent 里的 group-based 方法（GiGPO 一系）都靠**精确匹配**分组——观测一字不差才算同一组。我最初的改进是改成**按相似度软分组**。→ 查证后发现 **2026-06 的 BiPACE 已经做了这个方向**（隐状态 cosine 软聚类）。这一层被占了。
- **转向想法②——压缩 memory 空间的分组与聚合**：再核实 HGPO / BiPACE 的 future work，发现它们**亲手把下一层空位留了出来**：
  - HGPO（空位一）：历史一旦被摘要进 memory，层级分组就 intractable——*"…it is necessary to explore other ways for hierarchical grouping, e.g., the embedding similarity of the memory."*（用 memory 的 embedding 相似度分组）。
  - HGPO（空位二）：*"…develop a better adaptive weighting scheme for advantage aggregation from hierarchical groups by considering the uncertainty of the advantage estimate in each hierarchical group."*（考虑不确定性的自适应聚合权重）。
  - BiPACE：*"…extending the bisimulation-guided grouping to agents that compress history into a memory module (where direct observation hashing is intractable)…"*
- **我的切入**：解决"历史压缩进 memory 后，如何在压缩表示空间做分组信用分配"。
- **粗略实验的意外转折**：我本以为难点是"压缩空间难分组"；但实验显示——**只要 memory 压得准（忠实），分组反而更准**（可分性 0.52→0.92），是**粗糙压缩**才失效。→ 真正的枢纽是**压缩保真度**；定位从"造新度量"精修为"**忠实压缩 + 对不完美 memory 鲁棒的分组信用分配**"。

**两个创新点（占的都是同一团队 Bo An / Lang Feng 自己留的 future work）：**

| # | 创新点 | 来源空位 | 定位 |
|---|---|---|---|
| ① | 压缩历史场景下、按 **memory embedding 相似度的软分组信用分配** | BiPACE + HGPO 空位一 | 核心 |
| ② | **不确定性/保真度感知的自适应聚合**，与①结合 | HGPO 空位二 | 可结合、待验证 |



---

## 3. 数据驱动的回答

> "为什么一定得压缩 memory？memory 在这里是否是瓶颈？别人没压缩也成功了。"

**(i) 代码证据**：核对 verl-agent 源码，**默认 `SimpleMemory` 根本不压缩**——只取最近 K 步原始拼接（即 HGPO 的"原始 K-window"）。其 `memory/README.md` 明说这只是"最简单的起点"，把 dynamic summarization 列为鼓励开发者自行扩展的方向。→ **连这个主流 agentic-RL 框架都不默认压缩**，"必须压缩"确实是需要论证的假设，而非既定事实。

**(ii) 诊断实验**（下节）：**压缩本身不是瓶颈——memory 表示的保真度才是。**

---

## 4. 诊断实验（粗略版）

### 4.1 目的
在RL 训练前，先粗步判断：BiPACE 式表示分组在"观测唯一 + 压缩"下的效果。

### 4.2 设计
两旋钮：**U = 观测唯一度**（注入干扰使每步观测都不同）× **C = 压缩率**（把区分性细节抹掉）。指标用**阈值无关的可分性 AUC**（cos 能否预测"是否同一处境"）为主，辅以 CA-Acc（关键步信用符号正确率）。合成 web-agent 处境（加购物车/点搜索结果/填表单…），Qwen2.5-7B-Instruct 层-8 表示 + 朴素 cosine。

**关于"跟谁比、公不公平"**：baseline 用的表示（层-8 归一化隐状态）和度量（**朴素 cosine**）**与 BiPACE 完全一致**（BiPACE 就是 cosine、不是学习度量）——所以这是对 BiPACE 分组**忠实**的代理，不是"把 baseline 弄瘸"。设计成**受控三方对比**：`A1 原始(不压) / A2 粗糙压缩(真7B) / A3 忠实压缩(oracle上界)`，表示与度量三臂完全相同，只换输入。唯一的代理是：用可分性 AUC 代替"贪心聚类+算信用"（绝对效果留给层2用真 BiPACE 训练确认）。

### 4.3 三方对比的发现（层1，诊断层）
```
A1 原始/不压     AUC ≈ 0.52  （≈瞎猜，BiPACE式表示在开放环境坍缩）
A2 粗糙压缩(真7B) AUC ≈ 0.56  （随手压＝白压）
A3 忠实压缩(oracle) AUC ≈ 0.92 （用真值标出的上界）
```
1. **原始观测上分组坍缩（0.52）** → BiPACE 针对的开放环境难题**真实存在**。
2. **"压缩比不压好"成立、但有前提**：只有忠实压缩赢（A3），naive 白压（A2≈A1）。**A2→A3 的 0.56→0.92 鸿沟＝方法要吃的空间**。
3. **真摘要器会失败**：7B 零样本摘要经常把干扰当主体、丢任务核心（如"填邮箱"被摘成"展示促销推荐"），同处境不同变体还互相矛盾。

⚠️ **一敏感性**：A3 的 0.92 用的是 **mean 池化**；若换成 **BiPACE 忠实的 last-token（末 token）**，短压缩文本会坍缩成一个簇（cluster_probe 证实）。两种读法都成立、但含义不同——(i) BiPACE 自己的末-token 表示在压缩 memory 上**失效**（正是它 future work 的难点）；(ii) 换个池化又能把信息捞回（说明信息**没丢**、是表示看不见）。**所以合成层只能声称"方向 + 有空间"，压缩的确切增益必须由 M1（真 memory 表示）来定**。

### 4.4 结论
合成 embedding 探针**测不准**这个问题——先后做了7 个测量假象（池化方式、prompt 包裹、处境难度、奖励泄漏、短文本表示坍缩、理想 vs 真实压缩、摘要器质量），每一个都能把结论翻面。**可信的结论只能来自真系统**（见第 6 节）。

---

## 5. 重构后的选题

原想法「在难的压缩空间里造新度量做软分组」**站不稳**——因为干净压缩空间里 plain cosine 已经很好分。修正为：

> **命题**：开放环境里，分组式信用分配**应当在 agent 的 memory/摘要表示上做，而非原始观测上**；难点在于真实 memory 是有损带噪的，需要**对不完美 memory 鲁棒的分组机制**。

两个可选抓手：
- **(a) 让 memory 更忠实**（任务感知的压缩/摘要，保住 credit 所需的区分信息）；
- **(b) 造对不完美 memory 鲁棒的分组度量**（value-aware，容忍摘要噪声）。

这正是 BiPACE / Memory-R2 各自列为 future work 的交集，且有第 2.3 节的度量理论支撑。

### 5.1 分好组之后，CA 怎么改（CA 贡献本体）

分组只是底座，**信用分配（CA）才是本课题**。候选改进（可组合）：

1. **软成员优势**：不硬分组，用 memory 相似度做**核加权**——V̂(s)=Σⱼ wⱼRⱼ、Q̂(s,a)=同动作邻居的加权回报，wⱼ=memory 相似度。（BiPACE 软化的自然延伸）
2. **不确定性/保真度加权聚合（吃下 HGPO 空位二）**：每个邻居的贡献再乘一个"可信度"——邻居太少 / 回报方差大 / **该步 memory 压缩不可靠（表示坍缩、可分性低）→ 降权**。这能**防止从"压糊的、模糊的" memory 状态泄漏错误信用**。
3. **动作反事实（借 BiPACE 的 PACE）**：软组内按执行动作拆分算 Q̂−V̂，隔离动作专属信用。
4. **多粒度融合**：episode 级(GRPO) + memory 组级(本方法) + 动作级(PACE)，用不确定性自适应权重融合——等于把 HGPO 的层级从"历史步数"迁到"**memory 相似度空间**"。

**候选方法名**：*Memory-space Uncertainty-aware Soft Credit Assignment*——在压缩 memory 表示上按相似度软分组，按**压缩保真度 / 估计不确定性**自适应加权聚合优势。**一举吃下 BiPACE 与 HGPO 各自的 future work 空位。**

**方法流程（每一步对应一个被点名的 future work 空位）：**

```
观测 oₜ   （开放环境：每步近乎唯一 → 原始精确分组会坍缩）
   │
   ▼
┌────────────────────────────────┐
│ 步1  压缩进 memory              │  忠实/任务感知压缩，保住 credit 相关信息
│      mₜ = Compress(h₁..ₜ)       │  ◀ BiPACE 空位：压缩历史后如何分组
└────────────────────────────────┘
   │  取隐状态表示 φ(mₜ)
   ▼
┌────────────────────────────────┐
│ 步2  memory 空间相似度软分组     │  不精确匹配；邻居 j 权重 wⱼ = sim(φₜ, φⱼ)
│                                │  ◀ HGPO 空位一：memory embedding 相似度分组
└────────────────────────────────┘
   │
   ▼
┌────────────────────────────────┐
│ 步3  保真度 / 不确定性 加权      │  压糊、坍缩的状态降权 → 防信用泄漏
│      w'ⱼ = wⱼ · conf(mⱼ)        │  ◀ HGPO 空位二：不确定性自适应加权  ★你独有
└────────────────────────────────┘
   │
   ▼
┌────────────────────────────────┐
│ 步4  CA 优势（软成员+动作反事实）│  V̂(s)=Σⱼ w'ⱼRⱼ ；Q̂(s,a)=同动作邻居加权回报（借 BiPACE PACE）
│                                │  Aₜ = Q̂(s,a) − V̂(s)
└────────────────────────────────┘
   │
   ▼
group-based RL 更新（GRPO / GiGPO 家族）
```

一句话看图：**压缩(①) → 软分组(②) → 保真度加权(③) → 算 CA 优势(④)**；步1–3 分别对应 BiPACE、HGPO 的三个 future work 空位，步4 是 CA 本体。

---

## 6. 预想实验：verl-agent 真 memory 表示测法

用 agent 部署时**真实产出的 memory 表示**（不是合成摘要）来测量，取代脆弱的合成探针。四个里程碑：

- **M0 定位 memory（已完成）**：核对 verl-agent 源码——memory 怎么建（默认 `SimpleMemory` = 原始 K-window，不压缩）、在哪进 LLM（`env_manager.build_text_obs`）、真值状态标签用 `anchor`（去掉历史的当前观测；两步 anchor 相同 = 同一底层状态）。
- **M1 抽表示**：ALFWorld/WebShop 跑少量 rollout（仅推理、4bit、3060 可跑），逐步 dump `(memory 表示, anchor 真值状态, 动作, 回报)`。
- **M2 复用诊断**：把已写好的 sep-AUC / CA-Acc 套到**真** memory 表示上（含 BiPACE 忠实 last-token vs 池化对照、保真度加权消融）。
- **M3 判死活**：真 memory 上可分性高 → 修正后的故事稳、可进 RL 小实验；低 → 抓手 (a)/(b) 有更强动机。

---

## 7. 后期 RL：训练数据与环境（4×A100）

**澄清**：agentic RL 的"数据"= 交互**环境 + 任务 split**（靠 rollout 在线采样训练，不是静态数据集）。

- **必须（与 baseline 对齐、可直接比）**：**ALFWorld**、**WebShop**（GiGPO/HGPO/BiPACE 都报），**TextCraft**（BiPACE 用）——verl-agent 已集成。
- **最贴合"观测唯一 + 长历史"故事**：**WebShop / WebArena**（网页观测天然每步唯一、历史长 → memory 关键）。
- **可选更长程/更重**：AppWorld、ScienceWorld、SWE-Gym。
- **轻量跑通 pipeline**：MiniWoB++ / BabyAI。
- **冷启动**：常配少量**专家轨迹**做 BC 预热（HGPO 用 Qwen2.5-1.5B/7B + warmup）。

**算力路线**：4×A100 先训 **Qwen2.5-1.5B**（对齐 HGPO 的 1.5B 设置、最省），跑通再上 7B；环境先 ALFWorld + WebShop，再加 TextCraft 对比 BiPACE。

---

## 8. 风险与缓解（按严重度排序）

| # | 风险 | 说明 / 触发信号 | 缓解 / 应对 |
|---|---|---|---|
| ① | **方法真有效性存疑** | "保真度/不确定性加权"是假设，消融可能显示无增益。 | **早验证、便宜地 fail**：M1 + 降权消融先跑；正/负结果都指导方向，负结果本身可作诊断类贡献。 |
| ② | **合成 / 测量不可信** | 已踩 7 个测量假象；结论对表示/池化敏感（0.92↔0.5 会翻）。 | 合成只作 landscape 勘探、**不进论文当证据**；论文证据来自 verl-agent 真 memory（M1）+ RL 层真 BiPACE。 |
| ③ | **算力 / RL 不稳** | 4×A100 需租用；开放环境 RL 昂贵、易不稳、可能跑不完。 | 分级：先 1.5B、先短程环境；配 warmup 冷启动；主实验做不完则退守"诊断+基准+轻方法"。 |
| ④ | **工程集成复杂** | 需在 verl-agent 里接：忠实压缩 memory 模块 + 抽表示 + 软分组/加权进 RL loop。 | 复用 verl-agent 现成 memory/环境接口（M0 已定位挂载点）；先做**离线诊断**（不改训练环）把风险前移。 |

