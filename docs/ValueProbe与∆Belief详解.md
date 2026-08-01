# ValueProbe 与 ∆Belief：两种把握信号的原理与成本

---

## 1. 背景：为什么需要一个"把握信号"

agent 每走一步，我们需要判断这一步对最终成败的影响有多大——跳幅 $\Delta_t$ = 把握值的变化量。$\Delta_t$ 大 → 这一步重要 → 该保留原文；$\Delta_t \approx 0$ → 流水账 → 该压掉。

**问题在于：把握值 V_t 怎么来？** 两种路线：自己单独问一次（ValueProbe），或者从 agent 决策时顺手拿（∆Belief）。两条路的成本差了一个数量级。

---

## 2. ValueProbe：单独问一次

### 做法

每走一步，构造一个探针 prompt，问模型"这任务能成吗？是/否"，但不让模型生成文字——只读最后位置的 logits，看"是"和"否"两个 token 的相对概率：

$$V_t = \frac{P(\text{是})}{P(\text{是}) + P(\text{否})}$$

### 为什么比让模型逐字生成好

| 方式 | 操作 | 耗时 | 精度 |
|---|---|---|---|
| 生成式 | 逐个 decode token 直到完整回答 | 慢一个量级 | 受采样随机性影响 |
| logits 直读 | 一次 forward，读两个 token 的 logit | 毫秒级 | 连续值，无采样噪声 |

三大好处：
1. **快**：一次 forward 不 decode，比逐字生成快一个量级
2. **干净**：只在"是/否"两个 token 之间归一化，排除其他 token 的干扰
3. **连续**：得到 0.70 而非硬的是/否，跳幅正是靠这个连续性算出来的

### V_t 是主观判断，不是真实概率

V_t 是**模型的自我判断**，不是客观的成功率。但方法只用跳幅（相对变化），不用绝对值。只要能分出"这步比那步跳得多"就够，绝对值偏不偏无所谓。

---

## 3. ValueProbe 的成本：两次 forward

### 为什么是两次

agent 决策时已经跑了一次 forward：

```
Forward ①（agent 决策）:
  memory_0 + memory_1 + ... + memory_t + "请决定下一步动作" → 采样动作
```

ValueProbe 需要再跑一次：

```
Forward ②（价值探针）:
  memory_0 + memory_1 + ... + memory_t + "能成功吗？是/否" → 读 P(是)
```

**这两次的前半段（memory 部分）一模一样。** 如果不优化，第二次 forward 把整个 memory 重新算一遍 → 成本近乎翻倍。

### KV cache 复用能省多少

KV cache 的作用：第一次 forward 里 memory 部分的 key-value 向量存下来，第二次直接复用，不重算。因此：

```
第一次: [████████ memory ████████] [决策prompt] [生成动作...]
                                     ↑ 存 KV cache

第二次: [████████ memory ████████] [探针prompt] [读 是/否]
         ↑ 直接复用，不重算           ↑ 只算这一段
```

memory 部分虽然不用重算，但探针 prompt 的 token 还是要过全部 28 层 transformer，加上读 logits 的开销。**实测下来，探针 overhead 约占 agent 完整决策耗时的 13.9%。**

### 13.9% 的构成

```
完整决策耗时 = memory 计算 + 决策 prompt 计算 + 动作 token 逐位生成
探针耗时     =                      探针 prompt 计算 + 读 yes/no logits
            ≈ 13.9% × 完整决策耗时
```

为什么不是 0%？因为探针 prompt 那几个 token（"能成功吗？是/否"）虽然少，但要逐层过全部 28 层 transformer 的 self-attention 和 FFN，无法跳过。

---

## 4. ∆Belief：从决策 forward 里顺手拿

### 核心思路

**不单独问第二次。** 在 agent 做决策的同一次 forward 里，多拼几个候选答案 token，读它们的 log-probability。

### 具体操作

1. 在决策 prompt 末尾追加上下文，比如 `"正确的答案是："`
2. 准备候选答案 token（如钥匙-门任务中 4 个颜色词："暗红色"、"黄铜色"、"深绿色"、"银白色"）
3. 同一个 forward pass 里，读这四个 token 位置的 log-probability
4. $b_t = \text{softmax}(\log P(\text{正确答案} \mid h_t))$ 

**根本没有第二次 forward。** 增加的计算量 = 多拼的几个 token（通常 3~5 个）的 transformer 计算 → 额外成本 ~1%。

### 一张图对比

```
ValueProbe（两次 forward）:
  ┌──────────────┐    ┌──────────────┐
  │ Forward ①    │    │ Forward ②    │
  │ agent 决策    │    │ 探针"能成吗"  │
  │              │    │              │
  │ memory 部分   │───→│ KV cache 复用 │   13.9% 额外
  │ + 决策 prompt │    │ + 探针 prompt │
  └──────────────┘    └──────────────┘
                            ↓
                      V_t = P(是)

∆Belief（一次 forward）:
  ┌──────────────────────────────┐
  │ Forward ①（唯一一次）         │
  │ agent 决策 + 信念一起出        │
  │                              │
  │ memory + 决策prompt           │   ~1% 额外
  │ + "答案是:" + 候选token列表    │
  └──────────────────────────────┘
           ↓                ↓
      采样动作          b_t = P(正确答案)
```

---

## 5. ∆Belief 的数学

∆Belief-RL（Auzina et al., ICML 2026）原始公式用对数比：

$$\Delta\text{Belief}_t = |\log b_t - \log b_{t-1}|$$

符号说明：

| 符号 | 含义 | 初始值 |
|---|---|---|
| $b_t$ | $P(\text{正确答案} \mid h_t)$，actor 决策 forward 里直接读 | — |
| $h_t$ | 前 t 步的压缩 memory | — |
| $b_{-1}$ | 均匀先验，4 个颜色 → 0.25 | 0.25 |
| $\Delta\text{Belief}_t$ | 信念的跳幅——这一步让 agent 对"正确答案是什么"的把握变了多少 | — |

### 为什么取对数

| 场景 | 绝对差 $\vert b_t - b_{t-1}\vert$ | 对数比 $\vert\log b_t - \log b_{t-1}\vert$ |
|---|---|---|
| 从蒙到有线索 (0.25 → 0.50) | 0.25 | **0.69** |
| 从有把握到更有把握 (0.90 → 0.95) | 0.05 | **0.05** |
| 从几乎确定到确定 (0.98 → 0.99) | 0.01 | **0.01** |

对数把"从蒙到有点把握"这段放大了——早期线索的获取才是压缩该保留的，后期微调不值钱。这正是我们需要的行为。

### 两种口径

代码里实现了两种口径（`signal_check_test.py --belief-metric`）：

| 口径 | 公式 | 适用 |
|---|---|---|
| `logratio` | $\Delta_t = \vert\log b_t - \log b_{t-1}\vert$ | ∆Belief 原文，强调早期跳幅 |
| `abs` | $\Delta_t = \vert b_t - b_{t-1}\vert$ | 与当前 ValueProbe 口径对齐，便于对比 |

---

## 6. 适用边界

| 条件 | ValueProbe | ∆Belief |
|---|---|---|
| **正确答案可枚举**（钥匙-门: 4 个颜色词） | ✅ 能用 | ✅ **近乎免费** |
| **正确答案不可枚举**（ALFWorld: 开放动作空间） | ✅ 始终可问 | ❌ 没法列候选答案 token |
| **中间步骤无"正确答案"概念** | ✅ "能成吗"始终可问 | ⚠️ $b_t$ 只在答案揭晓时跳 |
| **需要捕获中间进度信号** | ✅ 问"能成吗"能感知进展 | ❌ 只对"正确答案"敏感 |

### 为什么 ∆Belief 在钥匙-门任务上 top-1 命中 100%

钥匙-门的设置：步 0 告示说了"真钥匙是暗红色的"，步 1 捡起一样东西。agent 在决策 forward 里读到的 $P(\text{暗红色} \mid h_t)$ 只在**看到告示时从 0.25 跳到 ~0.9**，后续步几乎不变。

→ 跳幅全集中在告示步 → **最关键的那一步必然排第一**。

### ∆Belief 的天花板

P(正确) 只在"答案揭晓"的步会跳——比如看到告示、拿到钥匙。而 ValueProbe 问"能成吗"可以捕获更丰富的中间信号——比如"番茄放进微波炉了"让把握从 0.6 涨到 0.8，虽然没有"答案揭晓"，但进展是真实的。

**在开放动作空间上（如 ALFWorld 拿番茄），ValueProbe 可能更合适。** 两种都保留，真实数据上择优。

---

## 7. 论文里的写法

> "We adopt the ∆Belief-RL framework (Auzina et al., ICML 2026): instead of a separate value probe, we read the agent's own belief $b_t = P(\text{correct} \mid h_t)$ from the decision forward pass at zero extra cost. For tasks where the correct outcome is enumerable, the belief shift alone suffices as a compression signal; for open-ended tasks, we fall back to the value probe."

---

## 8. 退路方案

如果 ∆Belief 和 ValueProbe 都因为工程原因接不上（比如 actor 模型无法改 forward pass、KV cache 接口不兼容），三条退路：

| 退路 | 做法 | 额外成本 |
|---|---|---|
| **1.5B 小模型探针** | 另起一个 1.5B 模型专做探针，不跟 actor 抢显存 | ~3% |
| **agent 口头输出把握** | prompt 里加"请同时输出你对成功的把握（0-1）" | 几个 token 的生成开销 |
| **TD-error 近似** | 不用探针，直接用 Monte Carlo 回报对步数的差分 | 零，但延迟到 rollout 结束 |
