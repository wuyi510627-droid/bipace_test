# credit.py —— 软分组 + 信用分配（落地实现_详细版 §2.5 / §2.6）
# ═════════════════════════════════════════════════════════════════════════
# 这个模块是【阶段 2】的核心：拿压缩后的 memory 表示，把一条轨迹的总回报
# 摊到每一步头上。
#
# 为什么单独抽成模块（而不是继续写在各个测试脚本里）：
#   credit_rank_test.py 里已经内联了一份 advantages()，ca_ablation_test 又要一份，
#   接进 verl-agent 还要第三份 —— 三份各自演化必然分叉。这里是【唯一权威实现】，
#   数值上与 credit_rank_test.py 的内联版等价（见文件末尾自测场景 5）。
#
# 两个开关就是实验⑤的四条臂，消融不用改代码：
#   use_conf=F, use_pace=F  →  a 臂: 纯软分组（V̂ 用全部邻居, A = R − V̂）
#   use_conf=F, use_pace=T  →  b 臂: + 动作反事实（A = Q̂ − V̂）
#   use_conf=T, use_pace=T  →  c 臂: + 保真度加权（完整版）
#   use_conf=T, use_pace=F  →  d 臂: 只加保真度
#
# ⚠️ 已知隐忧：use_conf 在 walkthrough 里的增量是 1.438 → 1.439（≈ 没用）。
#    实验⑤的第一要务就是确认这一点 —— 若坐实没用，这个开关就该从方案里删掉。
#
# 自测（无需 GPU / 模型）：python credit.py
# ═════════════════════════════════════════════════════════════════════════

import numpy as np

__all__ = ["SoftGrouper", "CreditAssigner", "ARM_SWITCHES", "ARM_NAMES",
           "credit_metrics", "effective_neighbors", "grouping_quality"]


# ══════════════════════════════════════════════════════════════════════
# 软分组
# ══════════════════════════════════════════════════════════════════════
class SoftGrouper:
    """按 memory 表示的相似度给邻居打权重, 不做硬分组.

    为什么不能沿用 GiGPO 的精确哈希分组:
      哈希要求状态字符串完全一致. 压缩是有损的, 同一个处境在两条轨迹里可能被
      压成不同文本 ⇒ 哈希对不上 ⇒ 组永远只有自己一个成员 ⇒ 分组失效.

    ⚠️ 编码 phi 时【必须用 last-token 池化】, 不能用 mean:
       实测(sep_auc_test) 同/不同处境的相似度差距 mean=+0.013 / last=+0.401.
       核加权 exp((S-1)/tau) 吃的是【绝对差距】, 0.013 下权重几乎均匀 ⇒ 软分组等于没做.
    """

    def __init__(self, tau: float = 0.05):
        self.tau = tau

    def weights(self, phi: np.ndarray, tau: float | None = None,
                block: np.ndarray | None = None) -> np.ndarray:
        """phi: (N, d) 已 L2 归一化 → 返回 (N, N) 行归一化的权重矩阵.

        block: 可选的 (N,) 轨迹 id. 给了就【屏蔽同轨迹内的邻居】——
               同一条轨迹里的步共享同一个回报 R, 拿它们互相当对照等于自己比自己,
               V̂ 会被自身回报污染. 跨轨迹对比才是信用信号的来源.
        """
        tau = self.tau if tau is None else tau
        S = phi @ phi.T                                   # 已归一化 ⇒ 就是余弦相似度
        W = np.exp((S - 1.0) / tau)                       # 越像权重越大, 指数衰减
        np.fill_diagonal(W, 0.0)                          # 不把自己算进去
        if block is not None:
            W = W * (block[:, None] != block[None, :])    # 屏蔽同轨迹
        return W / W.sum(1, keepdims=True).clip(1e-9)


# ══════════════════════════════════════════════════════════════════════
# 信用分配
# ══════════════════════════════════════════════════════════════════════
class CreditAssigner:
    """优势 A = Q̂(s,a) − V̂(s), 两者都用软分组的邻居加权平均估出来.

      V̂(s)   = Σ_j w_ij · R_j              这个处境【一般】能拿多少分
      Q̂(s,a) = Σ_{j: a_j=a_i} w_ij · R_j   这个处境下【做了这个动作】能拿多少分
      A      = Q̂ − V̂                       做这件事比平均水平好多少
    """

    def advantages(self, W: np.ndarray, conf: np.ndarray, returns: np.ndarray,
                   actions: np.ndarray, use_conf: bool = True,
                   use_pace: bool = True) -> np.ndarray:
        W = np.asarray(W, dtype=float).copy()
        returns = np.asarray(returns, dtype=float)
        actions = np.asarray(actions)

        if use_conf:                                      # ② 压得粗的邻居降权
            c = np.asarray(conf, dtype=float)
            W = W * c[None, :]                            # 列方向: 被参考者的可信度
            W = W / W.sum(1, keepdims=True).clip(1e-9)

        V = W @ returns                                   # V̂(s)

        if not use_pace:
            return returns - V                            # 退化: 用自己的回报当 Q̂

        Q = np.zeros_like(V)                              # ③ 同动作邻居 → Q̂(s,a)
        for i in range(len(V)):
            m = (actions == actions[i]).astype(float)
            m[i] = 0.0
            wm = W[i] * m
            Q[i] = wm @ returns / wm.sum() if wm.sum() > 1e-9 else returns[i]
        return Q - V


# 四条臂的开关表 —— 实验⑤直接遍历这个
ARM_SWITCHES = {
    "a_soft":        dict(use_conf=False, use_pace=False),   # 纯软分组
    "b_pace":        dict(use_conf=False, use_pace=True),    # + 动作反事实
    "c_full":        dict(use_conf=True,  use_pace=True),    # 完整版
    "d_conf_only":   dict(use_conf=True,  use_pace=False),   # 只加保真度
}
ARM_NAMES = list(ARM_SWITCHES)


# ══════════════════════════════════════════════════════════════════════
# 指标
# ══════════════════════════════════════════════════════════════════════
def credit_metrics(A: np.ndarray, key_idx: int) -> tuple[int, float, float]:
    """A: 一条轨迹各步的优势; key_idx: 关键步下标(ground truth).

    返回 (排名, 集中度, margin):
      排名   : 关键步的 |A| 在全部步里排第几. 理想 = 1
      集中度 : |A[key]| / Σ|A|. 理想 → 1.0(所有信用都落在关键步)
      margin : |A[key]| − max(其他步的|A|). >0 才算真的分对了
    """
    a = np.abs(np.asarray(A, dtype=float))
    rank = int((a > a[key_idx]).sum()) + 1
    conc = a[key_idx] / max(a.sum(), 1e-9)
    others = np.delete(a, key_idx)
    margin = a[key_idx] - (others.max() if len(others) else 0.0)
    return rank, float(conc), float(margin)


def effective_neighbors(W: np.ndarray) -> float:
    """有效邻居数(权重分布的熵指数). 用来判 tau 合不合适:
       ≈1   → 太集中, 只听一个邻居的, 等于没做平均;
       ≈N   → 太平均, 全组等权, 等于没分组.
    """
    W = np.asarray(W, dtype=float)
    return float(np.exp(-(W * np.log(W + 1e-12)).sum(1)).mean())


def grouping_quality(phi: np.ndarray, labels: np.ndarray) -> dict:
    """分组质量诊断. labels<0 的样本(噪声步)排除在外.

    ⚠️ 判读要看【差距】那一项, 不是 AUC —— 首轮实测踩过这个坑:
       mean 池化 AUC=0.863(看着合格), 但同/不同处境相似度 0.997 vs 0.984,
       差距仅 +0.013, 核加权下权重几乎均匀 ⇒ 软分组失效.
       AUC 只看【排序】, 排序对 ≠ 拉得开.
       (身高 180.0/179.9/179.8: 排名完全正确, 但谁也说不出谁高.)
    """
    from sklearn.metrics import roc_auc_score
    keep = np.asarray(labels) >= 0
    p, k = np.asarray(phi)[keep], np.asarray(labels)[keep]
    S = p @ p.T
    iu = np.triu_indices(len(p), k=1)
    same = (k[iu[0]] == k[iu[1]]).astype(int)
    if len(same) == 0 or same.min() == same.max():
        return dict(auc=float("nan"), same=0.0, diff=0.0, gap=0.0)
    same_m, diff_m = S[iu][same == 1].mean(), S[iu][same == 0].mean()
    return dict(auc=float(roc_auc_score(same, S[iu])), same=float(same_m),
                diff=float(diff_m), gap=float(same_m - diff_m))


# ══════════════════════════════════════════════════════════════════════
# 自测（无 GPU）：构造已知答案的场景，验证实现没写错
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    rng = np.random.RandomState(0)
    ok = lambda c: "✅" if c else "❌ 失败"

    def unit(x):
        x = np.asarray(x, dtype=float)
        return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-9)

    print("=" * 72)
    print("场景 1 · 软分组能否把同处境的样本聚到一起")
    print("=" * 72)
    # 3 个处境 × 4 个样本, 同处境向量接近
    centers = unit(rng.randn(3, 16))
    phi = unit(np.repeat(centers, 4, axis=0) + 0.05 * rng.randn(12, 16))
    lab = np.repeat([0, 1, 2], 4)
    W = SoftGrouper(tau=0.05).weights(phi)
    in_group = np.array([W[i][lab == lab[i]].sum() for i in range(12)]).mean()
    q = grouping_quality(phi, lab)
    print(f"  同处境权重占比 = {in_group:.3f}  (理想→1.0)          {ok(in_group > 0.9)}")
    print(f"  相似度差距     = {q['gap']:+.3f} (>0.3 算拉得开)      {ok(q['gap'] > 0.3)}")
    print(f"  有效邻居数     = {effective_neighbors(W):.2f} / 11")

    print()
    print("=" * 72)
    print("场景 2 · 动作反事实: 同处境下好动作该拿正优势")
    print("=" * 72)
    # 一个处境重复 8 次, 动作 X 全成功、动作 Y 全失败
    phi2 = unit(np.tile(centers[0], (8, 1)) + 0.02 * rng.randn(8, 16))
    acts = np.array(["X"] * 4 + ["Y"] * 4)
    rets = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    conf = np.ones(8)
    W2 = SoftGrouper(tau=0.05).weights(phi2)
    A = CreditAssigner().advantages(W2, conf, rets, acts)
    print(f"  动作 X 的平均优势 = {A[:4].mean():+.3f}  (应为正)      {ok(A[:4].mean() > 0.2)}")
    print(f"  动作 Y 的平均优势 = {A[4:].mean():+.3f}  (应为负)      {ok(A[4:].mean() < -0.2)}")

    print()
    print("=" * 72)
    print("场景 3 · 保真度加权: 压糊了的邻居应当被降权")
    print("=" * 72)
    # 邻居 1 保真度高但回报低, 邻居 2 保真度低但回报高 → 加权后 V̂ 应偏向邻居 1
    phi3 = unit(np.tile(centers[1], (3, 1)) + 0.02 * rng.randn(3, 16))
    rets3 = np.array([0.5, 0.0, 1.0])
    conf3 = np.array([1.0, 1.0, 0.3])                  # 第 3 个压得很狠
    W3 = SoftGrouper(tau=0.05).weights(phi3)
    v_off = (W3 @ rets3)[0]
    Wc = W3 * conf3[None, :]; Wc = Wc / Wc.sum(1, keepdims=True).clip(1e-9)
    v_on = (Wc @ rets3)[0]
    print(f"  不加权 V̂ = {v_off:.3f}")
    print(f"  加权后 V̂ = {v_on:.3f}  (应更低: 高回报那个被打折)   {ok(v_on < v_off)}")
    print(f"  变化量   = {v_on - v_off:+.3f}   ← ⚠️ 这个量级就是 use_conf 的全部作用")

    print()
    print("=" * 72)
    print("场景 4 · block 屏蔽: 同轨迹的邻居不该参与对比")
    print("=" * 72)
    phi4 = unit(np.tile(centers[2], (6, 1)) + 0.02 * rng.randn(6, 16))
    blk = np.array([0, 0, 0, 1, 1, 1])
    W4 = SoftGrouper(tau=0.05).weights(phi4, block=blk)
    leak = W4[0][blk == blk[0]].sum()
    print(f"  第 0 步分给同轨迹邻居的权重 = {leak:.6f}  (应为 0)   {ok(leak < 1e-9)}")

    print()
    print("=" * 72)
    print("场景 5 · 与 credit_rank_test.py 的内联实现数值等价")
    print("=" * 72)
    phi5 = unit(rng.randn(10, 16))
    rets5 = rng.rand(10)
    acts5 = np.array(list("XYXYXYXYXY"))

    def inline_advantages(phi, rets, acts, tau=0.05):     # 复制自 credit_rank_test.py
        S = phi @ phi.T
        W = np.exp((S - 1.0) / tau); np.fill_diagonal(W, 0.0)
        W = W / W.sum(1, keepdims=True).clip(1e-9)
        V = W @ rets
        Q = np.zeros_like(V)
        for i in range(len(V)):
            m = (acts == acts[i]).astype(float); m[i] = 0.0
            Q[i] = (W[i] * m) @ rets / max((W[i] * m).sum(), 1e-9) if m.sum() else rets[i]
        return Q - V

    A_ref = inline_advantages(phi5, rets5, acts5)
    A_new = CreditAssigner().advantages(SoftGrouper(0.05).weights(phi5),
                                        np.ones(10), rets5, acts5,
                                        use_conf=False, use_pace=True)
    d = np.abs(A_ref - A_new).max()
    print(f"  最大逐元素差 = {d:.2e}   (应 < 1e-12)              {ok(d < 1e-12)}")

    print()
    print("=" * 72)
    print("场景 6 · 四条臂都能跑, 且互不相同")
    print("=" * 72)
    conf6 = rng.uniform(0.3, 1.0, 10)
    W6 = SoftGrouper(0.05).weights(phi5)
    As = {}
    for name, sw in ARM_SWITCHES.items():
        As[name] = CreditAssigner().advantages(W6, conf6, rets5, acts5, **sw)
        print(f"  {name:<12} use_conf={str(sw['use_conf']):<5} use_pace={str(sw['use_pace']):<5} "
              f"|A|均值={np.abs(As[name]).mean():.4f}")
    pairs = [(x, y) for i, x in enumerate(ARM_NAMES) for y in ARM_NAMES[i + 1:]]
    alldiff = all(np.abs(As[x] - As[y]).max() > 1e-9 for x, y in pairs)
    print(f"  四条臂两两不同                                     {ok(alldiff)}")

    print()
    print("=" * 72)
    print("场景 7 · tau 敏感性: 有效邻居数随 tau 单调上升")
    print("=" * 72)
    effs = []
    for tau in (0.01, 0.05, 0.1, 0.3, 1.0):
        e = effective_neighbors(SoftGrouper(tau).weights(phi))
        effs.append(e)
        note = "太集中" if e < 2 else ("太平均" if e > len(phi) / 3 else "合理")
        print(f"  tau={tau:<5} 有效邻居数={e:>5.2f} / {len(phi)-1}   {note}")
    print(f"  单调上升                                           "
          f"{ok(all(effs[i] <= effs[i+1] + 1e-9 for i in range(len(effs)-1)))}")

    print(f"""
{'='*72}
自测结论
{'='*72}
  场景 1-2 过 → 软分组和动作反事实的机制是对的;
  场景 3   → 注意【变化量】那一行, 它是 use_conf 能起作用的上限. 上限本身很小
             的话, 实验⑤大概率也测不出增量 —— 这与 walkthrough 的 1.438→1.439 一致;
  场景 5   → 换用本模块不会改变 credit_rank_test 已跑出的数字, 可以放心替换;
  场景 7   → 接真实数据时先跑这个, 挑有效邻居数落在【合理】区间的 tau.""")
