# walkthrough.py —— 拿具体数据把【整体流程】端到端走一遍
# ─────────────────────────────────────────────────────────────────────────
# 对应 memory压缩.md §2.1 的两个阶段:
#   阶段A (rollout 内, 每步):  观测 → 价值探针 → 跳幅 → 预算重排 → memory + conf
#   阶段B (轨迹结束后, 批量):  编码 → 软分组 → conf 加权 → 优势 → (喂 GRPO)
#
# 用 4 条"加热番茄"轨迹(2成2败), 全部数字都是代码算出来的, 不是手写的.
# 探针用 ScriptedProbe 模拟(本机无 GPU); 换成真 ValueProbe 后主流程一行不用改.
# 编码用字符 bigram 词袋模拟(无需模型); 上真系统时换成探针的隐状态.
#
# 运行: python walkthrough.py
# ─────────────────────────────────────────────────────────────────────────

import numpy as np
from vtree_compressor import VTreeCompressor, ScriptedProbe

np.set_printoptions(precision=3, suppress=True)
BUDGET = 34          # 历史 memory 的 token 预算(此处 1 字 = 1 token)
KEY_OBS = "手拿生的番茄，走到微波炉前"      # 我们要考察的"关键处境"


class CharTok:
    """假分词器: 1 字 = 1 token. 真系统换成 HF tokenizer."""
    def __call__(self, t, add_special_tokens=False):
        return type("E", (), {"input_ids": list(t)})()


# ══════════════════════════════════════════════════════════════════════
# 数据: 4 条轨迹, 都经过"关键处境", 但后续动作/结局不同
# ══════════════════════════════════════════════════════════════════════
TRAJS = [
    # ── #1 成功: 干脆利落 ──
    dict(name="#1 成功", ret=1.0, action="放进微波炉",
         steps=["你在厨房里四处找番茄", "在料理台找到并拿起了生的番茄", KEY_OBS,
                "打开微波炉，把生的番茄放了进去", "微波炉加热完成，番茄已经是热的",
                "手拿热的番茄，走向桌子", "把热的番茄放到桌上，任务完成"],
         vals=[0.35, 0.55, 0.60, 0.85, 0.92, 0.93, 0.99]),
    # ── #2 成功: 略有波折 ──
    dict(name="#2 成功", ret=1.0, action="放进微波炉",
         steps=["你在厨房里四处找番茄", "在料理台找到并拿起了生的番茄", KEY_OBS,
                "打开微波炉，把生的番茄放了进去", "微波炉加热完成，番茄已经是热的",
                "手拿热的番茄，走向桌子", "把热的番茄放到桌上，任务完成"],
         vals=[0.33, 0.52, 0.58, 0.83, 0.90, 0.94, 0.98]),
    # ── #3 失败: 在关键处境走开了 ──
    dict(name="#3 失败", ret=0.0, action="走开拿盘子",
         steps=["你在厨房里四处找番茄", "在料理台找到并拿起了生的番茄", KEY_OBS,
                "放下番茄，转身去拿盘子", "手上拿着盘子，番茄还在台上",
                "又走回微波炉前，但手里没有番茄", "时间到，任务失败"],
         vals=[0.35, 0.55, 0.60, 0.30, 0.25, 0.20, 0.10]),
    # ── #4 失败: 动作对, 但全程价值判断模糊(探针一直拿不准) ──
    #    → 所有跳幅都小 → memory 被压得最狠 → conf 最低
    dict(name="#4 失败", ret=0.0, action="放进微波炉",
         steps=["你在厨房里四处找番茄", "在料理台找到并拿起了生的番茄", KEY_OBS,
                "打开微波炉，把生的番茄放了进去", "开门时番茄掉到了地上",
                "捡起番茄，但已经脏了", "把脏番茄放到桌上，任务失败"],
         vals=[0.50, 0.52, 0.54, 0.56, 0.55, 0.53, 0.51]),
]


def embed(texts):
    """字符 bigram 词袋 + L2 归一化. 模拟 φ(memory); 真系统用探针的隐状态."""
    vocab = sorted({t[i:i + 2] for t in texts for i in range(len(t) - 1)})
    idx = {g: i for i, g in enumerate(vocab)}
    M = np.zeros((len(texts), len(vocab)))
    for r, t in enumerate(texts):
        for i in range(len(t) - 1):
            M[r, idx[t[i:i + 2]]] += 1
    return M / np.linalg.norm(M, axis=1, keepdims=True).clip(1e-9)


# ══════════════════════════════════════════════════════════════════════
# 阶段 A · rollout 内: 每步 探针 → 跳幅 → 预算重排 → memory + conf
# ══════════════════════════════════════════════════════════════════════
print("=" * 78)
print("阶段 A · rollout 内 (每走一步做一遍)".center(70))
print("=" * 78)

records = []          # 收集"关键处境那一步"的 (memory, conf, 动作, 回报)
for tr in TRAJS:
    c = VTreeCompressor(ScriptedProbe(tr["vals"]), CharTok(), budget=BUDGET, tiers=3)
    print(f"\n【{tr['name']}】 终局回报 R={tr['ret']:.0f}")
    print(f"  {'t':>2} {'观测':<24} {'V_t':>5} {'Δ_t':>6}")
    key_t = None
    for t, obs in enumerate(tr["steps"]):
        c.push(obs)                                   # ← 每步 1 次探针调用
        if obs == KEY_OBS:
            key_t = t
        star = " ★关键处境" if obs == KEY_OBS else (" ←跳幅最大" if t == int(np.argmax(c.jumps)) and t == len(tr["steps"]) - 1 else "")
        print(f"  {t:>2} {obs:<24} {c.values[t]:>5.2f} {c.jumps[t]:>6.2f}{star}")

    top = int(np.argmax(c.jumps))
    print(f"  → 跳幅最大的是第 {top} 步 (Δ={c.jumps[top]:.2f}): 「{tr['steps'][top]}」")

    # 回到"关键处境"那一刻, 看当时 agent 手里的 memory 长什么样
    c2 = VTreeCompressor(ScriptedProbe(tr["vals"]), CharTok(), budget=BUDGET, tiers=3)
    for obs in tr["steps"][:key_t + 1]:
        c2.push(obs)
    mem, conf = c2.render()
    fid = c2.fidelity()
    print(f"  ── 第 {key_t} 步(关键处境)时, agent 看到的 memory ──")
    for line in mem.split("\n"):
        print(f"     │ {line}")
    print(f"     档位={''.join({1.0:'F',0.6:'S',0.3:'M'}[x] for x in conf)}  "
          f"整段保真度 conf={fid:.2f}  memory={len(mem)}字")
    records.append(dict(name=tr["name"], mem=mem, conf=fid,
                        action=tr["action"], ret=tr["ret"]))

# ══════════════════════════════════════════════════════════════════════
# 阶段 B · 轨迹结束后: 编码 → 软分组 → conf 加权 → 优势
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("阶段 B · 轨迹结束后 (在'关键处境'这一组上算信用)".center(66))
print("=" * 78)

names = [r["name"] for r in records]
conf = np.array([r["conf"] for r in records])
rets = np.array([r["ret"] for r in records])
acts = np.array([r["action"] for r in records])

print(f"\n  {'轨迹':<8} {'该步动作':<12} {'回报':>4} {'conf':>6}")
for n, a, R, cf in zip(names, acts, rets, conf):
    print(f"  {n:<8} {a:<12} {R:>4.0f} {cf:>6.2f}")

# ⑥ 编码
phi = embed([r["mem"] for r in records])
S = phi @ phi.T
print("\n  ⑥ 编码后两两余弦相似度:")
print("     " + "        ".join(names))
for i, n in enumerate(names):
    print(f"  {n}  " + "  ".join(f"{v:.3f}" for v in S[i]))

# ⑦ 软分组: 核加权
TAU = 0.05
W = np.exp((S - 1.0) / TAU)
np.fill_diagonal(W, 0.0)
W = W / W.sum(1, keepdims=True).clip(1e-9)
print(f"\n  ⑦ 软分组权重 w (核宽 τ={TAU}, 已行归一, 对角=0):")
for i, n in enumerate(names):
    print(f"  {n}  " + "  ".join(f"{v:.3f}" for v in W[i]))


def advantages(W, conf, rets, acts, use_conf, use_pace=True):
    Wx = W * conf[None, :] if use_conf else W.copy()        # ⑧ conf 加权
    Wx = Wx / Wx.sum(1, keepdims=True).clip(1e-9)
    V = Wx @ rets                                            # ⑨ V̂(s)
    if not use_pace:
        return rets - V, V, None
    Q = np.zeros_like(V)
    for i in range(len(V)):
        m = (acts == acts[i]).astype(float); m[i] = 0.0      # 同动作邻居
        Q[i] = (Wx[i] * m) @ rets / max((Wx[i] * m).sum(), 1e-9) if m.sum() else rets[i]
    return Q - V, V, Q


A_off, V_off, Q_off = advantages(W, conf, rets, acts, use_conf=False)
A_on, V_on, Q_on = advantages(W, conf, rets, acts, use_conf=True)

print("\n  ⑧⑨ 优势 A = Q̂(s,a) − V̂(s):")
print(f"  {'轨迹':<8} {'动作':<12} | {'V̂':>6} {'Q̂':>6} {'A':>7}  (不用conf) | "
      f"{'V̂':>6} {'Q̂':>6} {'A':>7}  (用conf)")
for i, n in enumerate(names):
    print(f"  {n:<8} {acts[i]:<12} | {V_off[i]:>6.3f} {Q_off[i]:>6.3f} {A_off[i]:>+7.3f} "
          f"           | {V_on[i]:>6.3f} {Q_on[i]:>6.3f} {A_on[i]:>+7.3f}")

print("\n" + "─" * 78)
good = [i for i, a in enumerate(acts) if a == "放进微波炉"]
bad = [i for i, a in enumerate(acts) if a != "放进微波炉"]
print(f"  「放进微波炉」平均优势: 不用conf {A_off[good].mean():+.3f} → 用conf {A_on[good].mean():+.3f}")
print(f"  「走开拿盘子」优势:     不用conf {A_off[bad].mean():+.3f} → 用conf {A_on[bad].mean():+.3f}")
gap_off = A_off[good].mean() - A_off[bad].mean()
gap_on = A_on[good].mean() - A_on[bad].mean()
print(f"  正确动作与错误动作的差距: {gap_off:+.3f} → {gap_on:+.3f}  "
      f"({'拉大 ✓' if gap_on > gap_off else '缩小 ✗'})")
print("─" * 78)
