# plot_tmaze.py —— 把 tmaze_text_test.py 的实测结果画成论文图(数据写死, 本地秒出, 不用重跑模型)
# ─────────────────────────────────────────────────────────────────────────
# 数据来源: Qwen2.5-7B-Instruct(4bit), N=100/seed, 3 seeds, τ_jump=0.25
# 相比脚本自带的两张折线图, 这里做了三处改进:
#   ① x 轴改 log2 —— 采样点 1,2,4,8,16 等间距, 断崖位置(L=2 / L=8)一眼对齐
#   ② token 图上【直接标注准确率】—— 不用来回对照两张图就能看出"trunc-8 又贵又错"
#   ③ 加第三张【帕累托图】—— (token, 准确率) 平面, vtree 落在左上角
# 运行: python plot_tmaze.py
# ─────────────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt

LS = np.array([1, 2, 4, 8, 16])

# ── 实测数据 (mean, std over 3 seeds) ─────────────────────────────────
ACC = {
    "no_mem":    ([0.51, 0.45, 0.51, 0.51, 0.52], [0.02, 0.02, 0.04, 0.03, 0.06]),
    "full":      ([1.00, 1.00, 1.00, 1.00, 1.00], [0.00] * 5),
    "vtree_val": ([1.00, 1.00, 1.00, 1.00, 1.00], [0.00] * 5),
    "vtree_bel": ([1.00, 1.00, 1.00, 1.00, 1.00], [0.00] * 5),
    "trunc2":    ([1.00, 0.45, 0.51, 0.51, 0.52], [0.00, 0.02, 0.04, 0.03, 0.06]),
    "trunc8":    ([1.00, 1.00, 1.00, 0.51, 0.52], [0.00, 0.00, 0.00, 0.03, 0.06]),
}
TOK = {
    "no_mem":    [19, 19, 19, 19, 19],
    "full":      [40, 57, 91, 159, 301],
    "vtree_val": [40, 40, 40, 57, 57],
    "vtree_bel": [40, 40, 40, 40, 40],
    "trunc2":    [40, 36, 36, 36, 37],
    "trunc8":    [40, 57, 91, 138, 144],
}

LBL = {"no_mem": "no memory", "full": "full history", "vtree_val": "vtree (value)",
       "vtree_bel": "vtree (belief)", "trunc2": "trunc-2", "trunc8": "trunc-8"}
STY = {"no_mem": ("o:", "gray"), "full": ("s-", "tab:green"),
       "vtree_val": ("D-", "tab:blue"), "vtree_bel": ("d-", "tab:cyan"),
       "trunc2": ("^--", "tab:red"), "trunc8": ("v--", "tab:orange")}
ARMS = list(LBL)

fig = plt.figure(figsize=(15, 4.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.85], wspace=0.28)

# ══ (a) 准确率 ════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[0])
for a in ARMS:
    fmt, c = STY[a]
    m, s = ACC[a]
    ax.errorbar(LS, m, yerr=s, fmt=fmt, color=c, label=LBL[a], capsize=3, ms=6)
ax.axhline(0.5, ls=":", c="k", alpha=.4)
ax.text(1.05, 0.515, "chance", fontsize=8, color="k", alpha=.6)
for L, txt in ((2, "trunc-2\nbreaks"), (8, "trunc-8\nbreaks")):
    ax.axvline(L, ls=":", c="gray", alpha=.35)
    ax.annotate(txt, xy=(L, 0.66), fontsize=8, ha="center", color="dimgray")
ax.set_xscale("log", base=2); ax.set_xticks(LS); ax.set_xticklabels(LS)
ax.set_xlabel("corridor length $L$  (= memory span)")
ax.set_ylabel("decision accuracy"); ax.set_ylim(0.38, 1.06)
ax.set_title("(a) truncation breaks at $L>k$, compression does not")
ax.legend(fontsize=7.5, loc="lower left", framealpha=.9, ncol=2)

# ══ (b) token, 末端直接标准确率 ════════════════════════════════════════
ax = fig.add_subplot(gs[1])
for a in ARMS:
    fmt, c = STY[a]
    ax.plot(LS, TOK[a], fmt, color=c, label=LBL[a], ms=6)
    # 只标关键 4 条; no_mem/trunc-2 与 trunc-8 同为 acc≈0.52, 不再重复标注(避免堆叠)
    dy = {"vtree_bel": -11, "vtree_val": 11, "trunc8": 0, "full": 0}.get(a)
    if dy is not None:
        ax.annotate(f"{TOK[a][-1]}tk  acc={ACC[a][0][-1]:.2f}", xy=(16, TOK[a][-1]),
                    xytext=(7, dy), textcoords="offset points", fontsize=7.5,
                    color=c, va="center",
                    fontweight="bold" if a.startswith("vtree") else "normal")
ax.set_xscale("log", base=2); ax.set_xticks(LS); ax.set_xticklabels(LS)
ax.set_xlim(0.9, 46)
ax.set_xlabel("corridor length $L$  (= memory span)")
ax.set_ylabel("memory tokens")
ax.set_title("(b) full history blows up; vtree stays flat")
ax.legend(fontsize=8, loc="upper left", framealpha=.9)

# ══ (c) 帕累托: (token, 准确率) ═══════════════════════════════════════
ax = fig.add_subplot(gs[2])
for a in ARMS:
    _, c = STY[a]
    x, y = TOK[a][-1], ACC[a][0][-1]
    star = a.startswith("vtree")
    ax.scatter(x, y, s=190 if star else 90, c=c, marker="*" if star else "o",
               edgecolors="k", linewidths=.8, zorder=3)
    off = {"no_mem": (0, -17), "trunc2": (2, -30), "trunc8": (0, 11),
           "full": (-6, -18), "vtree_bel": (-30, 13), "vtree_val": (34, -14)}[a]
    ax.annotate(LBL[a], xy=(x, y), xytext=off, textcoords="offset points",
                fontsize=8, ha="center",
                fontweight="bold" if a.startswith("vtree") else "normal")
ax.annotate("", xy=(55, 1.005), xytext=(295, 1.005),
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.6))
ax.text(175, 1.030, "7.5$\\times$ fewer tokens,\nsame accuracy", fontsize=8.5,
        color="tab:blue", ha="center")
ax.annotate("", xy=(45, 0.97), xytext=(45, 0.57),
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.6))
ax.text(52, 0.75, "better", fontsize=8.5, color="tab:blue", rotation=90, va="center")
ax.set_xlabel("memory tokens  ($L=16$)"); ax.set_ylabel("decision accuracy")
ax.set_ylim(0.36, 1.13); ax.set_xlim(0, 330)
ax.set_title("(c) cost–accuracy frontier at $L=16$")
ax.grid(alpha=.25)

plt.savefig("tmaze_result.png", dpi=150, bbox_inches="tight")
print("已存 tmaze_result.png")
print("\n关键读数 (L=16):")
for a in ARMS:
    print(f"  {LBL[a]:<16} {TOK[a][-1]:>4} tk   acc={ACC[a][0][-1]:.2f}")
print("\n  → trunc-8 花了 vtree(belief) 的 3.6 倍 token, 准确率却只有随机水平.")
print("  → vtree 位于左上角: 同样 100% 准确, token 只要 full 的 1/7.5.")
