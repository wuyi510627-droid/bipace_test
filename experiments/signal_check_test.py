# signal_check_test.py —— 实验②：价值跳幅这个信号到底可不可信
# ═════════════════════════════════════════════════════════════════════════
# 【这是整套方法的地基】
#   压缩靠"跳幅大 = 重要"来分配 token. 如果这个前提不成立, 后面全部白搭 ——
#   实验①③④测的都是"压完之后好不好", 唯独没人验过"凭什么这么压".
#
# 【怎么验】反事实干预, 不靠任何先验假设
#   问题: "第 t 步重不重要"没有现成标签, 我不能自己指定(会变成自证).
#   办法: 把第 t 步【抹掉】, 看任务成功率掉多少 —— 掉得多 = 这步真的重要.
#          真实重要性 imp[t] = acc(完整记忆) − acc(抹掉第 t 步)
#   这是【操作性定义】: 不问"我觉得哪步重要", 而是"拿掉它会不会出事".
#
#   然后检验: 跳幅 Δ_t 能不能预测 imp[t].
#
# 【必须有对照, 否则测了等于没测】
#   三个 baseline 打分函数, 跳幅必须赢过它们才算有信息:
#     · random   —— 随机分. 赢不过它 = 跳幅是纯噪声
#     · recency  —— 越近分越高. 赢不过它 = 跳幅只是在偷偷编码位置
#     · length   —— 原文越长分越高. 赢不过它 = 跳幅只是在偷偷编码长度
#   ⚠️ 第 2、3 条尤其要紧: 排序键是【跳幅 ÷ 长度】, 若跳幅本身就与长度相关,
#      那个除法就是在做无用功甚至反向操作.
#
# 【顺带解决一个待办】扫 3 种探针措辞, 选相关性最高的固定下来(详细版 §2.1 坑3).
#
# 【--belief 模式】用 ∆Belief-RL 的口径替代 ValueProbe:
#   b_t = P(正确答案 | actor决策prompt + 前t步记忆)
#   从 actor 决策 forward pass 里直接读 —— 部署时零额外成本.
#   ∆Belief_t = |log(b_t) − log(b_{t−1})| (或绝对差).
#   验证"信念跳幅"能不能像"价值跳幅"一样预测真实重要性.
#
# 运行(要 GPU + 模型):
#   python signal_check_test.py            # 标准模式: ValueProbe 三种措辞
#   python signal_check_test.py --belief   # ∆Belief 模式: P(正确答案|记忆)
# ═════════════════════════════════════════════════════════════════════════

import argparse, random, itertools
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "/home/wuyi/cuda12-dev/project/models/Qwen2.5-7B-Instruct"
BATCH, MAXLEN = 8, 1024
N_TRAJ = 12                 # 轨迹数
L_FILL = 6                  # 填充步数
N_EVAL = 6                  # 每个干预条件下重复问几次(降低单次判断的随机性)
SEEDS = [0, 1, 2]
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 三种探针措辞（本实验的第二个目的：选一个固定下来）────────────────
PROMPTS = {
    "success": "任务:{task}\n当前记忆:\n{mem}\n问:这个任务最终能成功吗?只回答一个字:是 或 否。\n答:",
    "enough":  "任务:{task}\n当前记忆:\n{mem}\n问:根据以上记忆,现在已经掌握完成任务所需的关键信息了吗?只回答一个字:是 或 否。\n答:",
    "ontrack": "任务:{task}\n当前记忆:\n{mem}\n问:目前的进展是在正确的轨道上吗?只回答一个字:是 或 否。\n答:",
}

# ══════════════════════════════════════════════════════════════════════
# 任务：钥匙-门。三类步骤的真实重要性【天然不同】，这样排序才有得比
#   ① 告示步  : 说明哪个颜色是钥匙 —— 抹掉必崩
#   ② 拾取步  : 捡起真钥匙        —— 抹掉必崩
#   ③ 干扰步  : 捡起同色系的假货  —— 抹掉无所谓
#   ④ 填充步  : 走廊废话          —— 抹掉无所谓
# 注意不预先假定谁重要, 上面只是设计意图; 真实重要性一律由干预实测.
# ══════════════════════════════════════════════════════════════════════
COLORS = ["暗红色", "黄铜色", "深绿色", "银白色"]
FILLER = ["你走过一段空走廊，两侧什么也没有",
          "你继续往前，脚步声在走廊里回响",
          "墙上有一道旧裂缝，没什么特别",
          "头顶的灯管闪了一下"]


def gen_traj(rng):
    """返回 steps / kinds / 正确答案。kinds 仅用于事后分类统计，不参与打分。"""
    key_c, *dec = rng.sample(COLORS, 3)
    steps, kinds = [], []

    steps.append(f"【告示】今天能打开尽头那扇门的，是{key_c}的钥匙")
    kinds.append("告示")

    events = ([("拾取", f"你在地上看到一个{key_c}的小物件，把它捡了起来")]
              + [("干扰", f"你在地上看到一个{c}的小物件，把它捡了起来") for c in dec]
              + [("填充", rng.choice(FILLER)) for _ in range(L_FILL)])
    rng.shuffle(events)
    for k, t in events:
        kinds.append(k); steps.append(t)

    steps.append("你走到走廊尽头，面前是一扇锁着的门")
    kinds.append("终点")
    return dict(steps=steps, kinds=kinds, key=key_c,
                task="找到正确的钥匙并打开走廊尽头的门")


# ── 模型 ────────────────────────────────────────────────────────────────
print(f"加载 {MODEL} (4bit) ...")
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"; tok.truncation_side = "left"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                             device_map={"": 0}).eval()
YES, NO = [tok(c, add_special_tokens=False).input_ids[0] for c in ("是", "否")]
COLOR_IDS = [tok(c, add_special_tokens=False).input_ids[0] for c in COLORS]

DECIDE_P = ("你面前是一扇锁着的门。这是你一路上的记忆:\n{mem}\n"
            "问:应该用哪个颜色的钥匙开门?只回答颜色的第一个字。\n答:")


@torch.no_grad()
def _last_logits(prompts):
    out = []
    for i in range(0, len(prompts), BATCH):
        enc = tok(prompts[i:i + BATCH], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAXLEN).to(device)
        out.append(model(**enc).logits[:, -1, :].float().cpu())
    return torch.cat(out, 0)


def probe(mems, task="", key="success"):
    """价值探针 → P(是)。key 选措辞。"""
    lg = _last_logits([PROMPTS[key].format(task=task, mem=m) for m in mems])
    return torch.softmax(lg[:, [YES, NO]], -1)[:, 0].numpy()


def decide_acc(mems, key_color):
    """把 memory 交给模型做最终决策, 返回选对钥匙的概率(不采样, 直接读分布).

    用概率而非硬 argmax: 抹掉一步往往只是让模型【变得没把握】,
    硬判断会把这种连续的退化压成 0/1, 损失分辨率.
    """
    lg = _last_logits([DECIDE_P.format(mem=m) for m in mems])
    p = torch.softmax(lg[:, COLOR_IDS], -1).numpy()
    return p[:, COLORS.index(key_color)]


# ══════════════════════════════════════════════════════════════════════
# 干预：抹掉第 t 步，量真实重要性
# ══════════════════════════════════════════════════════════════════════
def true_importance(tr):
    """imp[t] = acc(完整) − acc(抹掉第 t 步)。最后一步(终点)不参与干预。"""
    steps, n = tr["steps"], len(tr["steps"])
    full = "\n".join(steps)
    variants = [full] + ["\n".join(steps[:t] + steps[t + 1:]) for t in range(n - 1)]
    accs = decide_acc(variants, tr["key"])
    return accs[0] - accs[1:], accs[0]          # (n-1,), 基线准确率


def jumps_of(tr, prompt_key):
    """在线口径的跳幅: 逐前缀问一次, Δ_t = |V_t − V_{t−1}|。"""
    steps = tr["steps"]
    V = probe(["\n".join(steps[:t + 1]) for t in range(len(steps))],
              tr["task"], prompt_key)
    prev, J = 0.5, []
    for v in V:
        J.append(abs(v - prev)); prev = v
    return np.array(J[:len(steps) - 1]), V      # 与 imp 对齐(去掉终点步)


def belief_jumps_of(tr, metric="logratio"):
    """∆Belief 口径的跳幅: b_t = P(正确答案 | 决策prompt + 前t步记忆).

    不单独问"能成吗"——从 actor 的决策 forward pass 里直接读 P(正确钥匙)。
    部署时: b_t 和动作决策共享 KV cache ⇒ 零额外成本。
    这里为测信号质量仍做独立 forward pass（Cost 和 ValueProbe 相当）。

    metric="logratio"  → ∆_t = |log(b_t) − log(b_{t−1})|   (∆Belief-RL 原文)
    metric="abs"       → ∆_t = |b_t − b_{t−1}|             (与当前 ValueProbe 对齐)
    """
    steps = tr["steps"]
    mems = ["\n".join(steps[:t + 1]) for t in range(len(steps))]
    b = decide_acc(mems, tr["key"])          # P(正确钥匙 | 前t步记忆)
    b = np.clip(b, 1e-9, 1.0)               # 防 log(0)

    if metric == "logratio":
        prev_log = np.log(0.25)              # uniform prior over 4 colors
        J = []
        for bt in b:
            log_bt = np.log(bt)
            J.append(abs(log_bt - prev_log))
            prev_log = log_bt
    else:  # abs
        prev = 0.25                          # uniform prior
        J = []
        for bt in b:
            J.append(abs(bt - prev))
            prev = bt
    return np.array(J[:len(steps) - 1]), b   # 与 imp 对齐(去掉终点步)


# ── 三个对照打分 ────────────────────────────────────────────────────────
def baseline_scores(tr, rng):
    n = len(tr["steps"]) - 1
    return {
        "random":  np.array([rng.random() for _ in range(n)]),
        "recency": np.arange(n, dtype=float),                      # 越近越高
        "length":  np.array([len(s) for s in tr["steps"][:n]], float),
    }


def evaluate(score, imp, top_frac=0.25):
    """score 预测 imp 的两个指标。AUC 的正类 = 重要性排前 top_frac 的步。"""
    k = max(1, int(round(len(imp) * top_frac)))
    y = np.zeros(len(imp), int); y[np.argsort(-imp)[:k]] = 1
    rho = spearmanr(score, imp).correlation
    auc = roc_auc_score(y, score) if 0 < y.sum() < len(y) else float("nan")
    hit = int(np.argmax(score) == np.argmax(imp))                  # top-1 命中
    return rho, auc, hit


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--belief", action="store_true",
                    help="∆Belief 模式: P(正确答案|记忆) 替代 ValueProbe 探针")
    ap.add_argument("--belief-metric", choices=["logratio", "abs"], default="logratio",
                    help="logratio=∆Belief原文公式 / abs=绝对差(与ValueProbe对齐)")
    args = ap.parse_args()

    all_res = {}          # (prompt_key or baseline) → list of (rho, auc, hit)
    by_kind = {}          # kind → [真实重要性...]
    base_accs = []

    for sd in SEEDS:
        rng = random.Random(sd)
        prng = random.Random(sd + 999)
        for _ in range(N_TRAJ):
            tr = gen_traj(rng)
            imp, base = true_importance(tr)
            base_accs.append(base)
            for k, v in zip(tr["kinds"][:len(imp)], imp):
                by_kind.setdefault(k, []).append(v)

            if args.belief:
                # ── ∆Belief 模式 ──────────────────────────────────────
                for metric in ["logratio", "abs"]:
                    J, _ = belief_jumps_of(tr, metric)
                    tag = f"∆Belief·{metric}"
                    all_res.setdefault(tag, []).append(evaluate(J, imp))
            else:
                # ── 标准 ValueProbe 模式 ──────────────────────────────
                for pk in PROMPTS:                                   # 三种措辞
                    J, _ = jumps_of(tr, pk)
                    all_res.setdefault(f"跳幅·{pk}", []).append(evaluate(J, imp))

            for name, s in baseline_scores(tr, prng).items():        # 三个对照
                all_res.setdefault(f"对照·{name}", []).append(evaluate(s, imp))

    # ── 输出 ────────────────────────────────────────────────────────────
    print(f"\n{'='*76}")
    print(f"① 干预实测: 各类步骤的真实重要性  (抹掉它, 最终决策准确率掉多少)")
    print(f"{'='*76}")
    print(f"  完整记忆下的基线准确率 = {np.mean(base_accs):.3f}\n")
    print(f"  {'步骤类型':<8} | {'样本数':>6} | {'平均重要性':>10} | {'标准差':>8}")
    print("  " + "-" * 46)
    for k in ("告示", "拾取", "干扰", "填充"):
        if k not in by_kind: continue
        v = np.array(by_kind[k])
        print(f"  {k:<8} | {len(v):>6} | {v.mean():>+10.3f} | {v.std():>8.3f}")
    print("\n  ⚠️ 先看这张表: 若【告示/拾取】与【干扰/填充】的重要性没拉开,")
    print("     说明任务本身没造出重要性差异, 下面的相关性再高也没意义 —— 先改任务.")

    print(f"\n{'='*76}")
    mode_desc = "∆Belief 口径: P(正确答案|记忆)" if args.belief else "跳幅能否预测真实重要性  (三种措辞 vs 三个对照)"
    print(f"② {mode_desc}")
    print(f"{'='*76}")
    print(f"  {'打分方式':<18} | {'Spearman':>9} | {'AUC':>7} | {'top-1命中率':>10}")
    print("  " + "-" * 54)
    summary = {}
    for name, rows in all_res.items():
        a = np.array(rows, dtype=float)
        rho, auc, hit = np.nanmean(a, 0)
        summary[name] = (rho, auc, hit)
        star = " ★" if (name.startswith("跳幅") or name.startswith("∆Belief")) else ""
        print(f"  {name:<18} | {rho:>+9.3f} | {auc:>7.3f} | {hit:>10.1%}{star}")

    if args.belief:
        # ── ∆Belief 模式判读 ─────────────────────────────────────────
        believers = [n for n in summary if n.startswith("∆Belief")]
        base_lines = [n for n in summary if n.startswith("对照")]

        print(f"\n{'='*76}")
        print("判读 · ∆Belief 信号检验")
        print(f"{'='*76}")

        if len(believers) >= 2:
            logrho = summary["∆Belief·logratio"][1]
            absrho = summary["∆Belief·abs"][1]
            best_belief = "∆Belief·logratio" if logrho > absrho else "∆Belief·abs"
            print(f"  logratio AUC = {logrho:.3f}  |  abs AUC = {absrho:.3f}")
            print(f"  更好的口径: {best_belief}  (AUC {summary[best_belief][1]:.3f})")

        # 对比 ValueProbe 的口径（如果上次跑过，从已有结果里比对）
        if "跳幅·success" in summary:
            probe_auc = summary["跳幅·success"][1]
            belief_auc = summary.get("∆Belief·logratio", (0,0,0))[1]
            gap = probe_auc - belief_auc
            ratio = belief_auc / max(probe_auc, 1e-9)
            print(f"\n  对比 ValueProbe·success AUC={probe_auc:.3f}")
            print(f"        ∆Belief·logratio  AUC={belief_auc:.3f}")
            print(f"        差距 = {gap:+.3f}  (belief/probe = {ratio:.1%})")

        best_base = max(base_lines, key=lambda n: summary[n][1])
        print(f"""
  【出口条件】∆Belief 的 AUC > 0.7, 且 ≥ 最强对照的 90%.

  本轮: 最强对照 = {best_base}  (AUC {summary[best_base][1]:.3f})

  · ∆Belief AUC > 0.7 且 ≥ 对照的 90%
        → ✅ ∆Belief 信号站得住! 可以省掉 ValueProbe 那 13.9% 的额外成本.
          论文里这样写:
          "We adopt the ∆Belief-RL (Auzina et al., ICML 2026) framework:
           instead of a separate value probe, we read the agent's own belief
           b_t = P(correct|h_t) from the decision forward pass — zero extra cost.
           The belief shift |log(b_t/b_{t-1})| provides a reliable compression
           signal, matching or exceeding the separate-probe baseline."

  · ∆Belief ≈ 随机 (<0.3)
        → ❌ P(正确答案) 在早期步没有区分度 —— agent 一开始完全蒙, 概率
          稳定在 ~0.25, 到终点附近才蹦起来. 这意味着 trail-end bias:
          跳幅大部分集中在最后几步, 前期该保留的观测可能被错压.
          解决方案: 不用 ∆Belief 做压缩信号, 退回到小模型探针(路线二).

  · ∆Belief < Probe 但 > 0.7
        → ⚠️ 信号仍在, 但不如专问"能成吗". 论文里如实报告:
          "∆Belief provides a free compression signal at > $sp metric; the
           probe-based signal adds $diff at $cost extra computation."
          让读者自己选 —— 这是最诚实的写法.""")
    else:
        # ── 标准模式判读（不变）───────────────────────────────────────
        best = max((n for n in summary if n.startswith("跳幅")),
                   key=lambda n: summary[n][1])
        worst_base = max((n for n in summary if n.startswith("对照")),
                         key=lambda n: summary[n][1])

        print(f"\n{'='*76}")
        print("判读")
        print(f"{'='*76}")
        print(f"""  【出口条件】最好的跳幅措辞 AUC > 0.8, 且明显高于全部三个对照.

  本轮: 最好的措辞 = {best}  (AUC {summary[best][1]:.3f})
        最强的对照 = {worst_base}  (AUC {summary[worst_base][1]:.3f})
        差距 = {summary[best][1] - summary[worst_base][1]:+.3f}

  · 跳幅 > 三个对照 且 AUC > 0.8
        → ✅ 信号站得住, 地基通过. 把 {best.split('·')[1]} 这版措辞【固定下来】,
          之后不再改(改了跳幅尺度会变, 已跑的实验不可比).

  · 跳幅 ≈ random
        → ❌ 跳幅是噪声. 方法的前提不成立, 要么换信号(比如用真实 reward 的 TD-error),
          要么承认这条路走不通. 【这是最坏情况, 但必须诚实面对】.

  · 跳幅 ≈ recency (且都不错)
        → ⚠️ 跳幅只是在编码"离终点多近". 那 truncate 就够了, 不需要本方法 ——
          这会直接推翻实验①的解释(①的胜出可能另有原因).

  · 跳幅 ≈ length
        → ⚠️ 跳幅在编码长度. 而 render 的排序键正是【跳幅÷长度】,
          若两者本就相关, 那个除法在做无用功, 要重新设计排序键.

  · 三种措辞差异很大
        → 信号对措辞敏感, 是个 limitation, 论文里要报告这一点.
          (差异 < 0.05 可以说"对措辞不敏感", 这是加分项)""")
