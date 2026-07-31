# ca_ablation_test.py —— 实验⑤：CA 侧四条改法的消融
# ═════════════════════════════════════════════════════════════════════════
# 【要回答的问题】压缩之后, CA 侧那四条改法各自值不值得留
#     a_soft      : 纯软分组
#     b_pace      : + 动作反事实       ← b > a ?
#     c_full      : + 保真度加权(完整) ← c > b ?  【本实验的主问题】
#     d_conf_only : 只加保真度         ← 拆开看保真度单独的贡献
#
# 【主问题为什么是 c > b】
#   walkthrough 里保真度加权的效果是 1.438 → 1.439(差 0.001, ≈没用).
#   猜测原因: 软分组已经把压得狠的记忆排除掉了(它们表示本就不像, 权重自然低),
#             保真度加权【无事可做】. 若坐实, 这一条就该从方案里删掉 ——
#             少一个部件反而更干净, 也更好讲.
#   ⚠️ 本实验的合格产出【包含"确认它没用"】. 目标不是证明它有用.
#
# 【前置诊断: 先看 conf 的方差, 方差≈0 就别往下测了】
#   use_conf 干的事是 W ← W × conf 再行归一化. 若所有样本 conf 相同,
#   这就是【乘一个常数再归一化】= 恒等变换, 数学上必然零差异.
#   所以第一张表先报 conf 的分布. 方差太小 ⇒ 结论是"实验设计不成立", 不是"方法没用".
#
# 【数据来源】
#   优先读 M1 产出的真实 memory(mem_records.jsonl); 没有则退回合成轨迹.
#   ⚠️ 合成结果只能验机制, 不能进论文 —— 合成诊断踩过 7 个测量假象.
#
# 运行: python ca_ablation_test.py            (合成)
#       python ca_ablation_test.py --real mem_records.jsonl   (真实)
# ═════════════════════════════════════════════════════════════════════════

import argparse, json, os, random
import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from vtree_compressor import VTreeCompressor
from credit import (SoftGrouper, CreditAssigner, ARM_SWITCHES, ARM_NAMES,
                    effective_neighbors, grouping_quality)

MODEL = "/home/wuyi/cuda12-dev/project/models/Qwen2.5-7B-Instruct"
LAYER, MAXLEN, BATCH = -8, 1024, 8
# ★ G 要够大: 主指标在【has_key=True】的子集内再按动作二分, 有效样本 ≈ G/4.
#   dry-run 时 G=6 曾让两个子集之一为空 ⇒ 指标全 nan. G=24 时每格约 6 条.
G = 24                       # 每个 seed 的轨迹数
SEEDS = [0, 1, 2]
TAUS = [0.01, 0.05, 0.1, 0.3]
TAU_MAIN = 0.05
# ★ 预算【故意设成不同值】: conf 要有方差, use_conf 才可能起作用
BUDGETS = [40, 60, 90, 140]
device = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════
# 合成任务：处境识别【必须依赖历史】
#   当前观测只说"面前有扇门"——手里有没有钥匙, 只能从历史推出来.
#   （credit_rank_test 前两版的教训: 处境若能靠当前观测认出来, 历史压不压都一样,
#     压缩的价值根本测不出来。）
# ══════════════════════════════════════════════════════════════════════
COLORS = ["暗红色", "黄铜色", "深绿色", "银白色"]
FILLER = ["你走过一段空走廊，两侧什么也没有",
          "你继续往前，脚步声在走廊里回响",
          "墙上有一道旧裂缝，没什么特别",
          "头顶的灯管闪了一下",
          "地面有些积水，你绕了过去"]
ACTIONS = ["用钥匙开门", "推门", "敲门", "转身离开"]


def gen_traj(rng, L):
    """关键步 = 捡到真钥匙那一步。终点观测【不透露】手里有什么。"""
    key_c, *dec = rng.sample(COLORS, 3)
    steps = [f"【告示】能打开尽头那扇门的，是{key_c}的钥匙"]

    has_key = rng.random() < 0.5              # 一半轨迹拿到真钥匙
    ev = [(True, f"你在地上看到一个{key_c}的小物件，把它捡了起来")] if has_key else []
    ev += [(False, f"你在地上看到一个{c}的小物件，把它捡了起来") for c in dec[:1]]
    ev += [(False, rng.choice(FILLER)) for _ in range(L - len(ev) - 2)]
    rng.shuffle(ev)

    key_idx = -1
    for is_key, t in ev:
        if is_key: key_idx = len(steps)
        steps.append(t)
    steps.append("你走到走廊尽头，面前是一扇锁着的门")   # 不说手里有没有钥匙

    # ★ 动作【不能均匀四选一】: 主指标要在 has_key=True 内部再按"是否开门"二分,
    #   均匀采样下 (has_key ∧ 开门) 只占 1/8, G=16 时期望才 2 条 ⇒ 指标全是 nan.
    #   改成模拟一个半吊子 policy: 一半概率开门, 一半概率在其余动作里挑.
    act = "用钥匙开门" if rng.random() < 0.5 else rng.choice(ACTIONS[1:])
    # 回报规则: 只有【手里真有钥匙】且【选了用钥匙开门】才成功
    ret = 1.0 if (has_key and act == "用钥匙开门") else 0.0
    return dict(steps=steps, action=act, ret=ret,
                has_key=has_key, task="找到正确的钥匙并打开走廊尽头的门")


# ── 模型 ────────────────────────────────────────────────────────────────
def load_model():
    print(f"加载 {MODEL} (4bit) ...")
    tk = AutoTokenizer.from_pretrained(MODEL)
    if tk.pad_token is None: tk.pad_token = tk.eos_token
    tk.padding_side = "left"; tk.truncation_side = "left"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    md = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                              device_map={"": 0}).eval()
    return tk, md


tok, model = load_model()
YES, NO = [tok(c, add_special_tokens=False).input_ids[0] for c in ("是", "否")]
PROBE_P = "任务:{task}\n当前记忆:\n{mem}\n问:这个任务最终能成功吗?只回答一个字:是 或 否。\n答:"


@torch.no_grad()
def probe(mems, task=""):
    out = []
    for i in range(0, len(mems), BATCH):
        enc = tok([PROBE_P.format(task=task, mem=m) for m in mems[i:i + BATCH]],
                  return_tensors="pt", padding=True, truncation=True,
                  max_length=MAXLEN).to(device)
        lg = model(**enc).logits[:, -1, :].float()
        out.extend(torch.softmax(lg[:, [YES, NO]], -1)[:, 0].cpu().tolist())
    return np.array(out)


@torch.no_grad()
def embed(texts):
    """★ last-token 池化 —— 不能用 mean(实测差距 +0.401 vs +0.013)。"""
    vs = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAXLEN).to(device)
        h = model(**enc, output_hidden_states=True).hidden_states[LAYER]
        p = h[:, -1, :]                              # left padding ⇒ 末位就是末 token
        vs.append(torch.nn.functional.normalize(p.float(), dim=-1).cpu().numpy())
    return np.concatenate(vs, 0)


class _Tok:
    def __call__(self, t, add_special_tokens=False):
        return type("E", (), {"input_ids": tok(t, add_special_tokens=False).input_ids})()


class _Cached:
    """把批量算好的前缀价值按顺序喂给在线 push。"""
    def __init__(self, v): self.v, self.i = list(v), 0
    def __call__(self, mems, task=""):
        n = len(mems)
        o = np.array([self.v[min(self.i + k, len(self.v) - 1)] for k in range(n)])
        self.i += n; return o


# ══════════════════════════════════════════════════════════════════════
# 构造一批 (memory, conf, action, return, key_idx)
# ══════════════════════════════════════════════════════════════════════
def build_dataset(seed):
    rng = random.Random(seed)
    rows = []
    for g in range(G):
        L = rng.choice([8, 10, 12, 14])
        budget = BUDGETS[g % len(BUDGETS)]           # ★ 轮换预算 ⇒ conf 有方差
        tr = gen_traj(rng, L)

        V = probe(["\n".join(tr["steps"][:t + 1]) for t in range(len(tr["steps"]))],
                  tr["task"])
        c = VTreeCompressor(_Cached(V), _Tok(), budget=budget, task=tr["task"])
        for s in tr["steps"]:
            c.push(s)
        mem, _ = c.render()

        rows.append(dict(mem=mem, conf=c.fidelity(), action=tr["action"],
                         ret=tr["ret"], traj=g, budget=budget,
                         n_steps=len(tr["steps"]), has_key=tr["has_key"]))
    return rows


def load_real(path):
    """读 M1 产出的 mem_records.jsonl。每行需含: mem, conf, action, ret, traj。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append(dict(mem=d["mem"], conf=float(d.get("conf", 1.0)),
                             action=str(d["action"]), ret=float(d["ret"]),
                             traj=int(d.get("traj", 0)),
                             budget=d.get("budget", -1),
                             n_steps=d.get("n_steps", -1),
                             has_key=d.get("has_key", None)))
    return rows


# ══════════════════════════════════════════════════════════════════════
def subsets(rows):
    """主指标要用的两个子集. 分开返回, 好在诊断里报样本数.

    ★ 为什么只在 has_key=True 内部比:
      CA 该学到的是"【在有钥匙这个处境下】开门是好动作". 若把 has_key=False 的
      轨迹混进来一起平均, 比的就变成了"有没有钥匙"(处境差异), 而不是
      "同处境下动作好不好"(动作差异) —— 那是 V̂ 的活, 不是 A 的活.
    """
    hk = np.array([bool(r["has_key"]) for r in rows]) \
        if rows[0]["has_key"] is not None else None
    if hk is None:
        return None, None
    opened = np.array([r["action"] == "用钥匙开门" for r in rows])
    return hk & opened, hk & ~opened          # (该拿正优势, 该拿负优势)


def run_arms(rows, tau):
    phi = embed([r["mem"] for r in rows])
    conf = np.array([r["conf"] for r in rows])
    rets = np.array([r["ret"] for r in rows])
    acts = np.array([r["action"] for r in rows])
    blk = np.array([r["traj"] for r in rows])

    W = SoftGrouper(tau).weights(phi, block=blk)     # 屏蔽同轨迹自比
    ca = CreditAssigner()
    pos, neg = subsets(rows)
    usable = pos is not None and pos.sum() > 0 and neg.sum() > 0

    out = {}
    for arm, sw in ARM_SWITCHES.items():
        A = ca.advantages(W, conf, rets, acts, **sw)
        if usable:
            sep = float(A[pos].mean() - A[neg].mean())   # 主指标: 好坏动作的优势差
            y = np.zeros(len(A)); y[pos] = 1.0
            keep = pos | neg
            rho = spearmanr(A[keep], y[keep]).correlation
        else:
            sep = rho = float("nan")
        out[arm] = (rho, sep, float(np.abs(A).mean()))
    return out, W, phi, conf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None, help="mem_records.jsonl 路径")
    args = ap.parse_args()

    if args.real and os.path.exists(args.real):
        print(f"数据来源: 真实 memory  ({args.real})")
        datasets = [load_real(args.real)]
        src = "真实"
    else:
        if args.real:
            print(f"⚠️ 找不到 {args.real}, 退回合成数据")
        print("数据来源: 合成轨迹  ⚠️ 只能验机制, 不能进论文")
        datasets = [build_dataset(sd) for sd in SEEDS]
        src = "合成"

    # ── 前置诊断 1: conf 有没有方差 ──────────────────────────────────
    allconf = np.concatenate([[r["conf"] for r in d] for d in datasets])
    print(f"\n{'='*76}")
    print("前置诊断 ① · 保真度 conf 的分布  【方差≈0 则 use_conf 数学上必然无效】")
    print(f"{'='*76}")
    print(f"  样本数 {len(allconf)}   均值 {allconf.mean():.3f}   "
          f"标准差 {allconf.std():.4f}   范围 [{allconf.min():.3f}, {allconf.max():.3f}]")
    if allconf.std() < 0.02:
        print("  ❌ 方差太小: use_conf 退化成【乘常数再归一化】= 恒等变换.")
        print("     下面的 c≈b 是【实验设计不成立】, 不能解读成'保真度加权没用'.")
        print("     修法: 加大 BUDGETS 的跨度, 或混入更长的轨迹.")
    else:
        print("  ✅ conf 有区分度, use_conf 有起作用的空间, 下面的对比有效.")

    # ── 前置诊断 1.5: 主指标的两个子集有没有样本 ─────────────────────
    print(f"\n{'='*76}")
    print("前置诊断 ①ᵇ · 主指标的子集样本数  【任一为 0 则指标是 nan, 不是结论】")
    print(f"{'='*76}")
    n_pos = n_neg = 0
    for d in datasets:
        p, ng = subsets(d)
        if p is not None:
            n_pos += int(p.sum()); n_neg += int(ng.sum())
    print(f"  有钥匙 ∧ 开门 (该拿正优势) = {n_pos:>3} 条")
    print(f"  有钥匙 ∧ 没开门(该拿负优势) = {n_neg:>3} 条")
    if min(n_pos, n_neg) < 4:
        print(f"  ❌ 样本太少, 主指标不稳. 把 G 调大(现在 G={G}), 或提高开门动作的采样比例.")
    else:
        print("  ✅ 两个子集都够, 主指标可解读.")

    # ── 前置诊断 2: 分组到底成不成立 ─────────────────────────────────
    d0 = datasets[0]
    phi0 = embed([r["mem"] for r in d0])
    kinds0 = np.array([1 if r["has_key"] else 0 for r in d0]) \
        if d0[0]["has_key"] is not None else np.zeros(len(d0), int)
    q = grouping_quality(phi0, kinds0)
    print(f"\n{'='*76}")
    print("前置诊断 ② · 分组质量  【差距<0.05 则软分组必然退化, 先修表示】")
    print(f"{'='*76}")
    print(f"  sep-AUC {q['auc']:.3f} | 同类 {q['same']:.3f} | 异类 {q['diff']:.3f} | "
          f"差距 {q['gap']:+.3f}  {'✅' if q['gap'] > 0.05 else '❌'}")
    print("  ⚠️ 判据是【差距】不是 AUC —— AUC 只看排序, 排序对 ≠ 拉得开.")

    # ── 主结果: 四条臂 ──────────────────────────────────────────────
    acc = {a: [] for a in ARM_NAMES}
    for d in datasets:
        r, _, _, _ = run_arms(d, TAU_MAIN)
        for a in ARM_NAMES: acc[a].append(r[a])

    print(f"\n{'='*76}")
    print(f"主结果 · 四条臂  (tau={TAU_MAIN}, {src}数据, {len(datasets)} 次重复)")
    print(f"{'='*76}")
    print(f"  {'臂':<14} | {'Spearman':>9} | {'好坏动作优势差':>13} | {'|A|均值':>9}")
    print("  " + "-" * 58)
    m = {}
    for a in ARM_NAMES:
        v = np.nanmean(np.array(acc[a], dtype=float), 0)
        m[a] = v
        print(f"  {a:<14} | {v[0]:>+9.3f} | {v[1]:>+13.3f} | {v[2]:>9.4f}")

    print(f"\n  ── 逐条改法的净增量 ──")
    print(f"  b − a  (动作反事实)   = {m['b_pace'][1] - m['a_soft'][1]:+.4f}")
    print(f"  c − b  (保真度加权) ★ = {m['c_full'][1] - m['b_pace'][1]:+.4f}")
    print(f"  d − a  (保真度单独)   = {m['d_conf_only'][1] - m['a_soft'][1]:+.4f}")

    # ── tau 敏感性 ───────────────────────────────────────────────────
    print(f"\n{'='*76}")
    print("tau 敏感性  【两个方向都没预设答案, 扫出来再定】")
    print(f"{'='*76}")
    print(f"  {'tau':>6} | {'有效邻居数':>10} | " +
          " | ".join(f"{a:>11}" for a in ARM_NAMES))
    print("  " + "-" * 72)
    for tau in TAUS:
        rows_t, effs = [], []
        for d in datasets:
            r, W, _, _ = run_arms(d, tau)
            rows_t.append([r[a][1] for a in ARM_NAMES])
            effs.append(effective_neighbors(W))
        mt = np.nanmean(np.array(rows_t, dtype=float), 0)
        print(f"  {tau:>6.2f} | {np.mean(effs):>10.2f} | " +
              " | ".join(f"{x:>+11.3f}" for x in mt))

    print(f"""
{'='*76}
判读
{'='*76}
  【读表顺序】先看两个前置诊断, 任一不过就别解读主结果.

  主问题 c − b (保真度加权的净增量):
   · 显著 > 0        → ✅ 保真度加权有用, 保留. 这是"压缩+CA 一起做"独有的部件,
                        也是相比纯压缩方法最硬的差异点.
   · ≈ 0 (|Δ|<0.01) → ⚠️ 与 walkthrough 的 1.438→1.439 一致 ⇒ 【删掉它】.
                        理由: 软分组已把压得狠的记忆排除干净, 它无事可做.
                        删掉不是失败 —— 方案少一个部件更干净, 论文更好写,
                        而且省掉了"conf 的档位分怎么标定"这个悬而未决的问题.
   · < 0             → ❌ 它在帮倒忙. 必须删, 并在论文里说明为什么.

  b − a (动作反事实):
   · > 0  → 符合预期, 这是 CA 的标准做法;
   · ≈ 0  → 检查动作空间是不是太小(同动作邻居太多, Q̂ 退化成 V̂).

  d − a (保真度单独):
   · 若 d−a > 0 但 c−b ≈ 0 → 保真度的作用被动作反事实【吸收】了, 两者冗余.
     此时选一个留即可, 优先留 b(更通用, 不依赖压缩).

  tau:
   · 挑有效邻居数落在 2 ~ N/3 的那一档; 若各档结果差异很大, 是 limitation, 要报告.

  ⚠️ {src}数据: {'结论可进论文.' if src == '真实' else '只能验机制. 进论文的证据必须来自真实 memory(M1).'}""")
