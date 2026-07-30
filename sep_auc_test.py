# sep_auc_test.py —— 实验④：压完还认得出同处境吗（同时是 ③ 的诊断工具）
# ─────────────────────────────────────────────────────────────────────────
# 为什么先跑这个：
#   实验③(credit_rank_test) 跑出来四个臂几乎无差异, 集中度 0.28 —— 贴着"全糊掉"那一档.
#   说明【分组根本没工作】: 编码出来的向量分不出处境, 所有步被算进同一个大组,
#   压缩再好也没机会起作用. 所以要先回答: 表示到底能不能认出同处境?
#
# 测什么：sep-AUC(可分性)
#   把每步的 memory 编码成向量, 用"两步向量像不像"去预测"是不是同一个处境".
#     1.0 = 同处境的向量确实更像 → 认得出, 下游凑得成组
#     0.5 = 像不像和是不是同处境毫无关系 → 全糊成一团
#
# 顺带做三项诊断(定位 ③ 为什么没区分力)：
#   ① 相似度分布      —— 是不是全挤在 0.9+ 的窄锥里(各向异性)
#   ② 池化方式对比    —— mean vs last-token (BiPACE 用的是 last-token)
#   ③ 核宽 τ 敏感性   —— 有效邻居数随 τ 怎么变, 现用的 0.05 合不合适
#
# 运行(有 GPU+模型的机器): python sep_auc_test.py
# ─────────────────────────────────────────────────────────────────────────

import random, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import roc_auc_score
from vtree_compressor import VTreeCompressor

MODEL = "/home/wuyi/cuda12-dev/project/models/Qwen2.5-7B-Instruct"
LAYER, MAXLEN, BATCH = -8, 1024, 8
G, L = 12, 10             # 轨迹数 / 走廊长
TRUNC_K, BUDGET = 4, 80
SEEDS = [0, 1, 2]
TAUS = [0.01, 0.05, 0.1, 0.3]
device = "cuda" if torch.cuda.is_available() else "cpu"
ARMS = ["full", "naive", f"trunc{TRUNC_K}", "vtree"]

# ── 任务：同一处境跨轨迹复现（换物品名，处境标签相同）────────────────
OBJ = ["番茄", "鸡蛋", "土豆", "面包", "玉米", "茄子"]
STAGES = [                                    # (处境标签, 文本模板)
    (0, "你在厨房里四处找{o}"),
    (1, "在料理台找到并拿起了生的{o}"),
    (2, "手拿生的{o}，走到微波炉前"),
    (3, "打开微波炉，把生的{o}放了进去"),
    (4, "微波炉加热完成，{o}已经是热的"),
    (5, "手拿热的{o}，走向桌子"),
    (6, "把热的{o}放到桌上，任务完成"),
]
NOISE = ["你环顾四周，没看到别的东西", "你路过水槽，里面有些水渍",
         "你听到冰箱压缩机启动的声音", "你在厨房里又走了两步"]


def gen_traj(rng, L):
    """在 7 个固定处境之间插入噪声。处境标签跨轨迹复现 —— 这是 sep-AUC 的正样本对来源。"""
    o = rng.choice(OBJ)
    steps, labels = [], []
    for lab, tmpl in STAGES:
        for _ in range(rng.randint(0, max(0, (L - 7) // 3))):
            steps.append(rng.choice(NOISE)); labels.append(-1)      # -1 = 噪声, 不计入
        steps.append(tmpl.format(o=o)); labels.append(lab)
    return dict(steps=steps, labels=labels, task=f"把{o}加热后放到桌上")


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
def embed(texts, pool="mean"):
    """pool: mean = 掩码均值池化; last = 最后一个非 pad token(BiPACE 用的)"""
    vs = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAXLEN).to(device)
        h = model(**enc, output_hidden_states=True).hidden_states[LAYER]
        if pool == "mean":
            m = enc.attention_mask.unsqueeze(-1).float()
            p = (h * m).sum(1) / m.sum(1).clamp(min=1)
        else:                                   # left padding ⇒ 最后一位就是末 token
            p = h[:, -1, :]
        vs.append(torch.nn.functional.normalize(p.float(), dim=-1).cpu().numpy())
    return np.concatenate(vs, 0)


class _Tok:
    def __call__(self, t, add_special_tokens=False):
        return type("E", (), {"input_ids": tok(t, add_special_tokens=False).input_ids})()


class _Cached:
    def __init__(self, v): self.v, self.i = list(v), 0
    def __call__(self, mems, task=""):
        n = len(mems)
        o = np.array([self.v[min(self.i + k, len(self.v) - 1)] for k in range(n)])
        self.i += n; return o


def build_mems(tr, V):
    steps = tr["steps"]; out = {a: [] for a in ARMS}
    for t in range(len(steps)):
        pre = steps[:t + 1]
        out["full"].append("\n".join(pre))
        out["naive"].append("你在厨房里操作了一番。")
        out[f"trunc{TRUNC_K}"].append("\n".join(pre[max(0, len(pre) - TRUNC_K):]))
        c = VTreeCompressor(_Cached(V[:t + 1]), _Tok(), budget=BUDGET, task=tr["task"])
        for o in pre: c.push(o)
        out["vtree"].append(c.render()[0])
    return out


def sep_auc(phi, labels):
    """cos 相似度能否预测'是否同处境'. 只用 label>=0 的步(排除噪声步)."""
    keep = labels >= 0
    phi, labels = phi[keep], labels[keep]
    S = phi @ phi.T
    iu = np.triu_indices(len(phi), k=1)
    same = (labels[iu[0]] == labels[iu[1]]).astype(int)
    if same.min() == same.max(): return float("nan"), S, same, iu
    return roc_auc_score(same, S[iu]), S, same, iu


# ── 主流程 ──────────────────────────────────────────────────────────────
def run(seed, pool="mean"):
    rng = random.Random(seed)
    trajs = [gen_traj(rng, L) for _ in range(G)]
    for tr in trajs:
        tr["V"] = probe(["\n".join(tr["steps"][:t + 1]) for t in range(len(tr["steps"]))],
                        tr["task"])
    mems = {a: [] for a in ARMS}; labels = []
    for tr in trajs:
        bm = build_mems(tr, tr["V"])
        for a in ARMS: mems[a] += bm[a]
        labels += tr["labels"]
    labels = np.array(labels)
    res = {}
    for a in ARMS:
        auc, S, same, iu = sep_auc(embed(mems[a], pool), labels)
        res[a] = (auc, S[iu][same == 1].mean(), S[iu][same == 0].mean(), S[iu].mean())
    return res


if __name__ == "__main__":
    for pool in ("mean", "last"):
        print(f"\n{'='*78}\n池化方式 = {pool}   ({'BiPACE 用的就是 last-token' if pool=='last' else '当前 credit_rank_test 用的'})\n{'='*78}")
        print(f"  {'臂':<10} | {'sep-AUC':>9} | {'同处境相似度':>12} | {'不同处境':>10} | {'差距':>7}")
        print("  " + "-" * 62)
        acc = {a: [] for a in ARMS}
        for sd in SEEDS:
            r = run(sd, pool)
            for a in ARMS: acc[a].append(r[a])
        for a in ARMS:
            m = np.array(acc[a]).mean(0)
            star = " ★" if a == "vtree" else ""
            print(f"  {a:<10} | {m[0]:>9.3f} | {m[1]:>12.3f} | {m[2]:>10.3f} | "
                  f"{m[1]-m[2]:>+7.3f}{star}")

    # ── 核宽 τ 敏感性: 有效邻居数 ──────────────────────────────────────
    print(f"\n{'='*78}\n核宽 τ 敏感性 —— 有效邻居数(权重的熵指数), full 臂, mean 池化\n{'='*78}")
    rng = random.Random(0)
    trajs = [gen_traj(rng, L) for _ in range(G)]
    for tr in trajs:
        tr["V"] = probe(["\n".join(tr["steps"][:t + 1]) for t in range(len(tr["steps"]))], tr["task"])
    mm = []
    for tr in trajs: mm += build_mems(tr, tr["V"])["full"]
    phi = embed(mm, "mean"); S = phi @ phi.T
    print(f"  {'τ':>6} | {'有效邻居数':>10} | 说明")
    print("  " + "-" * 52)
    for tau in TAUS:
        W = np.exp((S - 1.0) / tau); np.fill_diagonal(W, 0.0)
        W = W / W.sum(1, keepdims=True).clip(1e-9)
        eff = np.exp(-(W * np.log(W + 1e-12)).sum(1)).mean()      # perplexity
        note = "太集中(只听1个邻居)" if eff < 2 else ("太平均(全组等权)" if eff > len(phi)/3 else "合理")
        print(f"  {tau:>6.2f} | {eff:>10.1f} | {note}  (总样本 {len(phi)})")

    print(f"""
{'='*78}
判读
{'='*78}
  ⚠️ 关键判据是【差距】那一列, 不是 sep-AUC —— 首轮实测就踩了这个坑:
     mean 池化 sep-AUC=0.863(看着合格!), 但同/不同处境相似度 0.997 vs 0.984,
     差距仅 +0.013. 核加权 exp((S-1)/τ) 吃的是绝对差距, 0.013 下权重几乎均匀
     ⇒ 软分组失效. AUC 只看【排序】, 排序对不等于【拉得开】.

  · 差距 < 0.05        → 向量全挤在窄锥里(各向异性), 软分组必然退化 ⇒ 先修表示;
  · 差距 > 0.3         → 拉得开, 核加权能正确集中 ⇒ 下游实验④ 才有意义;
  · last 明显优于 mean → 把④和方案里的池化都换成 last-token(首轮: 0.401 vs 0.013);
  · 有效邻居数 ≈ 全体   → τ 不对或表示不行; 结合"差距"那列判断是哪一个.""")
