# credit_rank_test.py —— 实验③：压缩之后，信用还分得对吗？（核心证据）
# ─────────────────────────────────────────────────────────────────────────
# 为什么是"最大的洞"：
#   ① 测的是【上游】—— 压完还能不能认出同处境(sep-AUC)
#   ① 测的是【下游】—— 任务成功率
#   中间那一环【信用分配本身】一次都没量过. 而"为信用而压"正是本方法区别于所有
#   现有压缩方法的地方 —— 不测它, 这个主张就只停留在嘴上.
#
# 任务：Key-to-Door + 干扰物（文本版）
#   走廊里散落【多个可拿的东西】, 每个都是"拿起/路过"二选一,
#   但【只有钥匙】能开门, 石头和易拉罐都是干扰:
#     步a  【地上有一块石头】   拿/不拿   ← 干扰
#     步b  【地上有一把钥匙】   拿/不拿   ← ★关键步
#     步c  【地上有一个易拉罐】 拿/不拿   ← 干扰
#     末步 【面前一扇锁着的门】 开门 —— 拿了钥匙才开得了
#   ⇒ 全部信用【都该】落在钥匙那步.
#
# ⚠️ 为什么非要放干扰物(第一版没有, 自测发现是设计缺陷):
#   若"拿起钥匙"是整条轨迹里独一无二的动作, 那么光按【动作】一分就能识别它,
#   根本不需要认出处境 —— 压缩糊不糊都不影响结果, 这个实验就白做了.
#   放上干扰物后,"拿起"这个动作出现在多个处境, 必须【认得出面对的是钥匙还是石头】
#   才分得对信用 —— 而"认不认得出处境"正是压缩影响的东西.
#
# 怎么算信用（沿用 GiGPO / BiPACE 那套 critic-free 的组内比较）：
#   ① 把所有 (轨迹,步) 的 memory 编码成向量
#   ② 软分组：向量相似的凑一堆（w_j = 核加权）
#   ③ 组内按【执行的动作】拆开：Q̂(s,a) = 同动作邻居的加权回报，V̂(s) = 全组加权回报
#   ④ 优势 A = Q̂ − V̂
#   ⇒ 压缩影响的是①②——压糊了就认不出同处境、分组乱、信用跟着错.
#
# 四个对照臂（同一批轨迹, 只换喂进去的 memory）：
#   full     全历史          naive  整段压成一句废话
#   trunc-k  只保最近 k 步     vtree  本方法（按价值跳幅分配 token）
#
# 三个指标（后两个防"排名饱和"）：
#   关键步排名   |A_key| 在所有 |A_t| 里排第几        —— 直观但会饱和
#   信用集中度   |A_key| / Σ|A_t|                    —— 不饱和
#   margin      |A_key| − 第二大的 |A_t|             —— 分得干不干脆
#
# 判读：
#   vtree ≈ full           → 只兑现动机一（压了不亏）
#   vtree > full（集中度/margin） → 才兑现动机二（压缩主动提纯信用信号）
#
# 运行(有 GPU+模型的机器): python credit_rank_test.py
# ─────────────────────────────────────────────────────────────────────────

import random, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from vtree_compressor import VTreeCompressor

MODEL = "/home/wuyi/cuda12-dev/project/models/Qwen2.5-7B-Instruct"
LAYER, MAXLEN, BATCH = -8, 1024, 8
G = 16                    # 每组轨迹数（要够大才凑得成组）
LS = [8, 14]              # 走廊总长（要放得下 3 个可拿物）
TRUNC_K = 4               # trunc 臂保留最近几步
BUDGET = 80               # vtree 的 token 上限
TAU = 0.05                # 软分组核宽
SEEDS = [0, 1, 2]
device = "cuda" if torch.cuda.is_available() else "cpu"
ARMS = ["full", "naive", f"trunc{TRUNC_K}", "vtree"]

FILLER = ["你走过一段空走廊，两侧什么也没有", "你继续往前，脚步声在走廊里回响",
          "你路过一扇紧闭的侧门", "你看到墙上有几道划痕", "你又往前走了几步",
          "头顶的灯管闪了一下", "地上积了些灰尘", "你听见远处有水滴声"]


ITEMS = [("【地上躺着一把黄铜钥匙】", True),      # 关键: 拿了才能开门
         ("【地上有一块灰色的石头】", False),      # 干扰: 动作一样, 但没用
         ("【地上有一个空易拉罐】", False)]        # 干扰


def gen_traj(rng, L):
    """走廊里散落 3 个可拿物(1 钥匙 + 2 干扰), 每个独立决定拿/不拿.
    只有拿了钥匙才成功 ⇒ 信用应当全部落在钥匙那步."""
    slots = sorted(rng.sample(range(1, L - 1), 3))      # 三个物品的位置
    order = list(range(3)); rng.shuffle(order)           # 谁在前谁在后随机
    steps, acts, key = [], [], None
    took_key = False
    for t in range(L):
        if t in slots:
            txt, is_key = ITEMS[order[slots.index(t)]]
            take = rng.random() < 0.5
            steps.append(txt)
            acts.append("拿起" if take else "路过")       # ← 同一套动作, 靠处境区分
            if is_key:
                key = t; took_key = take
        else:
            steps.append(rng.choice(FILLER))
            acts.append("往前走")
    steps.append("【面前是一扇锁着的门】")
    acts.append("开门")
    return dict(steps=steps, acts=acts, R=1.0 if took_key else 0.0,
                key=key, task="穿过走廊打开尽头那扇门")


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
def embed(texts):
    """memory → 向量（取晚期层隐状态 mean 池化 + L2 归一, 与 BiPACE 同法）."""
    vs = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAXLEN).to(device)
        h = model(**enc, output_hidden_states=True).hidden_states[LAYER]
        m = enc.attention_mask.unsqueeze(-1).float()
        p = (h * m).sum(1) / m.sum(1).clamp(min=1)
        vs.append(torch.nn.functional.normalize(p.float(), dim=-1).cpu().numpy())
    return np.concatenate(vs, 0)


class _Tok:                                    # 给 VTreeCompressor 用的分词器包装
    def __call__(self, t, add_special_tokens=False):
        return type("E", (), {"input_ids": tok(t, add_special_tokens=False).input_ids})()


class _CachedProbe:
    """把整条轨迹所有前缀的 V 预先算好, 供 VTreeCompressor 逐步取用(避免重复调用)."""
    def __init__(self, vals): self.vals, self.i = list(vals), 0
    def __call__(self, mems, task=""):
        n = len(mems)
        out = np.array([self.vals[min(self.i + k, len(self.vals) - 1)] for k in range(n)])
        self.i += n
        return out


# ── 各臂在第 t 步时的 memory ────────────────────────────────────────────
def build_mems(tr, prefix_V):
    """返回 {臂: [每一步的 memory]}"""
    steps = tr["steps"]; T = len(steps)
    out = {a: [] for a in ARMS}
    for t in range(T):
        pre = steps[:t + 1]
        out["full"].append("\n".join(pre))
        out["naive"].append("你在走廊里走了一段，做了一些事情。")
        out[f"trunc{TRUNC_K}"].append("\n".join(pre[max(0, len(pre) - TRUNC_K):]))
        c = VTreeCompressor(_CachedProbe(prefix_V[:t + 1]), _Tok(),
                            budget=BUDGET, task=tr["task"])
        for o in pre: c.push(o)
        out["vtree"].append(c.render()[0])
    return out


# ── 信用分配: 软分组 + 动作反事实 ───────────────────────────────────────
def advantages(phi, rets, acts, tau=TAU):
    S = phi @ phi.T
    W = np.exp((S - 1.0) / tau); np.fill_diagonal(W, 0.0)
    W = W / W.sum(1, keepdims=True).clip(1e-9)
    V = W @ rets                                            # V̂(s)
    Q = np.zeros_like(V)
    for i in range(len(V)):
        m = (acts == acts[i]).astype(float); m[i] = 0.0     # 同动作邻居
        Q[i] = (W[i] * m) @ rets / max((W[i] * m).sum(), 1e-9) if m.sum() else rets[i]
    return Q - V


def metrics(A, key_idx):
    """A: 一条轨迹各步的优势; key_idx: 关键步下标"""
    a = np.abs(A)
    rank = int((a > a[key_idx]).sum()) + 1                  # 关键步排第几
    conc = a[key_idx] / max(a.sum(), 1e-9)                  # 信用集中度
    others = np.delete(a, key_idx)
    margin = a[key_idx] - (others.max() if len(others) else 0.0)
    return rank, conc, margin


# ── 主流程 ──────────────────────────────────────────────────────────────
def run(seed, L):
    rng = random.Random(seed * 100 + L)
    trajs = [gen_traj(rng, L) for _ in range(G)]
    # 每条轨迹先批量算出所有前缀的 V（vtree 要用）
    for tr in trajs:
        tr["V"] = probe(["\n".join(tr["steps"][:t + 1]) for t in range(len(tr["steps"]))],
                        tr["task"])
    mems = {a: [] for a in ARMS}; rets, acts, keyflag, tid = [], [], [], []
    for gi, tr in enumerate(trajs):
        bm = build_mems(tr, tr["V"])
        for a in ARMS: mems[a] += bm[a]
        T = len(tr["steps"])
        rets += [tr["R"]] * T; acts += tr["acts"]
        keyflag += [t == tr["key"] for t in range(T)]; tid += [gi] * T
    rets = np.array(rets); acts = np.array(acts)
    keyflag = np.array(keyflag); tid = np.array(tid)

    res = {}
    for a in ARMS:
        A = advantages(embed(mems[a]), rets, acts)
        rk, cc, mg = [], [], []
        for gi in range(G):
            sel = tid == gi
            ki = int(np.where(keyflag[sel])[0][0])
            r, c, m = metrics(A[sel], ki)
            rk.append(r); cc.append(c); mg.append(m)
        res[a] = (np.mean(rk), np.mean(cc), np.mean(mg))
    return res


if __name__ == "__main__":
    for L in LS:
        print(f"\n{'='*74}\n走廊长 L={L}  (1 钥匙 + 2 干扰物; {G} 条轨迹/seed; {len(SEEDS)} seeds)\n{'='*74}")
        acc = {a: [] for a in ARMS}
        for sd in SEEDS:
            r = run(sd, L)
            for a in ARMS: acc[a].append(r[a])
            print(f"  seed{sd}: " + "  ".join(
                f"{a}(排名{r[a][0]:.1f}/集中{r[a][1]:.2f})" for a in ARMS))
        print(f"\n  {'臂':<10} | {'关键步排名':>10} | {'信用集中度':>11} | {'margin':>9}")
        print("  " + "-" * 50)
        for a in ARMS:
            m = np.array(acc[a]).mean(0); s = np.array(acc[a]).std(0)
            star = " ★" if a == "vtree" else ""
            print(f"  {a:<10} | {m[0]:>6.2f}±{s[0]:<3.1f} | "
                  f"{m[1]:>6.3f}±{s[1]:<4.3f} | {m[2]:>+7.3f}{star}")

    print(f"""
{'='*74}
判读
{'='*74}
  理想情况: 关键步排名=1, 集中度→1.0 (所有信用都落在关键步), margin 大.
  · vtree ≈ full            → 只兑现【动机一】: 压了不亏, token 省下来了;
  · vtree > full (集中度/margin 显著更高) → 兑现【动机二】: 压缩主动提纯了信用信号;
  · vtree ≈ naive           → ⚠️ 压缩没能保住信用信号, 方法的核心主张不成立;
  · trunc-{TRUNC_K} 应当明显最差 —— 关键步早被截出窗口, 无从分起.""")
