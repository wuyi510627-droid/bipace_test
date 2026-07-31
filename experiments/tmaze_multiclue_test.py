# tmaze_multiclue_test.py —— 多线索版:同样的预算, 比"挑哪几步"
# ─────────────────────────────────────────────────────────────────────────
# 为什么要这个(Passive 版的局限):
#   Passive T-Maze 只有【一个】关键步(第0步的线索), 其余全是废话 → 闭着眼挑都对.
#   它只能证"该留的留住了", 证不了"从一堆真假混杂的候选里挑对了".
#
# 本实验的设计要点 —— 唯一变量是【挑选策略】:
#   所有对照臂用【同一个 token 预算】、【同一套装箱逻辑】, 只有优先级不同.
#     vtree   : 按 价值跳幅/长度 (性价比) 排序        ← 本方法
#     random  : 随机排序                             ← "不会挑"的基线
#     recent  : 从后往前(= trunc-k 的等预算版)        ← 按位置挑
#     first   : 从前往后                             ← 另一个位置基线
#   外加 no_mem(下界) / full(上界, 不受预算约束).
#
# 任务 = 两跳推理: 必须【同时】记住两条线索才能答对
#     线索A「钥匙藏在{颜色}柜子里」+ 线索B「{颜色}柜子放在{楼层}楼」→ 问:钥匙在几楼?
#   两条线索随机散落在走廊里, 其余是噪声(有的还带迷惑性的具体细节).
#   预算设得【只够装 2~3 步】→ 挑错就答不对.
#
# 判读: vtree 应显著高于 random/recent/first(同预算), 并逼近 full(不限预算).
#       若 vtree ≈ random, 说明价值信号没有挑选能力 —— 这是本方法的证伪条件.
#
# 运行(有 GPU+模型的机器): python tmaze_multiclue_test.py
# ─────────────────────────────────────────────────────────────────────────

import random, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "/home/wuyi/cuda12-dev/project/models/Qwen2.5-7B-Instruct"
MAXLEN, BATCH = 1024, 16
LS = [4, 8, 16, 24]          # 走廊总长(线索散落其中)
BUDGETS = [60, 90]           # token 预算: 只够装 2~3 步
N_EPI = 100
SEEDS = [0, 1, 2]
device = "cuda" if torch.cuda.is_available() else "cpu"

COLORS = ["红色", "蓝色", "绿色"]
FLOORS = ["二", "三", "四"]

# 噪声: 有的带具体细节(对"信息量"型压缩很有诱惑, 但对答题无用)
NOISE = [
    "你走过一段空走廊，什么也没有。",
    "墙角堆着几个落满灰尘的纸箱。",
    "天花板的灯管闪了两下。",
    "地上有一张过期的外卖单，日期是三月十七号。",
    "你听见远处传来水管滴水的声音。",
    "布告栏上贴着一张褪色的消防演习通知。",
    "走廊的窗户开着，外面在下雨。",
    "有只猫从你脚边跑过去了。",
]


def gen_episode(L):
    """两跳推理: 线索A+线索B 随机散落, 其余噪声. 返回(答案楼层, 观测列表)."""
    color, floor = random.choice(COLORS), random.choice(FLOORS)
    clue_a = f"【告示】钥匙藏在{color}柜子里。"
    clue_b = f"【告示】{color}柜子放在{floor}楼。"
    obs = [random.choice(NOISE) for _ in range(L)]
    ia, ib = sorted(random.sample(range(L), 2))       # 两条线索随机位置
    obs[ia], obs[ib] = clue_a, clue_b
    obs.append("【岔路口】你要去找钥匙，必须选一层楼上去：二楼、三楼、还是四楼？")
    return floor, obs


# ── 模型 ────────────────────────────────────────────────────────────────
print(f"加载 {MODEL} (4bit) ...")
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"; tok.truncation_side = "left"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                             device_map={"": 0}).eval()
FLOOR_IDS = [tok(f, add_special_tokens=False).input_ids[0] for f in FLOORS]
YES_ID = tok("是", add_special_tokens=False).input_ids[0]
NO_ID = tok("否", add_special_tokens=False).input_ids[0]
_HEAD = "你是走迷宫的智能体。你目前掌握的记忆如下:\n{}\n"


@torch.no_grad()
def _logits(prompts):
    out = []
    for i in range(0, len(prompts), BATCH):
        enc = tok(prompts[i:i + BATCH], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAXLEN).to(device)
        out.append(model(**enc).logits[:, -1, :].float().cpu())
    return torch.cat(out, 0)


def answer_floor(mems):
    """决策: 钥匙在几楼 → 三选一"""
    lg = _logits([_HEAD.format(m) + "问:钥匙在几楼?只回答一个字:二 或 三 或 四。\n答:"
                  for m in mems])
    return [FLOORS[i] for i in lg[:, FLOOR_IDS].argmax(-1).tolist()]


def value_of(mems):
    """价值探针: 能否成功找到钥匙 → P(是)"""
    lg = _logits([_HEAD.format(m) + "问:你能成功找到钥匙吗?只回答一个字:是 或 否。\n答:"
                  for m in mems])
    return torch.softmax(lg[:, [YES_ID, NO_ID]], -1)[:, 0].numpy()


def ntok(t): return len(tok(t, add_special_tokens=False).input_ids)


# ── 统一装箱: 给定优先级顺序, 按预算挑步 ────────────────────────────────
FOLD = "(此处若干步已折叠)"


def pack(obs, order, budget):
    """所有对照臂共用: 当前步必留(不占预算), 其余按 order 优先级装到预算用完."""
    n = len(obs)
    keep, used = {n - 1}, 0
    for i in order:
        if i in keep:
            continue
        c = ntok(obs[i])
        if used + c <= budget:
            keep.add(i); used += c
    idx = sorted(keep)
    lines, prev = [], -1
    for i in idx:
        if i - prev > 1:
            lines.append(FOLD)
        lines.append(obs[i]); prev = i
    return "\n".join(lines)


def order_vtree(obs):
    """本方法: 按 价值跳幅/长度(性价比) 降序. 一次批量问所有前缀."""
    vs = value_of(["\n".join(obs[:t + 1]) for t in range(len(obs))])
    prev, jumps = 0.5, []
    for v in vs:
        jumps.append(abs(float(v) - prev)); prev = float(v)
    return sorted(range(len(obs) - 1),
                  key=lambda k: (-jumps[k] / max(ntok(obs[k]), 1), -k)), jumps


ARMS = ["no_mem", "full", "vtree", "random", "recent", "first"]


def run_seed(seed, budget):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    acc = {a: [] for a in ARMS}; tkn = {a: [] for a in ARMS}
    for L in LS:
        hit = {a: 0 for a in ARMS}; tot = {a: 0 for a in ARMS}
        for _ in range(N_EPI):
            goal, obs = gen_episode(L)
            n = len(obs)
            o_v, _ = order_vtree(obs)
            o_r = list(range(n - 1)); random.shuffle(o_r)
            mems = {
                "no_mem": obs[-1],
                "full":   "\n".join(obs),
                "vtree":  pack(obs, o_v, budget),
                "random": pack(obs, o_r, budget),
                "recent": pack(obs, list(range(n - 2, -1, -1)), budget),
                "first":  pack(obs, list(range(n - 1)), budget),
            }
            names = list(mems)
            for nm, pred in zip(names, answer_floor([mems[k] for k in names])):
                if pred == goal: hit[nm] += 1
                tot[nm] += ntok(mems[nm])
        for a in ARMS: acc[a].append(hit[a] / N_EPI); tkn[a].append(tot[a] / N_EPI)
        print(f"  L={L:>2}: " + "  ".join(f"{a}={acc[a][-1]:.2f}({tkn[a][-1]:.0f}tk)" for a in ARMS))
    return acc, tkn


if __name__ == "__main__":
    for budget in BUDGETS:
        print(f"\n{'='*72}\n预算 = {budget} tokens\n{'='*72}")
        A = {a: [] for a in ARMS}; T = {a: [] for a in ARMS}
        for s in SEEDS:
            print(f"seed {s}:")
            acc, tkn = run_seed(s, budget)
            for a in ARMS: A[a].append(acc[a]); T[a].append(tkn[a])
        Am = {a: np.mean(A[a], 0) for a in ARMS}; As = {a: np.std(A[a], 0) for a in ARMS}
        Tm = {a: np.mean(T[a], 0) for a in ARMS}
        print(f"\n--- 答对率 (mean±std over {len(SEEDS)} seeds, N={N_EPI}) ---")
        print("L\t" + "\t".join(ARMS))
        for i, L in enumerate(LS):
            print(f"{L}\t" + "\t".join(f"{Am[a][i]:.2f}±{As[a][i]:.2f}" for a in ARMS))
        print(f"--- 记忆 token ---")
        for i, L in enumerate(LS):
            print(f"{L}\t" + "\t".join(f"{Tm[a][i]:.0f}" for a in ARMS))

    print("\n判读:")
    print("  ① vtree 显著 > random/recent/first(同预算) → 价值信号【真的会挑】;")
    print("  ② vtree 逼近 full(不限预算) → 挑对了就够, 不必全留;")
    print("  ③ 若 vtree ≈ random → 价值信号没有挑选能力, 本方法的核心假设被证伪.")
    print("  ④ 随 L 增大, 噪声变多, vtree 与 random 的差距应【拉大】.")
