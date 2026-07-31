# credit_test.py —— Tier1: 表示降级 到底害不害"信用"? (不训练, 你的卡几分钟)
# 思路: 造带"关键步"的合成轨迹 → 4种方法各自分组算信用(Q̂-V̂) → 查关键步信用符号对不对
#   BiPACE随U掉到明显低于Oracle → 降级真害信用 → 题活, 差距=你的机会
#   BiPACE≈Oracle不随U降           → 不害 → 题危
# 依赖同 survival_test. 运行: python credit_test.py

import random, string
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt

# ---------------- 配置 ----------------
MODEL   = "/home/wuyi/cuda12-dev/project/models/Qwen2.5-7B-Instruct"  # 改成你的本地路径
USE_4BIT = True
LAYER   = -8
PROMPT_TMPL = "你是一个网页/桌面操作智能体。当前界面观测如下：\n{obs}\n\n请判断此刻应采取的操作。操作："
U_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
MAX_DISTRACTORS = 24
N_TRAJ = 150         # 轨迹数(每条=1个关键步). 注意: 一个seed内所有关键步被同一分组结构耦合,
                     #        有效样本≈seed数不是轨迹数 → 压方差要靠加seed, 不是加N_TRAJ
T_STEP = 6           # 每条轨迹步数
EPS = 0.15           # BiPACE贪心余弦聚类阈值(与survival_test一致, 忠实实现)
BATCH = 8
MAXLEN = 384
SEEDS = list(range(6))   # 6个seed: 用标准误SEM=std/√n判显著, seed是真正的采样单位
device = "cuda" if torch.cuda.is_available() else "cpu"

# 10个处境: index0=P1(要测的关键处境), index1=P2(混淆处境), 2..9=填充
CANON = [
    "在商品列表页, 把【笔记本电脑】加入购物车。",   # 0 P1
    "在商品列表页, 把【手机】加入购物车。",         # 1 P2
    "在商品列表页, 把【耳机】加入购物车。",
    "在搜索结果页, 点开【第1个】结果。",
    "在搜索结果页, 点开【第3个】结果。",
    "在表单页, 在【邮箱】字段填写。",
    "在表单页, 在【电话】字段填写。",
    "在设置页, 打开【深色模式】开关。",
    "在设置页, 打开【通知】开关。",
    "在文件页, 进入【Downloads】文件夹。",
]
DISTRACT_TEMPLATES = [
    "广告: {w}季大促, 立减{n}元!", "导航: 首页 > {w} > {w}", "页脚版权 (c) {n} {w}公司",
    "推荐商品: {w} {w} {w}", "时间戳: 2026-07-0{d} {n}:{n}", "session_id={tok}",
    "用户{tok}最近浏览: {w}", "cookie横幅: 本站使用{w}以改善体验", "热搜: {w}, {w}, {w}",
]
WORDS = ["蓝牙","促销","数码","家居","春装","会员","限时","旗舰","优选","清仓"]
def _rtok(k=6): return "".join(random.choices(string.ascii_lowercase+string.digits, k=k))
def _distractor():
    t = random.choice(DISTRACT_TEMPLATES)
    return t.format(w=random.choice(WORDS), n=random.randint(1,99), d=random.randint(1,9), tok=_rtok())
def make_obs(canon_text, U):
    k = int(round(U * MAX_DISTRACTORS))
    parts = [canon_text] + [_distractor() for _ in range(k)]
    random.shuffle(parts)
    return "\n".join(parts)

# ---------------- 造轨迹(带关键步+混淆步) ----------------
# 奖励 R = 0.5*(P1选对A) + 0.5*(P2选对C); action: 0=对(A/C), 1=错(B/D)
# → P2 是混淆项, 让"整条轨迹信用(GRPO)"不够用, 必须靠分组隔离P1
def build(U):
    obs, label, isp1, act, ret = [], [], [], [], []
    for _ in range(N_TRAJ):
        p1_pos, p2_pos = random.sample(range(T_STEP), 2)
        p1a, p2a = random.randint(0,1), random.randint(0,1)
        R = 0.5*(p1a==0) + 0.5*(p2a==0)
        for t in range(T_STEP):
            if t == p1_pos:   lb, o, ip, a = 0, make_obs(CANON[0],U), True,  p1a
            elif t == p2_pos: lb, o, ip, a = 1, make_obs(CANON[1],U), False, p2a
            else:
                lb = random.randint(2, len(CANON)-1); o = make_obs(CANON[lb],U); ip=False; a=random.randint(0,1)
            obs.append(o); label.append(lb); isp1.append(ip); act.append(a); ret.append(R)
    return obs, np.array(label), np.array(isp1), np.array(act), np.array(ret, float)

# ---------------- 抽表示(agent prompt + 决策位置最后token) ----------------
print(f"加载模型 {MODEL} (4bit={USE_4BIT}) ...")
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"; tok.truncation_side = "left"
if USE_4BIT and device == "cuda":
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModel.from_pretrained(MODEL, quantization_config=bnb, device_map={"":0}, output_hidden_states=True).eval()
else:
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16 if device=="cuda" else torch.float32,
                                      output_hidden_states=True).to(device).eval()

@torch.no_grad()
def embed(obs_list):
    vecs=[]
    for i in range(0, len(obs_list), BATCH):
        prompts = [PROMPT_TMPL.format(obs=o) for o in obs_list[i:i+BATCH]]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to(device)
        h = model(**enc).hidden_states[LAYER]
        pooled = torch.nn.functional.normalize(h[:, -1, :].float(), dim=-1)
        vecs.append(pooled.cpu().numpy())
    return np.concatenate(vecs, 0)

# ---------------- 4种方法的信用(优势) ----------------
def hard_qv(keys, act, ret):
    """按key硬分组, 优势 = Q̂(s,a) - V̂(s); GiGPO单点→优势0(无信号)"""
    adv = np.zeros(len(ret)); g = defaultdict(list)
    for i,k in enumerate(keys): g[k].append(i)
    for k, idx in g.items():
        idx = np.array(idx); v = ret[idx].mean()
        for a in (0,1):
            sa = idx[act[idx]==a]
            if len(sa)>0: adv[sa] = ret[sa].mean() - v
    return adv

def greedy_cosine_cluster(emb, eps):
    """BiPACE式在线贪心聚类: 与最近质心cos距离<eps则并入, 否则新组 (同survival_test)"""
    cents, members = [], []
    for i, x in enumerate(emb):
        if cents:
            sims = np.array([c @ x for c in cents]); j = int(sims.argmax())
            if (1 - sims[j]) < eps:
                members[j].append(i)
                c = emb[members[j]].mean(0); cents[j] = c/(np.linalg.norm(c)+1e-9)
                continue
        cents.append(x.copy()); members.append([i])
    pred = np.empty(len(emb), int)
    for cid, m in enumerate(members):
        for idx in m: pred[idx] = cid
    return pred

def bipace_qv(E, act, ret, eps):
    """忠实BiPACE: 先贪心余弦聚类得到组, 再在组内算 Q̂(s,a)-V̂(s) (认不出→单点→优势0)"""
    return hard_qv(greedy_cosine_cluster(E, eps), act, ret)

def ca_acc(adv, isp1, act):
    m=isp1; a=act[m]; d=adv[m]
    good = (d>0).astype(float) + 0.5*(d==0)        # 优势说"该动作是好的"的程度
    corr = np.where(a==0, good, 1.0-good)          # A(0)真好→要正; B(1)真坏→要负
    return corr.mean()

# ---------------- 扫描U (多seed) ----------------
METHODS = ["GRPO", "GiGPO", "BiPACE", "Oracle"]
# per-seed曲线: allres[method] = [ [U0..U4](seed0), [U0..U4](seed1), ... ]
allres = {m: [] for m in METHODS}
for sd in SEEDS:
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)
    row = {m: [] for m in METHODS}
    for U in U_LEVELS:
        obs, label, isp1, act, ret = build(U)
        E = embed(obs)
        row["GRPO"].append(  ca_acc(ret - ret.mean(),       isp1, act))  # episode级
        row["GiGPO"].append( ca_acc(hard_qv(obs,   act, ret), isp1, act))  # 精确观测分组
        row["BiPACE"].append(ca_acc(bipace_qv(E,   act, ret, EPS), isp1, act))  # 忠实BiPACE贪心聚类
        row["Oracle"].append(ca_acc(hard_qv(label, act, ret), isp1, act))  # 真值处境分组(上界)
        print(f"seed{sd} U={U:.2f} | Oracle={row['Oracle'][-1]:.3f} BiPACE={row['BiPACE'][-1]:.3f} "
              f"GiGPO={row['GiGPO'][-1]:.3f} GRPO={row['GRPO'][-1]:.3f}")
    for m in METHODS: allres[m].append(row[m])

n = len(SEEDS)
mean = {m: np.array(allres[m]).mean(0)         for m in METHODS}   # [n_U]
std  = {m: np.array(allres[m]).std(0, ddof=1)  for m in METHODS}   # 样本标准差(seed间散布)
sem  = {m: std[m]/np.sqrt(n)                    for m in METHODS}   # 标准误(均值的不确定度)
print(f"\n===== 均值±标准误SEM (over {n} seeds) =====")
for i, U in enumerate(U_LEVELS):
    print(f"U={U:.2f} | " + "  ".join(
        f"{m}={mean[m][i]:.3f}±{sem[m][i]:.3f}" for m in METHODS))

# ---------------- 画图 (误差棒=SEM) ----------------
U = U_LEVELS
plt.figure(figsize=(6.2,4.6))
style = {"Oracle":("o-","Oracle (true grouping, ceiling)"),
         "BiPACE":("s-","BiPACE (greedy cosine cluster)"),
         "GiGPO":("^--","GiGPO (exact hash)"),
         "GRPO":("x--","GRPO (episode)")}
for m,(fmt,lab) in style.items():
    plt.errorbar(U, mean[m], yerr=sem[m], fmt=fmt, capsize=3, label=lab)
plt.axhline(0.5, ls=":", c="gray"); plt.ylim(0.4,1.03)
plt.xlabel("uniqueness U"); plt.ylabel("CA-Acc (credit sign correct at pivotal step)")
plt.title(f"Does representation degradation HARM credit?  (mean±SEM, {n} seeds)\nBiPACE below Oracle = harm = topic ALIVE")
plt.legend(); plt.tight_layout(); plt.savefig("credit_result.png", dpi=140)
print("\n图已存 credit_result.png")

# ---------------- 裁决 (SEM显著性 + 整体下滑) ----------------
b = mean["BiPACE"]; o = mean["Oracle"]
gap1 = o[-1] - b[-1]                                   # U=1处的差距
gap1_sem = np.hypot(sem["Oracle"][-1], sem["BiPACE"][-1])  # 差距的标准误
drop = b[0] - b[-1]                                    # BiPACE从U=0到U=1掉了多少
diffs = np.diff(b); monotone = (diffs <= 0.05).mean() # 越接近1越单调下滑
print("\n===== 裁决 =====")
print(f"BiPACE曲线: {np.array2string(b, precision=3)}")
print(f"U=1: Oracle={o[-1]:.3f} BiPACE={b[-1]:.3f} gap={gap1:.3f}  (2×SEM阈={2*gap1_sem:.3f})")
print(f"整体下滑={drop:.3f}  单调度={monotone:.2f}  seed间散布std={std['BiPACE'][-1]:.3f}")
if gap1 > 2*gap1_sem and drop > 0.2:
    print("→ 【题活】: 平均效应显著(gap>2SEM) 且 BiPACE随U明显下滑. 可拿去汇报/进Tier-2")
    print("   注: 若seed间散布std仍很大, 汇报时如实说'效应稳健但单次波动大'")
elif gap1 <= gap1_sem:
    print("→ 【题危】: gap被标准误吃掉, BiPACE≈Oracle. 认真考虑换题")
else:
    print("→ 中间地带: 有方向但没到2SEM. 再加seed(seed才是采样单位)或换更难处境确认")
