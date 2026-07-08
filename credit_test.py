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
N_TRAJ = 80          # 轨迹数
T_STEP = 6           # 每条轨迹步数
TEMP = 0.5           # BiPACE软分组的温度(对z标准化后的相似度)
BATCH = 8
MAXLEN = 384
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
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

def soft_qv(E, act, ret, temp):
    """BiPACE: 用表示相似度做软分组(对z标准化后的cos), 优势=Q̂(s,a)-V̂(s)"""
    S = E @ E.T
    mu = S.mean(1,keepdims=True); sd = S.std(1,keepdims=True)+1e-6
    Z = (S-mu)/sd; np.fill_diagonal(Z, -1e9); Z/=temp; Z -= Z.max(1,keepdims=True)
    W = np.exp(Z); np.fill_diagonal(W, 0.0); W /= W.sum(1,keepdims=True)+1e-12
    V = W @ ret
    same = (act[None,:]==act[:,None]).astype(float)
    Ws = W*same
    Q = (Ws@ret)/(Ws.sum(1)+1e-12)
    return Q - V

def ca_acc(adv, isp1, act):
    m=isp1; a=act[m]; d=adv[m]
    good = (d>0).astype(float) + 0.5*(d==0)        # 优势说"该动作是好的"的程度
    corr = np.where(a==0, good, 1.0-good)          # A(0)真好→要正; B(1)真坏→要负
    return corr.mean()

# ---------------- 扫描U ----------------
res = {"U":[], "GRPO":[], "GiGPO":[], "BiPACE":[], "Oracle":[]}
for U in U_LEVELS:
    obs, label, isp1, act, ret = build(U)
    E = embed(obs)
    adv_grpo   = ret - ret.mean()                          # episode级
    adv_gigpo  = hard_qv(obs,   act, ret)                  # 精确观测分组
    adv_bipace = soft_qv(E,     act, ret, TEMP)            # 表示软分组
    adv_oracle = hard_qv(label, act, ret)                  # 真值处境分组(上界)
    res["U"].append(U)
    res["GRPO"].append(ca_acc(adv_grpo, isp1, act))
    res["GiGPO"].append(ca_acc(adv_gigpo, isp1, act))
    res["BiPACE"].append(ca_acc(adv_bipace, isp1, act))
    res["Oracle"].append(ca_acc(adv_oracle, isp1, act))
    print(f"U={U:.2f} | CA-Acc: Oracle={res['Oracle'][-1]:.3f} BiPACE={res['BiPACE'][-1]:.3f} "
          f"GiGPO={res['GiGPO'][-1]:.3f} GRPO={res['GRPO'][-1]:.3f}")

# ---------------- 画图 ----------------
U = res["U"]
plt.figure(figsize=(6,4.5))
plt.plot(U, res["Oracle"], "o-", label="Oracle (true grouping, ceiling)")
plt.plot(U, res["BiPACE"], "s-", label="BiPACE (repr. grouping)")
plt.plot(U, res["GiGPO"],  "^--", label="GiGPO (exact hash)")
plt.plot(U, res["GRPO"],   "x--", label="GRPO (episode)")
plt.axhline(0.5, ls=":", c="gray"); plt.ylim(0.4,1.03)
plt.xlabel("uniqueness U"); plt.ylabel("CA-Acc (credit sign correct at pivotal step)")
plt.title("Does representation degradation HARM credit?\nBiPACE drops below Oracle = harm = topic ALIVE")
plt.legend(); plt.tight_layout(); plt.savefig("credit_result.png", dpi=140)
print("\n图已存 credit_result.png")

# ---------------- 裁决 ----------------
o1, b1, b0 = res["Oracle"][-1], res["BiPACE"][-1], res["BiPACE"][0]
gap = o1 - b1
print("\n===== 裁决 (看U=1) =====")
print(f"Oracle={o1:.3f}  BiPACE={b1:.3f}  gap={gap:.3f}  (BiPACE U=0时={b0:.3f})")
if gap > 0.15 and b1 < b0 - 0.1:
    print("→ 表示降级【真的害了信用】: BiPACE明显低于Oracle且随U下滑")
    print("  【题活】gap就是你方法要补的空间")
elif gap < 0.08:
    print("→ 降级【没害到信用】: BiPACE≈Oracle. 【题危】认真考虑换题")
else:
    print("→ 中间地带: 有一定差距但不大, 需加大N/换更难处境再确认, 或谨慎推进")
