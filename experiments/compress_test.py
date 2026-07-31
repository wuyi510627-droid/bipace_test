# compress_test.py —— Tier1.5: 压缩memory 到底是不是瓶颈? (回答导师那句话)
# 固定 U=1(每步观测都唯一, 最真实), 扫"压缩率 C"这条轴。
# 关键三条线:
#   Oracle    = 用真值处境分组         → 绝对天花板(恒=1)
#   CompCeil  = 用"压缩后文本"精确分组 → 压缩后信息还在不在(信息天花板)
#   BiPACE_comp = 压缩摘要的表示上软分组 → 你的方法实际做到多少
#   BiPACE_raw  = 原始满观测上软分组(BiPACE今天的做法) → 参照横线
# 裁决:
#   CompCeil 随C塌  → 信息被压没了, 瓶颈在memory, 分组救不了 → 【题危】
#   CompCeil 稳住但 BiPACE_comp 掉下去 → 信息还在只是认不出 → 【题活】(你补的空间)
#
# 两种压缩(见 MODE):
#   "cheap" : 字符串层按概率C把区分性细节换成通用词(笔记本/手机→某件商品). 秒级, 扫全C.
#   "llm"   : 用同一个Qwen真生成一句话摘要(=verl-agent做法). 真实但慢, 只跑单点小规模验证.
# 运行: python compress_test.py

import random, string
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

# ---------------- 配置 ----------------
MODEL   = "/home/wuyi/cuda12-dev/project/models/Qwen2.5-7B-Instruct"
USE_4BIT = True
LAYER   = -8
MODE    = "cheap"          # "cheap"=廉价抹细节扫全C ; "llm"=真Qwen摘要单点验证
PROMPT_TMPL = "你是一个网页/桌面操作智能体。当前界面观测如下：\n{obs}\n\n请判断此刻应采取的操作。操作："
SUMM_TMPL   = "用一句话概括下面界面此刻要完成的操作，忽略广告、时间戳、导航等无关信息：\n{obs}\n一句话概括："
U_FIX   = 1.0             # 固定在"每步观测都唯一"(最真实的开放环境)
C_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]   # 压缩率(cheap模式用; llm模式忽略)
MAX_DISTRACTORS = 24
N_TRAJ = 400             # cheap用400; 换MODE="llm"请改小到~60(每步要额外生成)
T_STEP = 6
EPS = 0.15               # BiPACE贪心余弦聚类阈值(同survival/credit)
BATCH = 8
BATCH_GEN = 8
MAXLEN = 384
SEEDS = list(range(6))   # cheap模式6个seed; llm模式请改成 [0]
device = "cuda" if torch.cuda.is_available() else "cpu"

# 10个处境: (带细节的核心, 压掉细节后的通用核心). index0=P1, index1=P2 是易混对(同为"加购物车")
CANON = [
    ("在商品列表页, 把【笔记本电脑】加入购物车。", "在商品列表页, 把【某件商品】加入购物车。"),  # 0 P1
    ("在商品列表页, 把【手机】加入购物车。",       "在商品列表页, 把【某件商品】加入购物车。"),  # 1 P2
    ("在商品列表页, 把【耳机】加入购物车。",       "在商品列表页, 把【某件商品】加入购物车。"),
    ("在搜索结果页, 点开【第1个】结果。",          "在搜索结果页, 点开【某个】结果。"),
    ("在搜索结果页, 点开【第3个】结果。",          "在搜索结果页, 点开【某个】结果。"),
    ("在表单页, 在【邮箱】字段填写。",             "在表单页, 在【某个】字段填写。"),
    ("在表单页, 在【电话】字段填写。",             "在表单页, 在【某个】字段填写。"),
    ("在设置页, 打开【深色模式】开关。",           "在设置页, 打开【某个】开关。"),
    ("在设置页, 打开【通知】开关。",               "在设置页, 打开【某个】开关。"),
    ("在文件页, 进入【Downloads】文件夹。",        "在文件页, 进入【某个】文件夹。"),
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
def make_obs(core, U):
    k = int(round(U * MAX_DISTRACTORS))
    parts = [core] + [_distractor() for _ in range(k)]
    random.shuffle(parts)
    return "\n".join(parts)

# 压缩=摘要: 去掉所有干扰, 只留核心; 再按概率C把区分性细节也压掉
def compress_cheap(lb, C):
    detailed, generic = CANON[lb]
    return generic if random.random() < C else detailed

# ---------------- 造轨迹(带关键步P1 + 混淆步P2) ----------------
def build():
    label, isp1, act, ret = [], [], [], []
    for _ in range(N_TRAJ):
        p1_pos, p2_pos = random.sample(range(T_STEP), 2)
        p1a, p2a = random.randint(0,1), random.randint(0,1)
        # 关键: P1正确动作=0, P2正确动作=1(相反). 压缩把P1/P2混成一组时两者抵消→信用错, 才测得出伤害
        R = 0.5*(p1a==0) + 0.5*(p2a==1)
        for t in range(T_STEP):
            if t == p1_pos:   lb, ip, a = 0, True,  p1a
            elif t == p2_pos: lb, ip, a = 1, False, p2a
            else:             lb, ip, a = random.randint(2,len(CANON)-1), False, random.randint(0,1)
            label.append(lb); isp1.append(ip); act.append(a); ret.append(R)
    return np.array(label), np.array(isp1), np.array(act), np.array(ret, float)

# ---------------- 加载模型(一个CausalLM同时供 抽表示 + 生成摘要) ----------------
print(f"加载模型 {MODEL} (4bit={USE_4BIT}, MODE={MODE}) ...")
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"; tok.truncation_side = "left"
if USE_4BIT and device == "cuda":
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map={"":0}).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL,
        torch_dtype=torch.float16 if device=="cuda" else torch.float32).to(device).eval()

@torch.no_grad()
def embed(texts):
    vecs=[]
    for i in range(0, len(texts), BATCH):
        prompts = [PROMPT_TMPL.format(obs=o) for o in texts[i:i+BATCH]]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to(device)
        h = model(**enc, output_hidden_states=True).hidden_states[LAYER]
        pooled = torch.nn.functional.normalize(h[:, -1, :].float(), dim=-1)
        vecs.append(pooled.cpu().numpy())
    return np.concatenate(vecs, 0)

@torch.no_grad()
def compress_llm(raw_list):
    outs=[]
    for i in range(0, len(raw_list), BATCH_GEN):
        prompts = [SUMM_TMPL.format(obs=o) for o in raw_list[i:i+BATCH_GEN]]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to(device)
        gen = model.generate(**enc, max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id)
        for j in range(gen.shape[0]):
            new = gen[j][enc.input_ids.shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True).strip().replace("\n"," "))
    return outs

# ---------------- 4种信用(优势) ----------------
def hard_qv(keys, act, ret):
    adv = np.zeros(len(ret)); g = defaultdict(list)
    for i,k in enumerate(keys): g[k].append(i)
    for k, idx in g.items():
        idx = np.array(idx); v = ret[idx].mean()
        for a in (0,1):
            sa = idx[act[idx]==a]
            if len(sa)>0: adv[sa] = ret[sa].mean() - v
    return adv

def greedy_cosine_cluster(emb, eps):
    """BiPACE式在线贪心余弦聚类(同survival_test/credit_test)"""
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
    """忠实BiPACE: 贪心余弦聚类得到组, 再组内算 Q̂(s,a)-V̂(s)"""
    return hard_qv(greedy_cosine_cluster(E, eps), act, ret)

def ca_acc(adv, isp1, act):
    m=isp1; a=act[m]; d=adv[m]
    good = (d>0).astype(float) + 0.5*(d==0)
    corr = np.where(a==0, good, 1.0-good)
    return corr.mean()

# ---------------- 扫描 (多seed) ----------------
METHODS = ["Oracle", "CompCeil", "BiPACE_comp", "BiPACE_raw"]
Cs = C_LEVELS if MODE=="cheap" else [None]     # llm模式退化为单点
allres = {m: [] for m in METHODS}
for sd in SEEDS:
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)
    label, isp1, act, ret = build()
    raw_obs = [make_obs(CANON[lb][0], U_FIX) for lb in label]   # 满观测(BiPACE_raw用, 与C无关)
    E_raw = embed(raw_obs)
    bipace_raw = ca_acc(bipace_qv(E_raw, act, ret, EPS), isp1, act)
    oracle     = ca_acc(hard_qv(label, act, ret), isp1, act)
    row = {m: [] for m in METHODS}
    for C in Cs:
        if MODE=="cheap":
            comp_text = [compress_cheap(lb, C) for lb in label]
        else:
            comp_text = compress_llm(raw_obs)                    # 真摘要, C忽略
        E_comp = embed(comp_text)
        row["Oracle"].append(oracle)
        row["CompCeil"].append(ca_acc(hard_qv(comp_text, act, ret), isp1, act))   # 压缩文本精确分组=信息天花板
        row["BiPACE_comp"].append(ca_acc(bipace_qv(E_comp, act, ret, EPS), isp1, act))
        row["BiPACE_raw"].append(bipace_raw)
        tag = f"C={C:.2f}" if C is not None else "LLM摘要"
        print(f"seed{sd} {tag} | Oracle={row['Oracle'][-1]:.3f} CompCeil={row['CompCeil'][-1]:.3f} "
              f"BiPACE_comp={row['BiPACE_comp'][-1]:.3f} BiPACE_raw={row['BiPACE_raw'][-1]:.3f}")
    for m in METHODS: allres[m].append(row[m])

n = len(SEEDS)
mean = {m: np.array(allres[m]).mean(0)        for m in METHODS}
std  = {m: np.array(allres[m]).std(0, ddof=1) for m in METHODS}
sem  = {m: std[m]/np.sqrt(n)                   for m in METHODS}
print(f"\n===== 均值±标准误SEM (over {n} seeds) =====")
for i, C in enumerate(Cs):
    tag = f"C={C:.2f}" if C is not None else "LLM摘要"
    print(f"{tag} | " + "  ".join(f"{m}={mean[m][i]:.3f}±{sem[m][i]:.3f}" for m in METHODS))

# ---------------- 画图(cheap才有曲线) ----------------
if MODE=="cheap":
    plt.figure(figsize=(6.2,4.6))
    sty = {"Oracle":("o-","Oracle (true grouping, ceiling)"),
           "CompCeil":("D-","CompCeil (info left after compression)"),
           "BiPACE_comp":("s-","BiPACE on compressed memory (yours)"),
           "BiPACE_raw":("x--","BiPACE on raw obs (today)")}
    for m,(fmt,lab) in sty.items():
        plt.errorbar(C_LEVELS, mean[m], yerr=sem[m], fmt=fmt, capsize=3, label=lab)
    plt.axhline(0.5, ls=":", c="gray"); plt.ylim(0.4,1.03)
    plt.xlabel("compression rate C (detail dropped)"); plt.ylabel("CA-Acc at pivotal step")
    plt.title("Is memory compression the bottleneck?\nCompCeil drops=info gone(题危); CompCeil holds & BiPACE_comp drops=题活")
    plt.legend(); plt.tight_layout(); plt.savefig("compress_result.png", dpi=140)
    print("\n图已存 compress_result.png")

# ---------------- 裁决 ----------------
cc = mean["CompCeil"][-1]; bc = mean["BiPACE_comp"][-1]      # C=1(或LLM摘要)处
cc_err = sem["CompCeil"][-1]; bc_err = sem["BiPACE_comp"][-1]
print("\n===== 裁决 (看最压处) =====")
print(f"CompCeil={cc:.3f}±{cc_err:.3f}  BiPACE_comp={bc:.3f}±{bc_err:.3f}  (BiPACE_raw={mean['BiPACE_raw'][-1]:.3f})")
gap = cc - bc
if cc < 0.6:
    print("→ 【信息被压没了】: 压缩后连'精确分组'都救不回来(CompCeil塌).")
    print("  瓶颈在 memory 模块本身, 改分组救不了场. 【题危/需重定位】")
elif gap > 2*np.hypot(cc_err,bc_err) and gap > 0.15:
    print("→ 【信息还在, 但分组认不出】: CompCeil稳住而BiPACE_comp明显更低.")
    print("  压缩空间里的软分组有真实退化 → gap就是你方法要补的空间. 【题活】")
else:
    print("→ 中间地带: 压缩没造成明显分组损害(BiPACE_comp≈CompCeil).")
    print("  说明现成表示已够用, 你相对BiPACE的增量存疑. 谨慎, 或加大压缩难度再测.")
