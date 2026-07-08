# survival_test.py —— BiPACE 生死线测试（3070 可跑，只推理不训练）
# 问题: 当观测"表面越来越唯一"时, LLM表示还能不能把"其实同一处境"的认到一起?
#   能  -> BiPACE已解决, 你的题危
#   不能-> BiPACE也失败, 你的题活
#
# 依赖: pip install torch transformers scikit-learn numpy matplotlib
# 运行: python survival_test.py

import random, string, itertools
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, adjusted_rand_score
import matplotlib.pyplot as plt

# ---------------- 配置 ----------------
MODEL   = "/home/wuyi/cuda12-dev/project/models/Qwen2.5-7B-Instruct" # 12G显存: 7B开4bit能跑(最贴近BiPACE的7B); 嫌麻烦用 "Qwen/Qwen2.5-3B-Instruct"(fp16直接跑)
USE_4BIT = True                 # 7B必开4bit省显存; 若用3B/1.5B可设False走fp16
LAYER   = -8                    # BiPACE的7B用倒数第8层; 若换1.5B改成 -12
EPS     = 0.15                  # BiPACE式贪心聚类的cosine距离阈值(需按下面打印的分布微调)
M_CANON = 10                    # 10个"规范处境"(真值类别)
N_VAR   = 20                    # 每个处境20个变体
U_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]   # 唯一度旋钮
MAX_DISTRACTORS = 24            # U=1时注入的无关内容条数(制造真实表面唯一性)
BATCH = 8                       # 7B用8; 换3B/1.5B可调回16
MAXLEN = 384
SEED = 0
PROMPT_TMPL = "你是一个网页/桌面操作智能体。当前界面观测如下：\n{obs}\n\n请判断此刻应采取的操作。操作："
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- 1. 造测试题 ----------------
# 10个"任务处境"的核心语义(像web-agent的状态). 关键: 核心不变, 表面变.
CANON = [
    "在商品列表页, 把【笔记本电脑】加入购物车。",
    "在商品列表页, 把【手机】加入购物车。",
    "在商品列表页, 把【耳机】加入购物车。",
    "在搜索结果页, 点开【第1个】结果。",
    "在搜索结果页, 点开【第3个】结果。",
    "在表单页, 在【邮箱】字段填写。",
    "在表单页, 在【电话】字段填写。",
    "在设置页, 打开【深色模式】开关。",
    "在设置页, 打开【通知】开关。",
    "在文件页, 进入【Downloads】文件夹。",
]
# 无关"干扰内容"池(模拟真实网页里与任务无关、但每次都不同的表面信息)
DISTRACT_TEMPLATES = [
    "广告: {w}季大促, 立减{n}元!", "导航: 首页 > {w} > {w}", "页脚版权 (c) {n} {w}公司",
    "推荐商品: {w} {w} {w}", "时间戳: 2026-07-0{d} {n}:{n}", "session_id={tok}",
    "用户{tok}最近浏览: {w}", "cookie横幅: 本站使用{w}以改善体验", "热搜: {w}, {w}, {w}",
    "弹窗: 订阅{w}newsletter领取{n}优惠券", "面包屑: {w}/{w}/{w}", "评分: {n}星 共{n}条评论",
]
WORDS = ["蓝牙","促销","数码","家居","春装","会员","限时","旗舰","优选","清仓","国际","官方"]
def _rtok(k=6): return "".join(random.choices(string.ascii_lowercase+string.digits, k=k))
def _distractor():
    t = random.choice(DISTRACT_TEMPLATES)
    return t.format(w=random.choice(WORDS), n=random.randint(1,99), d=random.randint(1,9), tok=_rtok())

def make_obs(canon_text, U):
    """U=0: 就是核心句(各变体完全相同); U越大, 注入越多唯一的无关内容, 核心被淹没"""
    k = int(round(U * MAX_DISTRACTORS))
    parts = [canon_text] + [_distractor() for _ in range(k)]
    random.shuffle(parts)                 # 核心位置也随机, 更像真实页面
    return "\n".join(parts)

def build_dataset(U):
    obs, labels = [], []
    for c_idx, c_text in enumerate(CANON):
        for _ in range(N_VAR):
            obs.append(make_obs(c_text, U)); labels.append(c_idx)
    return obs, np.array(labels)

# ---------------- 2. 抽LLM表示(归一化), 即BiPACE的分组空间 ----------------
print(f"加载模型 {MODEL} (4bit={USE_4BIT}) ...")
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
tok.truncation_side = "left"   # 防止截断把结尾"操作："切掉
if USE_4BIT and device == "cuda":
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModel.from_pretrained(MODEL, quantization_config=bnb,
                                      device_map={"": 0}, output_hidden_states=True).eval()
else:
    model = AutoModel.from_pretrained(
        MODEL, torch_dtype=torch.float16 if device=="cuda" else torch.float32,
        output_hidden_states=True).to(device).eval()

@torch.no_grad()
def embed(obs_list):
    vecs = []
    for i in range(0, len(obs_list), BATCH):
        batch = obs_list[i:i+BATCH]
        prompts = [PROMPT_TMPL.format(obs=o) for o in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAXLEN).to(device)
        hs = model(**enc).hidden_states                 # tuple: 层数+1
        h = hs[LAYER]                                    # BiPACE用倒数第8层 [B,T,H]
        # mask = enc.attention_mask.unsqueeze(-1).float()  # [B,T,1]
        # pooled = (h*mask).sum(1) / mask.sum(1).clamp(min=1)  # 对token做mask均值池化 = f(s)
        pooled = h[:, -1, :]  # 简化: 不mask, 直接对token均值池化 = f(s)
        pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)  # phi=f/||f||
        vecs.append(pooled.cpu().numpy())
    return np.concatenate(vecs, 0)

# ---------------- 3. 两种分组 + 指标 ----------------
def greedy_cosine_cluster(emb, eps):
    """BiPACE式在线贪心聚类: 与最近质心cos距离<eps则并入, 否则新组"""
    cents, members = [], []
    for i, x in enumerate(emb):
        if cents:
            sims = np.array([c @ x for c in cents]); j = int(sims.argmax())
            if (1 - sims[j]) < eps:
                members[j].append(i)
                c = emb[members[j]].mean(0); cents[j] = c/ (np.linalg.norm(c)+1e-9)
                continue
        cents.append(x.copy()); members.append([i])
    pred = np.empty(len(emb), int)
    for cid, m in enumerate(members):
        for idx in m: pred[idx] = cid
    return pred, members

def exact_hash_cluster(obs):
    """GiGPO式: 观测字符串完全相同才一组(对照, 预期随U塌成单点)"""
    h2id, pred = {}, []
    for o in obs:
        pred.append(h2id.setdefault(o, len(h2id)))
    pred = np.array(pred)
    members = [list(np.where(pred==c)[0]) for c in range(pred.max()+1)]
    return pred, members

def group_stats(members, N):
    sizes = np.array([len(m) for m in members])
    eff = float((sizes**2).sum()/N)                     # 点加权平均组大小
    singleton_pts = int(sizes[sizes==1].sum())          # 落在单点组的样本数
    return eff, singleton_pts/N

def separability_auc(emb, labels):
    """★阈值无关的核心判据: cos相似度 能不能预测'是否同一处境'
       ~1: 表示能干净区分同/异 → BiPACE能work(题危)
       ~0.5: 分不开 → BiPACE必失败(题活)"""
    S = emb @ emb.T
    iu = np.triu_indices(len(emb), k=1)
    sims = S[iu]
    same = (labels[iu[0]] == labels[iu[1]]).astype(int)
    return roc_auc_score(same, sims)

# ---------------- 4. 扫描U, 收集曲线 ----------------
res = {"U":[], "AUC":[], "BiPACE_ARI":[], "BiPACE_eff":[], "BiPACE_singleton":[],
       "GiGPO_ARI":[], "GiGPO_singleton":[]}
for U in U_LEVELS:
    obs, labels = build_dataset(U)
    emb = embed(obs); N = len(obs)
    auc = separability_auc(emb, labels)
    bp_pred, bp_mem = greedy_cosine_cluster(emb, EPS)
    bp_eff, bp_sing = group_stats(bp_mem, N)
    gg_pred, gg_mem = exact_hash_cluster(obs)
    gg_eff, gg_sing = group_stats(gg_mem, N)
    res["U"].append(U); res["AUC"].append(auc)
    res["BiPACE_ARI"].append(adjusted_rand_score(labels, bp_pred))
    res["BiPACE_eff"].append(bp_eff); res["BiPACE_singleton"].append(bp_sing)
    res["GiGPO_ARI"].append(adjusted_rand_score(labels, gg_pred))
    res["GiGPO_singleton"].append(gg_sing)
    print(f"U={U:.2f} | 可分AUC={auc:.3f} | BiPACE: ARI={res['BiPACE_ARI'][-1]:.3f} "
          f"有效组={bp_eff:.1f} 单点率={bp_sing:.2f} | GiGPO单点率={gg_sing:.2f}")

# 提示: 若BiPACE有效组恒=1或恒=N, 说明EPS不合适, 按上面数值调EPS再跑

# ---------------- 5. 画图 ----------------
U = res["U"]
fig, ax = plt.subplots(1, 3, figsize=(15,4))
ax[0].plot(U, res["AUC"], "o-"); ax[0].axhline(0.5, ls="--", c="gray")
ax[0].set_title("(1) sep-AUC (threshold-free, key)\ndrop->0.5 = BiPACE fails = topic ALIVE"); ax[0].set_xlabel("uniqueness U"); ax[0].set_ylim(0.4,1.02)
ax[1].plot(U, res["BiPACE_ARI"], "o-", label="BiPACE"); ax[1].plot(U, res["GiGPO_ARI"], "s--", label="GiGPO")
ax[1].set_title("(2) grouping ARI\ndrop->0 = wrong groups"); ax[1].set_xlabel("uniqueness U"); ax[1].legend(); ax[1].set_ylim(-0.05,1.05)
ax[2].plot(U, res["BiPACE_singleton"], "o-", label="BiPACE"); ax[2].plot(U, res["GiGPO_singleton"], "s--", label="GiGPO")
ax[2].set_title("(3) singleton rate\nrise->1 = no groups = back to GRPO"); ax[2].set_xlabel("uniqueness U"); ax[2].legend(); ax[2].set_ylim(-0.05,1.05)
plt.tight_layout(); plt.savefig("survival_result.png", dpi=140); print("\n图已存 survival_result.png")

# ---------------- 6. 自动裁决 ----------------
auc1, ari1 = res["AUC"][-1], res["BiPACE_ARI"][-1]   # U=1处
print("\n===== 裁决 (看U=1) =====")
print(f"可分AUC={auc1:.3f}, BiPACE_ARI={ari1:.3f}")
if auc1 < 0.65 or ari1 < 0.2:
    print("→ BiPACE 崩了(认不出同处境). 【题活】: 可做方法(结局1)或诊断+基准(结局2)")
elif auc1 > 0.85 and ari1 > 0.5:
    print("→ BiPACE 扛住了(照样认得出). 【题危】: 它可能已解决, 考虑换题(结局3)")
else:
    print("→ 中间地带: 换更大模型(1.5B)/更真实的干扰再测一次, 别急着下结论")
