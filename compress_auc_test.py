# compress_auc_test.py —— 更可信版: 用 survival_test 验证过的 (mean池化 + 阈值无关AUC),
#   问压缩轴一个干净问题: 压缩后, 表示还能不能把"不同处境"分开?
# 抛弃了脆弱的 (聚类+CA-Acc信用符号); 只保留低方差、无阈值、被验证过的可分性AUC.
#
# 关键对照: AUC_raw(满观测,BiPACE今天) vs AUC_comp(压缩后) 随 C:
#   AUC_comp 明显 < AUC_raw → 压缩让分组更难 → 有空间 → 【题活】
#   AUC_comp ≥ AUC_raw     → 压缩反而更好分(去了干扰) → cos已够用 → 【题危】
#
# MODE: "cheap"=按概率C抹掉区分细节(扫全C, 秒级) ; "llm"=真Qwen一句话摘要(单点, 慢, 改小N_VAR/SEEDS)
# 运行: python compress_auc_test.py

import random, string
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

MODEL="/home/wuyi/cuda12-dev/project/models/Qwen2.5-7B-Instruct"; LAYER=-8; BATCH=8; MAXLEN=384
MODE="cheap"                    # "cheap" 扫全C ; "llm" 真摘要单点
U_FIX=1.0                       # 满观测(每步唯一)下测压缩
C_LEVELS=[0.0,0.25,0.5,0.75,1.0]
M_CANON=10; N_VAR=30            # 10个处境, 每个30个表面变体; llm模式请改小(如8)
MAX_DISTRACTORS=24
SEEDS=list(range(5))            # llm模式请改成 [0]
SUMM_TMPL="用一句话概括下面界面此刻要完成的操作，忽略广告、时间戳、导航等无关信息：\n{obs}\n一句话概括："
device="cuda" if torch.cuda.is_available() else "cpu"

CANON=[("在商品列表页, 把【笔记本电脑】加入购物车。","在商品列表页, 把【某件商品】加入购物车。"),
       ("在商品列表页, 把【手机】加入购物车。","在商品列表页, 把【某件商品】加入购物车。"),
       ("在商品列表页, 把【耳机】加入购物车。","在商品列表页, 把【某件商品】加入购物车。"),
       ("在搜索结果页, 点开【第1个】结果。","在搜索结果页, 点开【某个】结果。"),
       ("在搜索结果页, 点开【第3个】结果。","在搜索结果页, 点开【某个】结果。"),
       ("在表单页, 在【邮箱】字段填写。","在表单页, 在【某个】字段填写。"),
       ("在表单页, 在【电话】字段填写。","在表单页, 在【某个】字段填写。"),
       ("在设置页, 打开【深色模式】开关。","在设置页, 打开【某个】开关。"),
       ("在设置页, 打开【通知】开关。","在设置页, 打开【某个】开关。"),
       ("在文件页, 进入【Downloads】文件夹。","在文件页, 进入【某个】文件夹。")]
DISTRACT=["广告: {w}季大促, 立减{n}元!","导航: 首页 > {w} > {w}","页脚版权 (c) {n} {w}公司",
          "推荐商品: {w} {w} {w}","时间戳: 2026-07-0{d} {n}:{n}","session_id={tok}",
          "用户{tok}最近浏览: {w}","cookie横幅: 本站使用{w}以改善体验","热搜: {w}, {w}, {w}"]
WORDS=["蓝牙","促销","数码","家居","春装","会员","限时","旗舰","优选","清仓"]
def _rtok(k=6): return "".join(random.choices(string.ascii_lowercase+string.digits,k=k))
def _distractor():
    return random.choice(DISTRACT).format(w=random.choice(WORDS),n=random.randint(1,99),d=random.randint(1,9),tok=_rtok())
def make_obs(core,U):
    k=int(round(U*MAX_DISTRACTORS)); parts=[core]+[_distractor() for _ in range(k)]
    random.shuffle(parts); return "\n".join(parts)
def compress_cheap(c_idx,C):
    detailed,generic=CANON[c_idx]; return generic if random.random()<C else detailed

print(f"加载模型 {MODEL} (MODE={MODE}) ...")
tok=AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
tok.padding_side="left"; tok.truncation_side="left"
bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16)
model=AutoModelForCausalLM.from_pretrained(MODEL,quantization_config=bnb,device_map={"":0}).eval()

@torch.no_grad()
def embed(texts):
    """mean池化(对内容token做mask均值)=BiPACE的 f(s), 归一化. 不套prompt模板, 避免脚手架淹没短文本"""
    vecs=[]
    for i in range(0,len(texts),BATCH):
        enc=tok(texts[i:i+BATCH],return_tensors="pt",padding=True,truncation=True,max_length=MAXLEN).to(device)
        h=model(**enc,output_hidden_states=True).hidden_states[LAYER]
        mask=enc.attention_mask.unsqueeze(-1).float()
        pooled=(h*mask).sum(1)/mask.sum(1).clamp(min=1)
        vecs.append(torch.nn.functional.normalize(pooled.float(),dim=-1).cpu().numpy())
    return np.concatenate(vecs,0)

@torch.no_grad()
def compress_llm(raw_list):
    outs=[]
    for i in range(0,len(raw_list),BATCH):
        prompts=[SUMM_TMPL.format(obs=o) for o in raw_list[i:i+BATCH]]
        enc=tok(prompts,return_tensors="pt",padding=True,truncation=True,max_length=MAXLEN).to(device)
        gen=model.generate(**enc,max_new_tokens=32,do_sample=False,pad_token_id=tok.pad_token_id)
        for j in range(gen.shape[0]):
            outs.append(tok.decode(gen[j][enc.input_ids.shape[1]:],skip_special_tokens=True).strip().replace("\n"," "))
    return outs

def sep_auc(emb,labels):
    """阈值无关: cos相似度能否预测'是否同一处境'. 1=干净可分, 0.5=分不开"""
    S=emb@emb.T; iu=np.triu_indices(len(emb),k=1)
    same=(labels[iu[0]]==labels[iu[1]]).astype(int)
    if same.min()==same.max(): return float("nan")
    return roc_auc_score(same,S[iu])

def make_dataset_raw(U):
    obs,lab=[],[]
    for c in range(M_CANON):
        for _ in range(N_VAR): obs.append(make_obs(CANON[c][0],U)); lab.append(c)
    return obs,np.array(lab)
def make_dataset_comp(C):
    obs,lab,raws=[],[],[]
    for c in range(M_CANON):
        for _ in range(N_VAR):
            raw=make_obs(CANON[c][0],U_FIX); raws.append(raw); lab.append(c)
            obs.append(compress_cheap(c,C) if MODE=="cheap" else None)
    if MODE=="llm": obs=compress_llm(raws)
    return obs,np.array(lab)

Cs=C_LEVELS if MODE=="cheap" else [None]
res_raw=[]; res_comp={C:[] for C in Cs}
for sd in SEEDS:
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)
    obs_raw,lab=make_dataset_raw(U_FIX)
    res_raw.append(sep_auc(embed(obs_raw),lab))
    for C in Cs:
        obs_c,lab_c=make_dataset_comp(C)
        auc=sep_auc(embed(obs_c),lab_c); res_comp[C].append(auc)
        tag=f"C={C:.2f}" if C is not None else "LLM摘要"
        print(f"seed{sd} {tag} | AUC_comp={auc:.3f}  (AUC_raw={res_raw[-1]:.3f})")

n=len(SEEDS)
raw_m=np.nanmean(res_raw); raw_s=np.nanstd(res_raw,ddof=1)/np.sqrt(n)
comp_m={C:np.nanmean(res_comp[C]) for C in Cs}; comp_s={C:np.nanstd(res_comp[C],ddof=1)/np.sqrt(n) for C in Cs}
print(f"\n===== 均值±SEM (over {n} seeds) =====")
print(f"AUC_raw(满观测,BiPACE今天) = {raw_m:.3f}±{raw_s:.3f}")
for C in Cs:
    tag=f"C={C:.2f}" if C is not None else "LLM摘要"
    print(f"AUC_comp {tag} = {comp_m[C]:.3f}±{comp_s[C]:.3f}")

if MODE=="cheap":
    plt.figure(figsize=(6.2,4.6))
    plt.errorbar(C_LEVELS,[comp_m[C] for C in Cs],yerr=[comp_s[C] for C in Cs],fmt="s-",capsize=3,label="AUC_comp (compressed memory)")
    plt.axhline(raw_m,ls="--",c="tab:red",label=f"AUC_raw (raw obs, BiPACE today)={raw_m:.2f}")
    plt.axhline(0.5,ls=":",c="gray"); plt.ylim(0.45,1.03)
    plt.xlabel("compression rate C"); plt.ylabel("separability AUC (same vs diff situation)")
    plt.title("Can representation still separate situations after compression?\nAUC_comp < AUC_raw = compression hurts grouping = topic ALIVE")
    plt.legend(); plt.tight_layout(); plt.savefig("compress_auc_result.png",dpi=140); print("\n图已存 compress_auc_result.png")

c_last=comp_m[Cs[-1]]; gap=raw_m-c_last; gap_sem=np.hypot(raw_s,comp_s[Cs[-1]])
print("\n===== 裁决 (看最压处) =====")
print(f"AUC_raw={raw_m:.3f}  AUC_comp(最压)={c_last:.3f}  gap={gap:.3f}  (2SEM阈={2*gap_sem:.3f})")
if gap>2*gap_sem and gap>0.05:
    print("→ 【题活】: 压缩后可分性显著低于原始观测, 压缩让分组更难, 有你补的空间")
elif c_last>=raw_m-gap_sem:
    print("→ 【题危】: 压缩没让分组变难(甚至更好分), cos表示已够用, 你相对BiPACE增量存疑")
else:
    print("→ 中间地带: 有方向但没到2SEM, 加seed或用llm模式验证")
