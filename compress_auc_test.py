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

import random, string, sys
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

MODEL="/home/wuyi/cuda12-dev/project/models/Qwen2.5-7B-Instruct"; LAYER=-8; BATCH=8; MAXLEN=384
MODE = sys.argv[1] if len(sys.argv)>1 else "cheap"   # 用法: python compress_auc_test.py [cheap|llm]
U_FIX=1.0                       # 满观测(每步唯一)下测压缩
C_LEVELS=[0.0,0.25,0.5,0.75,1.0]
M_CANON=10
N_VAR = 8 if MODE=="llm" else 30     # llm每条要生成摘要, 自动减量
MAX_DISTRACTORS=24
SEEDS = [0] if MODE=="llm" else list(range(5))
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

PROMPT_TMPL="你是一个网页/桌面操作智能体。当前界面观测如下：\n{obs}\n\n请判断此刻应采取的操作。操作："

@torch.no_grad()
def embed(texts, pool):
    """pool='lasttok' = BiPACE忠实(套agent prompt, 取final prompt token=决策位, L2归一)
       pool='mean'    = 对内容token做mask均值(不套模板). 两种对照, 揭示表示敏感性."""
    vecs=[]
    for i in range(0,len(texts),BATCH):
        if pool=="lasttok":
            batch=[PROMPT_TMPL.format(obs=o) for o in texts[i:i+BATCH]]
        else:
            batch=list(texts[i:i+BATCH])
        enc=tok(batch,return_tensors="pt",padding=True,truncation=True,max_length=MAXLEN).to(device)
        h=model(**enc,output_hidden_states=True).hidden_states[LAYER]
        if pool=="lasttok":
            pooled=h[:,-1,:]                                    # 左pad → 末位=最后一个真token=决策位
        else:
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
POOLS=["lasttok","mean"]                         # lasttok=BiPACE忠实; mean=对照
res_raw={p:[] for p in POOLS}; res_comp={p:{C:[] for C in Cs} for p in POOLS}
for sd in SEEDS:
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)
    obs_raw,lab=make_dataset_raw(U_FIX)
    comp_sets={C:make_dataset_comp(C) for C in Cs}          # 文本只生成一次(llm摘要贵)
    for p in POOLS:
        res_raw[p].append(sep_auc(embed(obs_raw,p),lab))
        for C in Cs:
            obs_c,lab_c=comp_sets[C]
            res_comp[p][C].append(sep_auc(embed(obs_c,p),lab_c))
    print(f"seed{sd} done")

n=len(SEEDS)
mm=lambda x: float(np.nanmean(x)); se=lambda x: float(np.nanstd(x,ddof=1)/np.sqrt(n)) if n>1 else float("nan")
print(f"\n===== 均值±SEM (over {n} seeds), 两种池化对照 =====")
for p in POOLS:
    tagp="BiPACE忠实(last-token)" if p=="lasttok" else "对照(mean池化)"
    print(f"\n[{tagp}]  AUC_raw={mm(res_raw[p]):.3f}±{se(res_raw[p]):.3f}")
    for C in Cs:
        tag=f"C={C:.2f}" if C is not None else "LLM摘要"
        print(f"  AUC_comp {tag} = {mm(res_comp[p][C]):.3f}±{se(res_comp[p][C]):.3f}")

if MODE=="cheap":
    plt.figure(figsize=(6.6,4.7))
    for p,fmt,col in [("lasttok","s-","tab:blue"),("mean","o-","tab:green")]:
        yy=[mm(res_comp[p][C]) for C in Cs]; ee=[se(res_comp[p][C]) for C in Cs]
        nm="last-token(BiPACE)" if p=="lasttok" else "mean-pool"
        plt.errorbar(C_LEVELS,yy,yerr=ee,fmt=fmt,color=col,capsize=3,label=f"AUC_comp [{nm}]")
        plt.axhline(mm(res_raw[p]),ls="--",color=col,alpha=.5,label=f"AUC_raw [{nm}]={mm(res_raw[p]):.2f}")
    plt.axhline(0.5,ls=":",c="gray"); plt.ylim(0.45,1.03)
    plt.xlabel("compression rate C"); plt.ylabel("separability AUC")
    plt.title("Representation sensitivity on compressed memory:\nBiPACE last-token vs mean-pool")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig("compress_auc_result.png",dpi=140)
    print("\n图已存 compress_auc_result.png")

print("\n===== 裁决 (最压处, 两种池化对照) =====")
for p in POOLS:
    tagp="BiPACE忠实(last-token)" if p=="lasttok" else "mean池化"
    print(f"[{tagp}] AUC_raw={mm(res_raw[p]):.3f}  AUC_comp(最压)={mm(res_comp[p][Cs[-1]]):.3f}")
print("判读: last-token压缩后≈0.5(坍缩)而mean明显更高 → 信息没丢、是BiPACE表示看不见 → 支持(b)鲁棒度量;")
print("      两种池化压缩后都高 → 压缩确实恢复可分性; 都低 → 信息真丢(memory瓶颈, 偏(a))。")
