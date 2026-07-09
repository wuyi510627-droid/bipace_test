# cluster_probe.py —— 直接看真实embedding在 C=1 时的聚类结构, 找出 BiPACE_comp 为何=1.0
import random
from collections import defaultdict
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL="/home/wuyi/cuda12-dev/project/models/Qwen2.5-7B-Instruct"; LAYER=-8; EPS=0.15; BATCH=8
PROMPT_TMPL="你是一个网页/桌面操作智能体。当前界面观测如下：\n{obs}\n\n请判断此刻应采取的操作。操作："
N_TRAJ=100; T_STEP=6; U_FIX=1.0; MAX_DISTRACTORS=24
random.seed(0); np.random.seed(0); torch.manual_seed(0)

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
def build():
    label,isp1,act=[],[],[]
    for _ in range(N_TRAJ):
        p1,p2=random.sample(range(T_STEP),2); p1a,p2a=random.randint(0,1),random.randint(0,1)
        for t in range(T_STEP):
            if t==p1: lb,ip,a=0,True,p1a
            elif t==p2: lb,ip,a=1,False,p2a
            else: lb,ip,a=random.randint(2,9),False,random.randint(0,1)
            label.append(lb);isp1.append(ip);act.append(a)
    return np.array(label),np.array(isp1),np.array(act)

tok=AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
tok.padding_side="left"; tok.truncation_side="left"
bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16)
model=AutoModelForCausalLM.from_pretrained(MODEL,quantization_config=bnb,device_map={"":0}).eval()

@torch.no_grad()
def embed(texts):
    vecs=[]
    for i in range(0,len(texts),BATCH):
        prompts=[PROMPT_TMPL.format(obs=o) for o in texts[i:i+BATCH]]
        enc=tok(prompts,return_tensors="pt",padding=True,truncation=True,max_length=384).to(model.device)
        h=model(**enc,output_hidden_states=True).hidden_states[LAYER]
        vecs.append(torch.nn.functional.normalize(h[:,-1,:].float(),dim=-1).cpu().numpy())
    return np.concatenate(vecs,0)

def greedy_cosine_cluster(emb,eps):
    cents,members=[],[]
    for i,x in enumerate(emb):
        if cents:
            sims=np.array([c@x for c in cents]);j=int(sims.argmax())
            if (1-sims[j])<eps:
                members[j].append(i);c=emb[members[j]].mean(0);cents[j]=c/(np.linalg.norm(c)+1e-9);continue
        cents.append(x.copy());members.append([i])
    pred=np.empty(len(emb),int)
    for cid,m in enumerate(members):
        for idx in m:pred[idx]=cid
    return pred

label,isp1,act=build()
comp_text=[CANON[lb][1] for lb in label]        # C=1: 全部通用
E=embed(comp_text)
pred=greedy_cosine_cluster(E,EPS)

# 1) 文本层面: P1和P2的通用句是否真的一样?
print("P1通用句:", repr(CANON[0][1]))
print("P2通用句:", repr(CANON[1][1]))
print("字符串相等:", CANON[0][1]==CANON[1][1])
# 2) 向量层面: P1样本 vs P2样本 的cos
i_p1=np.where(label==0)[0]; i_p2=np.where(label==1)[0]
print(f"\ncos(P1样本, P2样本)均值 = {float((E[i_p1[0]]@E[i_p2].T).mean()):.4f}  (期望≈1)")
print(f"cos(P1样本, P1样本)均值 = {float((E[i_p1[0]]@E[i_p1].T).mean()):.4f}")
# 3) 聚类结果: P1和P2落在哪些簇, 有无重叠
c_p1=set(pred[i_p1].tolist()); c_p2=set(pred[i_p2].tolist())
print(f"\n总簇数={pred.max()+1}")
print(f"P1所在簇={sorted(c_p1)}  P2所在簇={sorted(c_p2)}  重叠={sorted(c_p1&c_p2)}")
print(f"P1/P2是否被聚到一起 = {len(c_p1&c_p2)>0}")
# 4) 每个P1簇里, P2占比(信用是否被P2污染)
for c in sorted(c_p1):
    idx=np.where(pred==c)[0]; labs=label[idx]
    print(f"  簇{c}: 共{len(idx)}个 | P1={ (labs==0).sum() } P2={ (labs==1).sum() } 其他={ (labs>1).sum() }")
