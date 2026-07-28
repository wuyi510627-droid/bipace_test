# value_tree_test.py —— B(价值失真记忆树)第一版机制原型 / 离线, 12G 可跑
# ─────────────────────────────────────────────────────────────────────────
# 一句话: 验证"按成功概率跳幅剪集锦式压缩(B)" 是否比"笨摘要" 更能保住处境可分性.
# 三种记忆对照:
#   raw   = 全历史(带填充噪声)         → 上界参照
#   naive = 一句泛泛的话(笨摘要)       → 下界(≈0.5, 对应你之前测的 0.56)
#   vtree = B: 只保"成功把握跳得最大"的那几步全貌, 其余压成一句  ← 本方法
# 判读: sep-AUC(vtree) 明显 > naive 且逼近/超过 raw
#       → "为价值而压" 真的保住了下游凑堆要用的区分信息 → B 成立.
#
# ⚠️ 合成数据 = 机制原型, 不作论文证据. 真证据要把数据换成 verl-agent dump 的
#    mem_records.jsonl(见文件末 M1 钩子); 主流程不用改.
# 运行(在有 GPU+模型的机器上, pull 后): python value_tree_test.py
# ─────────────────────────────────────────────────────────────────────────

import random, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import roc_auc_score

MODEL="/home/wuyi/cuda12-dev/project/models/Qwen2.5-7B-Instruct"
LAYER=-8; BATCH=8; MAXLEN=512
KEEP_RATIO=0.4                 # 记忆预算: 只把"跳幅最大"的这个比例的步留全貌, 其余压成一句
SEEDS=list(range(3))
device="cuda" if torch.cuda.is_available() else "cpu"

# ── 合成 ALFWorld 式轨迹: 每步 = (阶段标签stage, 处境核心文本) ──────────────
# stage = 真值处境标签, 跨物体复现, 用于 sep-AUC 的分组(同stage=正样本对).
# 0找物 1拿到 2炉前 3放入 4加热完 5桌前 6放好
OBJS=["番茄","鸡蛋","土豆","面包","玉米","茄子"]
FILLERS=["环顾四周","检查了冰箱","看了下水槽","经过料理台","瞥了眼窗外","整理了下站位","听到微波炉滴声"]
def make_traj(obj):
    return [
        (0, f"你在厨房里四处找{obj}"),
        (1, f"在料理台找到并拿起了生的{obj}"),
        (2, f"手拿生的{obj}, 走到微波炉前"),
        (3, f"打开微波炉, 把生的{obj}放了进去"),
        (4, f"微波炉加热完成, {obj}已经是热的, 拿在手上"),
        (5, f"手拿热的{obj}, 走向桌子"),
        (6, f"把热的{obj}放到桌上, 任务完成"),
    ]

def mem_raw(steps, t):         # 全历史: 到第t步, 每步核心 + 随机填充噪声
    lines=[]
    for i in range(t+1):
        lines.append(steps[i][1])
        if random.random()<0.7: lines.append(random.choice(FILLERS))
    return " ; ".join(lines)

def mem_naive(steps, t):       # 笨摘要: 整段压成一句无信息的话 → 处境糊成一团
    return "agent 在厨房里操作了一番。"

def mem_vtree(steps, t, keep):  # B: keep 中的步留全貌, 其余压成一句
    return " ; ".join(steps[i][1] if i in keep else "(略过一步例行操作)" for i in range(t+1))

# ── 加载模型 ────────────────────────────────────────────────────────────
print(f"加载 {MODEL} (4bit) ...")
tok=AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
tok.padding_side="left"; tok.truncation_side="left"
bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16)
model=AutoModelForCausalLM.from_pretrained(MODEL,quantization_config=bnb,device_map={"":0}).eval()
# 成功把握 = P(答"是") vs P(答"否"); 若分词非单token, 取首token(原型够用, 正式再核)
YES_ID=tok("是",add_special_tokens=False).input_ids[0]
NO_ID =tok("否",add_special_tokens=False).input_ids[0]

@torch.no_grad()
def value_of(mem_texts):       # 一批记忆 → 每条的成功把握 V∈[0,1]
    Vs=[]
    for i in range(0,len(mem_texts),BATCH):
        prompts=[f"你是家务智能体。当前记忆:\n{m}\n问:这个任务最终能成功吗?只回答一个字:是 或 否。\n答:"
                 for m in mem_texts[i:i+BATCH]]
        enc=tok(prompts,return_tensors="pt",padding=True,truncation=True,max_length=MAXLEN).to(device)
        logits=model(**enc).logits[:,-1,:].float()          # 末位=下一个token分布
        p=torch.softmax(logits[:,[YES_ID,NO_ID]],dim=-1)[:,0]  # P(是)=V
        Vs.extend(p.cpu().tolist())
    return Vs

# ── 表示 + 可分性 AUC (沿用 compress_auc_test 的做法) ──────────────────────
@torch.no_grad()
def embed(texts):
    vs=[]
    for i in range(0,len(texts),BATCH):
        enc=tok(texts[i:i+BATCH],return_tensors="pt",padding=True,truncation=True,max_length=MAXLEN).to(device)
        h=model(**enc,output_hidden_states=True).hidden_states[LAYER]
        mask=enc.attention_mask.unsqueeze(-1).float()
        p=(h*mask).sum(1)/mask.sum(1).clamp(min=1)
        vs.append(torch.nn.functional.normalize(p.float(),dim=-1).cpu().numpy())
    return np.concatenate(vs,0)

def sep_auc(emb,labels):        # cos 相似度能否预测"是否同一处境". 1=干净可分, 0.5=分不开
    S=emb@emb.T; iu=np.triu_indices(len(emb),k=1)
    same=(labels[iu[0]]==labels[iu[1]]).astype(int)
    if same.min()==same.max(): return float("nan")
    return roc_auc_score(same,S[iu])

# ── 主流程 ──────────────────────────────────────────────────────────────
def run(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    trajs=[make_traj(o) for o in OBJS]
    recs=[(ti,t,steps[t][0]) for ti,steps in enumerate(trajs) for t in range(len(steps))]
    labels=np.array([r[2] for r in recs])

    memR,memN,memV=[],[],[]
    for (ti,t,_) in recs:
        steps=trajs[ti]
        Vseq=value_of([mem_raw(steps,k) for k in range(t+1)])   # V_0..V_t
        jumps=[0.0]+[abs(Vseq[k]-Vseq[k-1]) for k in range(1,t+1)]
        K=max(1,int(round(KEEP_RATIO*(t+1))))
        keep=set(int(x) for x in np.argsort(jumps)[-K:])         # 跳幅最大的K步留全貌
        memR.append(mem_raw(steps,t)); memN.append(mem_naive(steps,t)); memV.append(mem_vtree(steps,t,keep))

    aR=sep_auc(embed(memR),labels); aN=sep_auc(embed(memN),labels); aV=sep_auc(embed(memV),labels)
    print(f"  seed{seed}:  raw={aR:.3f}  naive={aN:.3f}  vtree(B)={aV:.3f}")
    return [aR,aN,aV]

if __name__=="__main__":
    res=np.array([run(s) for s in SEEDS]); m=res.mean(0)
    print("\n===== 均值 (over %d seeds) =====" % len(SEEDS))
    print(f"raw   (全历史,上界参照) = {m[0]:.3f}")
    print(f"naive (笨摘要,下界)     = {m[1]:.3f}")
    print(f"vtree (B, 本方法)       = {m[2]:.3f}")
    print("判读: vtree 明显 > naive 且逼近/超过 raw → '为价值而压'保住了凑堆要用的区分信息 → B 成立.")

# ============ M1 真数据钩子(以后把合成换成真数据, 主流程不动)============
# 1) 用 memory_extract.py 从 verl-agent 的 total_batch_list dump 出 mem_records.jsonl,
#    每行 {"traj":i, "t":步号, "obs_text":含历史观测, "anchor":真值处境标签, "action":.., "ret":..}
# 2) recs 用 (traj,t,anchor); labels 用 anchor;
#    V 用 value_of([该轨迹 0..t 步的 obs_text]); mem_raw/naive/vtree 直接吃 obs_text.
# 3) 其余(建树/压缩/embed/sep_auc)一字不改.
