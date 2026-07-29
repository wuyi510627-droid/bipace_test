# tmaze_text_test.py —— 文本版 Passive T-Maze:一次证完三条主张
# ─────────────────────────────────────────────────────────────────────────
# 任务(照 Ni et al. 2023 的 Passive T-Maze 搬成文本):
#   第0步 = 起点, 墙上给一次线索"宝藏在上/下侧";  中间 = 空白走廊(纯 filler);
#   末尾 = 岔路口, 必须凭记忆选对上/下。 走廊越长 L, 线索离决策点越远。
#   记忆长度 = L, 信用分配长度 = 1(纯记忆诊断)。
#
# 三条主张:
#   ① truncate 会崩 : trunc-k 在 L>k 断崖到 ~0.5(线索滑出窗口)  = 打脸 G2PO/GraphGPO 的地基图
#   ② 压缩不崩     : vtree 一路 ~1.0(决定性步被保住)
#   ③ 压缩省 token : vtree 的 token 数是平的, full 随 L 线性爆炸 ← 动机图的另一半
#      (③ 才是 B 的动机命门: Ni 2023 已证"记忆长"逼不出压缩, 只有 token 硬上限能)
#
# 六条对照臂:
#   no_mem    : 只给当前(岔路口)观测           → 无线索, ≈50% 瞎猜
#   trunc-k   : 只留最近 k 步(=GraphGPO/G2PO)  → L>k 时线索滑出, ≈50%
#   full      : 全历史                         → ≈100%, 但 token 随 L 线性爆炸
#   vtree_val : B, 按【价值】(能否成功)跳幅留步  ← 主方法(定版决定1: 尺子=价值)
#   vtree_bel : B, 按【信念】(宝藏在哪)跳幅留步  ← 对照臂, 看两种信号是否等价
#
# ⚠️ Passive 是"送分题"(该留的步只有一个、太明显), 只证"必须压+管线通";
#    B 的真本事考在 Active/Key-to-Door(多个候选关键步里挑对的)。
# 运行(有 GPU+模型的机器): python tmaze_text_test.py
# ─────────────────────────────────────────────────────────────────────────

import random, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL="/home/wuyi/cuda12-dev/project/models/Qwen2.5-7B-Instruct"
MAXLEN=1024; BATCH=16       # BATCH: 前缀信号一次批量算完(vtree 每集要算 L+1 个前缀, 不批量会很慢)
LS=[1,2,4,8,16]             # 走廊长度扫描
TRUNC_KS=[2,8]              # 两个 truncate 窗口(2=GraphGPO 的"最近2步")
N_EPI=100                   # 每个 L 跑多少集(原为20, 噪声太大; 100 起步)
JUMP_TAU=0.25               # vtree: 跳幅 > 此值的步判为"决定性步", 保留
SEEDS=[0,1,2]               # 多 seed, 出误差棒
device="cuda" if torch.cuda.is_available() else "cpu"

ARMS=["no_mem","full","vtree_val","vtree_bel"]+[f"trunc{k}" for k in TRUNC_KS]

# ── 文本版 Passive T-Maze:生成一集的观测序列 ──────────────────────────────
SIDES=["上","下"]
def gen_episode(L):
    goal=random.choice(SIDES)
    obs=[f"【起点】你在走廊起点。墙上刻着一行字:宝藏在【{goal}侧】通道。"]      # 第0步:线索(仅此一次)
    for k in range(1,L):
        obs.append(f"你在走廊第{k}格,四周空无一物,只能继续往前走。")            # 空白走廊(filler)
    obs.append("【岔路口】前方分成上侧通道和下侧通道,必须选一个进去。")          # 末尾:决策点
    return goal, obs                                                          # len(obs)=L+1, 决策点=obs[L]

# ── 加载模型 ────────────────────────────────────────────────────────────
print(f"加载 {MODEL} (4bit) ...")
tok=AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
tok.padding_side="left"; tok.truncation_side="left"
bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16)
model=AutoModelForCausalLM.from_pretrained(MODEL,quantization_config=bnb,device_map={"":0}).eval()
UP_ID  =tok("上",add_special_tokens=False).input_ids[0]
DOWN_ID=tok("下",add_special_tokens=False).input_ids[0]
YES_ID =tok("是",add_special_tokens=False).input_ids[0]
NO_ID  =tok("否",add_special_tokens=False).input_ids[0]

@torch.no_grad()
def probe(prompts, id_a, id_b):     # 批量问二选一, 返回 P(选项a) 数组
    out=[]
    for i in range(0,len(prompts),BATCH):
        enc=tok(prompts[i:i+BATCH],return_tensors="pt",padding=True,
                truncation=True,max_length=MAXLEN).to(device)
        lg=model(**enc).logits[:,-1,:].float()          # 末位=下一个token分布
        out.extend(torch.softmax(lg[:,[id_a,id_b]],dim=-1)[:,0].cpu().tolist())
    return np.array(out)

_HEAD="你是走迷宫的智能体。你目前掌握的记忆如下:\n{}\n"
def belief(mems):   # 信念: 宝藏在上的概率(也是"决策"本身用的量)
    return probe([_HEAD.format(m)+"问:宝藏在上侧还是下侧通道?只回答一个字:上 或 下。\n答:" for m in mems],
                 UP_ID, DOWN_ID)
def value(mems):    # 价值: 能否成功找到宝藏的概率(定版决定1 的主信号)
    return probe([_HEAD.format(m)+"问:你能成功找到宝藏吗?只回答一个字:是 或 否。\n答:" for m in mems],
                 YES_ID, NO_ID)

# ── 各臂的记忆构造 ───────────────────────────────────────────────────────
def mem_full(obs):     return "\n".join(obs)                            # 全历史
def mem_trunc(obs,k):  return "\n".join(obs[max(0,len(obs)-k):])        # 最近 k 步
def mem_nomem(obs):    return obs[-1]                                   # 只当前观测

def mem_vtree(obs, sig_fn):
    """B: 按信号跳幅留决定性步, 丢空白走廊, 末尾岔路口必留.
    注: 第0步【不】无条件保留 —— 让它凭跳幅自己入选(线索恰在第0步, 硬留=作弊)."""
    prefixes=["\n".join(obs[:t+1]) for t in range(len(obs))]
    s=sig_fn(prefixes)                                                  # 一次批量算完所有前缀
    keep=[]; prev=0.5                                                   # 先验 0.5(还不知道上下/成败)
    for t in range(len(obs)):
        if abs(s[t]-prev)>=JUMP_TAU: keep.append(t)                     # 跳幅猛跳=决定性步
        prev=s[t]
    if (len(obs)-1) not in keep: keep.append(len(obs)-1)                # 决策点必留
    return "\n".join(obs[t] for t in sorted(set(keep)))

def ntok(text): return len(tok(text,add_special_tokens=False).input_ids)

# ── 单个 seed 的一轮 ─────────────────────────────────────────────────────
def run_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    acc={a:[] for a in ARMS}; tkn={a:[] for a in ARMS}
    for L in LS:
        hit={a:0 for a in ARMS}; tot={a:0 for a in ARMS}
        for _ in range(N_EPI):
            goal,obs=gen_episode(L)
            mems={"no_mem":mem_nomem(obs), "full":mem_full(obs),
                  "vtree_val":mem_vtree(obs,value), "vtree_bel":mem_vtree(obs,belief)}
            for k in TRUNC_KS: mems[f"trunc{k}"]=mem_trunc(obs,k)
            names=list(mems)
            p=belief([mems[n] for n in names])                          # 决策一律问上/下(任务本身)
            for n,pu in zip(names,p):
                if ("上" if pu>=0.5 else "下")==goal: hit[n]+=1
                tot[n]+=ntok(mems[n])
        for a in ARMS: acc[a].append(hit[a]/N_EPI); tkn[a].append(tot[a]/N_EPI)
        print(f"  L={L:>2}: " + "  ".join(f"{a}={acc[a][-1]:.2f}({tkn[a][-1]:.0f}tk)" for a in ARMS))
    return acc,tkn

# ── 主流程 ──────────────────────────────────────────────────────────────
if __name__=="__main__":
    A={a:[] for a in ARMS}; T={a:[] for a in ARMS}
    for s in SEEDS:
        print(f"seed {s}:")
        acc,tkn=run_seed(s)
        for a in ARMS: A[a].append(acc[a]); T[a].append(tkn[a])
    # 跨 seed 汇总
    Am={a:np.mean(A[a],0) for a in ARMS}; As={a:np.std(A[a],0) for a in ARMS}
    Tm={a:np.mean(T[a],0) for a in ARMS}; Ts={a:np.std(T[a],0) for a in ARMS}

    print(f"\n===== 选对率 (mean±std over {len(SEEDS)} seeds, N={N_EPI}/seed) =====")
    print("L\t" + "\t".join(ARMS))
    for i,L in enumerate(LS):
        print(f"{L}\t" + "\t".join(f"{Am[a][i]:.2f}±{As[a][i]:.2f}" for a in ARMS))
    print("\n===== 记忆 token 数 =====")
    print("L\t" + "\t".join(ARMS))
    for i,L in enumerate(LS):
        print(f"{L}\t" + "\t".join(f"{Tm[a][i]:.0f}" for a in ARMS))

    try:
        import matplotlib.pyplot as plt
        # 英文标签: 避免 CJK 字体缺失的 warning, 且论文图本来就要英文
        LBL={"no_mem":"no memory","full":"full history","vtree_val":"vtree (value)",
             "vtree_bel":"vtree (belief)","trunc2":"trunc-2","trunc8":"trunc-8"}
        STY={"no_mem":("o:","gray"),"full":("s-","tab:green"),"vtree_val":("D-","tab:blue"),
             "vtree_bel":("d--","tab:cyan"),"trunc2":("^--","tab:red"),"trunc8":("v--","tab:orange")}
        fig,ax=plt.subplots(1,2,figsize=(11,4.4))
        for a in ARMS:
            fmt,c=STY[a]
            ax[0].errorbar(LS,Am[a],yerr=As[a],fmt=fmt,color=c,label=LBL[a],capsize=3)
            ax[1].errorbar(LS,Tm[a],yerr=Ts[a],fmt=fmt,color=c,label=LBL[a],capsize=3)
        ax[0].axhline(0.5,ls=":",c="k",alpha=.4); ax[0].set_ylim(0.35,1.05)
        ax[0].set_xlabel("corridor length L (= memory span)"); ax[0].set_ylabel("decision accuracy")
        ax[0].set_title("(a) truncation breaks, compression does not")
        ax[1].set_xlabel("corridor length L (= memory span)"); ax[1].set_ylabel("memory tokens")
        ax[1].set_title("(b) full history blows up, vtree stays flat")
        ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
        plt.tight_layout(); plt.savefig("tmaze_result.png",dpi=140)
        print("\n图已存 tmaze_result.png  (左=准确率, 右=token数)")
    except Exception as e:
        print("画图跳过:",e)

    print("\n判读(三条一起看):")
    print("  ① trunc-k 在 L>k 掉到 ~0.5 → truncate(G2PO/GraphGPO的招)在'必须长程记忆'的任务上崩;")
    print("  ② vtree 一路 ~1.0        → 压缩保住了决定性线索;")
    print("  ③ full 的 token 随 L 线性涨、vtree 平 → 压缩真的解决 token 硬上限(B 的动机命门).")
    print("  ④ vtree_val vs vtree_bel → 若两条重合, 说明 Passive 下'价值信号'与'信念信号'等价.")

# ============ 下一步(Active 版, 留作 B 的真本事图) ============
# 把 gen_episode 改成 Active: 起点在 oracle 右边一格, 线索要"先走回去读"→ CA 长度也变 T;
# 再加一条"多线索"变体(多个决定性步), 看 vtree(B) 能不能只留对的那几步、且优于均匀压缩。
