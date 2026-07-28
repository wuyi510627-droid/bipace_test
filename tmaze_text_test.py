# tmaze_text_test.py —— 文本版 Passive T-Maze:证明"truncate 会崩、压缩不崩"(打脸 G2PO 的地基图)
# ─────────────────────────────────────────────────────────────────────────
# 任务(照 Ni et al. 2023 的 Passive T-Maze 搬成文本):
#   第0步 = 起点, 墙上给一次线索"宝藏在上/下侧";  中间 = 空白走廊(纯 filler);
#   末尾 = 岔路口, 必须凭记忆选对上/下。 走廊越长 L, 线索离决策点越远。
#   记忆长度 = L, 信用分配长度 = 1(纯记忆诊断)。
#
# 四条对照(在决策点问模型"宝藏在上还是下", 看选对率):
#   no_mem   : 只给当前(岔路口)观测        → 无线索, ≈50% 瞎猜
#   trunc-k  : 只留最近 k 步(=GraphGPO/G2PO 的招) → L>k 时线索滑出窗口, ≈50%
#   full     : 全历史                        → 线索在, ≈100%(但 token 随 L 爆炸)
#   vtree(B) : 按"信念跳变"留决定性步、丢空白走廊 → 线索在、token 极省, ≈100%
#
# 期望图: X=走廊长L, Y=选对率。 trunc-k 在 L>k 断崖, full/vtree 一路平。
# ⚠️ Passive 是"送分题"(该留的步太明显), 只证"必须压+管线通"; B 的真本事考在 Active/Key-to-Door。
# 运行(有 GPU+模型的机器): python tmaze_text_test.py
# ─────────────────────────────────────────────────────────────────────────

import random, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL="/home/wuyi/cuda12-dev/project/models/Qwen2.5-7B-Instruct"; MAXLEN=1024
LS=[1,2,4,8,16]              # 走廊长度扫描
TRUNC_KS=[2,8]              # 两个 truncate 窗口(2=GraphGPO 的"最近2步")
N_EPI=20                    # 每个 L 跑多少集
JUMP_TAU=0.25              # vtree: 信念跳幅 > 此值的步判为"决定性步", 保留
SEED=0
device="cuda" if torch.cuda.is_available() else "cpu"

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
UP_ID=tok("上",add_special_tokens=False).input_ids[0]
DOWN_ID=tok("下",add_special_tokens=False).input_ids[0]

@torch.no_grad()
def p_up(mem_text):     # 给一份记忆, 返回模型认为"宝藏在上"的概率
    prompt=(f"你是走迷宫的智能体。你目前掌握的记忆如下:\n{mem_text}\n"
            f"问:宝藏在上侧还是下侧通道?只回答一个字:上 或 下。\n答:")
    enc=tok(prompt,return_tensors="pt",padding=True,truncation=True,max_length=MAXLEN).to(device)
    logits=model(**enc).logits[:,-1,:].float()
    return torch.softmax(logits[:,[UP_ID,DOWN_ID]],dim=-1)[0,0].item()

def pred_side(mem_text):  # 决策:上 or 下
    return "上" if p_up(mem_text)>=0.5 else "下"

# ── 四种记忆在决策点的构造 ─────────────────────────────────────────────────
def mem_full(obs):            return "\n".join(obs)                              # 全历史
def mem_trunc(obs,k):         return "\n".join(obs[max(0,len(obs)-k):])          # 最近 k 步
def mem_nomem(obs):           return obs[-1]                                     # 只当前观测
def mem_vtree(obs):           # B: 按信念跳变留决定性步, 丢空白走廊, 末尾岔路口必留
    keep=[]; prev=0.5                                                            # 先验 0.5(还不知道上下)
    for t in range(len(obs)):
        b=p_up("\n".join(obs[:t+1]))
        if t==0 or abs(b-prev)>=JUMP_TAU: keep.append(t)                         # 信念猛跳=决定性步
        prev=b
    if (len(obs)-1) not in keep: keep.append(len(obs)-1)                         # 决策点必留
    return "\n".join(obs[t] for t in sorted(set(keep)))

# ── 主流程 ──────────────────────────────────────────────────────────────
def run():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    arms=["no_mem","full","vtree"]+[f"trunc{k}" for k in TRUNC_KS]
    acc={a:[] for a in arms}
    for L in LS:
        hit={a:0 for a in arms}
        for _ in range(N_EPI):
            goal,obs=gen_episode(L)
            mems={"no_mem":mem_nomem(obs),"full":mem_full(obs),"vtree":mem_vtree(obs)}
            for k in TRUNC_KS: mems[f"trunc{k}"]=mem_trunc(obs,k)
            for a in arms:
                if pred_side(mems[a])==goal: hit[a]+=1
        for a in arms: acc[a].append(hit[a]/N_EPI)
        print(f"L={L:>2}: " + "  ".join(f"{a}={acc[a][-1]:.2f}" for a in arms))
    return arms,acc

if __name__=="__main__":
    arms,acc=run()
    print("\n===== 选对率 vs 走廊长 L =====")
    print("L\t" + "\t".join(arms))
    for i,L in enumerate(LS):
        print(f"{L}\t" + "\t".join(f"{acc[a][i]:.2f}" for a in arms))
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6.4,4.6))
        sty={"no_mem":("o:","gray"),"full":("s-","tab:green"),"vtree":("D-","tab:blue"),
             "trunc2":("^--","tab:red"),"trunc8":("v--","tab:orange")}
        for a in arms:
            fmt,c=sty.get(a,("x-","black")); plt.plot(LS,acc[a],fmt,color=c,label=a)
        plt.axhline(0.5,ls=":",c="k",alpha=.4,label="瞎猜")
        plt.xlabel("走廊长度 L(=记忆长度)"); plt.ylabel("决策点选对率"); plt.ylim(0.4,1.03)
        plt.title("文本版 Passive T-Maze:truncate 崩、压缩不崩")
        plt.legend(); plt.tight_layout(); plt.savefig("tmaze_result.png",dpi=140)
        print("\n图已存 tmaze_result.png")
    except Exception as e:
        print("画图跳过:",e)
    print("\n判读: trunc-k 在 L>k 掉到 ~0.5(线索滑出窗口), 而 full/vtree 保持 ~1.0")
    print("      → truncate(G2PO的招)在'必须长程记忆'的任务上崩; 压缩(B)保住决定性线索、token 却极省 → memory 压缩的存在理由成立。")

# ============ 下一步(Active 版, 留作 B 的真本事图) ============
# 把 gen_episode 改成 Active: 起点在 oracle 右边一格, 线索要"先走回去读"→ CA 长度也变 T;
# 再加一条"多线索"变体(多个决定性步), 看 vtree(B) 能不能只留对的那几步、且优于均匀压缩。
