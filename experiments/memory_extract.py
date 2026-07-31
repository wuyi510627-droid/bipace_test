# memory_extract.py —— 真·memory 表示测法(基于对 verl-agent 源码的 M0 核对)
# 架构要点(已核对源码):
#   * 默认 SimpleMemory = 原始K-window拼接(不压缩); 压缩是官方"扩展点"(memory/README.md).
#   * memory 注入点 = env_manager.build_text_obs() 调 self.memory.fetch() 塞进 prompt.
#   * obs = {'text': 含历史的完整观测, 'anchor': 去历史的当前状态}  ← anchor=天生真值状态标签.
#   * rollout 用 vLLM(不吐hidden); 无需hook —— total_batch_list 每步已存
#       input_ids(解码=完整obs文本)/anchor_obs/responses(动作)/episode_rewards.
#   * 表示离线再取: HF 4bit forward + mean池化(同 compress_auc_test), 12G够.
#
# 两阶段:
#   Phase1(在verl-agent里): 跑少量rollout, 从 total_batch_list dump 逐步 (obs_text, anchor, action, ret) → mem_records.jsonl
#   Phase2(本脚本主体): 离线embed + 诊断. 与rollout解耦, 可反复跑.
#
# 压缩轴: 用默认SimpleMemory跑一次 → 再换 SummaryMemory(见 memory/README 扩展)跑一次 → 对比AUC.

import json, numpy as np, torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import roc_auc_score

MODEL="/home/wuyi/cuda12-dev/project/models/Qwen2.5-7B-Instruct"; LAYER=-8; BATCH=8; MAXLEN=1024
RECORDS="mem_records.jsonl"   # Phase1 产出; 每行: {"obs_text":..., "anchor":..., "action":..., "ret":...}

# ============================================================
# Phase 1 采集器 —— 贴进 verl-agent 的 rollout 驱动脚本里调用
# ============================================================
def dump_records_from_batch(total_batch_list, episode_rewards, tokenizer, out=RECORDS):
    """
    从 vanilla_multi_turn_loop 返回的 total_batch_list 逐步导出记录. 不改核心, 只后处理.
    total_batch_list[env][step] 里有: input_ids, anchor_obs, responses, active_masks.
    """
    n=0
    with open(out,"w") as f:
        for env_idx, steps in enumerate(total_batch_list):
            ret=float(episode_rewards[env_idx])
            for rec in steps:
                if not rec.get("active_masks", True): continue
                obs_text=tokenizer.decode(rec["input_ids"], skip_special_tokens=True)   # 完整含memory的观测
                anchor=rec.get("anchor_obs", None)                                      # 去历史的真值状态
                anchor=anchor.item() if hasattr(anchor,"item") else (str(anchor) if not isinstance(anchor,str) else anchor)
                action=tokenizer.decode(rec["responses"], skip_special_tokens=True)
                f.write(json.dumps({"obs_text":obs_text,"anchor":str(anchor),"action":action,"ret":ret},ensure_ascii=False)+"\n")
                n+=1
    print(f"[Phase1] 写出 {n} 条 → {out}")

# ============================================================
# Phase 2: 离线取表示 + 诊断
# ============================================================
def load_model():
    tok=AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tok.padding_side="left"; tok.truncation_side="left"
    bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16)
    m=AutoModelForCausalLM.from_pretrained(MODEL,quantization_config=bnb,device_map={"":0}).eval()
    return tok,m

@torch.no_grad()
def embed(texts, tok, model):
    vecs=[]
    for i in range(0,len(texts),BATCH):
        enc=tok(texts[i:i+BATCH],return_tensors="pt",padding=True,truncation=True,max_length=MAXLEN).to(model.device)
        h=model(**enc,output_hidden_states=True).hidden_states[LAYER]
        mask=enc.attention_mask.unsqueeze(-1).float()
        pooled=(h*mask).sum(1)/mask.sum(1).clamp(min=1)                 # mean池化 = f(s)
        vecs.append(torch.nn.functional.normalize(pooled.float(),dim=-1).cpu().numpy())
    return np.concatenate(vecs,0)

def sep_auc(emb, labels):
    S=emb@emb.T; iu=np.triu_indices(len(emb),k=1)
    same=(labels[iu[0]]==labels[iu[1]]).astype(int)
    if same.min()==same.max(): return float("nan")
    return roc_auc_score(same,S[iu])

def main():
    recs=[json.loads(l) for l in open(RECORDS)]
    labs_raw=[r["anchor"] for r in recs]
    u={s:i for i,s in enumerate(sorted(set(labs_raw)))}          # anchor→整数状态id
    lab=np.array([u[s] for s in labs_raw])
    tok,model=load_model()
    E=embed([r["obs_text"] for r in recs], tok, model)
    auc=sep_auc(E,lab)
    print(f"真memory表示 sep-AUC={auc:.3f}  样本={len(lab)} 真值状态数={lab.max()+1}")
    print("判读: AUC高→含memory的表示按真值状态可分, '在memory表示上分组'故事稳;")
    print("      AUC低→分不开, 需要更忠实memory或更鲁棒度量.")
    print("压缩对比: 用 SummaryMemory 重跑 Phase1 得到 mem_records_summary.jsonl, 改 RECORDS 再跑, 对比两条 AUC.")

if __name__=="__main__":
    main()
