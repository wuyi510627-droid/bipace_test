# embed_probe.py —— 实锤: left padding 下"同一句话"是否被embed成不同向量?
# 现象: 若同文本在 长batch(左pad多) vs 短batch(左pad少) 里 cos 明显<1, 则位置编码bug确认.
# 同时验证修复: 用 attention_mask 生成正确 position_ids 后 cos 应≈1.
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL="/home/wuyi/cuda12-dev/project/models/Qwen2.5-7B-Instruct"; LAYER=-8; BATCH=8
PROMPT_TMPL="你是一个网页/桌面操作智能体。当前界面观测如下：\n{obs}\n\n请判断此刻应采取的操作。操作："
tok=AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
tok.padding_side="left"; tok.truncation_side="left"
bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16)
model=AutoModelForCausalLM.from_pretrained(MODEL,quantization_config=bnb,device_map={"":0}).eval()

@torch.no_grad()
def embed(texts, fix_pos):
    vecs=[]
    for i in range(0,len(texts),BATCH):
        prompts=[PROMPT_TMPL.format(obs=o) for o in texts[i:i+BATCH]]
        enc=tok(prompts,return_tensors="pt",padding=True,truncation=True,max_length=512).to(model.device)
        kw={}
        if fix_pos:
            pos=enc.attention_mask.long().cumsum(-1)-1
            pos.masked_fill_(enc.attention_mask==0,1)
            kw["position_ids"]=pos
        h=model(**enc,output_hidden_states=True,**kw).hidden_states[LAYER]
        pooled=torch.nn.functional.normalize(h[:,-1,:].float(),dim=-1)
        vecs.append(pooled.cpu().numpy())
    return np.concatenate(vecs,0)

GEN="在商品列表页, 把【某件商品】加入购物车。"                 # 压缩后的通用句
LONG="广告: 蓝牙季大促立减99元! "*40 + GEN                    # 很长, 逼高 batch 的max长度
# batch0 = [LONG]+7*GEN (GEN被左pad很多) ; batch1 = 8*GEN (GEN几乎不pad)
texts=[LONG]+[GEN]*7 + [GEN]*8
for fix in [False, True]:
    E=embed(texts,fix)
    c_diff = float(E[1] @ E[8])    # 同一句GEN: 长batch里的 vs 短batch里的
    c_same = float(E[8] @ E[9])    # 同一句GEN: 都在短batch里
    print(f"fix_position_ids={fix}: cos(GEN跨batch)={c_diff:.4f}   cos(GEN同batch)={c_same:.4f}")
print("\n判读: 若 fix=False 时 cos(跨batch)明显<0.99 而 fix=True 时≈1.00 → 位置编码bug确认, 需给3个脚本都加position_ids修复")
