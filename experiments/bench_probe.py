# bench_probe.py —— 实测:价值探针到底比 agent 决策贵多少
# ─────────────────────────────────────────────────────────────────────────
# 回答一个必须用数据说话的问题:"每走一步都问一次大模型, 消耗是不是太大?"
#
# 测四件事(同一个模型、同样长度的记忆):
#   ① agent 决策   : 读一遍 + 【生成】动作(默认 30 token)   ← 基准
#   ② 探针(独立)   : 读一遍 + 【不生成】, 只取末位 logit
#   ③ 探针(共享)   : 复用①已读的前缀 KV cache, 只多读问句  ← 层次2
#   ④ LLM 摘要     : 读一遍 + 生成一小段(默认 32 token)     ← 中间档的代价
#
# 判读: 看 ②③④ 相对 ① 的百分比. 若 ③ < 2%, "每步一问"就不是问题.
# 运行(有 GPU+模型的机器): python bench_probe.py
# ─────────────────────────────────────────────────────────────────────────

import time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "/home/wuyi/cuda12-dev/project/models/Qwen2.5-7B-Instruct"
MEM_LENS = [200, 500, 1000, 2000]     # 记忆长度(token), 覆盖短程到长程
GEN_ACTION = 30                       # agent 生成动作的长度
GEN_SUMM = 32                         # 摘要生成长度
REPEAT = 10                           # 每项重复次数(取中位数, 抗抖动)
WARMUP = 3
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"加载 {MODEL} (4bit) ...")
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                             device_map={"": 0}).eval()
YES_ID = tok("是", add_special_tokens=False).input_ids[0]
NO_ID = tok("否", add_special_tokens=False).input_ids[0]

FILLER = "你走过一段走廊，看到墙边堆着一些杂物，继续往前走。"
Q_ACT = "\n问:接下来该做什么?请给出你的思考和动作。\n答:"
Q_VAL = "\n问:这个任务最终能成功吗?只回答一个字:是 或 否。\n答:"


def make_mem(n_tok):
    s = ""
    while len(tok(s, add_special_tokens=False).input_ids) < n_tok:
        s += FILLER + "\n"
    ids = tok(s, add_special_tokens=False).input_ids[:n_tok]
    return tok.decode(ids)


def timed(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize() if device == "cuda" else None
    ts = []
    for _ in range(REPEAT):
        t0 = time.perf_counter(); fn()
        torch.cuda.synchronize() if device == "cuda" else None
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000        # ms


@torch.no_grad()
def bench(mem):
    p_act, p_val = mem + Q_ACT, mem + Q_VAL

    def f_decide():                                   # ① 决策: 读 + 生成动作
        enc = tok(p_act, return_tensors="pt").to(device)
        model.generate(**enc, max_new_tokens=GEN_ACTION, do_sample=False,
                       pad_token_id=tok.pad_token_id)

    def f_probe():                                    # ② 探针(独立): 读, 不生成
        enc = tok(p_val, return_tensors="pt").to(device)
        lg = model(**enc).logits[:, -1, :]
        torch.softmax(lg[:, [YES_ID, NO_ID]].float(), -1)

    # ③ 探针(共享): 先把 mem 的 KV cache 算好(这步在真实系统里由 agent 决策时顺带完成),
    #    计时只算"多读问句 + 取 logit"的增量部分.
    enc_mem = tok(mem, return_tensors="pt").to(device)
    cache = model(**enc_mem, use_cache=True).past_key_values
    q_ids = tok(Q_VAL, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    def f_probe_shared():
        out = model(q_ids, past_key_values=cache, use_cache=True)
        lg = out.logits[:, -1, :]
        torch.softmax(lg[:, [YES_ID, NO_ID]].float(), -1)

    def f_summ():                                     # ④ 摘要: 读 + 生成一小段
        enc = tok(FILLER + "\n压缩成不超过12个字:", return_tensors="pt").to(device)
        model.generate(**enc, max_new_tokens=GEN_SUMM, do_sample=False,
                       pad_token_id=tok.pad_token_id)

    return (timed(f_decide), timed(f_probe), timed(f_probe_shared), timed(f_summ))


print(f"\n{'记忆长度':>8} | {'①决策':>9} | {'②探针独立':>11} | {'③探针共享':>11} | {'④LLM摘要':>10}")
print(f"{'(token)':>8} | {'(ms)':>9} | {'ms / 占比':>11} | {'ms / 占比':>11} | {'ms / 占比':>10}")
print("-" * 68)
rows = []
for n in MEM_LENS:
    mem = make_mem(n)
    d, p, ps, sm = bench(mem)
    rows.append((n, d, p, ps, sm))
    print(f"{n:>8} | {d:>9.1f} | {p:>6.1f} /{p/d*100:>4.1f}% | "
          f"{ps:>6.1f} /{ps/d*100:>4.1f}% | {sm:>5.1f} /{sm/d*100:>4.1f}%")

print("\n判读:")
avg_ps = np.mean([r[3] / r[1] for r in rows]) * 100
avg_p = np.mean([r[2] / r[1] for r in rows]) * 100
avg_sm = np.mean([r[4] / r[1] for r in rows]) * 100
print(f"  探针(独立) 平均占决策的 {avg_p:.1f}%  → 层次1")
print(f"  探针(共享) 平均占决策的 {avg_ps:.1f}%  → 层次2 (复用 KV cache)")
print(f"  LLM 摘要   平均占决策的 {avg_sm:.1f}%  → 中间档若同步做的代价")
print("\n  · 若『探针(共享)』< 2%: 每步一问不是问题, 按现方案做.")
print("  · 若『探针(独立)』就已很低: 连 KV cache 复用都不必, 实现更简单.")
print("  · 若『LLM 摘要』很高: 中间档必须用零成本摘要器或异步做(见 落地实现方案 D4).")
