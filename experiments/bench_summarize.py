# bench_summarize.py —— 实测:摘要怎么才能不慢(验证成本模型 + 批量到底管不管用)
# ─────────────────────────────────────────────────────────────────────────
# 上一轮 bench_probe.py 测出"LLM 摘要 ~1.3s, 占决策 70%". 随后我【纸上】推了一套
# 成本模型: 读 n 个 token ≈ 241 + 0.89n ms, 写 1 个 token ≈ 21 ms, 并据此断言
# "批量最划算、抽取式只快 1.5 倍". 这些都没验证过 —— 本脚本就是来验的.
#
# 测四件事:
#   ① 固定开销    : 扫极短输入(8~256 tok)的纯读耗时 → 拟合 截距 + 斜率
#                   (截距 = 每次调模型躲不掉的固定成本, 是"批量能不能省"的关键)
#   ② 批量加速比  : batch = 1/2/4/8/16, 看【单条】成本随批量怎么降
#   ③ 生成长度    : max_new = 8/16/32/48, 看写多少字差多少
#   ④ 抽取式      : 只 prefill 不生成(代表"挑词不写句"), 对比生成式
#
# 判读要点:
#   · 若①的截距很大 → 批量收益大 → 优先做批量;
#   · 若②的单条成本在 batch=8 已趋平 → 批量到 8 就够, 不必更大;
#   · 若④相对②的批量版没快多少 → 抽取式不值得做(它的实现复杂度高得多).
#
# 运行(有 GPU+模型的机器): python bench_summarize.py
# ─────────────────────────────────────────────────────────────────────────

import time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "/home/wuyi/cuda12-dev/project/models/Qwen2.5-7B-Instruct"
PREFILL_LENS = [8, 16, 32, 64, 128, 256]      # ① 拟合固定开销
BATCHES = [1, 2, 4, 8, 16]                    # ② 批量
GEN_LENS = [8, 16, 32, 48]                    # ③ 生成长度
REPEAT, WARMUP = 8, 3
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"加载 {MODEL} (4bit) ...")
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                             device_map={"": 0}).eval()

# 摘要的真实输入: 一条 ALFWorld 式观测 + 压缩指令
OBS = ("你看到台面上有一个苹果、一片面包、一把黄油刀、一个杯子、一个盘子和一个番茄，"
       "旁边的微波炉门是关着的。")
SUMM_PROMPT = "把下面这条观测压缩成不超过12个字，只保留会影响任务成败的信息，直接输出结果:\n{}\n结果:"


def pad_to(n_tok, base=OBS):
    s = base
    while len(tok(s, add_special_tokens=False).input_ids) < n_tok:
        s += "，" + base
    ids = tok(s, add_special_tokens=False).input_ids[:n_tok]
    return tok.decode(ids)


def timed(fn):
    for _ in range(WARMUP): fn()
    if device == "cuda": torch.cuda.synchronize()
    ts = []
    for _ in range(REPEAT):
        t0 = time.perf_counter(); fn()
        if device == "cuda": torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000


# ══ ① 固定开销: 纯读, 扫长度 ═══════════════════════════════════════════
print("\n" + "=" * 68)
print("① 固定开销 —— 纯读(prefill only), 扫输入长度")
print("=" * 68)
print(f"{'输入(tok)':>10} | {'耗时(ms)':>10}")
xs, ys = [], []


@torch.no_grad()
def _prefill(text):
    enc = tok(text, return_tensors="pt").to(device)
    model(**enc).logits[:, -1, :]


for n in PREFILL_LENS:
    txt = pad_to(n)
    t = timed(lambda: _prefill(txt))
    xs.append(n); ys.append(t)
    print(f"{n:>10} | {t:>10.1f}")

slope, intercept = np.polyfit(xs, ys, 1)
print(f"\n  线性拟合: 耗时 ≈ {intercept:.0f} + {slope:.3f} × n  (ms)")
print(f"  → 固定开销 ≈ {intercept:.0f} ms  (每次调模型躲不掉的部分)")
print(f"  → 边际成本 ≈ {slope:.3f} ms/token")

# ══ ② 批量加速比 ═══════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("② 批量加速比 —— 摘要(读一条观测 + 生成), 看【单条】成本")
print("=" * 68)
p_summ = SUMM_PROMPT.format(OBS)
n_in = len(tok(p_summ, add_special_tokens=False).input_ids)
print(f"(单条输入 {n_in} tok)")


@torch.no_grad()
def _gen(bs, gen):
    enc = tok([p_summ] * bs, return_tensors="pt", padding=True).to(device)
    # min_new_tokens 强制跑满: 否则 bs=1 会因 early stopping 提前结束,
    # 而 bs>1 要等所有序列结束 → bs=1 被低估, "相对 bs=1"的加速比失真.
    model.generate(**enc, max_new_tokens=gen, min_new_tokens=gen, do_sample=False,
                   pad_token_id=tok.pad_token_id)


for gen in (16, 48):
    print(f"\n  生成长度 = {gen} token")
    print(f"  {'batch':>6} | {'总耗时(ms)':>11} | {'单条(ms)':>10} | {'相对bs=1':>9}")
    base = None
    for bs in BATCHES:
        t = timed(lambda bs=bs, gen=gen: _gen(bs, gen))
        per = t / bs
        if base is None: base = per
        print(f"  {bs:>6} | {t:>11.1f} | {per:>10.1f} | {base/per:>8.1f}×")

# ══ ③④ 生成长度 & 抽取式 ══════════════════════════════════════════════
print("\n" + "=" * 68)
print("③④ 生成长度的代价, 以及抽取式(只读不写)")
print("=" * 68)


@torch.no_grad()
def _extract(bs):
    """抽取式代表: 只 prefill + 取最后一层 hidden(供打分挑词), 不生成."""
    enc = tok([p_summ] * bs, return_tensors="pt", padding=True).to(device)
    out = model(**enc, output_hidden_states=True)
    out.hidden_states[-1].float().mean(1)          # 模拟"逐 token 打分"的取用


print(f"  {'做法':<26} | {'bs=1 单条':>10} | {'bs=8 单条':>10}")
print("  " + "-" * 52)
rows = []
for gen in GEN_LENS:
    t1 = timed(lambda gen=gen: _gen(1, gen))
    t8 = timed(lambda gen=gen: _gen(8, gen)) / 8
    rows.append((f"生成式 (max_new={gen})", t1, t8))
    print(f"  {f'生成式 (max_new={gen})':<26} | {t1:>10.1f} | {t8:>10.1f}")
te1 = timed(lambda: _extract(1))
te8 = timed(lambda: _extract(8)) / 8
print(f"  {'抽取式 (只读不写)':<26} | {te1:>10.1f} | {te8:>10.1f}")

# ══ 结论 ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("结论")
print("=" * 68)
g16_1 = [r for r in rows if "16" in r[0]][0]
print(f"  · 固定开销 ≈ {intercept:.0f} ms —— 批量能摊薄的就是这部分")
print(f"  · 生成式 max_new=48 → 16: 单条 {rows[-1][1]:.0f} → {g16_1[1]:.0f} ms")
print(f"  · 批量 bs=8: 生成式(16) 单条降到 {g16_1[2]:.0f} ms")
print(f"  · 抽取式 bs=8: 单条 {te8:.0f} ms  (相对生成式(16)批量版 {g16_1[2]/max(te8,1e-9):.1f}×)")
print("\n  判读:")
print("   ① 若『抽取式 vs 生成式(16)批量版』差距 < 2×  → 抽取式不值得做(复杂度换不来速度);")
print("   ② 若批量在 bs=8 已趋平            → 批量开到 8 即可;")
print("   ③ 若单条降到决策(≈2000ms)的 <10%  → 摘要成本问题解决, 中间档可放心用 LLM.")
