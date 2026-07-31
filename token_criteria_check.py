# token_criteria_check.py —— §7.4 预检：「信息量」和「信用相关性」是不是两回事
# ═════════════════════════════════════════════════════════════════════════
# 【这是一个便宜的 go / no-go 判定, 约一小时】
#   详细版 §7.3 的 claim 是: 现有 prompt compression(LLMLingua / Selective-Context)
#   以"保留信息量"为目标, 而 agentic RL 该以"保留信用相关性"为目标.
#
#   这个 claim 【只有在两个判据确实不一致时才成立】. 若高度相关, 现成方法就够用了,
#   §7 整节应当降级成 related work 里的一句话 —— 别去开第二条战线.
#
# 【怎么测】对同一批观测, 给每个 token 打两种分, 算排序相关性
#   ① 信息量      = −log p(token | 前缀)        罕见 = 高. 一次 forward 全拿到.
#   ② 信用相关性  = |V(完整) − V(删掉这个 token)|
#                   删掉它, "能不能成"的判断变了多少. 变得多 = 对成败重要.
#                   用留一法而非注意力归因: 注意力当解释在学术上有争议
#                   ("attention is not explanation"), 留一法是直接的因果度量.
#
#   Spearman(①, ②) 低  → 两个判据确实不同 → claim 有实证支撑
#   Spearman(①, ②) 高  → 现成方法已经在做同一件事 → 放弃这一节
#
# 【顺带产出一张论文可用的定性表】: 两个判据各自的 top-5 词, 并排放.
#
# 运行: python token_criteria_check.py
#       python token_criteria_check.py --real obs_dump.txt   (一行一条真实观测)
# ═════════════════════════════════════════════════════════════════════════

import argparse, os
import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "/home/wuyi/cuda12-dev/project/models/Qwen2.5-7B-Instruct"
BATCH, MAXLEN = 16, 1024
TOPK_SHOW = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

TASK = "heat some tomato and put it on the table"

# ALFWorld 风格的观测 —— 故意混入两类 token:
#   · 杂物名(apple / butterknife / saltshaker): 罕见 ⇒ 信息量高, 但与番茄无关
#   · 关键动作(pick up / heat / put): 极常见 ⇒ 信息量低, 但决定成败
SYNTH_OBS = [
    "You see a apple 1, a bread 1, a butterknife 1, a cup 1, a fork 2, and a tomato 1.",
    "You pick up the tomato 1 from the countertop 1.",
    "You arrive at microwave 1. The microwave 1 is closed.",
    "You open the microwave 1. The microwave 1 is empty.",
    "You heat the tomato 1 with the microwave 1.",
    "You see a saltshaker 2, a peppershaker 3, a soapbottle 1, and a spatula 2.",
    "You put the tomato 1 in/on the diningtable 1.",
    "Nothing happens.",
    "You move to shelf 3. On the shelf 3, you see a vase 2 and a statue 1.",
    "You take the mug 1 from the cabinet 4.",
]

PROBE_P = ("Task: {task}\nCurrent memory:\n{mem}\n"
           "Q: Will this task eventually succeed? Answer one word: yes or no.\nA:")


def load():
    print(f"加载 {MODEL} (4bit) ...")
    tk = AutoTokenizer.from_pretrained(MODEL)
    if tk.pad_token is None: tk.pad_token = tk.eos_token
    tk.padding_side = "left"; tk.truncation_side = "left"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    md = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                              device_map={"": 0}).eval()
    return tk, md


tok, model = load()
YES = tok(" yes", add_special_tokens=False).input_ids[-1]
NO = tok(" no", add_special_tokens=False).input_ids[-1]


@torch.no_grad()
def value(mems):
    """价值探针 → P(yes)。"""
    out = []
    for i in range(0, len(mems), BATCH):
        enc = tok([PROBE_P.format(task=TASK, mem=m) for m in mems[i:i + BATCH]],
                  return_tensors="pt", padding=True, truncation=True,
                  max_length=MAXLEN).to(device)
        lg = model(**enc).logits[:, -1, :].float()
        out.extend(torch.softmax(lg[:, [YES, NO]], -1)[:, 0].cpu().tolist())
    return np.array(out)


@torch.no_grad()
def self_information(text):
    """每个 token 的 −log p(token | 前缀)。一次 forward 全拿到。

    这就是 Selective-Context 的判据; LLMLingua 用的 perplexity 是它的指数平均,
    在【排序】层面两者等价, 所以测一个就够.
    """
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    lg = model(ids).logits.float()
    lp = torch.log_softmax(lg[0, :-1], -1)                  # 预测下一位
    tgt = ids[0, 1:]
    si = -lp[torch.arange(len(tgt)), tgt]                   # (n-1,)
    toks = tok.convert_ids_to_tokens(ids[0])[1:]            # 与 si 对齐
    return si.detach().cpu().numpy(), toks, ids[0, 1:].tolist()


def credit_relevance(text, ids):
    """留一法: |V(完整) − V(删掉第 i 个 token)|。"""
    base = value([text])[0]
    variants = [tok.decode(ids[:i] + ids[i + 1:]) for i in range(len(ids))]
    vs = value(variants)
    return np.abs(base - vs), base


def clean(t):
    return t.replace("Ġ", "").replace("Ċ", "\\n").strip() or "␠"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None, help="一行一条真实观测的 txt")
    args = ap.parse_args()

    if args.real and os.path.exists(args.real):
        obs_list = [l.strip() for l in open(args.real, encoding="utf-8") if l.strip()]
        src = f"真实 ({args.real}, {len(obs_list)} 条)"
    else:
        if args.real:
            print(f"⚠️ 找不到 {args.real}, 退回合成观测")
        obs_list = SYNTH_OBS
        src = f"合成 ({len(obs_list)} 条 ALFWorld 风格)"

    print(f"\n数据来源: {src}\n")

    rhos, all_info, all_cred = [], [], []
    examples = []

    for obs in obs_list:
        si, toks, ids = self_information(obs)
        cr, base = credit_relevance(obs, ids)
        n = min(len(si), len(cr))
        si, cr, toks = si[:n], cr[:n], toks[:n]
        if n < 3:
            continue
        rho = spearmanr(si, cr).correlation
        rhos.append(rho)
        all_info.append(si); all_cred.append(cr)

        top_i = [clean(toks[k]) for k in np.argsort(-si)[:TOPK_SHOW]]
        top_c = [clean(toks[k]) for k in np.argsort(-cr)[:TOPK_SHOW]]
        overlap = len(set(top_i) & set(top_c)) / TOPK_SHOW
        examples.append((obs, rho, top_i, top_c, overlap, base))

    # ── 定性表 ──────────────────────────────────────────────────────
    print("=" * 92)
    print(f"① 定性对照 · 每条观测里, 两个判据各自的 top-{TOPK_SHOW} token")
    print("=" * 92)
    for obs, rho, ti, tc, ov, base in examples:
        print(f"\n  观测: {obs[:78]}{'…' if len(obs) > 78 else ''}")
        print(f"        V(完整)={base:.3f}   Spearman={rho:+.3f}   top-{TOPK_SHOW}重合率={ov:.0%}")
        print(f"    按信息量  : {', '.join(ti)}")
        print(f"    按信用相关: {', '.join(tc)}")

    # ── 定量 ────────────────────────────────────────────────────────
    rhos = np.array([r for r in rhos if not np.isnan(r)])
    ovs = np.array([e[4] for e in examples])
    pooled = spearmanr(np.concatenate(all_info), np.concatenate(all_cred)).correlation

    print(f"\n{'='*92}")
    print("② 定量结果")
    print(f"{'='*92}")
    print(f"  逐条 Spearman 的均值 = {rhos.mean():+.3f}  (标准差 {rhos.std():.3f}, "
          f"范围 [{rhos.min():+.3f}, {rhos.max():+.3f}])")
    print(f"  合并全部 token 的 Spearman = {pooled:+.3f}")
    print(f"  top-{TOPK_SHOW} 重合率均值 = {ovs.mean():.1%}")

    verdict = ("判据确实不同 → §7 的 claim 成立, 写进论文" if abs(pooled) < 0.3
               else ("边界情况, 再取一批真实观测复核" if abs(pooled) < 0.5
                     else "两者高度相关 → 现成方法够用, §7 降级为 related work 一句话"))

    print(f"""
{'='*92}
判读
{'='*92}
  【出口条件】|Spearman| < 0.3  ⇒ go;  > 0.5 ⇒ no-go.

  本轮: |{pooled:+.3f}| ⇒ 【{verdict}】

  · |ρ| < 0.3   → ✅ 两个判据确实测的是不同东西. §7.3 的说法有实证支撑:
                    "现有方法优化信息量, 我们优化信用相关性, 两者不一致".
                    定性表里的例子可以直接做成论文里的 Figure/Table.
  · 0.3~0.5     → ⚠️ 有部分重叠. 别急着下结论, 换一批【真实】观测再跑一次 ——
                    合成观测是我按"杂物名罕见、动作词常见"的直觉编的,
                    这本身就预设了结论, 有自证风险.
  · |ρ| > 0.5   → ❌ 现成的信息量判据已经在做类似的事. §7 不值得单独做,
                    降级成 related work 里的一句话, 把精力放回主线(实验②⑤).

  【无论结果如何都要注意】
   本预检只测【判据是否一致】, 不测【换判据后效果是否更好】.
   即使 ρ 很低, 也只说明"两者不同", 不说明"我们的更好" —— 后者要靠端到端实验.
   所以这一节的 claim 在论文里应当写成【设计动机】, 不是【实验结论】.

  【排期】放在主线(实验②⑤)之后. 段落级的核心主张还没验完之前, 不开第二条战线.""")
