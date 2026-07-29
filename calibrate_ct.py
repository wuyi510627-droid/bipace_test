# calibrate_ct.py —— 标定档位分 c_t：把代码里拍脑袋的 0.6 / 0.3 换成实测值
# ─────────────────────────────────────────────────────────────────────────
# 现状: vtree_compressor.py 里 CONF = {原文:1.0, 摘要:0.6, 折叠:0.3} 是凭感觉填的.
# 本脚本跑一遍真实探针, 直接输出该用什么数.
#
# 怎么测(一句话): 把某一步分别用【原文/摘要/折叠】三种形式接在同一段历史后面,
#   看它产生的【跳幅】排序还对不对. 排序全对=1, 全乱(随机)=0.
#   之所以测排序而不是数值: 方法只用跳幅比大小, 从不用 V 的绝对值.
#
# 运行(有 GPU+模型的机器): python calibrate_ct.py
# 输出: c_摘要 = 0.xx , c_折叠 = 0.xx  → 直接填回 vtree_compressor.py 的 CONF
# ─────────────────────────────────────────────────────────────────────────

import random, itertools, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from vtree_compressor import TruncSummarizer, FOLD_TMPL

MODEL = "/home/wuyi/cuda12-dev/project/models/Qwen2.5-7B-Instruct"
N_TRAJ, MAXLEN, BATCH, SEED = 60, 1024, 16, 0
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 造一批 ALFWorld 风格轨迹: 有关键步(推进任务) 也有流水账 ────────────
OBJ = ["番茄", "鸡蛋", "土豆", "面包", "玉米"]
KEY = ["在料理台找到并拿起了生的{o}",
       "打开微波炉，把生的{o}放了进去",
       "微波炉加热完成，{o}已经是热的，拿在手上",
       "把热的{o}放到桌上，任务完成"]
NOISE = ["你环顾四周，没有看到什么新东西", "你经过料理台，上面摆着几个空盘子",
         "你看了一眼水槽，里面有些水渍", "你听到冰箱压缩机启动的声音",
         "你在厨房里又走了两步", "窗外的天色似乎暗了一些",
         "你瞥见墙上挂着一份菜谱", "地上有一小块面包屑"]


def gen_traj(rng):
    o = rng.choice(OBJ)
    steps = [f"你在厨房里四处找{o}"]
    for k in KEY:
        steps += [rng.choice(NOISE) for _ in range(rng.randint(1, 3))]
        steps.append(k.format(o=o))
    return f"把{o}加热后放到桌上", steps


# ── 模型 ────────────────────────────────────────────────────────────────
print(f"加载 {MODEL} (4bit) ...")
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"; tok.truncation_side = "left"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                             device_map={"": 0}).eval()
YES, NO = [tok(c, add_special_tokens=False).input_ids[0] for c in ("是", "否")]
P = "任务:{task}\n当前记忆:\n{mem}\n问:这个任务最终能成功吗?只回答一个字:是 或 否。\n答:"


@torch.no_grad()
def probe(mems, task):
    out = []
    for i in range(0, len(mems), BATCH):
        enc = tok([P.format(task=task, mem=m) for m in mems[i:i + BATCH]],
                  return_tensors="pt", padding=True, truncation=True,
                  max_length=MAXLEN).to(device)
        lg = model(**enc).logits[:, -1, :].float()
        out.extend(torch.softmax(lg[:, [YES, NO]], -1)[:, 0].cpu().tolist())
    return np.array(out)


# ── 主流程 ──────────────────────────────────────────────────────────────
rng = random.Random(SEED)
summ = TruncSummarizer()
pairs = {"摘要": [0, 0], "折叠": [0, 0]}      # [一致数, 总数]
dv = {"摘要": [], "折叠": []}                  # 顺带记数值差, 作对照

for n in range(N_TRAJ):
    task, steps = gen_traj(rng)
    T = len(steps)
    # 每一步都造三个版本, 前缀一律用原文 —— 差异只归因于这一步
    mems = []
    for t in range(T):
        pre = "\n".join(steps[:t])
        s = steps[t]
        mems += [pre, (pre + "\n" + s).strip(),
                 (pre + "\n" + summ(s, max(8, int(len(s) * .45)))).strip(),
                 (pre + "\n" + FOLD_TMPL.format(n=1)).strip()]
    V = probe(mems, task).reshape(T, 4)                 # 列: 前缀 / 原文 / 摘要 / 折叠
    prev, full, sm, fd = V[:, 0], V[:, 1], V[:, 2], V[:, 3]
    d_full, d_sm, d_fd = np.abs(full - prev), np.abs(sm - prev), np.abs(fd - prev)
    dv["摘要"].append(np.abs(full - sm).mean()); dv["折叠"].append(np.abs(full - fd).mean())

    # 同一条轨迹内, 所有步对的跳幅排序是否一致
    for i, j in itertools.combinations(range(T), 2):
        ref = np.sign(d_full[i] - d_full[j])
        if ref == 0: continue
        for name, d in (("摘要", d_sm), ("折叠", d_fd)):
            pairs[name][1] += 1
            if np.sign(d[i] - d[j]) == ref: pairs[name][0] += 1
    if (n + 1) % 10 == 0:
        print(f"  {n+1}/{N_TRAJ} 条…", flush=True)

# ── 结果 ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"{'档位':<6} | {'排序一致率':>10} | {'c_t = 2(acc-0.5)':>17} | {'现用值':>7}")
print("-" * 60)
res = {}
for name, cur in (("摘要", 0.6), ("折叠", 0.3)):
    hit, tot = pairs[name]
    acc = hit / max(tot, 1)
    c = max(0.0, 2 * (acc - 0.5))
    res[name] = c
    print(f"{name:<6} | {acc:>10.3f} | {c:>17.2f} | {cur:>7.1f}")
print("=" * 60)
print(f"（原文档位恒为 1.0；样本 {N_TRAJ} 条轨迹、{pairs['摘要'][1]} 个步对）")
print(f"\n顺带对照·数值差 mean|V_原文 − V_压缩|:  "
      f"摘要 {np.mean(dv['摘要']):.3f}   折叠 {np.mean(dv['折叠']):.3f}")
print("  (数值差小≠判断力强 —— 模型若对什么都答 0.5~0.6, 数值差天然小, 故以排序为准)")

print(f"""
→ 把 vtree_compressor.py 里这一行改成:
   CONF = {{RES_FULL: 1.0, RES_SUMM: {res['摘要']:.2f}, RES_MERGE: {res['折叠']:.2f}}}""")
if res["摘要"] < 0.4:
    print("\n⚠️ c_摘要 < 0.4: 摘要档保不住跳幅排序 → §2.4「三档优于两档」的论证需重新审视.")
