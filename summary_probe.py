# summary_probe.py —— 把真Qwen生成的摘要打印出来, 判断 AUC=0.56 是"真难分"还是"摘要套话"
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL="/home/wuyi/cuda12-dev/project/models/Qwen2.5-7B-Instruct"; BATCH=8; MAXLEN=384
U_FIX=1.0; MAX_DISTRACTORS=24
SUMM_TMPL="用一句话概括下面界面此刻要完成的操作，忽略广告、时间戳、导航等无关信息：\n{obs}\n一句话概括："
random.seed(0); torch.manual_seed(0)

CANON=["在商品列表页, 把【笔记本电脑】加入购物车。","在商品列表页, 把【手机】加入购物车。",
       "在商品列表页, 把【耳机】加入购物车。","在搜索结果页, 点开【第1个】结果。",
       "在搜索结果页, 点开【第3个】结果。","在表单页, 在【邮箱】字段填写。",
       "在表单页, 在【电话】字段填写。","在设置页, 打开【深色模式】开关。",
       "在设置页, 打开【通知】开关。","在文件页, 进入【Downloads】文件夹。"]
import string
DISTRACT=["广告: {w}季大促, 立减{n}元!","导航: 首页 > {w} > {w}","页脚版权 (c) {n} {w}公司",
          "推荐商品: {w} {w} {w}","时间戳: 2026-07-0{d} {n}:{n}","session_id={tok}",
          "用户{tok}最近浏览: {w}","cookie横幅: 本站使用{w}以改善体验","热搜: {w}, {w}, {w}"]
WORDS=["蓝牙","促销","数码","家居","春装","会员","限时","旗舰","优选","清仓"]
def _rtok(k=6): return "".join(random.choices(string.ascii_lowercase+string.digits,k=k))
def _distractor(): return random.choice(DISTRACT).format(w=random.choice(WORDS),n=random.randint(1,99),d=random.randint(1,9),tok=_rtok())
def make_obs(core):
    parts=[core]+[_distractor() for _ in range(MAX_DISTRACTORS)]; random.shuffle(parts); return "\n".join(parts)

tok=AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
tok.padding_side="left"; tok.truncation_side="left"
bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16)
model=AutoModelForCausalLM.from_pretrained(MODEL,quantization_config=bnb,device_map={"":0}).eval()

@torch.no_grad()
def summarize(raws):
    prompts=[SUMM_TMPL.format(obs=o) for o in raws]
    enc=tok(prompts,return_tensors="pt",padding=True,truncation=True,max_length=MAXLEN).to(model.device)
    gen=model.generate(**enc,max_new_tokens=32,do_sample=False,pad_token_id=tok.pad_token_id)
    return [tok.decode(gen[j][enc.input_ids.shape[1]:],skip_special_tokens=True).strip().replace("\n"," ") for j in range(gen.shape[0])]

# 每个处境取2个不同表面变体, 看: (a)同处境两变体摘要是否一致 (b)不同处境摘要是否有区分
print("="*90)
for i,core in enumerate(CANON):
    raws=[make_obs(core) for _ in range(2)]
    sums=summarize(raws)
    print(f"[处境{i}] 真核心: {core}")
    print(f"   摘要变体A: {sums[0]}")
    print(f"   摘要变体B: {sums[1]}")
    print("-"*90)
