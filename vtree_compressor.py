# vtree_compressor.py —— 价值失真记忆树:在线增量压缩的共用模块
# ─────────────────────────────────────────────────────────────────────────
# 一句话: 给定固定 token 预算, 按"这一步让'能不能成'跳了多少"决定每段展开到多细.
#
# 设计约束(见 落地实现方案.md §0):
#   动机是 context 硬上限 ⇒ 压缩必须作用在 agent 决策所看的 memory 上
#   ⇒ 必须【在线】压(边走边压) ⇒ 探针要能增量算、预算重分配要便宜、摘要要带缓存.
#
# 三档分辨率(= 树的三层):
#   RES_FULL  原文一字不差            → 叶子      conf 1.0
#   RES_SUMM  摘要(可插拔摘要器)        → 中间层    conf 0.6
#   RES_MERGE 连续段捏成一句           → 近根      conf 0.3
#
# 用法(在线):                        用法(离线, 给实验脚本):
#   c = VTreeCompressor(probe, tok)    mem, conf = VTreeCompressor.compress(
#   for obs in stream:                     steps, probe, tok, budget=512, task=...)
#       c.push(obs)
#       mem, conf = c.render()
#
# 自测(无需 GPU): python vtree_compressor.py
# ─────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import re
import numpy as np

RES_FULL, RES_SUMM, RES_MERGE = 0, 1, 2
CONF = {RES_FULL: 1.0, RES_SUMM: 0.6, RES_MERGE: 0.3}     # 落地方案 D6: conf 由档位定
FOLD_TMPL = "(此处 {n} 步例行操作已折叠)"


# ══════════════════════════════════════════════════════════════════════
# 1. 价值探针
# ══════════════════════════════════════════════════════════════════════
class ValueProbe:
    """问模型: 当前记忆下这任务能成吗 → P(是). 只做 1 次 forward, 不生成.

    ⚠️ 落地方案 D2: 训练全程必须【冻结】探针行为(用训练开始时的权重快照, 或独立小模型).
       否则 V 的尺度随 actor 更新漂移, 跳幅失去可比性 —— 这条不做, 信号会中途失效.
    """

    PROMPT = "任务:{task}\n当前记忆:\n{mem}\n问:这个任务最终能成功吗?只回答一个字:是 或 否。\n答:"

    def __init__(self, model, tok, yes="是", no="否", max_len=2048, batch=16, device=None):
        import torch
        self.torch = torch
        self.model, self.tok = model, tok
        self.yes_id = tok(yes, add_special_tokens=False).input_ids[0]
        self.no_id = tok(no, add_special_tokens=False).input_ids[0]
        self.max_len, self.batch = max_len, batch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, mem_texts, task=""):
        torch = self.torch
        prompts = [self.PROMPT.format(task=task, mem=m) for m in mem_texts]
        out = []
        with torch.no_grad():
            for i in range(0, len(prompts), self.batch):
                enc = self.tok(prompts[i:i + self.batch], return_tensors="pt", padding=True,
                               truncation=True, max_length=self.max_len).to(self.device)
                lg = self.model(**enc).logits[:, -1, :].float()      # 末位 = 下一个 token 分布
                p = torch.softmax(lg[:, [self.yes_id, self.no_id]], dim=-1)[:, 0]
                out.extend(p.cpu().tolist())
        return np.asarray(out)


class ConstProbe:
    """自测/消融用的假探针: 按关键词给分, 不需要模型."""

    def __init__(self, key_words=("线索", "宝藏", "加热", "钥匙"), hit=0.95, miss=0.5):
        self.kw, self.hit, self.miss = key_words, hit, miss

    def __call__(self, mem_texts, task=""):
        return np.array([self.hit if any(k in m for k in self.kw) else self.miss
                         for m in mem_texts])


class ScriptedProbe:
    """自测用: 按预设价值序列返回, 精确控制每步跳幅(便于验降档顺序)."""

    def __init__(self, values):
        self.values = list(values)

    def __call__(self, mem_texts, task=""):
        return np.array(self.values[:len(mem_texts)])


# ══════════════════════════════════════════════════════════════════════
# 2. 摘要器(中间档) —— 可插拔, 与本方法正交
# ══════════════════════════════════════════════════════════════════════
# ⚠️ 定位很重要, 别搞混两件事:
#     "这一段该压多短" = 按价值跳幅分配预算 → ★本方法的创新, 在 VTreeCompressor 里
#     "怎么把一段话压短" = 文本压缩         → 已有成熟通用方案(LLMLingua/Selective-Context/
#                                            直接让 LLM 摘要), 本方法【不重新发明】
# 所以摘要器做成可换的组件: 换环境不改本方法, 只换/不换摘要器.
# 早期版本把它写成手写正则, 那等于"每个环境都要重新实现一档", 方法就没有通用性了.

class Summarizer:
    """接口: 把 text 压到 <= budget_chars, 返回压缩后的文本. 自带缓存."""

    def __init__(self):
        self._cache: dict[tuple[str, int], str] = {}

    def __call__(self, text: str, budget_chars: int) -> str:
        key = (text, budget_chars)
        if key not in self._cache:
            self._cache[key] = self._summarize(text, budget_chars)
        return self._cache[key]

    def _summarize(self, text: str, budget_chars: int) -> str:
        raise NotImplementedError


class TruncSummarizer(Summarizer):
    """通用兜底: 保首句主干 + 截断. 零成本、零依赖、任何环境都能用, 但质量最差.
    默认用它跑通管线; 质量不够再换 LLMSummarizer."""

    def _summarize(self, text, budget_chars):
        s = re.sub(r"[【】\[\]()（）]", "", text.strip().replace("\n", " "))
        core = re.split(r"[，,。;；]", s)[0]                # 首个子句 = 主干
        if len(core) <= budget_chars:
            return core
        return core[:max(1, budget_chars - 1)] + "…"


class LLMSummarizer(Summarizer):
    """通用: 让 LLM 摘要. 换环境不用改代码 —— 这是解决通用性的默认选项.

    成本可控(见 落地方案 D4 重算): 只有【落在中间档】的步需要摘要(FULL 用原文、
    MERGE 折叠成一句都不需要), 中间档步数 <= 预算/平均摘要长, 且结果进缓存,
    档位不变就不重算 ⇒ 每步摊销最多 1 次短生成.
    """

    PROMPT = "把下面这条 agent 观测压缩成不超过 {n} 个字, 只保留会影响任务成败的信息, 直接输出压缩结果:\n{t}\n压缩结果:"

    def __init__(self, model, tok, device=None, max_new=48):
        super().__init__()
        import torch
        self.torch, self.model, self.tok, self.max_new = torch, model, tok, max_new
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _summarize(self, text, budget_chars):
        enc = self.tok(self.PROMPT.format(n=budget_chars, t=text),
                       return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new,
                                      do_sample=False, pad_token_id=self.tok.pad_token_id)
        s = self.tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        return s[:budget_chars] if s else TruncSummarizer()._summarize(text, budget_chars)


_VERBS = r"(拿起|放下|打开|关上|加热|清洗|切开|走到|前往|查看|搜索|点击|购买|选择|使用|放入|取出)"


class RuleSummarizer(Summarizer):
    """按环境手写的规则(此处为 ALFWorld 式中文动作句). 抽"动作+对象", 丢修饰.

    ⚠️ 【不通用】—— 换环境要重写正则. 只作为: ① 无模型时的快速原型;
       ② 消融基线(证明"摘要器换成什么都不影响本方法的结论").
    """

    def _summarize(self, text, budget_chars):
        s = text.strip().replace("\n", " ")
        hits = re.findall(_VERBS + r"[了过]?\s*([^,，。;；]{0,12})", s)
        if hits:
            return "，".join(f"{v}{o}".strip() for v, o in hits)[:budget_chars]
        return TruncSummarizer()._summarize(text, budget_chars)     # 匹配不上就退回通用兜底


DEFAULT_SUMMARIZER = TruncSummarizer()


# ══════════════════════════════════════════════════════════════════════
# 3. 压缩器
# ══════════════════════════════════════════════════════════════════════
class VTreeCompressor:
    """在线增量的价值失真记忆树.

    每步 push 一次(1 次探针调用), 随时 render 出当前 memory 与每步 conf.
    """

    def __init__(self, probe, tok, budget: int = 512, task: str = "", v_prior: float = 0.5,
                 min_jump: float = 0.01, summarizer=None, summ_ratio: float = 0.45,
                 tiers: int = 3):
        self.probe, self.tok = probe, tok
        self.budget, self.task = budget, task
        self.min_jump = min_jump                # 跳幅低于此值 = 零信息步, 不值得占预算
        self.summarizer = summarizer or DEFAULT_SUMMARIZER   # 可插拔(§2): 与本方法正交
        self.summ_ratio = summ_ratio            # 中间档目标长度 = ratio × 原文长度
        self.tiers = tiers                      # 3=原文/摘要/折叠; 2=只有原文/折叠(消融用)
        self.steps: list[str] = []
        self.jumps: list[float] = []
        self.values: list[float] = []
        self.v_prev = v_prior
        self._n_full = self._n_summ = None      # 各档 token 数缓存
        self._last_mem = ""                     # 上一步 render 的结果(供增量探针用)
        self._fold_cost = self._ntok(FOLD_TMPL.format(n=99))
        self.budget_warning = None

    # ── 每走一步调一次 ────────────────────────────────────────────
    def push(self, obs_text: str) -> float:
        """加入新观测, 算这一步的跳幅. 返回 Δ_t.

        探针输入 = 【上一步的压缩 memory】+ 新观测原文, 不是全量原文.
        理由: ① 全量原文本身会超 context(压缩就是为了解决这个), 用它算价值自相矛盾;
              ② 与部署一致 —— agent 看到的就是压缩后的 memory;
              ③ 副作用是自纠错: 压缩若丢了关键信息, 价值判断会变、跳幅随之变化.
        """
        probe_in = (self._last_mem + "\n" + obs_text).strip() if self._last_mem else obs_text
        v = float(self.probe([probe_in], self.task)[0])
        self.steps.append(obs_text)
        self.values.append(v)
        self.jumps.append(abs(v - self.v_prev))
        self.v_prev = v
        self._n_full = self._n_summ = None                  # 缓存失效
        return self.jumps[-1]

    # ── 随时可调: 渲染当前 memory ─────────────────────────────────
    def render(self) -> tuple[str, list[float]]:
        """按固定预算贪心分配分辨率 → (memory 文本, 每步 conf). 纯 CPU, 无模型调用.

        预算语义: self.budget 是【历史】的预算; 当前观测(最后一步)必须全貌才能决策,
        它的开销【不占】历史预算 —— 否则当前观测一长(如网页文本), 就会把关键步挤掉,
        导致"预算越紧越丢关键步"的反向行为.
        """
        n = len(self.steps)
        if n == 0:
            return "", []
        self._ensure_costs()

        res = [RES_MERGE] * n
        res[n - 1] = RES_FULL                               # 当前步必须全貌(要拿来决策)
        cur_cost = self._n_full[n - 1]                      # 当前观测的开销, 单独记, 不占历史预算

        # 跳幅大的优先升档: MERGE → SUMM → FULL, 花光历史预算为止.
        # tie-break: 跳幅相同时【靠后(更近)的优先】—— 否则平局永远是最早的步胜出, 没道理.
        for i in sorted(range(n - 1), key=lambda k: (-self.jumps[k], -k)):
            if self.jumps[i] < self.min_jump:
                break        # 已按跳幅降序: 剩下的全是零信息步, 预算宁可剩着也不浪费
            for target in ((RES_FULL, RES_SUMM) if self.tiers == 3 else (RES_FULL,)):
                if target >= res[i]:                        # 只升不降
                    continue
                trial = res.copy()
                trial[i] = target
                if self._cost(trial) - cur_cost <= self.budget:
                    res = trial
                    break

        # 预算过紧的告警: 跳幅最大的历史步都没能升上去 = 预算装不下任何关键信息
        if n > 1:
            top = max(range(n - 1), key=lambda k: self.jumps[k])
            if res[top] == RES_MERGE and self.jumps[top] > 1e-6:
                self.budget_warning = (
                    f"预算 {self.budget} 太紧: 跳幅最大的第 {top} 步(Δ={self.jumps[top]:.3f}) "
                    f"仍被折叠, 关键信息已丢失")
            else:
                self.budget_warning = None

        self._last_mem = self._compose(res)
        return self._last_mem, [CONF[r] for r in res]

    # ── 离线一把梭(给实验脚本) ────────────────────────────────────
    @classmethod
    def compress(cls, steps, probe, tok, budget=512, task="", batch_probe=True,
                 summarizer=None, tiers=3):
        """给完整轨迹, 一次压完. batch_probe=True 时把所有前缀打包问, 快很多."""
        c = cls(probe, tok, budget, task, summarizer=summarizer, tiers=tiers)
        if not batch_probe:
            for s in steps:
                c.push(s)
            return c.render()
        # 批量版: 前缀用【原文】(离线无 context 压力), 一次问完
        prefixes = ["\n".join(steps[:i + 1]) for i in range(len(steps))]
        vs = probe(prefixes, task)
        c.steps = list(steps)
        c.values = [float(v) for v in vs]
        prev = 0.5
        c.jumps = []
        for v in c.values:
            c.jumps.append(abs(v - prev))
            prev = v
        c.v_prev = prev
        c._n_full = c._n_summ = None
        return c.render()

    # ── 内部 ──────────────────────────────────────────────────────
    def _summ(self, i: str | int) -> str:
        """第 i 步的中间档文本. 目标长度按压缩比给, 保证 S 档确实比 F 档省 ——
        否则摘要≈原文, 三档会塌成两档."""
        src = self.steps[i] if isinstance(i, int) else i
        return self.summarizer(src, max(8, int(len(src) * self.summ_ratio)))

    def _ntok(self, t): return len(self.tok(t, add_special_tokens=False).input_ids)

    def _ensure_costs(self):
        """预算 token 数预计算: 避免贪心循环里反复 tokenize(否则 O(n²) 次分词)."""
        if self._n_full is not None:
            return
        self._n_full = [self._ntok(s) for s in self.steps]
        self._n_summ = [self._ntok(self._summ(i)) for i in range(len(self.steps))]

    def _cost(self, res) -> int:
        """纯算术: 非 MERGE 步的 token 和 + 折叠段数 × 折叠句成本."""
        tot, in_run = 0, False
        for i, r in enumerate(res):
            if r == RES_MERGE:
                if not in_run:
                    tot += self._fold_cost
                    in_run = True
                continue
            in_run = False
            tot += self._n_full[i] if r == RES_FULL else self._n_summ[i]
        return tot

    def _compose(self, res) -> str:
        """连续 MERGE 段捏成一句 —— 这一句就是树上的中间节点."""
        out, run = [], 0
        for i, r in enumerate(res):
            if r == RES_MERGE:
                run += 1
                continue
            if run:
                out.append(FOLD_TMPL.format(n=run))
                run = 0
            out.append(self.steps[i] if r == RES_FULL else self._summ(i))
        if run:
            out.append(FOLD_TMPL.format(n=run))
        return "\n".join(out)

    # ── 诊断 ──────────────────────────────────────────────────────
    def stats(self) -> dict:
        _, conf = self.render()
        res = [RES_FULL if c == 1.0 else (RES_SUMM if c == 0.6 else RES_MERGE) for c in conf]
        return {"n_steps": len(self.steps), "tokens": self._ntok(self._last_mem),
                "n_full": res.count(RES_FULL), "n_summ": res.count(RES_SUMM),
                "n_merge": res.count(RES_MERGE),
                "top_jump_idx": int(np.argmax(self.jumps)) if self.jumps else -1}


# ══════════════════════════════════════════════════════════════════════
# 自测(无 GPU): 关键步能否在预算收紧时被留到最后
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    class _Tok:                                   # 假分词器: 1 字 = 1 token
        def __call__(self, t, add_special_tokens=False):
            return type("E", (), {"input_ids": list(t)})()

    D = {1.0: "F", 0.6: "S", 0.3: "M"}

    # ── 场景1: 单关键步(Passive T-maze 式) ──────────────────────────
    steps1 = ["你在走廊起点，墙上刻着:宝藏在上侧通道。"] + \
             [f"你在走廊第{k}格，四周空无一物，只能继续往前走。" for k in range(1, 9)] + \
             ["前方分成上侧通道和下侧通道，必须选一个进去。"]
    print("=== 场景1 单关键步: 预算扫描 ===")
    for b in (400, 200, 120, 80, 50):
        mem, conf = VTreeCompressor.compress(steps1, ConstProbe(), _Tok(), budget=b)
        print(f"  预算{b:>4}: 实占{len(mem):>4}字  关键步{'在' if '宝藏' in mem else '丢'}  "
              f"档位={''.join(D[c] for c in conf)}")
    print("  → 零跳幅步不占预算, 所以各预算结果相同、token 恒定(不因预算大就乱花).\n")

    # ── 场景2: 多关键步, 跳幅有大有小 → 看降档顺序 ──────────────────
    steps2 = ["【线索A】你看到墙上写着:钥匙在红色柜子里。",
              "你走过一条走廊，四周空无一物。",
              "你又走了一格，什么也没发生。",
              "【线索B】你听到有人说:红色柜子在二楼。",
              "你继续往前走，没有新发现。",
              "你路过一扇关着的门。",
              "【线索C】地上有张纸条,字迹模糊看不太清。",
              "你走到了楼梯口。",
              "你上了楼。",
              "面前有三个柜子,必须选一个打开。"]
    #                 步0    1     2     3     4     5     6     7     8     9
    vals = [0.90, 0.90, 0.90, 0.65, 0.65, 0.65, 0.72, 0.72, 0.72, 0.72]
    # 跳幅:        0.40  0.00  0.00  0.25  0.00  0.00  0.07  0.00  0.00  0.00
    print("=== 场景2 多关键步(跳幅 A=0.40 > B=0.25 > C=0.07): 预算收紧时谁先被牺牲 ===")
    for b in (300, 160, 110, 70, 40):
        mem, conf = VTreeCompressor.compress(steps2, ScriptedProbe(vals), _Tok(), budget=b)
        keep = [("A" if "线索A" in mem else "-"), ("B" if "线索B" in mem else "-"),
                ("C" if "线索C" in mem else "-")]
        print(f"  预算{b:>4}: 实占{len(mem):>4}字  留下={''.join(keep)}  "
              f"档位={''.join(D[c] for c in conf)}")
    print("  → 预算收紧的牺牲顺序应是 C(0.07) → B(0.25) → A(0.40), 即【跳幅小的先丢】.\n")

    print("=== 场景2 渲染示例(预算 110) ===")
    mem, conf = VTreeCompressor.compress(steps2, ScriptedProbe(vals), _Tok(), budget=110)
    print(mem)
    print("\nconf =", conf)
    print("\n判读: F=原文(叶子) S=摘要(中间层) M=折叠(近根); conf 随档位下降,")
    print("      供 §3.2② 的保真度加权使用 —— 压得粗的步在信用分配时降权.\n")

    # ── 场景3: 换摘要器, 分辨率分配结果应【不变】 ────────────────────
    print("=== 场景3 摘要器可插拔: 换实现, 档位分配不该变(摘要器与本方法正交) ===")
    for name, sm in (("Trunc(通用兜底)", TruncSummarizer()), ("Rule(手写正则,不通用)", RuleSummarizer())):
        mem, conf = VTreeCompressor.compress(steps2, ScriptedProbe(vals), _Tok(),
                                             budget=110, summarizer=sm)
        print(f"  {name:<22} 档位={''.join(D[c] for c in conf)}  实占{len(mem):>4}字")
    print("  → 档位序列一致 = 本方法只管【哪段压多短】; 摘要器只管【怎么压】, 可换.\n")

    # ── 场景4: 三档 vs 两档 —— 中间档到底有没有实质好处 ──────────────
    print("=== 场景4 三档 vs 两档(只留/删): 同预算下保住几条线索 ===")
    print(f"  {'预算':>4} | {'三档(F/S/M)':^26} | {'两档(F/M)':^26}")
    print("  " + "-" * 62)
    for b in (200, 140, 110, 90, 70, 50):
        row = []
        for t in (3, 2):
            mem, conf = VTreeCompressor.compress(steps2, ScriptedProbe(vals), _Tok(),
                                                 budget=b, tiers=t)
            k = "".join(x for x, n in zip("ABC", ("线索A", "线索B", "线索C")) if n in mem) or "-"
            row.append(f"留={k:<3} {len(mem):>3}字 {''.join(D[c] for c in conf)}")
        star = " ←三档多保住线索" if len(row[0].split()[0]) > len(row[1].split()[0]) else ""
        print(f"  {b:>4} | {row[0]:^26} | {row[1]:^26}{star}")
    print("  → 中间档的作用: 预算不够展原文时【降级保主干】, 而不是整条丢掉.")
