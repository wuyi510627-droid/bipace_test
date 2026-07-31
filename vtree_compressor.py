# vtree_compressor.py —— 价值失真记忆树:在线增量压缩的共用模块
# ─────────────────────────────────────────────────────────────────────────
# 一句话: 给定固定 token 预算, 按"这一步让'能不能成'跳了多少"决定每段展开到多细.
#
# 设计约束(见 落地实现方案.md §0):
#   动机是 context 硬上限 ⇒ 压缩必须作用在 agent 决策所看的 memory 上
#   ⇒ 必须【在线】压(边走边压) ⇒ 探针要能增量算、预算重分配要便宜、摘要要带缓存.
#
# 【树结构】mode="tree"(默认) —— 三层:
#   根   = 整条轨迹
#   中间 = 【段】, 边界由价值跳幅的【局部峰值】切出来(转折点 = 阶段切换).
#          段边界只看跳幅, 【与预算无关】⇒ 结构稳定, 不随预算漂移.
#   叶子 = 步
#   压缩 = 给每一步选分辨率, 但折叠块【不跨段】, 整段折叠时用【段标题】而非计数句.
#
# mode="flat" —— 旧行为(无层级): 折叠块由连续 MERGE 自动夹出来, 会跨越语义阶段.
#   ⚠️ 已跑完的实验①③④用的是 flat. 与本版 flat 逐字节一致(800 组随机对拍),
#      要复现旧数字就传 mode="flat"; tree vs flat 本身也是一条可测的消融.
#
# 三档分辨率:
#   RES_FULL  原文一字不差            → 叶子      conf 1.0
#   RES_SUMM  摘要(可插拔摘要器)        → 中间层    conf 0.6
#   RES_MERGE 折叠                    → 近根      conf 0.3
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
SEG_TMPL = "[{title}×{n}步]"          # 树模式: 段折叠成带语义的标题(比计数句更短且有信息)


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
    """自测用: 按预设价值序列返回, 精确控制每步跳幅(便于验降档顺序).

    ⚠️ 必须同时支持两种调用方式, 否则在线路径会静默出错:
       批量(compress): 一次传 T 个前缀  → 一次返回全部
       在线(push):     每次传 1 个文本  → 按游标依次返回 values[0], values[1], ...
       早期版本只按"取前 n 个"实现, 在线模式下每次都返回 values[0], 跳幅恒为 0.
    """

    def __init__(self, values):
        self.values = list(values)
        self.i = 0

    def __call__(self, mem_texts, task=""):
        n = len(mem_texts)
        out = np.array([self.values[min(self.i + k, len(self.values) - 1)] for k in range(n)])
        self.i += n
        return out

    def reset(self):
        self.i = 0
        return self


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
                 tiers: int = 3, mode: str = "tree", max_segs: int = 6):
        self.probe, self.tok = probe, tok
        self.budget, self.task = budget, task
        self.min_jump = min_jump                # 跳幅低于此值 = 零信息步, 不值得占预算
        self.summarizer = summarizer or DEFAULT_SUMMARIZER   # 可插拔(§2): 与本方法正交
        self.summ_ratio = summ_ratio            # 中间档目标长度 = ratio × 原文长度
        self.tiers = tiers                      # 3=原文/摘要/折叠; 2=只有原文/折叠(消融用)
        # mode: "tree" = 先按跳幅峰值切【段】, 再给每段选展开深度(真的有层级);
        #       "flat" = 旧行为, 逐步选档位, 折叠句由连续 MERGE 自动夹出来(无层级).
        #       ⚠️ 已跑完的实验①③④用的都是 flat, 复现旧数字要显式传 mode="flat".
        self.mode = mode
        self.max_segs = max_segs                # 段数上限, 防止短轨迹被切得太碎
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

    # ══════════════════════════════════════════════════════════════
    # 树结构: 分段
    # ══════════════════════════════════════════════════════════════
    def _seg_starts(self) -> list[int]:
        """段边界 = 价值跳幅的显著峰值。返回每段的起始下标。

        【为什么用跳幅分段, 而不是另写一个语义分段器】
          跳幅大的地方就是"事情起了变化"的地方 —— 它天然就是阶段切换点.
          代价为零: 跳幅在 push 时已经算好, 不需要任何新的模型调用, 也不需要
          手写规则(那是 Summarizer 踩过的通用性坑, 不能再踩第二次).

        【和 flat 模式的关键差别】
          flat: 包的边界 = 哪些步碰巧被升档了 ⇒ 预算一改, 结构就变;
          tree: 段的边界 = 跳幅峰值 ⇒ 【与预算无关】, 结构稳定.
          后者正是 credit_rank_test 测出"压缩一致性低"的修法 ——
          同一处境的两条轨迹, 只要价值曲线形状相近, 段结构就相近.
        """
        n = len(self.steps)
        if n <= 1:
            return [0]
        j = np.asarray(self.jumps, dtype=float)
        # 段首 = 跳幅的【局部峰值】: 比前一步大、且不小于后一步.
        #   为什么不用"跳幅 >= 均值"这种全局阈值(第一版就是, 错了):
        #   一条轨迹里各转折点的量级本就不同(0.40 / 0.25 / 0.07 都是真转折),
        #   全局阈值会把小的那个滤掉 —— 实测线索C(Δ=0.07)被均值0.072卡掉,
        #   结果线索B和C被塞进同一段. 局部峰值只问"这里是不是拐点", 不问"拐得够不够大".
        cand = [i for i in range(1, n)
                if j[i] > j[i - 1] and (i == n - 1 or j[i] >= j[i + 1])
                and j[i] >= self.min_jump]
        if len(cand) > self.max_segs - 1:                  # 段太多 ⇒ 只留跳幅最大的几个
            cand = sorted(sorted(cand, key=lambda i: -j[i])[:self.max_segs - 1])
        return [0] + cand

    def segments(self) -> list[tuple[int, int]]:
        """[(start, end_exclusive), ...] —— 树的第二层。"""
        st = self._seg_starts()
        return [(st[k], st[k + 1] if k + 1 < len(st) else len(self.steps))
                for k in range(len(st))]

    def _seg_title(self, a: int, b: int) -> str:
        """段标题 = 段内跳幅最大那一步的摘要（它就是这段的主角）。零额外成本。"""
        lead = max(range(a, b), key=lambda i: self.jumps[i])
        return self._summ(lead)

    def tree(self) -> dict:
        """把当前结构导出来看 —— 树模式下才有意义, 调试/画图用。"""
        segs = self.segments()
        _, conf = self.render()
        return {
            "root": self.task or "(整条轨迹)",
            "mode": self.mode,
            "segments": [
                {"range": [a, b], "n_steps": b - a,
                 "title": self._seg_title(a, b),
                 "max_jump": float(max(self.jumps[a:b])),
                 "level": ("展开到叶子" if max(conf[a:b]) == 1.0
                           else ("部分展开" if max(conf[a:b]) > 0.3 else "整段折叠")),
                 "steps": self.steps[a:b]}
                for k, (a, b) in enumerate(segs)],
        }

    # ── 树模式: 渲染块 ────────────────────────────────────────────
    def _blocks(self, res):
        """把档位数组切成渲染块。两条规则让"树"真正成立:

          ① 折叠块【不跨段】—— flat 模式下, 一个折叠句可能横跨两个语义阶段
             (实测见过"拿起番茄"和"走到微波炉"被捏进同一个包), 那不是树, 是碰巧连着;
          ② 整段被折叠时用【段标题】而非计数句 —— 标题带语义, 计数句不带.

        步级的档位分配【原样保留】(和 flat 用同一套贪心), 所以树模式不会比 flat 更粗糙 ——
        第一版按"整段选一个层级"做, 结果一个段里只要混进废话步就整段被牺牲, 反而更差.
        """
        raw = []
        for (a, b) in self.segments():
            run = []
            for i in range(a, b):
                if res[i] == RES_MERGE:
                    run.append(i)
                    continue
                if run:
                    raw.append(("fold", (a, b, len(run), len(run) == b - a)))
                    run = []
                raw.append(("step", (i, res[i])))
            if run:
                raw.append(("fold", (a, b, len(run), len(run) == b - a)))

        # ★ 相邻的【整段折叠】块合并成一个 —— 不这么做会超预算:
        #   贪心的起点是"全部 MERGE", 树模式下那等于【每段一个标题】.
        #   段一多, 起点成本就已经超了预算, 而贪心只能阻止升档、没法把起点降下来
        #   (随机对拍实测 500 组里 104 组超预算). 合并后起点退回单块, 与 flat 同量级.
        #   注意合并【只影响渲染】, 不动段边界 —— 结构稳定性(与预算无关)因此得以保留.
        out, run = [], []
        for kind, p in raw:
            if kind == "fold":                             # 相邻折叠块一律合并
                run.append(p)
                continue
            if run:
                out.append(("multifold", tuple(run))); run = []
            out.append((kind, p))
        if run:
            out.append(("multifold", tuple(run)))
        return out

    def _block_text(self, kind, p) -> str:
        if kind == "step":
            i, r = p
            return self.steps[i] if r == RES_FULL else self._summ(i)
        if kind == "multifold":
            counted = FOLD_TMPL.format(n=sum(x[2] for x in p))
            if not all(x[3] for x in p):
                return counted          # 含"段内部分折叠" ⇒ 段标题不成立, 只能计数
            # 全是整段折叠 ⇒ 标题拼接 vs 计数句取【短的那个】,
            # 自动在"有语义"和"省地方"之间择优, 且保证不比 flat 贵
            titled = "".join(SEG_TMPL.format(title=self._seg_title(a, b), n=n)
                             for a, b, n, _ in p)
            return titled if self._ntok(titled) <= self._ntok(counted) else counted
        a, b, n, whole = p
        if whole:                                  # 整段折叠 → 段标题(有语义)
            return SEG_TMPL.format(title=self._seg_title(a, b), n=n)
        return FOLD_TMPL.format(n=n)               # 段内部分折叠 → 计数句

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

        # 按【性价比】排序升档: MERGE → SUMM → FULL, 花光历史预算为止.
        #   性价比 = 跳幅 / 原文长度 —— 一步跳幅再大, 若要吃掉半个预算, 可能不如
        #   换两个次要但便宜的步. 纯按跳幅排会让"又长又重要"的步挤掉"又短又还行"的,
        #   走查里见过: Δ=0.07(19字) 被压, Δ=0.05(13字) 反而保住了原文.
        # tie-break: 相同性价比时【靠后(更近)的优先】—— 否则平局永远是最早的步胜出.
        for i in sorted(range(n - 1),
                        key=lambda k: (-self.jumps[k] / max(self._n_full[k], 1), -k)):
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
                 summarizer=None, tiers=3, mode="tree"):
        """给完整轨迹, 一次压完. batch_probe=True 时把所有前缀打包问, 快很多."""
        c = cls(probe, tok, budget, task, summarizer=summarizer, tiers=tiers, mode=mode)
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
        """当前档位分配的 token 开销. tree 走块口径, flat 走旧口径(保证旧实验可复现)."""
        if self.mode == "tree":
            tot = 0
            for kind, p in self._blocks(res):
                if kind == "step":
                    i, r = p
                    tot += self._n_full[i] if r == RES_FULL else self._n_summ[i]
                else:
                    tot += self._ntok(self._block_text(kind, p))
            return tot
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
        if self.mode == "tree":
            return "\n".join(self._block_text(k, p) for k, p in self._blocks(res))
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

    def fidelity(self) -> float:
        """当前 memory 的【整体保真度】∈(0,1] —— 这才是 §3.2② 要用的 conf.

        为什么不是"某一步的档位": 分组用的表示是【整段 memory】(该步当时的历史前缀),
        所以它的可靠性取决于整段被压得多狠, 而非某一步落在哪档.
        而且当前步恒为 FULL, 若取单步档位, conf 恒等于 1.0, 完全没有区分度.

        定义: 各步档位 conf 按【原文长度】加权平均 —— 长的步被压掉, 丢的信息更多.
        """
        _, conf = self.render()
        w = np.asarray(self._n_full, dtype=float)
        c = np.asarray(conf, dtype=float)
        return float((w * c).sum() / max(w.sum(), 1e-9))

    # ── 诊断 ──────────────────────────────────────────────────────
    def stats(self) -> dict:
        _, conf = self.render()
        res = [RES_FULL if c == 1.0 else (RES_SUMM if c == 0.6 else RES_MERGE) for c in conf]
        return {"n_steps": len(self.steps), "tokens": self._ntok(self._last_mem),
                "mode": self.mode, "n_segs": len(self.segments()),
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

    # ── 场景5: 树结构 —— tree 模式比 flat 多了什么 ──────────────────
    print("=== 场景5 tree vs flat: 层级结构 ===")
    c5 = VTreeCompressor(ScriptedProbe(vals).reset(), _Tok(), budget=70,
                         task="找到钥匙并打开正确的柜子", mode="tree")
    for _s in steps2:
        c5.push(_s)
    c5.render()
    t5 = c5.tree()
    print(f"  根: {t5['root']}")
    for k, sg in enumerate(t5["segments"]):
        print(f"  ├─ 段{k} [{sg['title']}]  {sg['n_steps']}步  "
              f"最大跳幅={sg['max_jump']:.2f}  → {sg['level']}")
        for st in sg["steps"]:
            print(f"  │     · {st[:24]}")
    print()
    # 一个最小对照(随机搜索 4000 组找出的最短例子): 同样长度, 信息量天差地别
    _st = ["从抽屉里拿起钥匙", "把钥匙插进了锁孔", "从抽屉里拿起钥匙",
           "环顾四周没有发现", "你走过一条空走廊"]
    _v = [0.35, 0.66, 0.68, 0.76, 0.74]
    print("  ── 最小对照 (预算 35) ──")
    for _mo in ("flat", "tree"):
        _m, _ = VTreeCompressor.compress(_st, ScriptedProbe(_v), _Tok(), budget=35, mode=_mo)
        print(f"    --{_mo}--")
        for line in _m.split("\n"):
            print(f"        {line}")
    print("""    ↑ 第二行是全部差别所在:
        flat "(此处 2 步例行操作已折叠)" —— 折叠了什么, 完全不知道
        tree "[把钥匙插进了锁孔×2步]"    —— 一样长, 但知道这段是关于什么的""")
    print("""
  → tree 相比 flat 的两处改动:
     ① 折叠块【不跨段】—— flat 会把分属不同阶段的步捏进同一句;
        (但相邻折叠块仍会合并, 否则起点就超预算 —— 实测 500 组里 104 组超)
     ② 整段折叠时【段标题 vs 计数句取短的那个】.
        ⚠️ 标题不是总能用上: 多个段一起折叠时拼接往往比计数句还长, 这时自动
           退回计数句 —— 刻意如此, 保证 tree 【任何情况下都不比 flat 贵】.
           随机对拍: 同预算下 tree 保住的原文步数 5.55 vs flat 5.39, 不吃亏.
     段边界由【跳幅的局部峰值】定, 与预算无关 ⇒ 结构稳定(300/300 组验证);
     flat 的包边界会随预算漂移 —— 这正是 credit_rank_test 测出"压缩一致性低"的原因.
     ⚠️ mode='flat' 与本次改动前逐字节一致(800 组随机对拍), 旧实验结论可复现.""")

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
