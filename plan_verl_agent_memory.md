# verl-agent 真·memory 表示测法 —— 构建计划（B）

目标：用 agent **部署时真实产出的 memory 表示**，取代脆弱的合成 embedding 探针，
可信地测「压缩 memory 空间的可分性 + 分组信用分配质量」。

---

## 里程碑

### M0 · 定位 memory（先做这一步，需要看源码）
在 verl-agent 里搞清三件事：
1. **history/memory 怎么构建**：是原始 K-window 拼接，还是 LLM 摘要/memory 模块？（recipe 目录 gigpo/hgpo/看 agent loop）
2. **它在哪成为 LLM 的输入**：prompt 里哪一段是 memory；
3. **hidden state 在哪能取**：actor forward 时，memory 对应 token 的隐状态如何拿到（挂 hook 或改 forward 返回 hidden_states）。

产出：一页笔记，标明 memory 的数据结构 + 抽表示的挂载点（填进骨架的 TODO）。

### M1 · 抽表示（少量 rollout，仅推理，4bit，12G 可跑）
在 ALFWorld/WebShop 跑 ~20–50 episode，每个决策点 dump：
`(memory_repr 向量, 真值状态标识 state_id, 动作 action, 该 episode 回报 return)`。
- state_id：用环境的底层状态（ALFWorld 有可读状态）当"真值处境标签"，供 AUC 用。

### M2 · 复用已有诊断
把 dump 的 `memory_repr` 喂进现成的：
- `sep_auc(emb, state_id)` —— 压缩 memory 空间里同处境是否可分（本仓库 compress_auc_test 里已有）；
- `bipace_qv / ca_acc` —— 分组信用分配质量（credit_test 里已有）。

### M3 · 判死活
- 真 memory 上 **AUC 高** → 修正故事（"该在 memory 表示上做分组"）稳 → 进 RL 小实验；
- 真 memory 上 **AUC 低** → 抓手 (a) 让 memory 忠实 / (b) 鲁棒度量 有强动机 → 做方法。

---

## 交付：骨架脚本
`memory_extract.py`（本仓库）已给出骨架，所有需要对着 verl-agent 源码填的地方标了 `# TODO(M0)`。

---

## 第一步行动（现在就能做）
在服务器上把 verl-agent 弄到手，让我读 memory 相关代码来完成 M0：
```
cd ~/cuda12-dev/project
git clone https://github.com/langfengq/verl-agent.git   # 若还没有
# 然后把 verl-agent 里 agent loop / recipe(gigpo,hgpo) / memory 构建相关文件路径告诉我, 我来定位挂载点
```
（若你本地已有 verl-agent，直接告诉我它的路径即可。）
