# memory_extract.py —— 骨架: 从 verl-agent 真实 rollout 抽 agent 的 memory 表示, 供诊断
# 所有需对着 verl-agent 源码填的地方标了 TODO(M0). 填完即可跑 M1→M2.
# 复用本仓库现成诊断: sep_auc(compress_auc_test), bipace_qv/ca_acc(credit_test).
#
# 用法(填完TODO后): python memory_extract.py  → 产出 mem_dump.npz + 诊断打印

import numpy as np, torch
from collections import defaultdict
from sklearn.metrics import roc_auc_score

# ============================================================
# 复用的诊断函数(和仓库其它脚本一致)
# ============================================================
def sep_auc(emb, labels):
    S = emb @ emb.T; iu = np.triu_indices(len(emb), k=1)
    same = (labels[iu[0]] == labels[iu[1]]).astype(int)
    if same.min()==same.max(): return float("nan")
    return roc_auc_score(same, S[iu])

def greedy_cosine_cluster(emb, eps):
    cents, members = [], []
    for i, x in enumerate(emb):
        if cents:
            sims = np.array([c @ x for c in cents]); j = int(sims.argmax())
            if (1 - sims[j]) < eps:
                members[j].append(i); c = emb[members[j]].mean(0); cents[j] = c/(np.linalg.norm(c)+1e-9); continue
        cents.append(x.copy()); members.append([i])
    pred = np.empty(len(emb), int)
    for cid, m in enumerate(members):
        for idx in m: pred[idx] = cid
    return pred

def hard_qv(keys, act, ret):
    adv = np.zeros(len(ret)); g = defaultdict(list)
    for i,k in enumerate(keys): g[k].append(i)
    for k, idx in g.items():
        idx = np.array(idx); v = ret[idx].mean()
        for a in set(act[idx].tolist()):
            sa = idx[act[idx]==a]
            if len(sa)>0: adv[sa] = ret[sa].mean() - v
    return adv

# ============================================================
# M0/M1: 从 verl-agent 抽 (memory表示, 真值状态id, 动作, 回报)
# ============================================================
def rollout_and_extract(n_episodes=30, eps_cluster=0.15):
    """
    TODO(M0): 对照 verl-agent 源码填三处挂载点.
    """
    # --- TODO(M0-1): 载入 verl-agent 的 actor 模型与环境 ---
    #   from verl_agent... import build_actor, make_env
    #   actor = build_actor(ckpt=..., load_in_4bit=True)      # 12G: 4bit 推理
    #   env   = make_env("alfworld")                          # 或 webshop
    raise NotImplementedError("填 M0-1: 载入 actor 与 env")

    records = []  # 每条: dict(repr=np.array, state_id=str/int, action=int, ret=float)
    for ep in range(n_episodes):
        obs = env.reset(); done = False; traj = []
        while not done:
            # --- TODO(M0-2): 构建 memory 输入 & 取隐状态 ---
            #   memory_text = agent.build_memory(traj_history)   # verl-agent 里 history/memory 的构建函数
            #   out = actor.forward(memory_text, output_hidden_states=True)
            #   h = out.hidden_states[LAYER]                      # [T,H]
            #   memory_repr = normalize(mean_pool_over_memory_tokens(h))  # = f(memory), 归一化
            #   action = actor.act(out)
            # --- TODO(M0-3): 取环境底层真值状态当标签(供AUC) ---
            #   state_id = env.underlying_state_id()             # ALFWorld 有可读状态
            memory_repr = None; state_id = None; action = None   # <- 由上面填出
            traj.append(dict(repr=memory_repr, state_id=state_id, action=action))
            obs, reward, done, info = env.step(action)
        ep_return = info["episode_return"]                       # TODO(M0): 取该 episode 回报
        for r in traj: r["ret"] = ep_return; records.append(r)

    # 打包
    E   = np.stack([r["repr"] for r in records]).astype(float)
    sid = np.array([r["state_id"] for r in records])
    act = np.array([r["action"] for r in records])
    ret = np.array([r["ret"] for r in records], float)
    # 把 state_id 映射成整数标签
    u = {s:i for i,s in enumerate(sorted(set(sid.tolist())))}
    lab = np.array([u[s] for s in sid])
    np.savez("mem_dump.npz", E=E, lab=lab, act=act, ret=ret)
    return E, lab, act, ret

# ============================================================
# M2: 在真 memory 表示上跑诊断
# ============================================================
def diagnose(E, lab, act, ret, eps_cluster=0.15):
    auc = sep_auc(E, lab)
    pred = greedy_cosine_cluster(E, eps_cluster)
    # 分组信用分配质量: 用真值状态分组(上界) vs memory聚类分组
    def ca_sign_match(adv):
        # 无合成"关键步", 这里粗看: memory聚类 vs 真值分组 给出的优势符号一致率
        return float((np.sign(adv) == np.sign(hard_qv(lab, act, ret))).mean())
    match = ca_sign_match(hard_qv(pred, act, ret))
    print(f"真memory表示: sep-AUC={auc:.3f}  (高→同处境可分, 修正故事稳; 低→有损, 抓手a/b有据)")
    print(f"memory聚类 vs 真值分组 的优势符号一致率={match:.3f}")
    print(f"簇数={pred.max()+1}  样本数={len(lab)}  真值处境数={lab.max()+1}")

if __name__ == "__main__":
    E, lab, act, ret = rollout_and_extract()
    diagnose(E, lab, act, ret)
