import numpy as np

# 可按需开关
DEBUG_REWARD = False

def _clip01(x):
    return float(max(0.0, min(1.0, x)))

def get_reward_vector(final_trajectory, ground_truth, max_steps):
    """
    返回 dict：{"correctness": [0,1], "efficiency": [0,1], "quality": [0,1]}
    所有分量已统一到 [0,1]，减少量纲不一致带来的震荡。
    """
    # correctness：完全匹配或松匹配
    answer_text = ""
    for sid, content in final_trajectory:
        if sid == "answering":
            answer_text = content
            break
    corr = 1.0 if _exact_hit(answer_text, ground_truth) else _fuzzy_hit(answer_text, ground_truth)
    correctness = _clip01(corr)

    # efficiency：步数越少越好
    steps = len(final_trajectory)
    efficiency = _clip01(1.0 - (steps - 1) / max(1, max_steps - 1))

    # quality：简单基于长度与标点结构的启发式分数，可按需替换为更复杂的打分器
    q = _quality_heuristic(answer_text)
    quality = _clip01(q)

    out = {"correctness": correctness, "efficiency": efficiency, "quality": quality}
    if DEBUG_REWARD:
        print("REWARD:", out)
    return out


def _exact_hit(ans, gt):
    a = _clean(ans); g = _clean(gt)
    return a == g and len(a) > 0

def _fuzzy_hit(ans, gt):
    a = _clean(ans); g = _clean(gt)
    if not a or not g:
        return 0.0
    # Jaccard 相似度（词级）
    aset, gset = set(a.split()), set(g.split())
    inter = len(aset & gset)
    union = len(aset | gset)
    return inter / union if union else 0.0

def _quality_heuristic(text):
    if not text:
        return 0.0
    t = _clean(text)
    n = len(t.split())
    # 20~80 词最优，超出或不足按线性降
    if n < 10:
        base = n / 10.0
    elif n <= 120:
        base = 1.0
    else:
        base = max(0.0, 1.0 - (n - 120) / 120.0)
    # 含结论性词汇给一点奖励
    bonus = 0.1 if any(k in t for k in ["therefore", "thus", "in conclusion", "综上", "因此"]) else 0.0
    return min(1.0, base + bonus)

def _clean(s):
    return " ".join(str(s).strip().lower().replace("\n", " ").split())