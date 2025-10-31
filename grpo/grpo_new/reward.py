import re

def extract_answer_pubmedqa(last_value: str) -> str:
    if not last_value or not isinstance(last_value, str):
        return ""
    text = last_value.strip()
    if text.startswith("[") and "]:" in text:
        text = text.split("]:", 1)[1].strip()
    if "\n" in text:
        text = text.split("\n")[0].strip()
    text_clean = re.sub(r"[.,!?;:'\"]", "", text.lower()).strip()
    words = text_clean.split()
    if words:
        first = words[0]
        if first in {"yes", "no", "maybe"}:
            return first
    for w in words[:3]:
        if w in {"yes", "no", "maybe"}:
            return w
    for tok in ("yes", "no", "maybe"):
        m = re.search(rf"\b{tok}\b", text_clean)
        if m:
            return tok
    return ""


def calculate_correctness_reward(final_output_text: str, ground_truth: str) -> float:
    pred = extract_answer_pubmedqa(final_output_text)
    return 2.0 if pred == ground_truth.lower() else 0.0


def calculate_efficiency_reward(num_steps: int, max_steps: int) -> float:
    return -0.2 * (num_steps / max_steps)


def calculate_quality_reward(agent_name: str, agent_output: str) -> float:
    s = 0.0
    low = agent_output.lower()
    if agent_name == "problem_understanding" and any(k in low for k in ["condition", "population", "endpoint", "answer type", "yes/no/maybe"]):
        s += 0.2
    if agent_name == "reasoning" and ("stance:" in low or "evidence" in low or "supports" in low):
        s += 0.3
    if agent_name == "computation" and ("summary:" in low or any(k in low for k in ["sensitivity", "specificity", "%", "ci", "rr", "n="])):
        s += 0.2
    if agent_name == "answering" and extract_answer_pubmedqa(agent_output) != "":
        s += 0.5
    if len(agent_output) > 600:
        s -= 0.1
    return s


def get_reward_vector(trajectory: list, ground_truth: str, max_steps: int) -> dict:
    if not trajectory:
        return {"correctness": 0.0, "efficiency": -0.2, "quality": 0.0}
    final_agent, final_output = trajectory[-1]
    num_steps = len(trajectory)
    r_correctness = calculate_correctness_reward(final_output, ground_truth) if final_agent == "answering" else 0.0
    r_efficiency = calculate_efficiency_reward(num_steps, max_steps)
    r_quality = sum(calculate_quality_reward(name, out) for name, out in trajectory) / max(num_steps, 1)
    return {"correctness": r_correctness, "efficiency": r_efficiency, "quality": r_quality}
