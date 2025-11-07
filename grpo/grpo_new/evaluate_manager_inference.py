# ===============================================================
# Evaluation Script
# Supports multi-dimensional rewards and robust English answer extraction
# ===============================================================

import os
import re
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from manager_agent import ManagerAgent, FixedSpecialistAgent
from utils import LocalHF
from reward import get_reward_vector
import subagents

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

MAX_NEW_ANSWER_TOKENS = 200


# ===============================================================
# Robust English Answer Extraction
# ===============================================================
def extract_answer_pubmedqa(text: str) -> str:
    """
    Extract the final answer from model output.
    Handles patterns such as:
    - "Answer: Yes"
    - "Final Answer: No"
    - "The correct answer is (C)"
    - "Therefore, the final answer is True"
    """

    if not text:
        return ""

    t = text.strip().lower()
    t = re.sub(r"[\*\[\]\(\)\{\}\:\-\_\"\'<>]+", " ", t)
    t = re.sub(r"\s+", " ", t)

    # Common answer patterns
    patterns = [
        r"(?:final|correct|true|predicted)?\s*answer\s*(?:is|:)?\s*([a-d]|yes|no|true|false)",
        r"the\s+answer\s+is\s+([a-d]|yes|no|true|false)",
        r"option\s*([a-d])",
    ]

    for p in patterns:
        m = re.search(p, t)
        if m:
            return m.group(1).strip().lower()

    # Fallback: take last sentence containing answer tokens
    sents = re.split(r"[.\n]", t)
    for s in reversed(sents):
        s = s.strip()
        if len(s) > 1 and any(k in s for k in ["yes", "no", "true", "false", "a", "b", "c", "d"]):
            return s.split()[-1]

    # Final fallback
    return t.split()[0] if t else ""


# ===============================================================
# Evaluation Core
# ===============================================================
def run_inference(manager, specialists, dataset, device, tag, output_dir, max_steps=5):
    """Run inference with the manager and compute rewards."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{tag}_trajectories.jsonl")

    correct_count = 0
    total = 0
    reward_sums = {"correctness": 0.0, "efficiency": 0.0, "quality": 0.0}

    with open(log_path, "w", encoding="utf-8") as f_log:
        for idx, sample in enumerate(tqdm(dataset, desc=f"Evaluating [{tag}]")):
            state = {"problem": sample.get("problem", ""), "context": sample.get("context", "")}
            ground_truth = sample.get("answer", "").strip().lower()
            history, trajectory = [], []

            for step in range(max_steps):
                try:
                    manager_action, _, _, _ = manager.act(state, history)
                    specialist_id = manager_action["specialist_id"]
                    specialist = specialists[specialist_id]
                    current_state = {**state, "instruction": manager_action["input"]}

                    output_text, _, _ = specialist.generate(current_state, history)
                    if specialist_id == "answering":
                        output_text = output_text.strip()[:200]

                    trajectory.append({
                        "step": step + 1,
                        "specialist": specialist_id,
                        "content": output_text
                    })
                    history.append({"role": "assistant", "content": f"[{specialist_id}]: {output_text}"})

                    if specialist_id == "answering":
                        break
                except Exception as e:
                    print(f"❌ Error at step {step}: {e}")
                    break

            # Extract final answer
            answer_text = ""
            for t in trajectory:
                if t["specialist"] == "answering":
                    answer_text = extract_answer_pubmedqa(t["content"])
                    break

            # Compute rewards
            traj_pairs = [(t["specialist"], t["content"]) for t in trajectory]
            rewards = get_reward_vector(traj_pairs, ground_truth, max_steps)

            correct = rewards["correctness"] >= 0.95
            if correct:
                correct_count += 1
            total += 1
            for k in rewards:
                reward_sums[k] += rewards[k]

            f_log.write(json.dumps({
                "id": idx,
                "problem": state["problem"],
                "ground_truth": ground_truth,
                "final_answer": answer_text,
                "trajectory": trajectory,
                "manager_type": tag,
                "rewards": rewards,
                "correct": correct
            }, ensure_ascii=False) + "\n")

    acc = correct_count / max(total, 1)
    mean_rewards = {k: v / max(total, 1) for k, v in reward_sums.items()}

    print(f"\n✅ [{tag}] Accuracy: {acc:.2%} ({correct_count}/{total})")
    print(f"📈 Avg Correctness={mean_rewards['correctness']:.3f} | "
          f"Efficiency={mean_rewards['efficiency']:.3f} | "
          f"Quality={mean_rewards['quality']:.3f}")
    print(f"📜 Logs saved to: {log_path}")

    return acc, mean_rewards, log_path


def run_fixed_pipeline(specialists, dataset, output_dir, max_steps=5):
    """Run specialists sequentially (no manager)."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "fixed_pipeline_trajectories.jsonl")

    order = ["problem_understanding", "reasoning", "computation", "answering"]
    correct_count, total = 0, 0
    reward_sums = {"correctness": 0.0, "efficiency": 0.0, "quality": 0.0}

    with open(log_path, "w", encoding="utf-8") as f_log:
        for idx, sample in enumerate(tqdm(dataset, desc="Evaluating [fixed_pipeline]")):
            state = {"problem": sample.get("problem", ""), "context": sample.get("context", "")}
            ground_truth = sample.get("answer", "").strip().lower()
            history, trajectory = [], []

            for step, specialist_id in enumerate(order):
                try:
                    specialist = specialists[specialist_id]
                    current_state = {**state, "instruction": "Analyze and process"}
                    output_text, _, _ = specialist.generate(current_state, history)
                    if specialist_id == "answering":
                        output_text = output_text.strip()[:200]

                    trajectory.append({
                        "step": step + 1,
                        "specialist": specialist_id,
                        "content": output_text
                    })
                    history.append({"role": "assistant", "content": f"[{specialist_id}]: {output_text}"})

                    if specialist_id == "answering":
                        break
                except Exception as e:
                    print(f"❌ Error at step {step}: {e}")
                    break

            answer_text = extract_answer_pubmedqa(
                next((t["content"] for t in trajectory if t["specialist"] == "answering"), "")
            )

            traj_pairs = [(t["specialist"], t["content"]) for t in trajectory]
            rewards = get_reward_vector(traj_pairs, ground_truth, max_steps)

            correct = rewards["correctness"] >= 0.95
            if correct:
                correct_count += 1
            total += 1
            for k in rewards:
                reward_sums[k] += rewards[k]

            f_log.write(json.dumps({
                "id": idx,
                "problem": state["problem"],
                "ground_truth": ground_truth,
                "final_answer": answer_text,
                "trajectory": trajectory,
                "manager_type": "fixed_pipeline",
                "rewards": rewards,
                "correct": correct
            }, ensure_ascii=False) + "\n")

    acc = correct_count / max(total, 1)
    mean_rewards = {k: v / max(total, 1) for k, v in reward_sums.items()}

    print(f"\n✅ [fixed_pipeline] Accuracy: {acc:.2%} ({correct_count}/{total})")
    print(f"📈 Avg Correctness={mean_rewards['correctness']:.3f} | "
          f"Efficiency={mean_rewards['efficiency']:.3f} | "
          f"Quality={mean_rewards['quality']:.3f}")
    print(f"📜 Logs saved to: {log_path}")

    return acc, mean_rewards, log_path


# ===============================================================
# Main Entry
# ===============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    trained_manager_dir = "./trained_manager_grpo/manager"
    validation_path = "./test_indices_grpo.json"
    output_dir = "./eval_logs"

    # Load validation data
    with open(validation_path, "r", encoding="utf-8") as f:
        val_json = json.load(f)
    test_data = val_json.get("test_samples", val_json)

    backend = LocalHF(model_name, max_tokens=150)
    backend.model.eval()
    for p in backend.model.parameters():
        p.requires_grad = False

    specialist_names = ["problem_understanding", "reasoning", "computation", "answering"]
    specialists = {n: FixedSpecialistAgent(agent_name=n, model_backend=backend) for n in specialist_names}

    # Baseline manager
    baseline_manager = ManagerAgent(model_name, specialist_names, num_rewards=3, freeze_manager_backbone=True)
    baseline_manager.to(device)
    baseline_acc, baseline_r, _ = run_inference(baseline_manager, specialists, test_data, device, "baseline", output_dir)

    # Trained manager
    trained_manager = ManagerAgent(model_name, specialist_names, num_rewards=3, freeze_manager_backbone=True)
    checkpoint = torch.load(os.path.join(trained_manager_dir, "heads.pt"), map_location=device)
    trained_manager.policy_head.load_state_dict(checkpoint["policy_head"])
    trained_manager.value_head.load_state_dict(checkpoint["value_head"])
    trained_manager.to(device)
    trained_acc, trained_r, _ = run_inference(trained_manager, specialists, test_data, device, "trained", output_dir)

    # Fixed pipeline (no manager)
    fixed_acc, fixed_r, _ = run_fixed_pipeline(specialists, test_data, output_dir)

    print("\n" + "=" * 90)
    print("📊 PERFORMANCE COMPARISON")
    print("=" * 90)
    print(f"Baseline:  Acc={baseline_acc:.2%}  | Rewards={baseline_r}")
    print(f"Trained:   Acc={trained_acc:.2%}  | Rewards={trained_r}")
    print(f"Fixed:     Acc={fixed_acc:.2%}    | Rewards={fixed_r}")
    print("=" * 90)


if __name__ == "__main__":
    main()