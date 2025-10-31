# ===============================================================
# evaluate_manager_inference_final_with_fixed.py
# New mode:
#   3️⃣ Fixed Pipeline — Sequential execution of all Specialists, no Manager involvement
# ===============================================================

import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from manager_agent import ManagerAgent, FixedSpecialistAgent
from utils import LocalHF
from reward import extract_answer_pubmedqa
import subagents

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

MAX_NEW_ANSWER_TOKENS = 200


def run_inference(manager, specialists, dataset, device, tag, output_dir, max_steps=5):
    """Standard Manager inference (baseline or trained)"""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{tag}_trajectories.jsonl")
    correct_count = 0
    total = 0

    with open(log_path, "w", encoding="utf-8") as f_log:
        for idx, sample in enumerate(tqdm(dataset, desc=f"Evaluating [{tag}]")):
            state = {
                "problem": sample.get("problem", ""),
                "context": sample.get("context", "")
            }
            ground_truth = sample.get("answer", "").strip().lower()
            history, trajectory, final_answer = [], [], ""

            for step in range(max_steps):
                try:
                    manager_action, _, _, _ = manager.act(state, history)
                    specialist_id = manager_action["specialist_id"]
                    specialist = specialists[specialist_id]
                    current_state = {**state, "instruction": manager_action["input"]}

                    if specialist_id == "answering":
                        output_text, _, _ = specialist.generate(current_state, history)
                        output_text = output_text.strip().split("\n")[0][:20]
                    else:
                        output_text, _, _ = specialist.generate(current_state, history)

                    trajectory.append({
                        "step": step + 1,
                        "specialist": specialist_id,
                        "content": output_text
                    })
                    history.append({"role": "assistant", "content": f"[{specialist_id}]: {output_text}"})

                    if specialist_id == "answering":
                        final_answer = extract_answer_pubmedqa(output_text)
                        break
                except Exception as e:
                    print(f"❌ Error at step {step}: {e}")
                    break

            correct = (final_answer.strip().lower() == ground_truth.strip().lower())
            if correct:
                correct_count += 1
            total += 1

            f_log.write(json.dumps({
                "id": idx,
                "problem": state["problem"],
                "ground_truth": ground_truth,
                "manager_type": tag,
                "trajectory": trajectory,
                "final_answer": final_answer,
                "correct": correct
            }, ensure_ascii=False) + "\n")

    acc = correct_count / max(total, 1)
    print(f"\n✅ [{tag}] Accuracy: {acc:.2%} ({correct_count}/{total})")
    print(f"📜 Logs saved to: {log_path}")
    return acc, log_path


def run_fixed_pipeline(specialists, dataset, output_dir):
    """Execute four Specialists sequentially, without Manager"""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "fixed_pipeline_trajectories.jsonl")
    correct_count, total = 0, 0

    order = ["problem_understanding", "reasoning", "computation", "answering"]

    with open(log_path, "w", encoding="utf-8") as f_log:
        for idx, sample in enumerate(tqdm(dataset, desc="Evaluating [fixed_pipeline]")):
            state = {"problem": sample.get("problem", ""), "context": sample.get("context", "")}
            ground_truth = sample.get("answer", "").strip().lower()

            history, trajectory, final_answer = [], [], ""

            for step, specialist_id in enumerate(order):
                try:
                    specialist = specialists[specialist_id]
                    current_state = {**state, "instruction": "Analyze and process"}
                    output_text, _, _ = specialist.generate(current_state, history)
                    if specialist_id == "answering":
                        output_text = output_text.strip().split("\n")[0][:20]

                    trajectory.append({
                        "step": step + 1,
                        "specialist": specialist_id,
                        "content": output_text
                    })
                    history.append({"role": "assistant", "content": f"[{specialist_id}]: {output_text}"})

                    if specialist_id == "answering":
                        final_answer = extract_answer_pubmedqa(output_text)
                        break
                except Exception as e:
                    print(f"❌ Error at step {step}: {e}")
                    break

            correct = (final_answer.strip().lower() == ground_truth.strip().lower())
            if correct:
                correct_count += 1
            total += 1

            f_log.write(json.dumps({
                "id": idx,
                "problem": state["problem"],
                "ground_truth": ground_truth,
                "manager_type": "fixed_pipeline",
                "trajectory": trajectory,
                "final_answer": final_answer,
                "correct": correct
            }, ensure_ascii=False) + "\n")

    acc = correct_count / max(total, 1)
    print(f"\n✅ [fixed_pipeline] Accuracy: {acc:.2%} ({correct_count}/{total})")
    print(f"📜 Logs saved to: {log_path}")
    return acc, log_path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    trained_manager_dir = "./trained_manager_grpo/manager"
    validation_path = "./test_indices_grpo.json"
    output_dir = "./eval_logs"

    # Load validation set
    with open(validation_path, "r", encoding="utf-8") as f:
        val_json = json.load(f)
    test_data = val_json.get("test_samples", val_json)

    backend = LocalHF(model_name, max_tokens=150)
    backend.model.eval()
    for p in backend.model.parameters():
        p.requires_grad = False

    specialist_names = ["problem_understanding", "reasoning", "computation", "answering"]
    specialists = {n: FixedSpecialistAgent(agent_name=n, model_backend=backend) for n in specialist_names}

    # 1️⃣ Baseline
    baseline_manager = ManagerAgent(model_name, specialist_names, num_rewards=3, freeze_manager_backbone=True)
    baseline_manager.to(device)
    baseline_acc, _ = run_inference(baseline_manager, specialists, test_data, device, "baseline", output_dir)

    # 2️⃣ Trained
    trained_manager = ManagerAgent(model_name, specialist_names, num_rewards=3, freeze_manager_backbone=True)
    checkpoint = torch.load(os.path.join(trained_manager_dir, "heads.pt"), map_location=device)
    trained_manager.policy_head.load_state_dict(checkpoint["policy_head"])
    trained_manager.value_head.load_state_dict(checkpoint["value_head"])
    trained_manager.to(device)
    trained_acc, _ = run_inference(trained_manager, specialists, test_data, device, "trained", output_dir)

    # 3️⃣ Fixed Pipeline
    fixed_acc, _ = run_fixed_pipeline(specialists, test_data, output_dir)

    print("\n" + "=" * 80)
    print("📊 PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"Baseline Manager (random):  {baseline_acc:.2%}")
    print(f"Trained Manager (GRPO):     {trained_acc:.2%}")
    print(f"Fixed Pipeline (sequential):{fixed_acc:.2%}")
    print(f"GRPO vs Baseline Gain:      {trained_acc - baseline_acc:+.2%}")
    print(f"Fixed vs Baseline Gain:     {fixed_acc - baseline_acc:+.2%}")
    print("=" * 80)


if __name__ == "__main__":
    main()