# ===============================================================
# evaluate_trained_manager_fast.py
# Only evaluate the trained Manager pipeline (7B model, epoch_2)
# ===============================================================

import os
import re
import json
import time
import torch
import random
import numpy as np
from tqdm import tqdm
from manager_agent import ManagerAgent, FixedSpecialistAgent
from utils import LocalHF
from reward import get_reward_vector
import subagents
import argparse

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

MAX_NEW_ANSWER_TOKENS = 80


# ===============================================================
# 1️⃣ Robust English Answer Extraction
# ===============================================================
def extract_answer_pubmedqa(text: str) -> str:
    if not text:
        return ""
    t = text.strip().lower()
    t = re.sub(r"[\*\[\]\(\)\{\}\:\-\_\"\'<>]+", " ", t)
    t = re.sub(r"\s+", " ", t)
    patterns = [
        r"(?:final|correct|predicted)?\s*answer\s*(?:is|:)?\s*([a-d]|yes|no|true|false)",
        r"the\s+answer\s+is\s+([a-d]|yes|no|true|false)",
        r"option\s*([a-d])",
    ]
    for p in patterns:
        m = re.search(p, t)
        if m:
            return m.group(1).strip().lower()
    sents = re.split(r"[.\n]", t)
    for s in reversed(sents):
        if any(k in s for k in ["yes", "no", "true", "false", "a", "b", "c", "d"]):
            return s.split()[-1]
    return t.split()[0] if t else ""


# ===============================================================
# 2️⃣ Batched Generator
# ===============================================================
def batch_generate(backend, prompts, device):
    """Batch generate using FP16 and deterministic decoding"""
    inputs = backend.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = backend.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_ANSWER_TOKENS,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            num_beams=1
        )
    return backend.tokenizer.batch_decode(outputs, skip_special_tokens=True)


# ===============================================================
# 3️⃣ Manager-driven Evaluation (multi-step, trained only)
# ===============================================================
def run_trained_pipeline(manager, specialists, dataset, device, output_dir, batch_size=1, max_steps=5):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"trained_manager_7b_epoch2.jsonl")

    correct_count, total = 0, 0
    reward_sums = {"correctness": 0.0, "efficiency": 0.0, "quality": 0.0}

    start_time = time.time()

    with open(log_path, "w", encoding="utf-8") as f_log:
        for idx, sample in enumerate(tqdm(dataset, desc="Evaluating [trained_manager_7b]")):
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

                    # Generate output (batched per step if needed)
                    output_text, _, _ = specialist.generate(current_state, history)
                    if specialist_id == "answering":
                        output_text = output_text.strip().split("\n")[0][:50]

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

            rewards = get_reward_vector([(s["specialist"], s["content"]) for s in trajectory],
                                        ground_truth, max_steps)
            correct = rewards["correctness"] >= 0.95
            total += 1
            if correct:
                correct_count += 1
            for k in rewards:
                reward_sums[k] += rewards[k]

            f_log.write(json.dumps({
                "id": idx,
                "problem": state["problem"],
                "ground_truth": ground_truth,
                "trajectory": trajectory,
                "final_answer": final_answer,
                "rewards": rewards,
                "correct": correct
            }, ensure_ascii=False) + "\n")

    acc = correct_count / max(total, 1)
    mean_rewards = {k: v / max(total, 1) for k, v in reward_sums.items()}
    print(f"\n✅ Trained Manager Accuracy: {acc:.2%} ({correct_count}/{total})")
    print(f"📈 Avg Correctness={mean_rewards['correctness']:.3f} | "
          f"Efficiency={mean_rewards['efficiency']:.3f} | "
          f"Quality={mean_rewards['quality']:.3f}")
    print(f"⏱ Total Time: {(time.time()-start_time)/60:.2f} min")
    print(f"📜 Logs saved to: {log_path}")
    return acc, mean_rewards, log_path


# ===============================================================
# 4️⃣ Main
# ===============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    trained_manager_dir = "./trained_manager_grpo/manager/epoch_2"
    validation_path = "./test_indices_grpo.json"
    output_dir = "./eval_logs_7b_fast"

    with open(validation_path, "r", encoding="utf-8") as f:
        val_json = json.load(f)
    test_data = val_json.get("test_samples", val_json)

    backend = LocalHF(model_name, max_tokens=MAX_NEW_ANSWER_TOKENS)
    backend.model.eval()
    backend.model.half()                # FP16 inference
    backend.model.to(device)
    for p in backend.model.parameters():
        p.requires_grad = False

    specialist_names = ["problem_understanding", "reasoning", "computation", "answering"]
    specialists = {n: FixedSpecialistAgent(agent_name=n, model_backend=backend) for n in specialist_names}

    trained_manager = ManagerAgent(model_name, specialist_names, num_rewards=3, freeze_manager_backbone=True)
    checkpoint_path = os.path.join(trained_manager_dir, "heads.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    trained_manager.policy_head.load_state_dict(checkpoint["policy_head"])
    trained_manager.value_head.load_state_dict(checkpoint["value_head"])
    trained_manager.to(device)

    run_trained_pipeline(trained_manager, specialists, test_data, device, output_dir,
                         batch_size=args.batch_size, max_steps=5)


if __name__ == "__main__":
    main()