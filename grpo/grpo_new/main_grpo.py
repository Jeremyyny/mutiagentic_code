# ====== 环境变量必须放在一切 import 前 ======
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
# 如果你的机箱/网络拓扑一般，建议打开以下两行提升稳定性（可能略慢）
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"

import torch
import json
import random
import warnings
from accelerate import Accelerator
from manager_agent import ManagerAgent, FixedSpecialistAgent
from grpo_trainer import GRPOTrainer
from utils import LocalHF

warnings.filterwarnings('ignore', message='Attempting to unscale FP16 gradients')


def load_dataset_from_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except FileNotFoundError:
        print(f"Wrong: File '{path}' could not be found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: File '{path}' is not valid JSON file.")
        return []

    dataset = []
    for row in full_data:
        dataset.append({
            "problem": row.get("question", ""),
            "context": row.get("context", ""),
            "answer": row.get("ground_truth", "")
        })
    return dataset


def main():
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    if accelerator.is_main_process:
        print(f"Using accelerator device: {device}")
        print(f"Distributed setup: {accelerator.state.num_processes} processes")

    config = {
        "manager_model_path": "Qwen/Qwen2.5-7B-Instruct",
        "specialist_model_path": "Qwen/Qwen2.5-7B-Instruct",
        "dataset_path": r"C:\Users\yyn07\Desktop\multi_agent_test\Codes\data\golden_dataset_pubmedqa_qwen2.5_pro_test_500.json",

        "num_epochs": 5,
        "max_steps": 5,
        "manager_lr": 3e-6,
        "specialist_max_tokens": 300,

        "num_samples_per_prompt": 4,
        "grpo_epochs": 3,
        "minibatch_size": 4,
        "ent_coef": 0.01,
        "max_grad_norm": 1.0,

        "verbose_trajectory": True,   # 打开后会把轨迹写到 wandb（仅 rank0）
        "verbose_frequency": 10,

        "reward_dims": ["correctness", "efficiency", "quality"],
        "manager_preference": [1.0, 0.2, 0.3],

        "random_seed": 42,
        "train_ratio": 0.8,

        "freeze_manager_backbone": True,
        "use_value_baseline": True,
        "lambda_coef": 0.5,
        "value_coef": 0.5,
        "normalize_adv": False,
        "manager_token_max_len": 2048,
        "verbose_trajectory" :True,
        "use_wandb": True,
        "wandb_project": "multi-agent-pubmedqa-7b",
        "wandb_run_name": "grpo-qwen2.5-7b-improved-prompts",
        "wandb_tags": ["grpo", "7b", "pubmedqa", "improved-prompts"],
        "wandb_notes": "7B model with enhanced prompts and better history utilization",
    }

    if accelerator.is_main_process:
        print("=" * 60)
        print("LOADING AND SPLITTING DATASET")
        print("=" * 60)

    full_dataset = load_dataset_from_json(config['dataset_path'])
    if not full_dataset:
        print("Error: The dataset is empty")
        return

    random.seed(config['random_seed'])
    shuffled_dataset = full_dataset.copy()
    random.shuffle(shuffled_dataset)

    num_train = int(len(shuffled_dataset) * config['train_ratio'])
    train_dataset = shuffled_dataset[:num_train]
    test_dataset = shuffled_dataset[num_train:]

    if accelerator.is_main_process:
        print(f"✓ Loaded {len(full_dataset)} samples")
        print(f"Training set: {len(train_dataset)}, Test set: {len(test_dataset)}")

    test_indices_file = "./test_indices_grpo_7b.json"
    if accelerator.is_main_process:
        with open(test_indices_file, 'w', encoding='utf-8') as f:
            json.dump({
                "random_seed": config['random_seed'],
                "test_samples": [{"problem": s["problem"][:100], "answer": s["answer"]} for s in test_dataset]
            }, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved test set info to {test_indices_file}")

    if accelerator.is_main_process:
        print("Initializing model backend for 7B...")

    specialist_backend = LocalHF(config['specialist_model_path'], max_tokens=config['specialist_max_tokens'])
    specialist_backend.model.eval()
    for param in specialist_backend.model.parameters():
        param.requires_grad = False

    specialist_names = ["problem_understanding", "reasoning", "computation", "answering"]
    manager = ManagerAgent(
        model_path=config['manager_model_path'],
        specialist_names=specialist_names,
        num_rewards=len(config['reward_dims']),
        freeze_manager_backbone=config['freeze_manager_backbone'],
        manager_token_max_len=config['manager_token_max_len']
    )

    specialists = {
        name: FixedSpecialistAgent(agent_name=name, model_backend=specialist_backend)
        for name in specialist_names
    }

    trainer = GRPOTrainer(
        config=config,
        manager=manager,
        specialists=specialists,
        model_backend=specialist_backend,
        train_dataset=train_dataset,
        accelerator=accelerator
    )

    if accelerator.is_main_process:
        print("\n🚀 Starting GRPO Training (rank0 rollout + multi-GPU sync update, bf16)...")

    trainer.train()

    if accelerator.is_main_process:
        print("\n✅ Training Completed! Checkpoints saved under ./checkpoints_grpo/")
        print(f"✓ Test info: {test_indices_file}")


if __name__ == "__main__":
    main()