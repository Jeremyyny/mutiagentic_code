import torch
import json
import random
import warnings
from accelerate import Accelerator
from manager_agent import ManagerAgent, FixedSpecialistAgent
from grpo_trainer import GRPOTrainer
from utils import LocalHF
from accelerate.utils import set_seed


warnings.filterwarnings("ignore", message="Attempting to unscale FP16 gradients")


def load_dataset_from_json(path):
    """Load dataset safely."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return []

    dataset = []
    for row in data:
        dataset.append({
            "problem": row.get("question", ""),
            "context": row.get("context", ""),
            "answer": row.get("ground_truth", "")
        })
    return dataset


def main():
    # ======================================================
    # 1️⃣ Initialize Accelerator (bf16 for RTX 6000 Blackwell)
    # ======================================================
    accelerator = Accelerator(mixed_precision="bf16")
    set_seed(42 + accelerator.process_index, device_specific=True)
    device = accelerator.device
    if accelerator.is_main_process:
        print(f"Using device: {device}")
        print(f"Distributed processes: {accelerator.state.num_processes}")

    # ======================================================
    # 2️⃣ Config
    # ======================================================
    config = {
        "manager_model_path": "Qwen/Qwen2.5-7B-Instruct",
        "specialist_model_path": "Qwen/Qwen2.5-7B-Instruct",
        "dataset_path": r"C:\Users\yyn07\Desktop\multi_agent_test\Codes\data\golden_dataset_pubmedqa_qwen2.5_pro_test_500.json",
        "num_epochs": 5,
        "max_steps": 5,
        "manager_lr": 2e-6,  # ✅ 稍微调低，防止震荡
        "specialist_max_tokens": 200,
        "num_samples_per_prompt": 6,  # ✅ 提高样本稳定性
        "grpo_epochs": 3,
        "minibatch_size": 8,
        "ent_coef": 0.005,
        "max_grad_norm": 1.0,
        "reward_dims": ["correctness", "efficiency", "quality"],
        "manager_preference": [1.0, 0.2, 0.3],
        "random_seed": 42,
        "train_ratio": 0.8,
        "freeze_manager_backbone": True,
        "use_value_baseline": True,
        "lambda_coef": 0.3,
        "value_coef": 0.1,
        "normalize_adv": False,
        "manager_token_max_len": 1024,
        "baseline_warmup_steps": 200,
        "log_interval": 5,
        "verbose_trajectory": True,
        "use_wandb": True,
        "wandb_project": "multi-agent-pubmedqa-7b",
        "wandb_run_name": "grpo-qwen2.5-7b-v2",
        "wandb_tags": ["grpo", "7b", "bf16", "optimized"],
        "wandb_notes": "stable reward baseline + short prompts + accelerated specialists"
    }

    # ======================================================
    # 3️⃣ Dataset
    # ======================================================
    if accelerator.is_main_process:
        print("\n📘 Loading dataset...")

    dataset = load_dataset_from_json(config["dataset_path"])
    if not dataset:
        print("Dataset empty or invalid.")
        return

    random.seed(config["random_seed"])
    random.shuffle(dataset)
    n_train = int(len(dataset) * config["train_ratio"])
    train_set, test_set = dataset[:n_train], dataset[n_train:]

    if accelerator.is_main_process:
        print(f"✓ Train: {len(train_set)}, Test: {len(test_set)}")

    with open("./test_indices.json", "w", encoding="utf-8") as f:
        json.dump({"test_samples": test_set[:10]}, f, indent=2, ensure_ascii=False)

    # ======================================================
    # 4️⃣ Models
    # ======================================================
    if accelerator.is_main_process:
        print("\n⚙️ Initializing models...")

    specialist_backend = LocalHF(config["specialist_model_path"], max_tokens=config["specialist_max_tokens"])

    specialist_names = ["problem_understanding", "reasoning", "computation", "answering"]
    manager = ManagerAgent(
        model_path=config["manager_model_path"],
        specialist_names=specialist_names,
        num_rewards=len(config["reward_dims"]),
        freeze_manager_backbone=config["freeze_manager_backbone"],
        manager_token_max_len=config["manager_token_max_len"],
    )
    specialists = {
        name: FixedSpecialistAgent(agent_name=name, model_backend=specialist_backend)
        for name in specialist_names
    }

    # ======================================================
    # 5️⃣ Trainer
    # ======================================================
    trainer = GRPOTrainer(
        config=config,
        manager=manager,
        specialists=specialists,
        model_backend=specialist_backend,
        train_dataset=train_set,
        accelerator=accelerator,
    )

    if accelerator.is_main_process:
        print("\n🚀 Starting GRPO Training...\n")

    trainer.train()

    if accelerator.is_main_process:
        print("\n✅ Training finished! Checkpoints under ./checkpoints_grpo/")
        print("Test samples saved to ./test_indices.json")


if __name__ == "__main__":
    main()