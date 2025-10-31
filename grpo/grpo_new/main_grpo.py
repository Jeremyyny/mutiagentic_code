
# =============================
# main_grpo.py (改进版)
# 变化点：
# 1) 新增 freeze_manager_backbone、use_value_baseline、lambda/value 系数、manager_token_max_len
# 2) specialist_max_tokens 默认 150
# 3) 仅训练 heads 的说明
# =============================

import torch
import json
import random
import warnings
from manager_agent import ManagerAgent, FixedSpecialistAgent
from grpo_trainer import GRPOTrainer
from utils import LocalHF

warnings.filterwarnings('ignore', message='Attempting to unscale FP16 gradients')

def load_dataset_from_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件 '{path}' 未找到。")
        return []
    except json.JSONDecodeError:
        print(f"错误: 文件 '{path}' 不是有效的 JSON 文件。")
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
    config = {
        # === 1. 模型路径 ===
        "manager_model_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "specialist_model_path": "Qwen/Qwen2.5-0.5B-Instruct",
        
        # === 2. 数据路径 ===
        "dataset_path": r"C:\\Users\\yyn07\\Desktop\\multi_agent_test\\Codes\\data\\golden_dataset_pubmedqa_qwen2.5_pro_test_500.json",

        # === 3. 训练超参数 ===
        "num_epochs": 3,
        "max_steps": 5,
        "manager_lr": 5e-6,  # 更稳健（只训 heads）
        "specialist_max_tokens": 150,
        
        # === 4. GRPO 参数 ===
        "num_samples_per_prompt": 4,
        "grpo_epochs": 3,
        "minibatch_size": 8,
        "ent_coef": 0.01,
        "max_grad_norm": 1.0,
        
        # === 5. 输出控制 ===
        "verbose_trajectory": True,
        "verbose_frequency": 5,
        
        # === 6. 混合精度 ===
        "use_amp": True,
        
        # === 7. 奖励维度和偏好 ===
        "reward_dims": ["correctness", "efficiency", "quality"],
        "manager_preference": [1.0, 0.1, 0.2],
        
        # === 8. 数据划分 ===
        "random_seed": 42,
        "train_ratio": 0.8,
        
        # === 9. Manager 训练范围与 Baseline ===
        "freeze_manager_backbone": True,      # 仅训练 heads
        "use_value_baseline": True,           # 启用 GRPO-λ 和 value loss
        "lambda_coef": 0.5,                   # GRPO-λ 中 V 的权重
        "value_coef": 0.5,                    # value loss 权重
        "normalize_adv": False,               # 是否标准化组优势
        "manager_token_max_len": 1024,
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("="*60)
    print("LOADING AND SPLITTING DATASET")
    print("="*60)
    
    full_dataset = load_dataset_from_json(config['dataset_path'])
    
    if not full_dataset:
        print("错误: 数据集为空!")
        return
    
    print(f"✓ Loaded {len(full_dataset)} samples")
    
    random.seed(config['random_seed'])
    print(f"✓ Set random seed to {config['random_seed']}")
    
    shuffled_dataset = full_dataset.copy()
    random.shuffle(shuffled_dataset)
    
    num_train = int(len(shuffled_dataset) * config['train_ratio'])
    train_dataset = shuffled_dataset[:num_train]
    test_dataset = shuffled_dataset[num_train:]
    
    print(f"\n✓ Split completed:")
    print(f"  Training set:   {len(train_dataset)} samples")
    print(f"  Test set:       {len(test_dataset)} samples")
    print("="*60)
    
    test_indices_file = "./test_indices_grpo.json"
    with open(test_indices_file, 'w', encoding='utf-8') as f:
        json.dump({
            "random_seed": config['random_seed'],
            "test_samples": [
                {"problem": s["problem"][:100], "answer": s["answer"]} 
                for s in test_dataset
            ]
        }, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved test set info to {test_indices_file}\n")

    print("Initializing model backend...")
    specialist_backend = LocalHF(
        config['specialist_model_path'], 
        max_tokens=config['specialist_max_tokens']
    )
    specialist_backend.model.eval()
    for param in specialist_backend.model.parameters():
        param.requires_grad = False
    print("✓ Specialist model frozen (no training)")

    print("\nInitializing Manager and Specialists...")
    specialist_names = ["problem_understanding", "reasoning", "computation", "answering"]
    
    manager = ManagerAgent(
        model_path=config['manager_model_path'], 
        specialist_names=specialist_names,
        num_rewards=len(config['reward_dims']),
        freeze_manager_backbone=config['freeze_manager_backbone'],
        manager_token_max_len=config['manager_token_max_len']
    )
    
    print("="*60)
    print("Manager Configuration")
    print("="*60)
    print(f"Specialists: {specialist_names}")
    print(f"Reward dims: {config['reward_dims']}")
    print(f"Preference:  {config['manager_preference']}")
    print(f"Freeze backbone: {config['freeze_manager_backbone']}")
    print(f"Use value baseline: {config['use_value_baseline']} (lambda={config['lambda_coef']}, value_coef={config['value_coef']})")
    print("="*60)
    
    print("\nTesting Manager...")
    test_state = {"problem": "Test: What is 2+2?", "context": ""}
    with torch.no_grad():
        test_action, _, _, _ = manager.act(test_state, [])
        print(f"✓ Manager selected: {test_action['specialist_id']}")
    
    specialists = {
        name: FixedSpecialistAgent(agent_name=name, model_backend=specialist_backend)
        for name in specialist_names
    }
    print(f"✓ Created {len(specialists)} fixed specialists")

    print("\nInitializing GRPO Trainer...")
    trainer = GRPOTrainer(
        config=config,
        manager=manager,
        specialists=specialists,
        model_backend=specialist_backend,
        train_dataset=train_dataset
    )
    
    print("="*60)
    print("GRPO Configuration")
    print("="*60)
    print(f"Samples per prompt:  {config['num_samples_per_prompt']}")
    print(f"GRPO epochs:         {config['grpo_epochs']}")
    print(f"Manager LR:          {config['manager_lr']}")
    print(f"Minibatch size:      {config['minibatch_size']}")
    print("="*60)

    print("\n🚀 Starting GRPO Training...")
    print("💡 Note: Only Manager HEADS are being trained (backbone frozen)\n")
    
    trainer.train()
    
    print("\n💾 Saving trained Manager...")
    trainer.save_manager("./trained_manager_grpo")
    print("✓ Training completed!")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print(f"✓ Manager saved to: ./trained_manager_grpo")
    print(f"✓ Test set info saved to: {test_indices_file}")
    print(f"\nTo evaluate:")
    print(f"  1. Load the trained Manager")
    print(f"  2. Use the test set (last {len(test_dataset)} samples)")
    print(f"  3. Run inference with fixed Specialists")
    print("="*60)


if __name__ == "__main__":
    main()
