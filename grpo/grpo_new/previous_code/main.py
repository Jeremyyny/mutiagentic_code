import torch
import csv
import json
from agents import ConductorAgent, SpecialistAgent
from trainer import CoEvolutionTrainer
from utils import LocalHF
import subagents 
import random
import warnings

# 忽略 FP16 警告
warnings.filterwarnings('ignore', message='Attempting to unscale FP16 gradients')

def load_dataset_from_json(path):
    """从JSON文件加载并格式化数据集。"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件 '{path}' 未找到。请检查路径。")
        return []
    except json.JSONDecodeError:
        print(f"错误: 文件 '{path}' 不是一个有效的JSON文件。")
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
        # --- 1. 模型路径 ---
        "conductor_model_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "reasoning_model_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "general_model_path": "Qwen/Qwen2.5-0.5B-Instruct",
        
        # --- 2. 数据路径 ---
        "dataset_path": r"C:\Users\yyn07\Desktop\multi_agent_test\Codes\data\golden_dataset_pubmedqa_qwen2.5_pro_test_500.json",

        # --- 3. 训练超参数 ---
        "num_epochs": 1,
        "batch_size": 1,
        "max_steps": 5,
        "conductor_lr": 1e-5,
        "specialist_lr": 3e-5,
        "ppo_epochs": 4,
        "minibatch_size": 4,
        "verbose_trajectory": True,
        "verbose_frequency": 5,
        
        # --- 4. PPO / GAE 参数 ---
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.02,
        "max_grad_norm": 1.0,
        
        # --- 5. 混合精度 ---
        "use_amp": True,  # 保持开启 AMP
        
        # --- 6. GRPO 偏好向量 ---
        "reward_dims": ["correctness", "efficiency", "quality"],
        "conductor_preference": [1.0, 0.1, 0.2],
        "specialist_preferences": {
            "problem_understanding": [0.8, 0.3, 0.5],
            "reasoning": [1.0, 0.1, 0.5],
            "computation": [1.0, 0.5, 0.1],
            "answering": [1.0, 0.1, 0.3],
            "default": [1.0, 0.1, 0.1]
        },
        
        # ⭐ 新增: 随机种子配置
        "random_seed": 42,  # 固定随机种子,保证可复现
        "train_ratio": 0.8,  # 80% 训练, 20% 测试
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========================================================================
    # ⭐ 1. 加载完整数据集并随机划分
    # ========================================================================
    
    print("="*60)
    print("LOADING AND SPLITTING DATASET")
    print("="*60)
    
    print(f"Loading full dataset from {config['dataset_path']}...")
    full_dataset = load_dataset_from_json(config['dataset_path'])
    
    if not full_dataset:
        print("错误: 数据集为空!")
        return
    
    print(f"✓ Loaded {len(full_dataset)} samples")
    
    # ⭐ 设置随机种子,保证可复现
    random.seed(config['random_seed'])
    print(f"✓ Set random seed to {config['random_seed']} for reproducibility")
    
    # ⭐ 随机打乱数据集
    shuffled_dataset = full_dataset.copy()
    random.shuffle(shuffled_dataset)
    print(f"✓ Shuffled dataset")
    
    # ⭐ 划分训练集和测试集
    num_train = int(len(shuffled_dataset) * config['train_ratio'])
    train_dataset = shuffled_dataset[:num_train]
    test_dataset = shuffled_dataset[num_train:]
    
    print(f"\n✓ Split completed:")
    print(f"  Training set:   {len(train_dataset)} samples")
    print(f"  Test set:       {len(test_dataset)} samples")
    print(f"  Train ratio:    {config['train_ratio']*100:.0f}%")
    
    # ⭐ 保存测试集索引,用于后续评测
    test_indices_file = "./test_indices.json"
    test_indices = {
        "random_seed": config['random_seed'],
        "test_samples": [
            {"problem": sample["problem"][:100], "answer": sample["answer"]} 
            for sample in test_dataset
        ]
    }
    with open(test_indices_file, 'w', encoding='utf-8') as f:
        json.dump(test_indices, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved test set info to {test_indices_file}")
    
    print("="*60)
    
    # ========================================================================
    # 2. 初始化模型后端
    # ========================================================================
    
    print("\nInitializing model backends...")
    specialist_max_tokens = config.get('specialist_max_tokens', 300)
    model_backends = {
        "conductor_hf": LocalHF(config['conductor_model_path'], max_tokens=60),
        "reasoning_specialist_hf": LocalHF(config['reasoning_model_path'], max_tokens=specialist_max_tokens),
        "general_specialist_hf": LocalHF(config['general_model_path'], max_tokens=20)
    }

    # ========================================================================
    # 3. 初始化智能体
    # ========================================================================
    
    print("Initializing agents...")
    specialist_names = ["problem_understanding", "reasoning", "computation", "answering"]
    
    conductor = ConductorAgent(
        model_path=config['conductor_model_path'], 
        specialist_names=specialist_names,
        num_rewards=len(config['reward_dims'])
    )
    
    print("=" * 60)
    print("Conductor Configuration Check")
    print("=" * 60)
    print(f"Specialist names: {conductor.specialist_names}")
    print(f"Number of specialists: {len(conductor.specialist_names)}")
    print(f"Model: {conductor.model.config.name_or_path if hasattr(conductor.model.config, 'name_or_path') else 'Unknown'}")
    print("=" * 60)

    # 测试 Conductor 的动作生成
    print("\nTesting Conductor's action generation...")
    test_state = {
        "problem": "Test problem: What is 2+2?",
        "options": "A) 3 B) 4 C) 5"
    }
    test_history = []

    with torch.no_grad():
        test_action, test_ids, test_logprobs, test_values = conductor.act(test_state, test_history)
        print(f"Generated action: {test_action}")
        print(f"Specialist ID: {test_action.get('specialist_id')}")
        print(f"Input: {test_action.get('input')}")
    print("=" * 60)
    
    specialists = {
        "problem_understanding": SpecialistAgent(agent_name="problem_understanding", model_alias="general_specialist_hf"),
        "reasoning": SpecialistAgent(agent_name="reasoning", model_alias="reasoning_specialist_hf"),
        "computation": SpecialistAgent(agent_name="computation", model_alias="general_specialist_hf"),
        "answering": SpecialistAgent(agent_name="answering", model_alias="general_specialist_hf")
    }

    # ========================================================================
    # 4. 初始化训练器 (使用随机划分的训练集)
    # ========================================================================
    
    print("Initializing trainer...")
    trainer = CoEvolutionTrainer(
        config=config,
        conductor=conductor,
        specialists=specialists,
        model_backends=model_backends,
        train_dataset=train_dataset
    )
    
    # ========================================================================
    # 5. 开始训练
    # ========================================================================
    
    print("\nStarting training...")
    trainer.train()
    
    # ========================================================================
    # 6. 保存模型
    # ========================================================================
    
    print("\nSaving trained models...")
    trainer.save_models("./trained_models_1017")
    print("Training completed!")
    
    # ========================================================================
    # ⭐ 7. 提示如何使用测试集
    # ========================================================================
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print(f"✓ Models saved to: ./trained_models_1017")
    print(f"✓ Test set info saved to: {test_indices_file}")
    print(f"\nTo evaluate on the test set:")
    print(f"  1. The test set is the last {len(test_dataset)} samples from the shuffled dataset")
    print(f"  2. Run: python evaluate.py")
    print(f"  3. Make sure evaluate.py uses the same random seed ({config['random_seed']})")
    print("="*60)

if __name__ == "__main__":
    main()