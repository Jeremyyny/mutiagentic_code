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
    config = {
        # === 1. Model Paths ===
        "manager_model_path": "Qwen/Qwen2.5-7B-Instruct",  # Changed to 7B
        "specialist_model_path": "Qwen/Qwen2.5-7B-Instruct",  # Changed to 7B
        
        # === 2. Data Path ===
        "dataset_path": r"C:\Users\yyn07\Desktop\multi_agent_test\Codes\data\golden_dataset_pubmedqa_qwen2.5_pro_test_500.json",

        # === 3. Training Hyperparameters ===
        "num_epochs": 5,  # Increased for 7B
        "max_steps": 5,
        "manager_lr": 3e-6,  # Lower LR for larger model
        "specialist_max_tokens": 300,  # Increased for richer responses
        
        # === 4. GRPO Parameters ===
        "num_samples_per_prompt": 4,
        "grpo_epochs": 3,
        "minibatch_size": 4,  # Reduced for 7B memory
        "ent_coef": 0.01,
        "max_grad_norm": 1.0,
        
        # === 5. Output Control ===
        "verbose_trajectory": True,
        "verbose_frequency": 10,  # Log less frequently
        
        # === 6. Mixed Precision ===
        "use_amp": True,
        
        # === 7. Reward Dimensions and Preferences ===
        "reward_dims": ["correctness", "efficiency", "quality"],
        "manager_preference": [1.0, 0.2, 0.3],  # Increased quality weight
        
        # === 8. Data Split ===
        "random_seed": 42,
        "train_ratio": 0.8,
        
        # === 9. Manager Training Scope ===
        "freeze_manager_backbone": True,
        "use_value_baseline": True,
        "lambda_coef": 0.5,
        "value_coef": 0.5,
        "normalize_adv": False,
        "manager_token_max_len": 2048,  # Increased for 7B
        
        # === 10. WandB Configuration ===
        "use_wandb": True,
        "wandb_project": "multi-agent-pubmedqa-7b",
        "wandb_run_name": "grpo-qwen2.5-7b-improved-prompts",
        "wandb_tags": ["grpo", "7b", "pubmedqa", "improved-prompts"],
        "wandb_notes": "7B model with enhanced prompts and better history utilization"
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    print("="*60)
    print("LOADING AND SPLITTING DATASET")
    print("="*60)
    
    full_dataset = load_dataset_from_json(config['dataset_path'])
    
    if not full_dataset:
        print("Error: The dataset is empty")
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
    
    test_indices_file = "./test_indices_grpo_7b.json"
    with open(test_indices_file, 'w', encoding='utf-8') as f:
        json.dump({
            "random_seed": config['random_seed'],
            "test_samples": [
                {"problem": s["problem"][:100], "answer": s["answer"]} 
                for s in test_dataset
            ]
        }, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved test set info to {test_indices_file}\n")

    print("Initializing model backend for 7B...")
    specialist_backend = LocalHF(
        config['specialist_model_path'], 
        max_tokens=config['specialist_max_tokens']
    )
    specialist_backend.model.eval()
    for param in specialist_backend.model.parameters():
        param.requires_grad = False
    print("✓ Specialist model (7B) frozen")

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
    print("Manager Configuration (7B)")
    print("="*60)
    print(f"Model: {config['manager_model_path']}")
    print(f"Specialists: {specialist_names}")
    print(f"Reward dims: {config['reward_dims']}")
    print(f"Preference:  {config['manager_preference']}")
    print(f"Freeze backbone: {config['freeze_manager_backbone']}")
    print(f"Value baseline: {config['use_value_baseline']} (λ={config['lambda_coef']}, value_coef={config['value_coef']})")
    print(f"Token max length: {config['manager_token_max_len']}")
    print("="*60)
    
    print("\nTesting Manager...")
    test_state = {"problem": "Test: What is 2+2?", "context": ""}
    with torch.no_grad():
        test_action, _, _, _ = manager.act(test_state, [])
        print(f"✓ Manager selected: {test_action['specialist_id']}")
    
    # Import improved subagents prompts
    print("\nLoading improved subagent prompts (7B-optimized)...")
    import subagents as subagents
    
    specialists = {
        name: FixedSpecialistAgent(agent_name=name, model_backend=specialist_backend)
        for name in specialist_names
    }
    print(f"✓ Created {len(specialists)} specialists with improved prompts")

    print("\nInitializing GRPO Trainer with WandB...")
    trainer = GRPOTrainer(
        config=config,
        manager=manager,
        specialists=specialists,
        model_backend=specialist_backend,
        train_dataset=train_dataset,
        use_wandb=config['use_wandb']
    )
    
    print("="*60)
    print("GRPO Configuration")
    print("="*60)
    print(f"Samples per prompt:  {config['num_samples_per_prompt']}")
    print(f"GRPO epochs:         {config['grpo_epochs']}")
    print(f"Manager LR:          {config['manager_lr']}")
    print(f"Minibatch size:      {config['minibatch_size']}")
    print(f"WandB enabled:       {config['use_wandb']}")
    if config['use_wandb']:
        print(f"WandB project:       {config['wandb_project']}")
        print(f"WandB run:           {config['wandb_run_name']}")
    print("="*60)

    print("\n🚀 Starting GRPO Training (7B model)...")
    print("💡 Note: Only Manager HEADS are being trained (backbone frozen)")
    print("📊 Training metrics will be logged to WandB\n")
    
    trainer.train()
    
    print("\n💾 Final save completed!")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"✓ Manager saved to: ./trained_manager_grpo")
    print(f"✓ Best model saved to: ./checkpoints_grpo/best_model")
    print(f"✓ Test set info: {test_indices_file}")
    if config['use_wandb']:
        print(f"✓ View training metrics at: https://wandb.ai")
    print("\nNext: Run evaluate_manager_inference.py to test the trained model")
    print("="*60)


if __name__ == "__main__":
    main()