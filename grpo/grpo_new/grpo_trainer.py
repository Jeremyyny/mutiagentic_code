# =============================
# grpo_trainer.py (improved version)
# Changes:
# 1) Only optimize policy/value heads (coordinated with freeze_manager_backbone in main)
# 2) Optional GRPO-λ: adv = (1-λ)*(r - group_mean) + λ*(r - V(s))
# 3) Optional value_loss auxiliary training
# 4) Manager/state max_length increased to 1024
# 5) Stored action index / logprob use scalars to avoid shape confusion
# =============================

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from typing import List, Dict
import os
import reward
from manager_agent import ManagerAgent, FixedSpecialistAgent
import subagents

class GRPOTrainer:
    def __init__(self, config: Dict, manager: ManagerAgent, specialists: Dict[str, FixedSpecialistAgent], 
                 model_backend, train_dataset: List[Dict]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Manager (trainable)
        self.manager = manager.to(self.device)
        
        # Specialists (fixed)
        self.specialists = specialists
        self.model_backend = model_backend
        self.model_backend.model.to(self.device)

        self.train_dataset = train_dataset

        # Only optimize heads (backbone frozen by default)
        head_params = list(self.manager.policy_head.parameters()) + list(self.manager.value_head.parameters())
        self.manager_optimizer = Adam(head_params, lr=config['manager_lr'])
        
        # AMP
        self.use_amp = config.get('use_amp', True)
        if self.use_amp:
            self.scaler = GradScaler()
            print("✓ Using Automatic Mixed Precision (AMP)")
        
        # GRPO parameters
        self.num_samples_per_prompt = config.get('num_samples_per_prompt', 4)
        self.reward_dims = config['reward_dims']
        self.manager_preference = torch.tensor(config['manager_preference'], device=self.device, dtype=torch.float32)
        
        # GRPO-λ / Value configuration
        self.use_value_baseline = config.get('use_value_baseline', True)
        self.lambda_coef = float(config.get('lambda_coef', 0.5))  # 0~1
        self.value_coef = float(config.get('value_coef', 0.5))
        
        # Output control
        self.verbose_trajectory = config.get('verbose_trajectory', True)
        self.trajectory_counter = 0
        self.manager_token_max_len = int(config.get('manager_token_max_len', 1024))

    def _compute_group_advantages(self, group_rewards: torch.Tensor) -> torch.Tensor:
        group_mean = group_rewards.mean()
        advantages = group_rewards - group_mean
        # Optional normalization (small K brings noise, disabled by default)
        if self.config.get('normalize_adv', False) and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def collect_trajectory_group(self, sample: Dict, show_details: bool = False) -> List[Dict]:
        state = {
            "problem": sample["problem"], 
            "context": sample.get("context", ""),
            "options": sample.get("options", "")
        }
        ground_truth = sample["answer"]
        
        group_trajectories = []
        
        if show_details:
            tqdm.write(f"\n{'='*80}")
            tqdm.write(f"📊 COLLECTING {self.num_samples_per_prompt} TRAJECTORIES")
            tqdm.write(f"{'='*80}")
            tqdm.write(f"Problem: {state['problem'][:200]}...")
            tqdm.write(f"Ground Truth: '{ground_truth}'")
            tqdm.write(f"{'-'*80}")
        
        for k in range(self.num_samples_per_prompt):
            history = []
            episode_steps = []
            
            if show_details:
                tqdm.write(f"\n🎲 Trajectory {k+1}/{self.num_samples_per_prompt}:")
            
            for step in range(self.config['max_steps']):
                try:
                    manager_action, action_index, action_log_prob, value_scalar = self.manager.act(state, history)
                    
                    if not manager_action or manager_action.get('specialist_id') is None:
                        break
                    
                    specialist_id = manager_action["specialist_id"]
                    
                    if show_details:
                        tqdm.write(f"   Step {step+1}: {specialist_id}")
                    
                    # Specialist generates output (fixed model, no gradients)
                    specialist = self.specialists[specialist_id]
                    current_state = {**state, "instruction": manager_action["input"]}
                    output_text, output_ids, _ = specialist.generate(current_state, history)
                    
                    if show_details and k == 0:
                        tqdm.write(f"      Output: {output_text}")
                    
                    # Record step data
                    step_data = {
                        "state_prompt": self.manager._build_prompt(state, history),
                        "manager_action_index": int(action_index),
                        "manager_log_prob": float(action_log_prob),
                        "specialist_id": specialist_id,
                    }
                    episode_steps.append(step_data)
                    
                    history.append({"role": "assistant", "content": f"[{specialist_id}]: {output_text}"})
                    
                    if specialist_id == "answering" or step == self.config['max_steps'] - 1:
                        break
                except Exception as e:
                    tqdm.write(f"❌ Error in step {step}: {e}")
                    break
            
            # Calculate trajectory reward
            if episode_steps:
                final_trajectory = [(step["specialist_id"], history[i]["content"]) for i, step in enumerate(episode_steps)]
                reward_vec_dict = reward.get_reward_vector(final_trajectory, ground_truth, self.config['max_steps'])
                reward_vec = torch.tensor([reward_vec_dict[dim] for dim in self.reward_dims], device=self.device)
                scalar_reward = torch.dot(reward_vec, self.manager_preference).item()
                
                if show_details:
                    tqdm.write(f"   Reward: {scalar_reward:.3f} (correctness={reward_vec_dict['correctness']:.1f})")
                
                group_trajectories.append({
                    "steps": episode_steps,
                    "scalar_reward": float(scalar_reward),
                    "reward_dict": {k: float(v) for k, v in reward_vec_dict.items()}
                })
        
        if show_details:
            tqdm.write(f"{'='*80}\n")
        
        return group_trajectories

    def update_manager_grpo(self, group_trajectories: List[Dict]) -> Dict:
        if not group_trajectories:
            return {}
        
        group_rewards = torch.tensor([traj['scalar_reward'] for traj in group_trajectories], device=self.device)
        # Group mean baseline
        group_adv = self._compute_group_advantages(group_rewards)  # [K]
        
        # Collect all steps
        all_steps = []
        for traj_idx, trajectory in enumerate(group_trajectories):
            for step in trajectory['steps']:
                record = dict(step)
                record['scalar_reward'] = float(trajectory['scalar_reward'])
                record['group_advantage'] = float(group_adv[traj_idx].item())
                all_steps.append(record)
        
        if not all_steps:
            return {}
        
        total_loss = 0.0
        num_updates = 0
        
        num_total_steps = len(all_steps)
        minibatch_size = self.config.get('minibatch_size', 8)
        
        for epoch in range(self.config.get('grpo_epochs', 3)):
            indices = np.random.permutation(num_total_steps)
            for start in range(0, num_total_steps, minibatch_size):
                end = min(start + minibatch_size, num_total_steps)
                minibatch_indices = indices[start:end]
                mb = [all_steps[i] for i in minibatch_indices]
                
                # Prepare batch data
                state_prompts = [s['state_prompt'] for s in mb]
                tokenized = self.manager.tokenizer(
                    state_prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.manager_token_max_len
                )
                state_ids = tokenized['input_ids'].to(self.device)
                state_mask = tokenized['attention_mask'].to(self.device)
                action_indices = torch.tensor([s['manager_action_index'] for s in mb], dtype=torch.long, device=self.device)
                old_log_probs = torch.tensor([s['manager_log_prob'] for s in mb], dtype=torch.float32, device=self.device)
                scalar_rewards = torch.tensor([s['scalar_reward'] for s in mb], dtype=torch.float32, device=self.device)
                group_advs = torch.tensor([s['group_advantage'] for s in mb], dtype=torch.float32, device=self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        new_log_probs, new_values, entropy = self.manager.evaluate_batch(state_ids, state_mask, action_indices)
                        
                        # GRPO objective (basic): maximize logprob * group_adv
                        adv = group_advs
                        
                        # Optional: GRPO-λ fusion with value baseline
                        if self.use_value_baseline:
                            # r - V(s)
                            adv_v = (scalar_rewards - new_values.detach())
                            adv = (1.0 - self.lambda_coef) * adv + self.lambda_coef * adv_v
                        
                        policy_loss = -(new_log_probs * adv).mean()
                        
                        # Optional value loss
                        value_loss = torch.tensor(0.0, device=self.device)
                        if self.use_value_baseline and self.value_coef > 0.0:
                            # Target uses scalar reward (can also use group_mean/GAE etc.)
                            value_loss = F.mse_loss(new_values, scalar_rewards)
                        
                        ent_coef = self.config.get('ent_coef', 0.01)
                        loss = policy_loss + self.value_coef * value_loss - ent_coef * entropy
                    
                    self.manager_optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.manager_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.manager.policy_head.parameters()) + list(self.manager.value_head.parameters()),
                        self.config.get('max_grad_norm', 1.0)
                    )
                    self.scaler.step(self.manager_optimizer)
                    self.scaler.update()
                else:
                    new_log_probs, new_values, entropy = self.manager.evaluate_batch(state_ids, state_mask, action_indices)
                    adv = group_advs
                    if self.use_value_baseline:
                        adv_v = (scalar_rewards - new_values.detach())
                        adv = (1.0 - self.lambda_coef) * adv + self.lambda_coef * adv_v
                    policy_loss = -(new_log_probs * adv).mean()
                    value_loss = torch.tensor(0.0, device=self.device)
                    if self.use_value_baseline and self.value_coef > 0.0:
                        value_loss = F.mse_loss(new_values, scalar_rewards)
                    ent_coef = self.config.get('ent_coef', 0.01)
                    loss = policy_loss + self.value_coef * value_loss - ent_coef * entropy
                    
                    self.manager_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.manager.policy_head.parameters()) + list(self.manager.value_head.parameters()),
                        self.config.get('max_grad_norm', 1.0)
                    )
                    self.manager_optimizer.step()
                
                total_loss += loss.item()
                num_updates += 1
        
        return {
            "loss_manager": total_loss / num_updates if num_updates > 0 else 0,
            "avg_reward": group_rewards.mean().item(),
            "reward_std": group_rewards.std().item()
        }

    def train(self):
        print("Starting GRPO Training (Manager Heads Only)...")
        print(f"Sampling {self.num_samples_per_prompt} trajectories per problem")
        
        dataloader = DataLoader(
            self.train_dataset, 
            batch_size=1,
            shuffle=True, 
            drop_last=True
        )

        for epoch in range(self.config['num_epochs']):
            print(f"\n{'='*60}\nEpoch {epoch+1}/{self.config['num_epochs']}\n{'='*60}")
            
            epoch_rewards = []
            epoch_losses = []
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            for batch_idx, batch in enumerate(pbar):
                try:
                    sample = {key: batch[key][0] for key in batch}
                    show_details = (self.verbose_trajectory and batch_idx % self.config.get('verbose_frequency', 5) == 0)
                    group_trajectories = self.collect_trajectory_group(sample, show_details)
                    if not group_trajectories:
                        continue
                    
                    avg_reward = np.mean([t['scalar_reward'] for t in group_trajectories])
                    epoch_rewards.append(avg_reward)
                    
                    metrics = self.update_manager_grpo(group_trajectories)
                    if metrics:
                        epoch_losses.append(metrics['loss_manager'])
                        pbar.set_postfix({
                            'loss': f"{metrics['loss_manager']:.4f}",
                            'reward': f"{metrics['avg_reward']:.3f}",
                            'std': f"{metrics['reward_std']:.3f}"
                        })
                        with open("train_log.txt", "a", encoding="utf-8") as f:
                            f.write(f"Epoch {epoch+1}, Step {batch_idx}, "
                                f"Loss={metrics['loss_manager']:.4f}, "
                                f"Reward={metrics['avg_reward']:.3f}, "
                                f"Std={metrics['reward_std']:.3f}\n")

                except Exception as e:
                    tqdm.write(f"\n❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if epoch_rewards:
                avg_reward = np.mean(epoch_rewards)
                avg_loss = np.mean(epoch_losses) if epoch_losses else 0
                print(f"\n{'='*60}")
                print(f"Epoch {epoch+1} Summary:")
                print(f"  Avg Reward:  {avg_reward:.4f}")
                print(f"  Avg Loss:    {avg_loss:.4f}")
                print(f"{'='*60}")
                if (epoch + 1) % 2 == 0:
                    self.save_manager(f"./checkpoints_grpo/epoch_{epoch+1}")

    def save_manager(self, output_dir: str):
        print(f"\n💾 Saving Manager to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        manager_dir = os.path.join(output_dir, "manager")
        self.manager.model.save_pretrained(manager_dir)
        self.manager.tokenizer.save_pretrained(manager_dir)
        torch.save({
            'policy_head': self.manager.policy_head.state_dict(),
            'value_head': self.manager.value_head.state_dict()
        }, os.path.join(manager_dir, "heads.pt"))
        print("✓ Manager saved successfully")