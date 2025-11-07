# =============================
# grpo_trainer_wandb.py (with WandB integration)
# New features:
# 1. WandB logging for metrics, gradients, and system stats
# 2. Better trajectory logging
# 3. Per-step reward breakdown
# 4. Model comparison tracking
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
import wandb

class GRPOTrainer:
    def __init__(self, config: Dict, manager: ManagerAgent, specialists: Dict[str, FixedSpecialistAgent], 
                 model_backend, train_dataset: List[Dict], use_wandb: bool = True):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_wandb = use_wandb

        # Initialize WandB
        if self.use_wandb:
            wandb.init(
                project=config.get('wandb_project', 'multi-agent-grpo'),
                name=config.get('wandb_run_name', None),
                config=config,
                tags=config.get('wandb_tags', ['grpo', 'multi-agent']),
                notes=config.get('wandb_notes', 'GRPO training with Manager-Specialist architecture')
            )
            print("✓ WandB initialized successfully")

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
        
        # Log parameter counts
        if self.use_wandb:
            total_params = sum(p.numel() for p in self.manager.parameters())
            trainable_params = sum(p.numel() for p in head_params)
            wandb.config.update({
                'total_manager_params': total_params,
                'trainable_manager_params': trainable_params,
                'frozen_backbone_params': total_params - trainable_params
            })
        
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
        self.lambda_coef = float(config.get('lambda_coef', 0.5))
        self.value_coef = float(config.get('value_coef', 0.5))
        
        # Output control
        self.verbose_trajectory = config.get('verbose_trajectory', True)
        self.trajectory_counter = 0
        self.manager_token_max_len = int(config.get('manager_token_max_len', 1024))
        
        # Training statistics
        self.global_step = 0
        self.best_reward = float('-inf')

    def _compute_group_advantages(self, group_rewards: torch.Tensor) -> torch.Tensor:
        group_mean = group_rewards.mean()
        advantages = group_rewards - group_mean
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
        
        trajectory_rewards = []
        trajectory_details = []
        
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
                    
                    # Specialist generates output
                    specialist = self.specialists[specialist_id]
                    current_state = {**state, "instruction": manager_action["input"]}
                    output_text, output_ids, _ = specialist.generate(current_state, history)
                    
                    if show_details and k == 0:
                        tqdm.write(f"      Output: {output_text[:100]}...")
                    
                    # Record step data
                    step_data = {
                        "state_prompt": self.manager._build_prompt(state, history),
                        "manager_action_index": int(action_index),
                        "manager_log_prob": float(action_log_prob),
                        "specialist_id": specialist_id,
                        "output_length": len(output_text)
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
                
                trajectory_rewards.append(scalar_reward)
                trajectory_details.append({
                    'correctness': reward_vec_dict['correctness'],
                    'efficiency': reward_vec_dict['efficiency'],
                    'quality': reward_vec_dict['quality'],
                    'num_steps': len(episode_steps)
                })
                
                if show_details:
                    tqdm.write(f"   Reward: {scalar_reward:.3f} | Correct: {reward_vec_dict['correctness']:.1f} | Eff: {reward_vec_dict['efficiency']:.2f} | Qual: {reward_vec_dict['quality']:.2f}")
                
                group_trajectories.append({
                    "steps": episode_steps,
                    "scalar_reward": float(scalar_reward),
                    "reward_dict": {k: float(v) for k, v in reward_vec_dict.items()}
                })
        
        # Log trajectory statistics to WandB
        if self.use_wandb and trajectory_rewards:
            wandb.log({
                'trajectory_group/mean_reward': np.mean(trajectory_rewards),
                'trajectory_group/std_reward': np.std(trajectory_rewards),
                'trajectory_group/max_reward': np.max(trajectory_rewards),
                'trajectory_group/min_reward': np.min(trajectory_rewards),
                'trajectory_group/mean_correctness': np.mean([d['correctness'] for d in trajectory_details]),
                'trajectory_group/mean_efficiency': np.mean([d['efficiency'] for d in trajectory_details]),
                'trajectory_group/mean_quality': np.mean([d['quality'] for d in trajectory_details]),
                'trajectory_group/mean_steps': np.mean([d['num_steps'] for d in trajectory_details]),
            }, step=self.global_step)
        
        if show_details:
            tqdm.write(f"{'='*80}\n")
        
        return group_trajectories

    def update_manager_grpo(self, group_trajectories: List[Dict]) -> Dict:
        if not group_trajectories:
            return {}
        
        group_rewards = torch.tensor([traj['scalar_reward'] for traj in group_trajectories], device=self.device)
        group_adv = self._compute_group_advantages(group_rewards)
        
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
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        
        num_total_steps = len(all_steps)
        minibatch_size = self.config.get('minibatch_size', 8)
        
        # Track gradient norms
        grad_norms = []
        
        for epoch in range(self.config.get('grpo_epochs', 3)):
            indices = np.random.permutation(num_total_steps)
            for start in range(0, num_total_steps, minibatch_size):
                end = min(start + minibatch_size, num_total_steps)
                minibatch_indices = indices[start:end]
                mb = [all_steps[i] for i in minibatch_indices]
                
                # Prepare batch
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
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.manager_optimizer)
                    
                    # Track gradient norm
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(self.manager.policy_head.parameters()) + list(self.manager.value_head.parameters()),
                        self.config.get('max_grad_norm', 1.0)
                    )
                    grad_norms.append(grad_norm.item())
                    
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
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(self.manager.policy_head.parameters()) + list(self.manager.value_head.parameters()),
                        self.config.get('max_grad_norm', 1.0)
                    )
                    grad_norms.append(grad_norm.item())
                    
                    self.manager_optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1
                self.global_step += 1
        
        metrics = {
            "loss_manager": total_loss / num_updates if num_updates > 0 else 0,
            "policy_loss": total_policy_loss / num_updates if num_updates > 0 else 0,
            "value_loss": total_value_loss / num_updates if num_updates > 0 else 0,
            "entropy": total_entropy / num_updates if num_updates > 0 else 0,
            "avg_reward": group_rewards.mean().item(),
            "reward_std": group_rewards.std().item(),
            "avg_grad_norm": np.mean(grad_norms) if grad_norms else 0,
            "max_grad_norm": np.max(grad_norms) if grad_norms else 0
        }
        
        # Log to WandB
        if self.use_wandb:
            wandb.log({
                'train/total_loss': metrics['loss_manager'],
                'train/policy_loss': metrics['policy_loss'],
                'train/value_loss': metrics['value_loss'],
                'train/entropy': metrics['entropy'],
                'train/avg_reward': metrics['avg_reward'],
                'train/reward_std': metrics['reward_std'],
                'train/grad_norm_avg': metrics['avg_grad_norm'],
                'train/grad_norm_max': metrics['max_grad_norm'],
                'train/learning_rate': self.manager_optimizer.param_groups[0]['lr']
            }, step=self.global_step)
        
        return metrics

    def train(self):
        print("Starting GRPO Training with WandB logging...")
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
            epoch_correctness = []
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            for batch_idx, batch in enumerate(pbar):
                try:
                    sample = {key: batch[key][0] for key in batch}
                    show_details = (self.verbose_trajectory and batch_idx % self.config.get('verbose_frequency', 10) == 0)
                    
                    group_trajectories = self.collect_trajectory_group(sample, show_details)
                    if not group_trajectories:
                        continue
                    
                    avg_reward = np.mean([t['scalar_reward'] for t in group_trajectories])
                    avg_correctness = np.mean([t['reward_dict']['correctness'] for t in group_trajectories])
                    epoch_rewards.append(avg_reward)
                    epoch_correctness.append(avg_correctness)
                    
                    metrics = self.update_manager_grpo(group_trajectories)
                    if metrics:
                        epoch_losses.append(metrics['loss_manager'])
                        pbar.set_postfix({
                            'loss': f"{metrics['loss_manager']:.4f}",
                            'reward': f"{metrics['avg_reward']:.3f}",
                            'correct': f"{avg_correctness:.2f}"
                        })

                except Exception as e:
                    tqdm.write(f"\n❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Epoch summary
            if epoch_rewards:
                avg_reward = np.mean(epoch_rewards)
                avg_loss = np.mean(epoch_losses) if epoch_losses else 0
                avg_correct = np.mean(epoch_correctness)
                
                print(f"\n{'='*60}")
                print(f"Epoch {epoch+1} Summary:")
                print(f"  Avg Reward:      {avg_reward:.4f}")
                print(f"  Avg Loss:        {avg_loss:.4f}")
                print(f"  Avg Correctness: {avg_correct:.4f}")
                print(f"{'='*60}")
                
                # Log epoch summary to WandB
                if self.use_wandb:
                    wandb.log({
                        'epoch/avg_reward': avg_reward,
                        'epoch/avg_loss': avg_loss,
                        'epoch/avg_correctness': avg_correct,
                        'epoch/num': epoch + 1
                    }, step=self.global_step)
                
                # Save best model
                if avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    self.save_manager(f"./checkpoints_grpo/best_model")
                    if self.use_wandb:
                        wandb.run.summary["best_reward"] = self.best_reward
                
                # Periodic checkpoint
                if (epoch + 1) % 2 == 0:
                    self.save_manager(f"./checkpoints_grpo/epoch_{epoch+1}")
        
        # Final save
        self.save_manager("./trained_manager_grpo")
        
        if self.use_wandb:
            wandb.finish()

    def save_manager(self, output_dir: str):
        print(f"\n💾 Saving Manager to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        manager_dir = os.path.join(output_dir, "manager")
        self.manager.model.save_pretrained(manager_dir)
        self.manager.tokenizer.save_pretrained(manager_dir)
        torch.save({
            'policy_head': self.manager.policy_head.state_dict(),
            'value_head': self.manager.value_head.state_dict(),
            'optimizer': self.manager_optimizer.state_dict(),
            'global_step': self.global_step,
            'best_reward': self.best_reward
        }, os.path.join(manager_dir, "heads.pt"))
        print("✓ Manager saved successfully")
        
        # Save checkpoint artifact to WandB
        if self.use_wandb:
            artifact = wandb.Artifact(
                name=f'manager-checkpoint',
                type='model',
                description=f'Manager checkpoint at step {self.global_step}'
            )
            artifact.add_dir(manager_dir)
            wandb.log_artifact(artifact)