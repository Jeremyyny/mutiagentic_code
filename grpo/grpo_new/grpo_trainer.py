import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import os
import reward
from manager_agent import ManagerAgent, FixedSpecialistAgent
import subagents


class GRPOTrainer:
    def __init__(self, config, manager, specialists, model_backend, train_dataset, accelerator=None):
        self.config = config
        self.accelerator = accelerator
        self.device = accelerator.device if accelerator else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.manager = manager.to(self.device)
        self.specialists = specialists
        self.model_backend = model_backend
        self.model_backend.model.to(self.device)
        self.train_dataset = train_dataset

        # Optimize only heads
        head_params = list(self.manager.policy_head.parameters()) + list(self.manager.value_head.parameters())
        self.manager_optimizer = Adam(head_params, lr=config['manager_lr'])

        # Prepare for distributed training
        if self.accelerator:
            self.manager, self.manager_optimizer = self.accelerator.prepare(self.manager, self.manager_optimizer)

        # GRPO parameters
        self.num_samples_per_prompt = config.get('num_samples_per_prompt', 4)
        self.reward_dims = config['reward_dims']
        self.manager_preference = torch.tensor(config['manager_preference'], device=self.device, dtype=torch.float32)
        self.use_value_baseline = config.get('use_value_baseline', True)
        self.lambda_coef = float(config.get('lambda_coef', 0.5))
        self.value_coef = float(config.get('value_coef', 0.5))
        self.verbose_trajectory = config.get('verbose_trajectory', True)
        self.manager_token_max_len = int(config.get('manager_token_max_len', 1024))

        # ---------------------- WandB 初始化 ----------------------
        self.use_wandb = config.get("use_wandb", False)
        self.wandb = None
        if self.use_wandb and (not self.accelerator or self.accelerator.is_main_process):
            import wandb
            wandb.init(
                project=config["wandb_project"],
                name=config["wandb_run_name"],
                tags=config["wandb_tags"],
                notes=config["wandb_notes"],
                config=config,
                reinit=True,
            )
            # 定义 metric schema（让曲线自动分组显示）
            wandb.define_metric("step")
            wandb.define_metric("epoch")
            wandb.define_metric("loss_manager", step_metric="step")
            wandb.define_metric("avg_reward", step_metric="step")
            wandb.define_metric("reward_std", step_metric="step")
            wandb.define_metric("policy_loss", step_metric="step")
            wandb.define_metric("value_loss", step_metric="step")
            wandb.define_metric("entropy", step_metric="step")
            wandb.define_metric("epoch/avg_reward", step_metric="epoch")
            wandb.define_metric("epoch/avg_loss", step_metric="epoch")
            self.wandb = wandb

    # ---------------------------- Advantage ----------------------------
    def _compute_group_advantages(self, group_rewards):
        group_mean = group_rewards.mean()
        advantages = group_rewards - group_mean
        if self.config.get('normalize_adv', False) and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    # ---------------------------- Collect ----------------------------
    def collect_trajectory_group(self, sample):
        state = {"problem": sample["problem"], "context": sample.get("context", "")}
        ground_truth = sample["answer"]
        group_trajectories = []

        for _ in range(self.num_samples_per_prompt):
            history, episode_steps = [], []
            for step in range(self.config['max_steps']):
                try:
                    manager_action, action_index, action_log_prob, _ = self.manager.act(state, history)
                    if not manager_action or manager_action.get('specialist_id') is None:
                        break
                    specialist_id = manager_action["specialist_id"]
                    specialist = self.specialists[specialist_id]
                    current_state = {**state, "instruction": manager_action["input"]}
                    output_text, _, _ = specialist.generate(current_state, history)

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

            if episode_steps:
                traj = [(s["specialist_id"], history[i]["content"]) for i, s in enumerate(episode_steps)]
                reward_vec_dict = reward.get_reward_vector(traj, ground_truth, self.config['max_steps'])
                reward_vec = torch.tensor([reward_vec_dict[dim] for dim in self.reward_dims], device=self.device)
                scalar_reward = torch.dot(reward_vec, self.manager_preference).item()
                group_trajectories.append({
                    "steps": episode_steps,
                    "scalar_reward": float(scalar_reward),
                })
        return group_trajectories

    # ---------------------------- Update ----------------------------
    def update_manager_grpo(self, group_trajectories, global_step):
        if not group_trajectories:
            return {}
        group_rewards = torch.tensor([t['scalar_reward'] for t in group_trajectories], device=self.device)
        group_adv = self._compute_group_advantages(group_rewards)
        all_steps = []
        for traj_idx, traj in enumerate(group_trajectories):
            for step in traj['steps']:
                record = dict(step)
                record['scalar_reward'] = float(traj['scalar_reward'])
                record['group_advantage'] = float(group_adv[traj_idx].item())
                all_steps.append(record)
        if not all_steps:
            return {}

        total_loss, num_updates = 0.0, 0
        total_policy, total_value, total_entropy = 0.0, 0.0, 0.0
        minibatch_size = self.config.get('minibatch_size', 8)
        num_total_steps = len(all_steps)

        for epoch in range(self.config.get('grpo_epochs', 3)):
            indices = np.random.permutation(num_total_steps)
            for start in range(0, num_total_steps, minibatch_size):
                mb = [all_steps[i] for i in indices[start:start + minibatch_size]]
                state_prompts = [s['state_prompt'] for s in mb]
                tokenized = self.manager.tokenizer(
                    state_prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=self.manager_token_max_len
                )
                state_ids = tokenized['input_ids'].to(self.device)
                state_mask = tokenized['attention_mask'].to(self.device)
                action_indices = torch.tensor([s['manager_action_index'] for s in mb], dtype=torch.long, device=self.device)
                scalar_rewards = torch.tensor([s['scalar_reward'] for s in mb], dtype=torch.float32, device=self.device)
                group_advs = torch.tensor([s['group_advantage'] for s in mb], dtype=torch.float32, device=self.device)

                new_log_probs, new_values, entropy = self.manager.evaluate_batch(state_ids, state_mask, action_indices)
                adv = group_advs
                if self.use_value_baseline:
                    adv_v = (scalar_rewards - new_values.detach())
                    adv = (1 - self.lambda_coef) * adv + self.lambda_coef * adv_v
                policy_loss = -(new_log_probs * adv).mean()
                value_loss = torch.tensor(0.0, device=self.device)
                if self.use_value_baseline and self.value_coef > 0.0:
                    value_loss = F.mse_loss(new_values, scalar_rewards)
                ent_coef = self.config.get('ent_coef', 0.01)
                loss = policy_loss + self.value_coef * value_loss - ent_coef * entropy

                self.manager_optimizer.zero_grad()
                if self.accelerator:
                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(
                        list(self.manager.policy_head.parameters()) + list(self.manager.value_head.parameters()),
                        self.config.get('max_grad_norm', 1.0)
                    )
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.manager.policy_head.parameters()) + list(self.manager.value_head.parameters()),
                        self.config.get('max_grad_norm', 1.0)
                    )
                self.manager_optimizer.step()

                total_loss += loss.item()
                total_policy += policy_loss.item()
                total_value += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        metrics = {
            "loss_manager": total_loss / max(1, num_updates),
            "avg_reward": group_rewards.mean().item(),
            "reward_std": group_rewards.std().item(),
            "policy_loss": total_policy / max(1, num_updates),
            "value_loss": total_value / max(1, num_updates),
            "entropy": total_entropy / max(1, num_updates),
        }

        # Log to WandB
        if self.wandb:
            self.wandb.log(metrics | {"step": global_step})

        return metrics

    # ---------------------------- Train ----------------------------
    def train(self):
        dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=True, drop_last=True)
        if self.accelerator:
            dataloader = self.accelerator.prepare(dataloader)

        global_step = 0
        for epoch in range(self.config['num_epochs']):
            if not self.accelerator or self.accelerator.is_main_process:
                print(f"\n{'='*60}\nEpoch {epoch+1}/{self.config['num_epochs']}\n{'='*60}")

            epoch_rewards, epoch_losses = [], []
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}",
                        disable=not (self.accelerator is None or self.accelerator.is_main_process))

            for batch in pbar:
                sample = {k: batch[k][0] for k in batch}
                group_trajectories = self.collect_trajectory_group(sample)
                if not group_trajectories:
                    continue
                metrics = self.update_manager_grpo(group_trajectories, global_step)
                global_step += 1
                if metrics:
                    epoch_losses.append(metrics['loss_manager'])
                    epoch_rewards.append(metrics['avg_reward'])
                    if self.accelerator is None or self.accelerator.is_main_process:
                        pbar.set_postfix({
                            'loss': f"{metrics['loss_manager']:.4f}",
                            'reward': f"{metrics['avg_reward']:.3f}"
                        })

            # Epoch summary
            if self.accelerator is None or self.accelerator.is_main_process:
                avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
                avg_loss = np.mean(epoch_losses) if epoch_losses else 0
                print(f"\nEpoch {epoch+1} Summary: Avg Reward={avg_reward:.4f}, Avg Loss={avg_loss:.4f}")
                if self.wandb:
                    self.wandb.log({
                        "epoch": epoch + 1,
                        "epoch/avg_reward": avg_reward,
                        "epoch/avg_loss": avg_loss
                    })
                if (epoch + 1) % 2 == 0:
                    self.save_manager(f"./checkpoints_grpo/epoch_{epoch+1}")

    # ---------------------------- Save ----------------------------
    def save_manager(self, output_dir):
        if self.accelerator and not self.accelerator.is_main_process:
            return
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