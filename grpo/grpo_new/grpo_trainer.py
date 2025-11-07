import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
import numpy as np
import reward
import torch.distributed as dist


class GRPOTrainer:
    def __init__(self, config, manager, specialists, model_backend, train_dataset, accelerator=None):
        self.config = config
        self.accelerator = accelerator

        # ---- 核心对象 ----
        self.manager = manager
        self.specialists = specialists
        self.model_backend = model_backend
        self.train_dataset = train_dataset

        # ---- 只训练 heads ----
        head_params = list(self.manager.policy_head.parameters()) + list(self.manager.value_head.parameters())
        self.manager_optimizer = Adam(head_params, lr=config['manager_lr'])

        # ---- accelerate prepare（仅 heads + optimizer）----
        if self.accelerator:
            self.manager.policy_head = self.accelerator.prepare(self.manager.policy_head)
            self.manager.value_head = self.accelerator.prepare(self.manager.value_head)
            self.manager_optimizer = self.accelerator.prepare(self.manager_optimizer)

        # ---- 冻结 backbone ----
        device = self.accelerator.device if self.accelerator else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if hasattr(self.manager, "model"):
            self.manager.model.to(device)
            self.manager.model.eval()
            for p in self.manager.model.parameters():
                p.requires_grad = False

        if hasattr(self.model_backend, "model"):
            self.model_backend.model.to(device)
            self.model_backend.model.eval()
            for p in self.model_backend.model.parameters():
                p.requires_grad = False

        # ---- GRPO 参数 ----
        self.num_samples_per_prompt = int(config.get('num_samples_per_prompt', 4))
        self.reward_dims = config['reward_dims']
        self.manager_preference = torch.tensor(config['manager_preference'], dtype=torch.float32)

        self.use_value_baseline = bool(config.get('use_value_baseline', True))
        self.lambda_coef = float(config.get('lambda_coef', 0.5))
        self.value_coef = float(config.get('value_coef', 0.5))
        self.ent_coef = float(config.get('ent_coef', 0.01))
        self.max_grad_norm = float(config.get('max_grad_norm', 1.0))
        self.grpo_epochs = int(config.get('grpo_epochs', 3))
        self.minibatch_size = int(config.get('minibatch_size', 8))
        self.manager_token_max_len = int(config.get('manager_token_max_len', 1024))
        self.log_interval = int(config.get('log_interval', 5))

        # ---- wandb 初始化 ----
        self.use_wandb = bool(config.get("use_wandb", False))
        self.wandb = None
        if self.use_wandb and (self.accelerator is None or self.accelerator.is_main_process):
            import wandb
            wandb.init(
                project=config["wandb_project"],
                name=config["wandb_run_name"],
                tags=config["wandb_tags"],
                notes=config["wandb_notes"],
                config=config,
                reinit=True,
            )
            wandb.define_metric("step")
            wandb.define_metric("epoch")
            for k in ["loss_manager", "avg_reward", "reward_std", "policy_loss", "value_loss", "entropy"]:
                wandb.define_metric(k, step_metric="step")
            wandb.define_metric("epoch/avg_reward", step_metric="epoch")
            wandb.define_metric("epoch/avg_loss", step_metric="epoch")
            self.wandb = wandb

    # ---------------------------- Advantage ----------------------------
    def _compute_group_advantages(self, group_rewards: torch.Tensor):
        advantages = group_rewards - group_rewards.mean()
        if self.config.get('normalize_adv', False) and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    # ---------------------------- Rollout（仅 rank0） ----------------------------
    def collect_trajectory_group(self, sample):
        """仅 rank0 rollout"""
        state = {"problem": sample["problem"], "context": sample.get("context", "")}
        ground_truth = sample["answer"]
        group_trajectories = []

        for k in range(self.num_samples_per_prompt):
            history, episode_steps = [], []
            for step in range(self.config['max_steps']):
                try:
                    act, action_index, action_log_prob, _ = self.manager.act(state, history)
                    if not act or act.get('specialist_id') is None:
                        break
                    sid = act["specialist_id"]
                    specialist = self.specialists[sid]
                    current_state = {**state, "instruction": act["input"]}
                    output_text, _, _ = specialist.generate(current_state, history)

                    episode_steps.append({
                        "state_prompt": self.manager._build_prompt(state, history),
                        "manager_action_index": int(action_index),
                        "manager_log_prob": float(action_log_prob),
                        "specialist_id": sid,
                    })
                    history.append({"role": "assistant", "content": f"[{sid}]: {output_text}"})

                    # 可选 wandb log
                    if self.wandb and self.config.get("verbose_trajectory", False):
                        self.wandb.log({
                            "trajectory/step": step,
                            "trajectory/specialist": sid,
                            "trajectory/logp": float(action_log_prob),
                            "trajectory/output": output_text[:100],
                        })

                    if sid == "answering" or step == self.config['max_steps'] - 1:
                        break
                except Exception as e:
                    tqdm.write(f"❌ rollout error: {e}")
                    break

            if episode_steps:
                traj = [(s["specialist_id"], history[i]["content"]) for i, s in enumerate(episode_steps)]
                rdict = reward.get_reward_vector(traj, ground_truth, self.config['max_steps'])
                rvec = torch.tensor([rdict[d] for d in self.reward_dims], dtype=torch.float32)
                scalar_reward = float(torch.dot(rvec, self.manager_preference).item())
                group_trajectories.append({"steps": episode_steps, "scalar_reward": scalar_reward})

        return group_trajectories

    # ---------------------------- GRPO update ----------------------------
    def update_manager_grpo(self, group_trajectories, global_step):
        if not group_trajectories:
            return {}

        model_device = next(self.manager.policy_head.parameters()).device
        group_rewards = torch.tensor([t['scalar_reward'] for t in group_trajectories],
                                     dtype=torch.float32, device=model_device)
        group_adv = self._compute_group_advantages(group_rewards)

        all_steps = []
        for i, traj in enumerate(group_trajectories):
            for st in traj['steps']:
                rec = dict(st)
                rec['scalar_reward'] = float(group_rewards[i].item())
                rec['group_advantage'] = float(group_adv[i].item())
                all_steps.append(rec)

        total_loss = total_policy = total_value = total_entropy = 0.0
        num_updates = 0
        n = len(all_steps)

        for _ in range(self.grpo_epochs):
            idx = np.random.permutation(n)
            for s in range(0, n, self.minibatch_size):
                mb = [all_steps[i] for i in idx[s:s + self.minibatch_size]]

                toks = self.manager.tokenizer(
                    [x['state_prompt'] for x in mb],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.manager_token_max_len
                )
                state_ids = toks['input_ids'].to(model_device)
                state_mask = toks['attention_mask'].to(model_device)
                act_idx = torch.tensor([x['manager_action_index'] for x in mb], dtype=torch.long, device=model_device)
                scalar_rewards = torch.tensor([x['scalar_reward'] for x in mb], dtype=torch.float32, device=model_device)
                grp_adv = torch.tensor([x['group_advantage'] for x in mb], dtype=torch.float32, device=model_device)

                new_logp, new_values, entropy = self.manager.evaluate_batch(state_ids, state_mask, act_idx)

                adv = grp_adv
                if self.use_value_baseline:
                    adv_v = (scalar_rewards - new_values.detach())
                    adv = (1.0 - self.lambda_coef) * adv + self.lambda_coef * adv_v

                policy_loss = -(new_logp * adv).mean()
                value_loss = torch.tensor(0.0, device=model_device)
                if self.use_value_baseline and self.value_coef > 0.0:
                    value_loss = F.mse_loss(new_values, scalar_rewards)
                loss = policy_loss + self.value_coef * value_loss - self.ent_coef * entropy

                self.manager_optimizer.zero_grad(set_to_none=True)
                if self.accelerator:
                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(
                        list(self.manager.policy_head.parameters()) + list(self.manager.value_head.parameters()),
                        self.max_grad_norm
                    )
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.manager.policy_head.parameters()) + list(self.manager.value_head.parameters()),
                        self.max_grad_norm
                    )
                self.manager_optimizer.step()

                total_loss += float(loss.item())
                total_policy += float(policy_loss.item())
                total_value += float(value_loss.item())
                total_entropy += float(entropy.mean().item())
                num_updates += 1

        metrics = {
            "loss_manager": total_loss / max(1, num_updates),
            "avg_reward": float(group_rewards.mean().item()),
            "reward_std": float(group_rewards.std().item()),
            "policy_loss": total_policy / max(1, num_updates),
            "value_loss": total_value / max(1, num_updates),
            "entropy": total_entropy / max(1, num_updates),
        }

        if self.wandb and (self.accelerator is None or self.accelerator.is_main_process):
            self.wandb.log({**metrics, "step": global_step})

        return metrics

    # ---------------------------- Training loop ----------------------------
    def train(self):
        if self.accelerator and self.accelerator.state.num_processes > 1:
            sampler = DistributedSampler(self.train_dataset, shuffle=True, drop_last=True)
            dataloader = DataLoader(self.train_dataset, batch_size=1, sampler=sampler,
                                    num_workers=2, pin_memory=True, persistent_workers=False)
        else:
            dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=True,
                                    drop_last=True, num_workers=2, pin_memory=True, persistent_workers=False)

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
                # ---- rank0 rollout ----
                if self.accelerator is None or self.accelerator.is_main_process:
                    sample = {k: batch[k][0] for k in batch}
                    gtraj = self.collect_trajectory_group(sample)
                else:
                    gtraj = []

                # ---- 用 torch.distributed.all_gather_object 同步轨迹 ----
                if self.accelerator and self.accelerator.num_processes > 1 and dist.is_initialized():
                    world_size = self.accelerator.num_processes
                    obj_list = [None for _ in range(world_size)]
                    dist.all_gather_object(obj_list, gtraj)
                    gtraj = []
                    for gt in obj_list:
                        if isinstance(gt, list):
                            gtraj.extend(gt)

                if not gtraj:
                    if self.accelerator:
                        self.accelerator.wait_for_everyone()
                    continue

                metrics = self.update_manager_grpo(gtraj, global_step)
                global_step += 1

                if metrics:
                    epoch_losses.append(metrics['loss_manager'])
                    epoch_rewards.append(metrics['avg_reward'])
                    if self.accelerator is None or self.accelerator.is_main_process:
                        if global_step % self.log_interval == 0:
                            pbar.set_postfix({
                                'loss': f"{metrics['loss_manager']:.4f}",
                                'reward': f"{metrics['avg_reward']:.3f}"
                            })

            if self.accelerator is None or self.accelerator.is_main_process:
                avg_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
                avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
                print(f"\nEpoch {epoch+1} Summary: Avg Reward={avg_reward:.4f}, Avg Loss={avg_loss:.4f}")
                if self.wandb:
                    self.wandb.log({
                        "epoch": epoch + 1,
                        "epoch/avg_reward": avg_reward,
                        "epoch/avg_loss": avg_loss
                    })
                if (epoch + 1) % 2 == 0:
                    self.save_manager(f"./checkpoints_grpo/epoch_{epoch+1}")

            if self.accelerator:
                self.accelerator.wait_for_everyone()

    # ---------------------------- Save ----------------------------
    def save_manager(self, output_dir):
        if self.accelerator and not self.accelerator.is_main_process:
            return
        os.makedirs(output_dir, exist_ok=True)
        manager_dir = os.path.join(output_dir, "manager")
        self.manager.model.save_pretrained(manager_dir)
        self.manager.tokenizer.save_pretrained(manager_dir)
        torch.save({
            'policy_head': self.manager.policy_head.state_dict(),
            'value_head': self.manager.value_head.state_dict()
        }, os.path.join(manager_dir, "heads.pt"))
        print(f"Saved Manager to {output_dir}")