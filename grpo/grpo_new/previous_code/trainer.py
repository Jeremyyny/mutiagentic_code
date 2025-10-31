# trainer.py (Final version - includes detailed output)

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any
import os
import reward 
from agents import ConductorAgent, SpecialistAgent
from utils import LocalHF
import subagents

class CoEvolutionTrainer:
    def __init__(self, config: Dict, conductor: ConductorAgent, specialists: Dict[str, SpecialistAgent], 
                 model_backends: Dict[str, LocalHF], train_dataset: List[Dict]):
        
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conductor = conductor.to(self.device)
        self.specialists = specialists
        for backend in model_backends.values():
            backend.model.to(self.device)
        self.model_backends = model_backends

        self.train_dataset = train_dataset

        self.conductor_optimizer = Adam(self.conductor.parameters(), lr=config['conductor_lr'])
        self.specialist_optimizers = {
            alias: Adam(backend.model.parameters(), lr=config['specialist_lr'])
            for alias, backend in self.model_backends.items()
        }
        
        self.use_amp = config.get('use_amp', True)
        if self.use_amp:
            self.scaler = GradScaler()
            print("✓ Using Automatic Mixed Precision (AMP)")
        
        self.reward_dims = config['reward_dims']
        self.conductor_preference = torch.tensor(config['conductor_preference'], device=self.device, dtype=torch.float32)
        self.specialist_preferences = {
            name: torch.tensor(pref, device=self.device, dtype=torch.float32) 
            for name, pref in config['specialist_preferences'].items()
        }
        
        # Detailed output control
        self.verbose_trajectory = config.get('verbose_trajectory', True)
        self.trajectory_counter = 0

    def _safe_normalize(self, advantages: torch.Tensor) -> torch.Tensor:
        if advantages.numel() <= 1:
            return torch.zeros_like(advantages)
        std = advantages.std(unbiased=True)
        if std < 1e-8:
            return torch.zeros_like(advantages)
        return (advantages - advantages.mean()) / (std + 1e-8)

    def _compute_advantages_and_returns(self, rewards_per_dim: Dict[str, torch.Tensor], 
                                        values_per_dim: Dict[str, torch.Tensor], 
                                        dones: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        num_steps = len(dones)
        advantages_list = []
        returns_list = []
        
        gamma = self.config['gamma']
        lam = self.config['lambda']
        
        for dim in self.reward_dims:
            rewards = rewards_per_dim[dim]
            values = values_per_dim[dim]
            advantages = torch.zeros_like(rewards)
            
            gae = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_value = 0
                    next_non_terminal = 0
                else:
                    next_value = values[t + 1]
                    next_non_terminal = 1 - dones[t + 1]
                
                delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
                gae = delta + gamma * lam * next_non_terminal * gae
                advantages[t] = gae
            
            returns = advantages + values
            advantages_list.append(advantages)
            returns_list.append(returns)
        
        return torch.stack(advantages_list, dim=1), torch.stack(returns_list, dim=1)

    def _collate_conductor_data(self, minibatch: List[Dict]) -> Dict[str, torch.Tensor]:
        state_prompts = [step['state_prompt'] for step in minibatch]
        tokenized_states = self.conductor.tokenizer(
            state_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        
        action_indices = torch.stack([step['conductor_action_ids'] for step in minibatch])
        old_log_probs = torch.stack([step['conductor_log_prob'] for step in minibatch])
        advantages_vecs = torch.stack([step['advantages_vec'] for step in minibatch])
        returns_vecs = torch.stack([step['returns_vec'] for step in minibatch])

        return {
            "state_ids": tokenized_states['input_ids'].to(self.device, non_blocking=True),
            "state_mask": tokenized_states['attention_mask'].to(self.device, non_blocking=True),
            "action_indices": action_indices.to(self.device, non_blocking=True),
            "old_log_probs": old_log_probs.to(self.device, non_blocking=True),
            "advantages_vecs": advantages_vecs.to(self.device, non_blocking=True),
            "returns_vecs": returns_vecs.to(self.device, non_blocking=True)
        }

    def collect_trajectories(self, batch: Dict) -> (List[Dict], Dict):
        """Collect trajectories - with detailed output"""
        all_trajectories_data = []
        
        try:
            num_samples_in_batch = len(batch["problem"])
        except (KeyError, TypeError):
            tqdm.write("Error: Batch format is incorrect.")
            return [], {}

        for i in range(num_samples_in_batch):
            sample = {key: batch[key][i] for key in batch}
            
            state = {"problem": sample["problem"], "options": sample.get("options", "")}
            ground_truth = sample["answer"]
            history = []
            episode_buffer = []
            
            # Decide whether to show detailed information
            if i == 0:
                self.trajectory_counter += 1
            verbose_freq = self.config.get('verbose_frequency', 1)  # Default: show every batch
            show_details = (self.verbose_trajectory and i == 0 and self.trajectory_counter % verbose_freq == 0)

            if show_details:
                tqdm.write(f"\n{'='*80}")
                tqdm.write(f"📊 TRAJECTORY #{self.trajectory_counter}")
                tqdm.write(f"{'='*80}")
                tqdm.write(f"Problem: {state['problem'][:200]}...")
                tqdm.write(f"Ground Truth: '{ground_truth}'")
                tqdm.write(f"{'-'*80}")

            for step in range(self.config['max_steps']):
                try:
                    conductor_action, action_ids, action_log_prob, value_vec = self.conductor.act(state, history)
                    
                    if not conductor_action or conductor_action.get('specialist_id') is None:
                        break 

                    specialist_id = conductor_action["specialist_id"]
                    specialist_input = conductor_action["input"]
                    
                    if show_details:
                        tqdm.write(f"\n🎯 Step {step+1}:")
                        tqdm.write(f"   Conductor → {specialist_id}")
                        tqdm.write(f"   Instruction: {specialist_input}")
                    
                    specialist = self.specialists[specialist_id]
                    model_backend = self.model_backends[specialist.model_alias]

                    current_state_for_specialist = {**state, "instruction": specialist_input}
                    agent_prompt_fn = subagents.AGENT_PROMPT_FUNCTIONS.get(specialist_id)
                    messages = agent_prompt_fn(current_state_for_specialist, history)
                    prompt_text = model_backend.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    
                    max_tokens = self.config.get('specialist_max_tokens', 80)
                    output_text, output_ids, _ = model_backend.generate_with_logprobs(
                        prompt_text, max_new_tokens=max_tokens
                    )
                    
                    if show_details:
                        tqdm.write(f"   Output: {output_text}")

                    step_data = {
                        "state_prompt": self.conductor._build_prompt(state, history),
                        "conductor_action_ids": action_ids.squeeze(0),
                        "conductor_log_prob": action_log_prob.squeeze(0),
                        "conductor_value_vec": value_vec.squeeze(0),
                        "specialist_id": specialist_id,
                        "specialist_prompt_text": prompt_text,
                        "specialist_output_ids": output_ids.squeeze(0),
                    }
                    episode_buffer.append(step_data)
                    
                    history.append({"role": "assistant", "content": f"[{specialist_id}]: {output_text}"})
                    
                    if specialist_id == "answering" or step == self.config['max_steps'] - 1:
                        break
                
                except Exception as e:
                    tqdm.write(f"Error in step {step}: {e}")
                    break
            
            num_steps = len(episode_buffer)
            if num_steps > 0:
                final_trajectory = [(step["specialist_id"], history[i]["content"]) for i, step in enumerate(episode_buffer)]
                reward_vec_dict = reward.get_reward_vector(final_trajectory, ground_truth, self.config['max_steps'])
                
                if show_details:
                    tqdm.write(f"\n{'-'*80}")
                    tqdm.write("📈 REWARDS:")
                    for dim, value in reward_vec_dict.items():
                        symbol = "✓" if value > 0 else "✗"
                        tqdm.write(f"   {symbol} {dim:15s}: {value:6.3f}")
                    
                    # Analyze correctness
                    if reward_vec_dict.get('correctness', 0) == 0:
                        tqdm.write(f"\n⚠️  CORRECTNESS = 0:")
                        tqdm.write(f"   Expected: '{ground_truth}'")
                        answering_outputs = [content for agent_id, content in final_trajectory if agent_id == "answering"]
                        if answering_outputs:
                            tqdm.write(f"   Got: '{answering_outputs[-1][:100]}'")
                        else:
                            tqdm.write(f"   ❌ 'answering' agent was never called!")
                    
                    tqdm.write(f"{'='*80}\n")
                
                rewards_per_dim = {dim: torch.zeros(num_steps, device=self.device) for dim in self.reward_dims}
                for dim in self.reward_dims:
                    rewards_per_dim[dim][-1] = reward_vec_dict[dim]
                
                all_trajectories_data.append({"steps": episode_buffer, "rewards_per_dim": rewards_per_dim})

        avg_rewards = {f"reward_{dim}": 0.0 for dim in self.reward_dims}
        if all_trajectories_data:
            num_trajectories = len(all_trajectories_data)
            for dim in self.reward_dims:
                total_dim_reward = sum(traj['rewards_per_dim'][dim][-1].item() for traj in all_trajectories_data)
                avg_rewards[f"reward_{dim}"] = total_dim_reward / num_trajectories
        
        return all_trajectories_data, avg_rewards

    def update_models(self, trajectories: List[Dict]) -> Dict:
        all_step_data = []
        for trajectory in trajectories:
            steps_data = trajectory["steps"]
            if not steps_data: continue
            
            rewards_per_dim = trajectory["rewards_per_dim"]
            num_steps = len(steps_data)
            values_per_dim = {}
            
            for i, dim in enumerate(self.reward_dims):
                dim_values = torch.stack([step["conductor_value_vec"][i] for step in steps_data])
                values_per_dim[dim] = dim_values
            
            dones = torch.zeros(num_steps, device=self.device)
            dones[-1] = 1
            
            advantages_vecs, returns_vecs = self._compute_advantages_and_returns(rewards_per_dim, values_per_dim, dones)
            
            for i, step in enumerate(steps_data):
                step['advantages_vec'] = advantages_vecs[i]
                step['returns_vec'] = returns_vecs[i]
                all_step_data.append(step)

        if not all_step_data: return {}

        total_loss_conductor, total_loss_specialists, num_updates = 0.0, 0.0, 0
        num_total_steps = len(all_step_data)
        
        for _ in range(self.config['ppo_epochs']):
            indices = np.random.permutation(num_total_steps)
            for start in range(0, num_total_steps, self.config['minibatch_size']):
                end = start + self.config['minibatch_size']
                minibatch_indices = indices[start:end]
                minibatch = [all_step_data[i] for i in minibatch_indices]

                self.conductor.train()
                conductor_batch = self._collate_conductor_data(minibatch)
                
                if self.use_amp:
                    with autocast():
                        new_log_probs, new_value_vecs, entropy = self.conductor.evaluate(
                            conductor_batch['state_ids'], conductor_batch['state_mask'], conductor_batch['action_indices']
                        )
                        adv_vecs = conductor_batch['advantages_vecs']
                        scalar_advantages_conductor = torch.tensordot(adv_vecs, self.conductor_preference, dims=1)
                        scalar_advantages_conductor = self._safe_normalize(scalar_advantages_conductor)
                        ratio = (new_log_probs - conductor_batch['old_log_probs']).exp()
                        surr1 = ratio * scalar_advantages_conductor
                        surr2 = torch.clamp(ratio, 1.0 - self.config['clip_param'], 1.0 + self.config['clip_param']) * scalar_advantages_conductor
                        policy_loss_conductor = -torch.min(surr1, surr2).mean()
                        value_loss_conductor = F.mse_loss(new_value_vecs, conductor_batch['returns_vecs'])
                        loss_conductor = policy_loss_conductor + self.config['vf_coef'] * value_loss_conductor - self.config['ent_coef'] * entropy
                    
                    self.conductor_optimizer.zero_grad()
                    self.scaler.scale(loss_conductor).backward()
                    self.scaler.unscale_(self.conductor_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.conductor.parameters(), self.config['max_grad_norm'])
                    self.scaler.step(self.conductor_optimizer)
                    self.scaler.update()
                else:
                    new_log_probs, new_value_vecs, entropy = self.conductor.evaluate(
                        conductor_batch['state_ids'], conductor_batch['state_mask'], conductor_batch['action_indices']
                    )
                    adv_vecs = conductor_batch['advantages_vecs']
                    scalar_advantages_conductor = torch.tensordot(adv_vecs, self.conductor_preference, dims=1)
                    scalar_advantages_conductor = self._safe_normalize(scalar_advantages_conductor)
                    ratio = (new_log_probs - conductor_batch['old_log_probs']).exp()
                    surr1 = ratio * scalar_advantages_conductor
                    surr2 = torch.clamp(ratio, 1.0 - self.config['clip_param'], 1.0 + self.config['clip_param']) * scalar_advantages_conductor
                    policy_loss_conductor = -torch.min(surr1, surr2).mean()
                    value_loss_conductor = F.mse_loss(new_value_vecs, conductor_batch['returns_vecs'])
                    loss_conductor = policy_loss_conductor + self.config['vf_coef'] * value_loss_conductor - self.config['ent_coef'] * entropy
                    
                    self.conductor_optimizer.zero_grad()
                    loss_conductor.backward()
                    torch.nn.utils.clip_grad_norm_(self.conductor.parameters(), self.config['max_grad_norm'])
                    self.conductor_optimizer.step()
                
                total_loss_conductor += loss_conductor.item()
                
                for specialist_id in self.specialists.keys():
                    specialist_steps_indices = [i for i, step in enumerate(minibatch) if step['specialist_id'] == specialist_id]
                    if not specialist_steps_indices: continue

                    model_alias = self.specialists[specialist_id].model_alias
                    optimizer = self.specialist_optimizers[model_alias]
                    backend = self.model_backends[model_alias]
                    backend.model.train()

                    prompts = [minibatch[i]['specialist_prompt_text'] for i in specialist_steps_indices]
                    output_ids = [minibatch[i]['specialist_output_ids'] for i in specialist_steps_indices]
                    adv_vecs_specialist = conductor_batch['advantages_vecs'][specialist_steps_indices]
                    
                    inputs = backend.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=512)
                    padded_output_ids = torch.nn.utils.rnn.pad_sequence(
                        output_ids, batch_first=True, padding_value=backend.tokenizer.pad_token_id
                    ).to(self.device)
                    
                    full_input_ids = torch.cat([inputs['input_ids'].to(self.device), padded_output_ids], dim=1)
                    full_attention_mask = torch.cat([
                        inputs['attention_mask'].to(self.device), 
                        (padded_output_ids != backend.tokenizer.pad_token_id).long()
                    ], dim=1)
                    
                    if self.use_amp:
                        with autocast():
                            outputs = backend.model(input_ids=full_input_ids, attention_mask=full_attention_mask)
                            logits = outputs.logits
                            output_logits = logits[:, inputs['input_ids'].shape[1]-1:-1, :]
                            log_probs = F.log_softmax(output_logits, dim=-1)
                            new_log_probs = log_probs.gather(dim=-1, index=padded_output_ids.unsqueeze(-1)).squeeze(-1)
                            output_mask = (padded_output_ids != backend.tokenizer.pad_token_id).float()
                            sequence_log_probs = (new_log_probs * output_mask).sum(dim=1)
                            pref = self.specialist_preferences.get(specialist_id, self.specialist_preferences['default'])
                            scalar_advantages_specialist = torch.tensordot(adv_vecs_specialist, pref, dims=1)
                            scalar_advantages_specialist = self._safe_normalize(scalar_advantages_specialist)
                            loss_specialist = -(sequence_log_probs * scalar_advantages_specialist.detach()).mean()
                        
                        optimizer.zero_grad()
                        self.scaler.scale(loss_specialist).backward()
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(backend.model.parameters(), self.config['max_grad_norm'])
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        outputs = backend.model(input_ids=full_input_ids, attention_mask=full_attention_mask)
                        logits = outputs.logits
                        output_logits = logits[:, inputs['input_ids'].shape[1]-1:-1, :]
                        log_probs = F.log_softmax(output_logits, dim=-1)
                        new_log_probs = log_probs.gather(dim=-1, index=padded_output_ids.unsqueeze(-1)).squeeze(-1)
                        output_mask = (padded_output_ids != backend.tokenizer.pad_token_id).float()
                        sequence_log_probs = (new_log_probs * output_mask).sum(dim=1)
                        pref = self.specialist_preferences.get(specialist_id, self.specialist_preferences['default'])
                        scalar_advantages_specialist = torch.tensordot(adv_vecs_specialist, pref, dims=1)
                        scalar_advantages_specialist = self._safe_normalize(scalar_advantages_specialist)
                        loss_specialist = -(sequence_log_probs * scalar_advantages_specialist.detach()).mean()

                        optimizer.zero_grad()
                        loss_specialist.backward()
                        torch.nn.utils.clip_grad_norm_(backend.model.parameters(), self.config['max_grad_norm'])
                        optimizer.step()
                    
                    total_loss_specialists += loss_specialist.item()
                
                num_updates += 1
        
        return {
            "loss_conductor": total_loss_conductor / num_updates if num_updates > 0 else 0,
            "loss_specialists": total_loss_specialists / num_updates if num_updates > 0 else 0
        }

    def train(self):
        print("Starting Training...")
        dataloader = DataLoader(self.train_dataset, batch_size=self.config['batch_size'], 
                                shuffle=True, drop_last=True, pin_memory=True, num_workers=4,
                                prefetch_factor=2, persistent_workers=True)

        for epoch in range(self.config['num_epochs']):
            print(f"\n{'='*60}\nEpoch {epoch+1}/{self.config['num_epochs']}\n{'='*60}")
            
            # Collect epoch statistics
            epoch_rewards = []
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            for batch in pbar:
                try:
                    trajectories, reward_metrics = self.collect_trajectories(batch)
                    if not trajectories: continue
                    
                    # Record rewards
                    epoch_rewards.append(reward_metrics.get('reward_correctness', 0))
                    
                    loss_metrics = self.update_models(trajectories)
                    all_metrics = {**reward_metrics, **loss_metrics}
                    pbar.set_postfix({k: f"{v:.4f}" for k, v in all_metrics.items()})
                except Exception as e:
                    tqdm.write(f"\nError: {e}")
                    continue
            
            # Epoch summary
            if epoch_rewards:
                avg_correctness = np.mean(epoch_rewards)
                print(f"\n{'='*60}")
                print(f"Epoch {epoch+1} Summary:")
                print(f"  Avg Correctness: {avg_correctness:.4f}")
                print(f"  Accuracy: {(avg_correctness/2)*100:.1f}%")  # correctness=2 means 100% accuracy
                print(f"{'='*60}")
                
                # Save checkpoints
                if (epoch + 1) % 2 == 0:  # Save every 2 epochs
                    self.save_models(f"./checkpoints/epoch_{epoch+1}")

    def save_models(self, output_dir: str):
        print(f"\nSaving models to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        conductor_dir = os.path.join(output_dir, "conductor")
        self.conductor.model.save_pretrained(conductor_dir)
        self.conductor.tokenizer.save_pretrained(conductor_dir)
        for alias, backend in self.model_backends.items():
            specialist_dir = os.path.join(output_dir, alias)
            backend.model.save_pretrained(specialist_dir)
            backend.tokenizer.save_pretrained(specialist_dir)
        print("Models saved.")