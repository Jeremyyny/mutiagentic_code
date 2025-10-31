import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributions import Categorical
import subagents

class ManagerAgent(nn.Module):
    def __init__(self, model_path, specialist_names, num_rewards, freeze_manager_backbone=True, manager_token_max_len=1024):
        super().__init__()
        self.specialist_names = specialist_names
        self.num_specialists = len(specialist_names)
        self.manager_token_max_len = manager_token_max_len
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        hidden_size = self.model.config.hidden_size
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, self.num_specialists)
        )
        for m in self.policy_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0)
        self.value_head = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.value_head.weight)
        nn.init.constant_(self.value_head.bias, 0)

        if freeze_manager_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

    def _build_prompt(self, state, history):
        problem_info = state.get("problem", state.get("question", ""))
        context_info = state.get("context", "")
        current_step = len(history)

        if history:
            recent_history = history[-2:]
            history_text = "\n\nCompleted steps:\n" + "\n".join([
                f"Step {current_step - len(recent_history) + i + 1}: {h['content'][:100]}..."
                for i, h in enumerate(recent_history)
            ])
        else:
            history_text = "\n\nNo steps completed yet."

        if current_step == 0:
            hint = "\n\nNext: Start with 'problem_understanding'."
        elif current_step == 1:
            hint = "\n\nNext: Call 'reasoning' to evaluate evidence briefly."
        elif current_step == 2:
            hint = "\n\nNext: Call 'computation' if numbers exist, otherwise skip."
        else:
            hint = "\n\nNext: Call 'answering' to produce ONLY one word: yes, no, or maybe."

        prompt = (
            f"Problem: {problem_info}\n"
            f"{f'Context: {context_info}' if context_info else ''}{history_text}{hint}\n\n"
            f"Available specialists: {', '.join(self.specialist_names)}\n\n"
            "Determine which specialist should act next based on the current progress."
        )

        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        mask_exp = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        pooled = (last_hidden_state * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp_min(1e-9)
        logits = self.policy_head(pooled)
        value = self.value_head(pooled).squeeze(-1)
        return logits, value

    def act(self, state, history):
        self.eval()
        prompt = self._build_prompt(state, history)
        device = next(self.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=self.manager_token_max_len)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        with torch.no_grad():
            logits, value = self.forward(input_ids, attention_mask)
            temperature = 1.0
            logits = logits / temperature
            current_step = len(history)
            bias = torch.zeros_like(logits)
            if current_step == 0:
                bias[0, self.specialist_names.index("problem_understanding")] += 2.0
            if current_step == 1:
                bias[0, self.specialist_names.index("reasoning")] += 2.0
            if current_step == 2:
                bias[0, self.specialist_names.index("computation")] += 1.5
            if current_step >= 3:
                bias[0, self.specialist_names.index("answering")] += 4.0
            logits = logits + bias
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            a_idx = dist.sample()
            logp = dist.log_prob(a_idx)
            chosen = self.specialist_names[a_idx.item()]
            return {"specialist_id": chosen, "input": "Analyze and process"}, int(a_idx.item()), float(logp.item()), float(value.squeeze(0).item())

    def evaluate_batch(self, state_ids, state_mask, action_indices):
        self.train()
        logits, values = self.forward(state_ids, state_mask)
        dist = Categorical(torch.softmax(logits, dim=-1))
        new_log_probs = dist.log_prob(action_indices)
        entropy = dist.entropy().mean()
        return new_log_probs, values, entropy


class FixedSpecialistAgent(nn.Module):
    def __init__(self, agent_name, model_backend):
        super().__init__()
        self.agent_name = agent_name
        self.model_backend = model_backend

    def generate(self, state, history):
        self.eval()
        agent_prompt_fn = subagents.AGENT_PROMPT_FUNCTIONS.get(self.agent_name)
        if not agent_prompt_fn:
            raise ValueError(f"Prompt function for {self.agent_name} not found.")
        messages = agent_prompt_fn(state, history)
        prompt_text = self.model_backend.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        max_new = 10 if self.agent_name == "answering" else self.model_backend.max_tokens
        with torch.no_grad():
            output_text, output_ids, log_probs = self.model_backend.generate_with_logprobs(prompt_text, max_new_tokens=max_new)
        return output_text, output_ids, log_probs
