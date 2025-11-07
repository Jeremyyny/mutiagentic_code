import torch
import torch.nn as nn
from torch.distributions import Categorical
from transformers import AutoModelForCausalLM, AutoTokenizer
import subagents


class ManagerAgent(nn.Module):
    def __init__(
        self,
        model_path,
        specialist_names,
        num_rewards,
        freeze_manager_backbone=True,
        manager_token_max_len=1024,
    ):
        super().__init__()
        self.specialist_names = specialist_names
        self.num_specialists = len(specialist_names)
        self.manager_token_max_len = manager_token_max_len

        # backbone
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=None,  # accelerate 管
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        # policy / value heads
        hidden = self.model.config.hidden_size
        self.policy_head = nn.Linear(hidden, self.num_specialists)
        self.value_head = nn.Linear(hidden, 1)
        nn.init.xavier_uniform_(self.policy_head.weight)
        nn.init.xavier_uniform_(self.value_head.weight)

        if freeze_manager_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

    # ----------------------------------------------------------------------
    # 🔹 改进 prompt：短、层次化、更易学
    # ----------------------------------------------------------------------
    def _build_prompt(self, state, history):
        problem = state.get("problem", state.get("question", ""))
        context = state.get("context", "")
        step = len(history)

        recent = history[-2:] if history else []
        hist_text = ""
        if recent:
            for i, h in enumerate(recent):
                content = h.get("content", "")
                hist_text += f"Step {step - len(recent) + i + 1}: {content[:160]}...\n"

        hint = {
            0: "Start with 'problem_understanding'.",
            1: "Then move to 'reasoning'.",
            2: "Use 'computation' if numbers appear, else skip.",
        }.get(step, "Finally call 'answering' to conclude (yes/no/maybe).")

        return (
            f"You are the MANAGER of a medical reasoning system.\n"
            f"Specialists: [problem_understanding, reasoning, computation, answering].\n"
            f"Task:\nProblem: {problem}\nContext: {context}\n"
            f"{'Recent steps:\n' + hist_text if hist_text else ''}"
            f"Next hint: {hint}\n\n"
            "Choose exactly one specialist for the next step."
        )

    # ----------------------------------------------------------------------
    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last = out.hidden_states[-1][:, -1, :]
        logits = self.policy_head(last)
        value = self.value_head(last).squeeze(-1)
        return logits, value

    # ----------------------------------------------------------------------
    def act(self, state, history):
        self.eval()
        prompt = self._build_prompt(state, history)
        device = next(self.policy_head.parameters()).device
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=self.manager_token_max_len
        ).to(device)

        with torch.no_grad():
            logits, value = self.forward(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            a_idx = dist.sample()
            logp = dist.log_prob(a_idx)

        chosen = self.specialist_names[a_idx.item()]
        return {"specialist_id": chosen, "input": "Continue with next analysis"}, int(a_idx), logp, value.item()

    # ----------------------------------------------------------------------
    def evaluate_batch(self, state_ids, state_mask, action_indices):
        self.train()
        logits, values = self.forward(state_ids, state_mask)
        dist = Categorical(torch.softmax(logits, dim=-1))
        log_probs = dist.log_prob(action_indices)
        entropy = dist.entropy().mean()
        return log_probs, values, entropy


# ----------------------------------------------------------------------
# ✅ Specialist wrapper: 纯推理 + prompt 注册机制
# ----------------------------------------------------------------------
class FixedSpecialistAgent(nn.Module):
    def __init__(self, agent_name, model_backend):
        super().__init__()
        self.agent_name = agent_name
        self.model_backend = model_backend

    def generate(self, state, history):
        self.eval()
        fn = subagents.AGENT_PROMPT_FUNCTIONS.get(self.agent_name)
        if not fn:
            raise ValueError(f"Prompt function for {self.agent_name} not found.")
        messages = fn(state, history)
        prompt_text = self.model_backend.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        max_new = 10 if self.agent_name == "answering" else self.model_backend.max_tokens
        with torch.no_grad():
            text, ids, logp = self.model_backend.generate_with_logprobs(prompt_text, max_new_tokens=max_new)
        return text, ids, logp