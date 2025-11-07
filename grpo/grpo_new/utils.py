# =============================
# utils.py (7B / multi-GPU optimized)
# =============================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalHF:
    """Local HuggingFace backend for specialists (bf16 + accelerate safe)."""

    def __init__(self, model_path, max_tokens=300):
        print(f"🔹 Loading specialist model from {model_path} ...")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,   # ✅ Blackwell 优化
            device_map=None               # ✅ 不使用 auto map（避免 accelerate 冲突）
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.max_tokens = max_tokens

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        print("✓ Specialist backend ready.")

    def generate_with_logprobs(self, prompt_text, max_new_tokens=None):
        """Greedy decoding + optional fallback (fast deterministic mode)."""
        if max_new_tokens is None:
            max_new_tokens = self.max_tokens

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt_text, return_tensors="pt",
                                truncation=True, max_length=1024).to(device)

        # Greedy decode (deterministic, faster)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,                 # ✅ 不采样，提高一致性和速度
                temperature=0.7,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
            gen_ids = outputs.sequences[:, inputs["input_ids"].shape[1]:]
            gen_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        # log-probs 计算（近似）
        scores = outputs.scores
        log_probs = []
        for i, logits in enumerate(scores):
            if i >= gen_ids.shape[1]:
                break
            log_softmax = torch.log_softmax(logits, dim=-1)
            tid = gen_ids[:, i].unsqueeze(-1)
            token_log_prob = log_softmax.gather(dim=1, index=tid)
            log_probs.append(token_log_prob)
        if log_probs:
            total_logp = torch.stack(log_probs, dim=1).sum(dim=1)
        else:
            total_logp = torch.zeros(1, device=device)

        return gen_text, gen_ids, total_logp