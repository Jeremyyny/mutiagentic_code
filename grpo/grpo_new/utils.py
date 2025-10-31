# =============================
# utils.py (minor changes)
# Changes:
# 1) Default max_tokens = 150
# 2) Tokenization truncation max_length = 1024
# 3) Generation quality fallback logic retained
# =============================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LocalHF:
    """Local HuggingFace model backend (fixed + length increased)"""
    
    def __init__(self, model_path, max_tokens=300):
        print(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_tokens = max_tokens
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"✓ Model loaded successfully")
    
    def generate_with_logprobs(self, prompt_text, max_new_tokens=None):
        if max_new_tokens is None:
            max_new_tokens = self.max_tokens
        device = next(self.model.parameters()).device
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=5,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
            generated_ids = outputs.sequences[:, input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            if len(generated_text.strip()) < 10 or not any(c.isalnum() for c in generated_text):
                print(f"⚠️  Low quality output detected, regenerating with greedy decoding...")
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=min(max_new_tokens, 40),
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                generated_ids = outputs.sequences[:, input_ids.shape[1]:]
            
            scores = outputs.scores
            log_probs_list = []
            for i, token_logits in enumerate(scores):
                if i >= generated_ids.shape[1]:
                    break
                log_softmax = torch.log_softmax(token_logits, dim=-1)
                token_id = generated_ids[:, i].unsqueeze(-1)
                token_log_prob = log_softmax.gather(dim=1, index=token_id)
                log_probs_list.append(token_log_prob)
            
            if log_probs_list:
                log_probs = torch.stack(log_probs_list, dim=1).sum(dim=1)
            else:
                log_probs = torch.zeros(1, device=device)
            
            output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return output_text, generated_ids, log_probs