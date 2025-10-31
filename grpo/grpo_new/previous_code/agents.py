# agents.py (修复版 - 解决策略崩溃和输出乱码问题)
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributions import Categorical
import subagents
import utils
import re

class ConductorAgent(nn.Module):
    def __init__(self, model_path, specialist_names, num_rewards):
        super().__init__()
        self.specialist_names = specialist_names
        self.num_specialists = len(specialist_names)
        
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 添加 pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        # === 关键改进 1: 初始化策略头,添加更强的正则化 ===
        hidden_size = self.model.config.hidden_size
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),  # 添加 LayerNorm
            nn.ReLU(),
            nn.Dropout(0.2),  # 增加 dropout
            nn.Linear(hidden_size // 2, self.num_specialists)
        )
        
        # === 关键改进 2: 策略头使用 Xavier 初始化,避免极端值 ===
        for module in self.policy_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # 小gain避免初始logits过大
                nn.init.constant_(module.bias, 0)
        
        # 价值网络头部
        self.value_head = nn.Linear(hidden_size, num_rewards)
        nn.init.xavier_uniform_(self.value_head.weight)
        nn.init.constant_(self.value_head.bias, 0)

    def _build_prompt(self, state, history):
        """构建更清晰的状态描述 prompt"""
        problem_info = state.get("problem", state.get("question", ""))
        
        # === 关键改进 3: 在 prompt 中明确告诉模型当前进度 ===
        current_step = len(history)
        
        if history:
            # 只显示最近的步骤
            recent_history = history[-2:]
            context_info = "\n\nCompleted steps:\n" + "\n".join([
                f"Step {current_step - len(recent_history) + i + 1}: {h['content'][:100]}..."
                for i, h in enumerate(recent_history)
            ])
        else:
            context_info = "\n\nNo steps completed yet."
        
        # === 关键改进 4: 明确提示下一步应该做什么 ===
        if current_step == 0:
            hint = "\n\nNext: Start by understanding the problem."
        elif current_step >= 3:
            hint = "\n\nNext: Provide the final answer (yes/no/maybe)."
        else:
            hint = "\n\nNext: Continue reasoning or computation."
        
        prompt = f"""Problem: {problem_info}{context_info}{hint}

Available agents: {', '.join(self.specialist_names)}

Determine which agent should act next based on the current progress."""
        
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], 
            tokenize=False, 
            add_generation_prompt=True
        )

    def forward(self, input_ids, attention_mask):
        """前向传播,返回 agent 选择 logits 和价值向量"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # 使用最后一层的平均池化而不是最后一个 token
        # === 关键改进 5: 使用 mean pooling 更稳定 ===
        last_hidden_state = outputs.hidden_states[-1]
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_hidden = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        pooled_hidden = sum_hidden / sum_mask
        
        # 策略 logits (用于选择 agent)
        policy_logits = self.policy_head(pooled_hidden)
        
        # 价值向量
        value_vector = self.value_head(pooled_hidden)
        
        return policy_logits, value_vector

    def act(self, state, history):
        """使用策略网络选择 agent"""
        self.eval()
        prompt = self._build_prompt(state, history)
        device = next(self.model.parameters()).device
        
        # Tokenize 输入
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            # 获取策略 logits 和价值
            policy_logits, value_vector = self.forward(input_ids, attention_mask)
            
            # === 关键改进 6: 添加温度参数,避免策略过于确定 ===
            temperature = 1.0
            policy_logits = policy_logits / temperature
            
            # === 关键改进 7: 基于步骤数的启发式偏置 ===
            current_step = len(history)
            bias = torch.zeros_like(policy_logits)
            
            # 第一步倾向于选择 problem_understanding
            if current_step == 0 and "problem_understanding" in self.specialist_names:
                idx = self.specialist_names.index("problem_understanding")
                bias[0, idx] += 2.0
            
            # 后期倾向于选择 answering
            if current_step >= 3 and "answering" in self.specialist_names:
                idx = self.specialist_names.index("answering")
                bias[0, idx] += 3.0
            
            # 应用偏置
            policy_logits = policy_logits + bias
            
            # 使用 Categorical 分布进行采样
            policy_probs = torch.softmax(policy_logits, dim=-1)
            dist = Categorical(policy_probs)
            
            # 采样一个 agent
            action_idx = dist.sample()
            action_log_prob = dist.log_prob(action_idx)
            
            # 获取对应的 specialist 名称
            chosen_specialist = self.specialist_names[action_idx.item()]
            
            # 构造动作
            action_dict = {
                "specialist_id": chosen_specialist,
                "input": "Analyze and process"
            }
            
            # 为了与原来的接口兼容,创建一个虚拟的 action_ids
            action_ids = action_idx.unsqueeze(0).unsqueeze(0)
        
        return action_dict, action_ids, action_log_prob, value_vector.squeeze(0)
    
    def evaluate(self, state_ids, state_mask, action_indices):
        """在 PPO 更新时,重新评估动作的 log_probs"""
        self.train()
        
        # 获取策略 logits 和价值
        policy_logits, new_value_vecs = self.forward(state_ids, state_mask)
        
        # 计算 log_probs
        policy_probs = torch.softmax(policy_logits, dim=-1)
        dist = Categorical(policy_probs)
        
        # action_indices 的形状是 [batch_size, 1],需要 squeeze
        action_indices = action_indices.squeeze(-1)
        new_log_probs = dist.log_prob(action_indices)
        
        # 计算熵(鼓励探索)
        entropy = dist.entropy().mean()
        
        return new_log_probs, new_value_vecs, entropy


class SpecialistAgent(nn.Module):
    """专家 Agent 的包装器"""
    def __init__(self, agent_name, model_alias):
        super().__init__()
        self.agent_name = agent_name
        self.model_alias = model_alias

    def generate(self, state, history, model_backend: utils.LocalHF):
        """使用指定的模型后端生成文本"""
        self.eval()
        agent_prompt_fn = subagents.AGENT_PROMPT_FUNCTIONS.get(self.agent_name)
        if not agent_prompt_fn:
            raise ValueError(f"Prompt function for {self.agent_name} not found.")

        messages = agent_prompt_fn(state, history)
        prompt_text = model_backend.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        output_text, output_ids, log_probs = model_backend.generate_with_logprobs(prompt_text)
        
        return output_text, output_ids, log_probs