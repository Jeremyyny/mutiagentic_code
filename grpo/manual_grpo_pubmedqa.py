# manual_grpo_pubmedqa.py

# ==============================================================================
# SECTION 1: IMPORTS AND SETUP
# ==============================================================================

# Basic Python libraries
import random
import copy
import re
import os
import numpy as np
import wandb # Optional, for logging
import json

# PyTorch and related libraries
import torch
import torch.nn as nn

# Hugging Face libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def set_random_seed(seed: int = 42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed for consistent results
set_random_seed(42)

# Set environment variables for Weights & Biases (wandb) logging
# Replace with your own key or comment out if not using wandb
# os.environ["WANDB_API_KEY"] = "YOUR_WANDB_API_KEY"
# os.environ["WANDB_PROJECT"] = "GRPO-Qwen-PubMedQA-Manual"
# ==============================================================================
# SECTION 2: PROMPT AND DATA PREPARATION (ADAPTED FOR PUBMEDQA)
# ==============================================================================

# Define the structured prompt format for our task
SYSTEM_PROMPT = """You are an expert biomedical researcher. Your task is to answer a question based on a provided context.
First, write out a step-by-step reasoning process within <reasoning> tags.
Then, provide the final answer (either "yes" or "no") within <answer> tags.
"""

def build_prompt(messages):
    """Builds a single prompt string from a list of messages."""
    return "\n".join([msg["content"].strip() for msg in messages])

def prepare_pubmedqa_dataset(json_file_path="golden_dataset_pubmedqa_qwen2.5_pro_test_500.json"):
    """Loads and prepares the PubMedQA dataset from your local JSON file."""
    formatted_data = []
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for entry in data:
        if all(k in entry for k in ['question', 'context', 'ground_truth']):
            user_content = f"Context:\n{entry['context']}\n\nQuestion:\n{entry['question']}"
            
            prompt_str = build_prompt([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ])
            
            formatted_example = {
                "prompt": prompt_str,
                "answer": entry["ground_truth"].strip().lower()
            }
            formatted_data.append(formatted_example)
    return formatted_data

# ==============================================================================
# SECTION 3: REWARD FUNCTIONS (ADAPTED FOR PUBMEDQA)
# ==============================================================================

def extract_answer_from_model_output(text):
    """Extracts the value from the last <answer> tag in the text."""
    # parts = text.split("<answer>")
    # if len(parts) < 2:
    #     return None
    # last_part = parts[-1]
    # if "</answer>" not in last_part:
    #     return None
    # answer = last_part.split("</answer>")[0].strip().lower()
    
    # # Be strict: only return if it's clearly 'yes' or 'no'
    if "yes" in text: return "yes"
    if "no" in text: return "no"
    return None

def pubmedqa_correctness_reward(completions, answer, **kwargs):
    """Assigns a reward based on the correctness of the 'yes'/'no' answer."""
    responses = [comp[0]['content'] for comp in completions]
    extracted_answers = [extract_answer_from_model_output(r) for r in responses]
    rewards = []
    for extracted, expected in zip(extracted_answers, answer):
        if extracted and extracted == expected:
            rewards.append(2.0)  # High reward for an exact match
        else:
            rewards.append(0.0)  # No reward for wrong or missing answer
    return rewards

def format_reward(completions, **kwargs):
    """Assigns a reward for adhering to the desired XML format."""
    responses = [comp[0]['content'] for comp in completions]
    rewards = []
    for response in responses:
        score = 0.0
        if "<reasoning>" in response: score += 0.2
        if "</reasoning>" in response: score += 0.2
        if "<answer>" in response: score += 0.2
        if "</answer>" in response: score += 0.2
        rewards.append(score) # Max format score = 0.8
    return rewards

def combined_reward(prompts, completions, answer, **kwargs):
    """Combines correctness and format rewards."""
    correctness_scores = pubmedqa_correctness_reward(completions=completions, answer=answer)
    format_scores = format_reward(completions=completions)

    combined_rewards = [c_score + f_score for c_score, f_score in zip(correctness_scores, format_scores)]
    return combined_rewards

# ==============================================================================
# SECTION 4: CORE GRPO/PPO LOGIC (IMITATED FROM EXAMPLE)
# ==============================================================================

def selective_log_softmax(logits, input_ids):
    """Computes log probabilities for specific tokens."""
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

def compute_log_probs(model, input_ids, attention_mask, logits_to_keep):
    """Computes the log probabilities for a batch of tokens."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return selective_log_softmax(logits, input_ids)

def create_completion_mask(completion_ids, eos_token_id):
    """Creates a mask for completion tokens, stopping after the first EOS token."""
    is_eos = completion_ids == eos_token_id
    # Find the index of the first EOS token for each sequence
    eos_indices = torch.argmax(is_eos.int(), dim=1)
    # If no EOS is found, argmax returns 0. We need to handle this.
    # We set the index to max_length if no EOS is found.
    eos_indices[~is_eos.any(dim=1)] = completion_ids.size(1)

    # Create a range tensor to compare with indices
    seq_indices = torch.arange(completion_ids.size(1), device=completion_ids.device).expand_as(completion_ids)
    
    # The mask is 1 for all tokens up to and including the first EOS
    mask = (seq_indices <= eos_indices.unsqueeze(1)).int()
    return mask

def generate_completions(model, tokenizer, prompts, num_generations=4, max_completion_length=128):
    """Generates multiple completions for each prompt."""
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)
    
    prompt_length = prompt_ids.size(1)
    
    # Repeat prompts to generate multiple completions in one batch
    repeated_prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    repeated_prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
    
    outputs = model.generate(
        repeated_prompt_ids,
        attention_mask=repeated_prompt_mask,
        max_new_tokens=max_completion_length,
        do_sample=True,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    completion_ids = outputs[:, prompt_length:]
    completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)
    
    return prompt_ids, prompt_mask, completion_ids, completion_mask

def generate_rollout_data(model, ref_model, tokenizer, batch_samples, num_generations, max_completion_length):
    """Generates data for GRPO rollouts including completions and log probabilities."""
    prompts = [sample["prompt"] for sample in batch_samples]
    answers = [sample["answer"] for sample in batch_samples]
    
    with torch.no_grad():
        prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
            model, tokenizer, prompts, num_generations, max_completion_length
        )
        
        # We need the original prompts repeated for log prob calculation
        repeated_prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
        repeated_prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
        
        input_ids = torch.cat([repeated_prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([repeated_prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # compute_log_probs needs a model on a single device, so we use .module
        # if it is wrapped in DataParallel
        policy_model = model.module if isinstance(model, nn.DataParallel) else model
        reference_model = ref_model.module if isinstance(ref_model, nn.DataParallel) else ref_model

        old_log_probs = compute_log_probs(policy_model, input_ids, attention_mask, logits_to_keep)
        ref_log_probs = compute_log_probs(reference_model, input_ids, attention_mask, logits_to_keep)
        
    formatted_completions = [[{'content': tokenizer.decode(ids, skip_special_tokens=True)}] for ids in completion_ids]
    repeated_prompts = [p for p in prompts for _ in range(num_generations)]
    repeated_answers = [a for a in answers for _ in range(num_generations)]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "formatted_completions": formatted_completions,
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "logits_to_keep": logits_to_keep,
        "batch_size": len(prompts),
        "num_generations": num_generations
    }

def grpo_loss(model, ref_model, rollout_data, reward_function, beta=0.01, epsilon=0.2):
    """Computes the GRPO loss for updating the policy model."""
    device = next(model.parameters()).device
    
    # Unpack rollout data
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    logits_to_keep = rollout_data["logits_to_keep"]
    old_log_probs = rollout_data["old_log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]
    
    # Compute current log probs
    policy_model = model.module if isinstance(model, nn.DataParallel) else model
    token_log_probs = compute_log_probs(policy_model, input_ids, attention_mask, logits_to_keep)
    
    # Calculate ratio and rewards
    ratio = torch.exp(token_log_probs - old_log_probs)
    rewards = torch.tensor(
        reward_function(
            prompts=rollout_data["repeated_prompts"], 
            completions=rollout_data["formatted_completions"], 
            answer=rollout_data["repeated_answers"]
        ),
        dtype=torch.float32,
        device=device
    )
    
    # Standardize rewards at the group level (GRPO)
    batch_size = rollout_data["batch_size"]
    num_generations = rollout_data["num_generations"]
    rewards_grouped = rewards.view(batch_size, num_generations)
    
    mean_rewards = rewards_grouped.mean(dim=1, keepdim=True)
    std_rewards = rewards_grouped.std(dim=1, keepdim=True)
    advantages = (rewards_grouped - mean_rewards) / (std_rewards + 1e-8)
    advantages = advantages.view(-1).unsqueeze(1) # Flatten back for token-wise multiplication
    
    # PPO Clipped Surrogate Objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    surrogate_loss = torch.min(surr1, surr2)
    
    # KL Penalty
    kl_div = torch.exp(ref_log_probs - token_log_probs) - (ref_log_probs - token_log_probs) - 1
    
    # Combine and mask the loss
    per_token_loss = surrogate_loss - beta * kl_div
    # We only care about the loss for the completion tokens
    masked_loss = per_token_loss * completion_mask
    loss = -(masked_loss.sum() / completion_mask.sum())
    
    avg_reward = rewards.mean().item()
    return loss, avg_reward

# ==============================================================================
# SECTION 5: TRAINING LOOP (IMITATED AND ADAPTED FOR SINGLE/MULTI GPU)
# ==============================================================================
 
def train_with_grpo(model, tokenizer, train_data, num_iterations=1, num_steps=500, batch_size=4,
                    num_generations=4, max_completion_length=128, beta=0.1,
                    learning_rate=5e-6, mu=3, epsilon=0.2, reward_function=None):
    """Main GRPO training loop."""
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Outer loop for updating the reference model
    for iteration in range(num_iterations):
        print(f"\n--- Starting GRPO Iteration {iteration + 1}/{num_iterations} ---")

        # Create a deep copy of the current model to act as the reference model
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        print("Reference model created for this iteration.")

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        model.train()

        # Inner loop for batch updates
        for step in range(num_steps):
            # Sample a random batch of prompts
            batch_samples = random.sample(train_data, batch_size)
            
            # 1. Generate rollouts (completions, log_probs)
            rollout_data = generate_rollout_data(
                model,
                ref_model,
                tokenizer,
                batch_samples,
                num_generations,
                max_completion_length
            )
            
            # 2. Perform multiple optimization steps (PPO-style)
            for _ in range(mu):
                loss, avg_reward = grpo_loss(
                    model,
                    ref_model,
                    rollout_data,
                    reward_function,
                    beta=beta,
                    epsilon=epsilon
                )
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Logging
            print(f"Iter {iteration+1}, Step {step+1}/{num_steps}, Loss: {loss.item():.4f}, Avg Reward: {avg_reward:.2f}")
            # if os.environ.get("WANDB_API_KEY"):
            #     wandb.log({
            #         "loss": loss.item(),
            #         "average_reward": avg_reward,
            #         "iteration": iteration + 1,
            #         "step": step + 1,
            #     })
    return model

# ==============================================================================
# SECTION 6: EVALUATION (ADAPTED FOR PUBMEDQA)
# ==============================================================================

def evaluate_model(model, tokenizer, eval_examples, device=None):
    """Evaluates the model on 'yes'/'no' accuracy."""
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    correct = 0
    total = len(eval_examples)
    print("\n" + "="*50)
    print(f"STARTING EVALUATION ON {total} EXAMPLES")
    print("="*50)

    for example in eval_examples:
        full_prompt = example["prompt"]
        expected_answer = example["answer"]

        inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                temperature=0.1, # Use low temperature for deterministic evaluation
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        predicted_answer = extract_answer_from_model_output(response)
        
        is_correct = (predicted_answer == expected_answer)
        if is_correct:
            correct += 1

        print(f"\nPrompt:\n{full_prompt}")
        print(f"\nExpected Answer: {expected_answer}")
        print(f"Predicted Answer: {predicted_answer}")
        print(f"Full Response:\n{response}")
        print(f"Correct: {'✓' if is_correct else '✗'}")
        print("-"*50)

    accuracy = (correct / total) * 100
    print(f"\nEvaluation Complete. Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print("="*50)
    
    model.train()
    return accuracy

# ==============================================================================
# SECTION 7: MAIN EXECUTION BLOCK
# ==============================================================================

def main():
    """Main function to orchestrate the entire process."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using primary device: {device}")

    # --- Model and Tokenizer Loading ---
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading model: {model_name}...")
    # Load in default FP32 precision, as mixed precision will be handled by the training loop if enabled.
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model and tokenizer loaded.")
    
    # Move model to device FIRST before evaluation
    model.to(device)

    # --- Data Preparation ---
    all_data = prepare_pubmedqa_dataset()
    random.shuffle(all_data)
    eval_data_size = 100
    eval_data = all_data[:eval_data_size]
    train_data = all_data[eval_data_size:]
    print(f"Data prepared. Training examples: {len(train_data)}, Evaluation examples: {len(eval_data)}")

    # --- Pre-Training Evaluation ---
    print("\nEvaluating model before fine-tuning...")
    evaluate_model(model, tokenizer, eval_data, device)

    # --- Training Configuration ---
    # This config is designed for a single GPU with ~16-24GB VRAM. Adjust if needed.
    training_config = {
        'num_iterations': 1,        # Number of times to update the reference model
        'num_steps': 100,           # Batches per iteration. Increase for more training.
        'batch_size': 2,            # Prompts per batch. Decrease if OOM.
        'num_generations': 4,       # Completions per prompt. Decrease if OOM.
        'max_completion_length': 300, # Decrease if OOM.
        'beta': 0.01,               # KL penalty strength
        'learning_rate': 5e-6,      # Optimizer learning rate
        'mu': 2,                    # Number of optimization steps per batch
        'epsilon': 0.2              # PPO clipping value
    }
    
    # Initialize wandb if API key is set
    # if os.environ.get("WANDB_API_KEY"):
    #     wandb.init(project=os.environ["WANDB_PROJECT"], config=training_config, reinit=True)
    #     print("Weights & Biases initialized.")

    # --- Start Training ---
    print("\nStarting GRPO fine-tuning...")
    trained_model = train_with_grpo(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        reward_function=combined_reward,
        **training_config
    )
    # if os.environ.get("WANDB_API_KEY"):
    #     wandb.finish()

    # --- Post-Training Evaluation ---
    print("\nEvaluating model after GRPO fine-tuning...")
    evaluate_model(trained_model, tokenizer, eval_data, device)

    # --- Save Final Model ---
    output_dir = "grpo_pubmedqa_finetuned_model"
    print(f"\nSaving fine-tuned model to {output_dir}...")
    trained_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()