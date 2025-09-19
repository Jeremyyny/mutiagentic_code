# GRPO PubMedQA - SingleCard - Fine-tuning: Quick Start Guide

A reinforcement learning approach to fine-tune language models on biomedical question-answering tasks using Group Relative Policy Optimization (GRPO).

## 🎯 What This Code Does

This script implements **GRPO (Group Relative Policy Optimization)** to fine-tune language models on the PubMedQA dataset. Key features:

- 🧠 **Reinforcement Learning**: Uses PPO-based optimization with group-relative rewards
- 📊 **Multi-Generation Strategy**: Generates multiple answers per question for better learning
- ⚖️ **Dual Reward System**: Combines answer correctness + XML format adherence
- 🎯 **Biomedical Focus**: Specialized for medical literature yes/no questions
- 📋 **Structured Output**: Enforces `<reasoning>` and `<answer>` tag format

## ✅ Prerequisites

### System Requirements
- **GPU**: 16-24GB VRAM recommended (RTX 3090/4090, A6000, V100, etc.)
- **RAM**: 32GB+ system memory
- **Storage**: 10GB+ free space for models and datasets

### Software Dependencies
```bash
# PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Hugging Face libraries
pip install transformers datasets

# Additional dependencies
pip install numpy wandb  # wandb is optional for logging
```

## 📦 Data Preparation

### 1. Prepare Your PubMedQA Dataset

Your JSON file should be named `golden_dataset_pubmedqa_qwen2.5_pro_test_500.json` and follow this format:

```json
[
  {
    "question": "Does aspirin reduce cardiovascular risk?",
    "context": "Multiple studies have shown that low-dose aspirin therapy can reduce the risk of cardiovascular events...",
    "ground_truth": "yes"
  },
  {
    "question": "Is vitamin D supplementation effective for bone health?",
    "context": "Research indicates that vitamin D plays a crucial role in calcium absorption...",
    "ground_truth": "no"
  }
]
```

### 2. Required Fields
- `question`: The biomedical question to answer
- `context`: Relevant biomedical context/literature
- `ground_truth`: Either "yes" or "no" (case-insensitive)

## 🚀 Quick Start

### Basic Usage

1. **Clone/Download** the script to your working directory
2. **Prepare your dataset** (see format above)
3. **Run the training**:

```bash
python manual_grpo_pubmedqa.py
```

That's it! The script will:
- Load the Qwen2.5-0.5B model
- Evaluate baseline performance
- Train using GRPO for 100 steps
- Evaluate final performance
- Save the fine-tuned model

### Expected Output

```
Using primary device: cuda:0
Loading model: Qwen/Qwen2.5-0.5B-Instruct...
Model and tokenizer loaded.
Data prepared. Training examples: 400, Evaluation examples: 100

Evaluating model before fine-tuning...
==================================================
STARTING EVALUATION ON 100 EXAMPLES
==================================================
...
Evaluation Complete. Accuracy: 65.00% (65/100)

--- Starting GRPO Iteration 1/1 ---
Reference model created for this iteration.

Starting GRPO fine-tuning...
Iter 1, Step 1/100, Loss: 0.2340, Avg Reward: 1.25
Iter 1, Step 2/100, Loss: 0.2180, Avg Reward: 1.42
...

Evaluating model after GRPO fine-tuning...
Evaluation Complete. Accuracy: 78.00% (78/100)

Saving fine-tuned model to grpo_pubmedqa_finetuned_model...
Model saved successfully.
```

## ⚙️ Configuration Options

### Memory Optimization

If you encounter **Out of Memory (OOM)** errors, adjust these parameters in the `training_config` dict:

```python
training_config = {
    'batch_size': 1,            # Reduce from 2 → 1
    'num_generations': 2,       # Reduce from 4 → 2
    'max_completion_length': 200, # Reduce from 300 → 200
    'num_steps': 50,            # Reduce training steps if needed
}
```

### Training Intensity

For **more thorough training**:

```python
training_config = {
    'num_iterations': 2,        # Update reference model twice
    'num_steps': 200,          # More training steps
    'batch_size': 4,           # Larger batches (if GPU allows)
    'learning_rate': 1e-5,     # Higher learning rate
}
```

### Reward Balance

To adjust the **reward weighting**:

```python
# In pubmedqa_correctness_reward function
rewards.append(3.0)  # Increase correctness reward (default: 2.0)

# In format_reward function
if "<reasoning>" in response: score += 0.5  # Increase format reward (default: 0.2)
```

## 📊 Monitoring and Logging

### Weights & Biases Integration

To enable W&B logging, uncomment and set:

```python
os.environ["WANDB_API_KEY"] = "your_wandb_api_key_here"
os.environ["WANDB_PROJECT"] = "GRPO-PubMedQA-Experiment"
```

Then uncomment the wandb initialization and logging lines throughout the code.

### Manual Monitoring

The script prints detailed progress including:
- Loss values per step
- Average rewards
- Individual evaluation examples
- Final accuracy metrics

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce `batch_size`, `num_generations`, or `max_completion_length`

**2. File Not Found**
```
FileNotFoundError: golden_dataset_pubmedqa_qwen2.5_pro_test_500.json
```
**Solution**: Ensure your JSON file is in the same directory and named correctly

**3. Invalid JSON Format**
```
json.JSONDecodeError
```
**Solution**: Validate your JSON file structure matches the expected format

**4. Low GPU Utilization**
```
Training very slow on GPU
```
**Solution**: Increase `batch_size` and `num_generations` if memory allows

### Performance Tips

- **GPU Memory**: Monitor with `nvidia-smi` during training
- **Data Size**: Start with smaller datasets (100-500 examples) for testing
- **Model Size**: Consider Qwen2.5-1.5B or 3B for better performance if you have more VRAM
- **Evaluation Frequency**: Reduce evaluation size during development

## 📈 Expected Results

### Typical Performance Gains
- **Baseline**: 60-70% accuracy (untrained model)
- **After GRPO**: 75-85% accuracy (depending on dataset quality)
- **Training Time**: 1-3 hours on RTX 4090 for 500 examples

### Output Format
The model learns to generate structured responses like:
```
<reasoning>
Based on the provided context about cardiovascular studies, 
aspirin has been shown to reduce the risk of heart attacks 
and strokes in multiple clinical trials...
</reasoning>

<answer>yes</answer>
```

## 🎛️ Advanced Customization

### Custom Reward Functions

Modify the reward system by editing:
- `pubmedqa_correctness_reward()`: Answer accuracy scoring
- `format_reward()`: XML structure compliance
- `combined_reward()`: Overall reward combination

### Different Model Backends

Change the model by modifying:
```python
model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Larger model
# or
model_name = "microsoft/DialoGPT-medium"    # Different architecture
```

### Custom Datasets

Adapt for other yes/no QA datasets by:
1. Modifying `prepare_pubmedqa_dataset()` function
2. Updating `extract_answer_from_model_output()` for your answer format
3. Adjusting the system prompt for your domain

---

## 🎯 Next Steps

1. **Start Small**: Test with 50-100 examples first
2. **Monitor Training**: Watch loss curves and reward trends
3. **Iterate**: Adjust hyperparameters based on initial results
4. **Scale Up**: Increase dataset size and model complexity gradually
5. **Deploy**: Use the saved model for inference on new biomedical questions

Happy fine-tuning! 🚀