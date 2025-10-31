# Multi-Agent GRPO – PubMedQA Fine-Tuning Quick Start Guide

A hierarchical multi-agent reinforcement-learning pipeline that fine-tunes a lightweight language model (Qwen 0.5 B) for biomedical yes/no/maybe question answering on **PubMedQA**, using **Group Relative Policy Optimization (GRPO)**.

---

## 🎯 What This Code Does

- 🧩 Implements a **Manager–Specialist architecture**  
  Manager decides which sub-agent to call → Specialists (`problem_understanding`, `reasoning`, `computation`, `answering`) produce text.

- 🧠 **GRPO training**: Manager’s policy/value heads are optimized from reward signals (correctness + efficiency + quality).

- ⚕️ **Biomedical focus**: Agents reason over PubMed abstracts to output *yes/no/maybe.*

- 📊 **Evaluation utilities**: Compare Baseline (random Manager), Trained (GRPO-fine-tuned), and Fixed Pipeline (deterministic 4-step).

---

## ✅ Repository Structure & File Roles

| File | Role |
|------|------|
| `main_grpo.py` | Entry script to start GRPO training (loads dataset → initialize Manager + Specialists → train → save). |
| `manager_agent.py` | Defines `ManagerAgent` (policy/value heads + decision logic) and `FixedSpecialistAgent` (wrapper for LLM generation). |
| `subagents.py` | Prompt templates for each Specialist (PubMedQA-specific instructions + short output constraints). |
| `reward.py` | Reward computation (correctness + efficiency + format quality dense shaping). |
| `grpo_trainer.py` | Implements GRPO update loop (policy loss + value loss + entropy bonus). |
| `utils.py` | Helper class `LocalHF` for local HF model loading/generation (`temperature`, `max_tokens`, device setup). |
| `evaluate_manager_inference_final.py` | Evaluation script → Baseline vs Trained accuracy, saves JSONL logs. |
| `evaluate_manager_inference_final_with_fixed.py` | Adds Fixed Pipeline baseline for three-way comparison. |
| `test_indices_grpo.json` | Validation/test subset indices used after training. |

---

## ⚙️ System Requirements

| Resource | Recommended |
|-----------|--------------|
| GPU | RTX 3090/4090 (≥24 GB VRAM) |
| CPU RAM | ≥ 32 GB |
| Disk | ≥ 10 GB free |
| Python | 3.10 + PyTorch ≥ 2.1 |

### Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets numpy tqdm
```

---

## 📦 Dataset Preparation

Dataset (JSON) format:

```json
[
  {
    "question": "Does aspirin reduce stroke risk?",
    "context": "Randomized trials suggest low-dose aspirin lowers ischemic events...",
    "ground_truth": "yes"
  },
  {
    "question": "Is vitamin D useful for fractures?",
    "context": "Meta-analysis shows limited efficacy of vitamin D supplementation...",
    "ground_truth": "no"
  }
]
```

Save as:  
`data/golden_dataset_pubmedqa_qwen2.5_pro_test_500.json`

---

## 🚀 Training Workflow

### 1️⃣ Run Training
```bash
python main_grpo.py
```

It will:
1. Load dataset & Qwen 0.5 B backbone  
2. Initialize Manager + Specialists  
3. Run GRPO optimization (Epoch × Steps)  
4. Save fine-tuned Manager → `./trained_manager_grpo/manager/`

**Outputs**
```
Epoch 3 Summary:
  Avg Reward: 0.87
  Avg Loss: 1.96
✓ Manager saved to ./trained_manager_grpo
✓ Validation indices saved to test_indices_grpo.json
```

---

## 🧪 Evaluation Workflow

### 2️⃣ Run Validation (2-way comparison)
```bash
python evaluate_manager_inference_final.py
```
Produces:
```
Baseline Manager (Qwen 0.5B): 27.00%
Trained Manager (GRPO): 42.00%
Accuracy Gain: +15.00%
Logs → ./eval_logs/
```

### 3️⃣ Run 3-way Evaluation (including Fixed Pipeline)
```bash
python evaluate_manager_inference_final_with_fixed.py
```
Example output:
```
Baseline (random): 27%
Trained (GRPO): 42%
Fixed (sequential): 34%
```

---

## 🔧 Key Hyperparameters

| Parameter | Location | Recommended Value | Notes |
|------------|-----------|-------------------|-------|
| `temperature` | `utils.LocalHF` | 0.7–0.8 (train), 0.3–0.5 (test) | High T → exploration ; Low T → deterministic evaluation |
| `max_steps` | `main_grpo.py` / eval scripts | 5 | Max Manager decision depth |
| `learning_rate` | `grpo_trainer.py` | 5e-6–3e-6 | Lower for stability |
| `ent_coef` | `grpo_trainer.py` | 0.01–0.02 | Higher → more exploration |
| `freeze_manager_backbone` | `manager_agent.py` | True | Only train policy/value heads |

---

## 🔬 Temperature Notes (Important)

- **Training T ≈ 0.8:** Encourages exploration → better reward learning  
- **Inference T ≈ 0.3–0.5:** Measures average ability under mild stochasticity  
- **T = 0.1** may cause *policy collapse* (distribution shift / entropy collapse)

---

## 📊 Expected Performance

| Setting | Accuracy (100 val samples) | Behavior |
|----------|----------------------------|-----------|
| Baseline Manager (random) | 25–30 % | Random routing |
| Fixed Pipeline (sequential) | 33–40 % | Hard-coded 4-step flow |
| Trained Manager (GRPO) | 40–45 % (typical) | Learned routing & structured answers |

---

## 🧩 File Outputs

| File / Folder | Content |
|----------------|----------|
| `trained_manager_grpo/manager/heads.pt` | Saved policy / value head weights |
| `test_indices_grpo.json` | Validation set indices |
| `eval_logs/baseline_trajectories.jsonl` | Per-sample baseline trajectories |
| `eval_logs/trained_trajectories.jsonl` | Per-sample trained trajectories |
| `eval_logs/fixed_pipeline_trajectories.jsonl` | Fixed-order runs |
| `summary_*.txt` | Accuracy summaries |

---

## 💡 Tips & Best Practices

- Start with 100 samples to verify pipeline; then scale to 500–1000.  
- Monitor reward → should rise steadily (0 → 1 +).  
- Use `temperature annealing` (high → low across epochs) to balance exploration/exploitation.  
- Use `--deterministic` flag (or set seeds) for reproducible runs.  
- Use stochastic mode to measure *realistic capability under sampling*.  

---

## ⚠️ Troubleshooting

| Issue | Likely Cause | Fix |
|--------|--------------|-----|
| `RuntimeError: CUDA out of memory` | Too many tokens / large batch | Lower `max_tokens` or `batch_size` |
| Reward ≈ 0 throughout | Answer format invalid (“maybe yes because…”) | Check `answering_prompt` → force one-word output |
| Accuracy drops when T ↓ to 0.1 | Distribution shift / entropy collapse | Evaluate with T ≈ 0.4–0.8 |
| Trained slower than Fixed | Manager forward adds overhead | Normal; Manager does extra reasoning |
| Baseline too high | Random seed fixed → deterministic behavior | Enable `do_sample=True` for stochastic evaluation |

---

## 📈 Next Steps

1. **Refine Reward Design** (add dense signals for sub-agents).  
2. **Expand Dataset** → 500 + examples for robust training.  
3. **Add Temperature Sweep Eval** → plot accuracy vs T curve.  
4. **Scale Model** → Qwen 1.5 B or 3 B for higher ceiling.  
5. **Experiment with Entropy Regularization & λ-baseline** for stability.

---

> 🧭 **Summary:**  
> - `main_grpo.py` → train Manager heads via GRPO  
> - `evaluate_manager_inference_final_with_fixed.py` → compare Baseline / Trained / Fixed  
> - Keep training T≈0.8, test T≈0.4 for true performance  
> - Expect ~15 pp accuracy gain on PubMedQA with Qwen 0.5 B  

Happy experimenting with your multi-agent GRPO pipeline 🚀
