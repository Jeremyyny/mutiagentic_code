
# Multi-Agent QA System: Command-Line Manual

This tool allows you to run multi-agent reasoning pipelines over datasets such as **MathQA**, **PubMedQA**, **GPQA**, or **HumanEval** using static (hierarchical) or dynamic (supervisor) agent orchestration.

---

## ✅ Setup

Make sure your directory includes:
- `main.py`
- `architecture.py`
- `subagents.py`
- `utils.py`
- `logger.py`
- `evaluate.py`
- Your model backend running (e.g., **TextGen WebUI** or **Ollama**)

---

## 🚀 Basic Usage

### Run Hierarchical Pipeline (MathQA-style)  
You can choose any subset of subagents for hierarchical mode — just make sure the last one is `answering`.

**Default MathQA agents:**
```python
["problem_understanding", "mathematical_formulation", "computation", "answering"]
````

**Example:**

```bash
python main.py --hierarchical --csv mathqa.csv --data mathqa
```

---

### Run Supervisor Pipeline (MathQA-style)

Use as many or as few subagents as you want, but make sure `answering` is last.

```bash
python main.py --supervisor --csv mathqa.csv --data mathqa \
  --structure problem_understanding,mathematical_formulation,computation,answering
```

---

### Run Supervisor Pipeline (PubMedQA-style)

Choose from:

```python
[question_understanding, context_analysis, reasoning, answering]
```

**Example:**

```bash
python main.py --supervisor --csv pubmedqa.csv --data pubmedqa \
  --structure question_understanding,context_analysis,reasoning,answering
```

---

### Run Pipeline for GPQA

We use the same data mode as MathQA, but agent options can be:

```python
[question_understanding, knowledge_grounding, option_elimination, answering]
```

**Example:**

```bash
python main.py --supervisor --csv gpqa.csv --data mathqa \
  --structure question_understanding,knowledge_grounding,option_elimination,answering
```

---

### Run HumanEval (Code Generation)

```bash
python main.py --hierarchical --data human_eval \
  --jsonl_path HumanEval.jsonl \
  --log_path samples_output.jsonl
```

Then run the official evaluation:

```bash
evaluate_functional_correctness samples_output.jsonl
```

---

### Logging

Add `--log_path` to save step-by-step outputs.
Example for a quick test:

```bash
python main.py --hierarchical --csv mathqa_test_5.csv --data mathqa --log_path output.csv
```

You will get the outputs of the first 5 MathQA questions in `output.csv`.

---

## 🧠 Key Arguments

| Argument         | Type   | Description                                           |
| ---------------- | ------ | ----------------------------------------------------- |
| `--hierarchical` | flag   | Use static agent order                                |
| `--supervisor`   | flag   | Use dynamic agent decisions                           |
| `--csv`          | string | Path to input CSV file                                |
| `--jsonl_path`   | string | Path to HumanEval JSONL file                          |
| `--data`         | string | One of: `mathqa`, `pubmedqa`, `human_eval`, or `auto` |
| `--structure`    | string | Comma-separated list of agent names                   |
| `--max_steps`    | int    | (Supervisor only) Max decision steps (default: 5)     |
| `--log_path`     | string | Path to save conversation trace (CSV/JSONL)           |
| `--verbose`      | flag   | Show detailed outputs                                 |

---

## 📦 Example Input Formats

**MathQA CSV:**

```csv
problem,options
"If a car travels 60 miles in 1.5 hours...", "A. 30 mph B. 40 mph C. 50 mph D. 60 mph E. 70 mph"
```

**PubMedQA CSV:**

```csv
context,question
"The disease is caused by...", "Is it treatable?"
```

**GPQA CSV:**
(same as MathQA)

```csv
problem,options
"Which planet is the largest?", "A. Earth B. Mars C. Jupiter D. Venus"
```

**HumanEval JSONL:**

```json
{"task_id": "HumanEval/0", "prompt": "def add(a, b):\n    \"\"\"Add two numbers and return the sum.\"\"\"\n", "code_context": ""}
{"task_id": "HumanEval/1", "prompt": "def is_even(n):\n    \"\"\"Return True if n is even, otherwise False.\"\"\"\n", "code_context": ""}
```

* Each line is a JSON object.
* `task_id`: unique identifier for the problem.
* `prompt`: function signature and docstring.
* `code_context` (optional): partial code context to complete.

---

## 📝 Output Files

* **CSV logging** (MathQA, PubMedQA, GPQA): Contains step-by-step agent outputs.
* **JSONL output** (HumanEval): Each line contains `{"task_id": "...", "completion": "..."}`.

---

## 🧪 Evaluation

### 1. Evaluating QA Tasks (MathQA, PubMedQA, GPQA)

After generating predictions, run:

```bash
python evaluate.py --pred pubmedqa_output.csv --label pubmedqa_labels.csv --task pubmedqa
```

**Supported tasks:**

* `mathqa` → uses `Correct Answer` column
* `gpqa` → uses `correct_answer` column
* `pubmedqa` → uses `final decision` column

**Example output:**

```bash
✅ Task: pubmedqa
Total examples: 500
Correct answers: 421
Accuracy: 84.20%
```

---

### 2. Evaluating HumanEval

#### 📥 Download HumanEval Dataset

```bash
# Clone the HumanEval repository (only needed once)
git clone https://github.com/openai/human-eval.git

# Move into the folder
cd human-eval

# (Optional) Create a Python virtual environment for evaluation
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Install requirements for HumanEval
pip install -e .
```

The dataset file is located at:

```
human-eval/data/HumanEval.jsonl.gz
```

---

#### 🧪 Running HumanEval with This Project

1. **Unzip the dataset into your project folder:**

```bash
gunzip human-eval/data/HumanEval.jsonl.gz -c > HumanEval.jsonl
```

2. **Run your pipeline in `human_eval` mode:**

```bash
python main.py --hierarchical \
  --data human_eval \
  --jsonl_path HumanEval.jsonl \
  --log_path samples_output.jsonl
```

* You can replace `--hierarchical` with `--supervisor` if testing supervisor mode.
* If `--structure` is not given, defaults to `["code_generation"]` for HumanEval.

3. **Evaluate using the official script:**

```bash
evaluate_functional_correctness samples_output.jsonl
```

---

#### 📊 Example Pass\@k Output

```
Calculating pass@k...
pass@1: 0.1585
pass@10: 0.2317
pass@100: 0.3475
```

---

## 🔧 Notes

* Backend must be running before calling `main.py`.
* For **TextGen WebUI**, provide the correct port when prompted.
* For **Ollama**, ensure the model is downloaded.
* Keep filenames lowercase to match imports (`logger.py`, `utils.py`).
* For HumanEval, use `["code_generation"]` as the structure to avoid mixing reasoning into code.
* You can add new agents in `subagents.py` and register them in `AGENT_FUNCTIONS`.

---

```
```
