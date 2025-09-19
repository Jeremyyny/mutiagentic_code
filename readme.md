# Multi-Agent QA System

A powerful command-line tool for running multi-agent reasoning pipelines over various datasets including MathQA, PubMedQA, GPQA, and HumanEval. Supports both static (hierarchical) and dynamic (supervisor) agent orchestration.

## Features

- 🤖 **Multi-Agent Architecture**: Choose between hierarchical and supervisor orchestration
- 📊 **Multiple Dataset Support**: MathQA, PubMedQA, GPQA, HumanEval
- 🔄 **Flexible Agent Combinations**: Mix and match specialized agents for different tasks
- 📝 **Comprehensive Logging**: Step-by-step output tracking and evaluation
- 🦙 **Ollama Integration**: Easy local model deployment

## ✅ Setup

### Requirements

Ensure your project directory contains:
- `main.py`
- `architecture.py` 
- `subagents.py`
- `utils.py`
- `logger.py`
- `evaluate.py`

### Backend Setup

You need a model backend running (TextGen WebUI or Ollama recommended).

## 🦙 Ollama Quick Start Guide

Ollama is the recommended way to run models locally for beginners.

### 1. Install Ollama

**macOS & Windows**: Download from [Ollama official website](https://ollama.com)

**Linux**:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Download and Run a Model

```bash
ollama run llama3
```

The first run will download the model (may take time). Once you see `>>> Send a message (/? for help)`, the model is ready.

### 3. Keep Ollama Running

⚠️ **Important**: Keep the Ollama terminal window open while using this tool. The main.py script connects to this running instance.

## 🚀 Usage Examples

### MathQA - Hierarchical Pipeline

Default agents: `problem_understanding` → `mathematical_formulation` → `computation` → `answering`

```bash
python main.py --hierarchical --csv mathqa.csv --data mathqa
```

### MathQA - Supervisor Pipeline

```bash
python main.py --supervisor --csv mathqa.csv --data mathqa \
  --structure problem_understanding,mathematical_formulation,computation,answering
```

### PubMedQA Pipeline

Available agents: `question_understanding`, `context_analysis`, `reasoning`, `answering`

```bash
python main.py --supervisor --csv pubmedqa.csv --data pubmedqa \
  --structure question_understanding,context_analysis,reasoning,answering
```

### GPQA Pipeline

Available agents: `question_understanding`, `knowledge_grounding`, `option_elimination`, `answering`

```bash
python main.py --supervisor --csv gpqa.csv --data mathqa \
  --structure question_understanding,knowledge_grounding,option_elimination,answering
```

### HumanEval (Code Generation)

```bash
python main.py --hierarchical --data human_eval \
  --jsonl_path HumanEval.jsonl \
  --log_path samples_output.jsonl
```

Then evaluate with the official script:
```bash
evaluate_functional_correctness samples_output.jsonl
```

### Quick Test with Logging

```bash
python main.py --hierarchical --csv mathqa_test_5.csv --data mathqa --log_path output.csv
```

## 🧠 Command Line Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--hierarchical` | flag | Use static agent order |
| `--supervisor` | flag | Use dynamic agent decisions |
| `--csv` | string | Path to input CSV file |
| `--jsonl_path` | string | Path to HumanEval JSONL file |
| `--data` | string | Dataset type: `mathqa`, `pubmedqa`, `human_eval`, or `auto` |
| `--structure` | string | Comma-separated list of agent names |
| `--max_steps` | int | (Supervisor only) Max decision steps (default: 5) |
| `--log_path` | string | Path to save conversation trace (CSV/JSONL) |
| `--verbose` | flag | Show detailed outputs |

## 📦 Input Format Examples

### MathQA CSV
```csv
problem,options
"If a car travels 60 miles in 1.5 hours...", "A. 30 mph B. 40 mph C. 50 mph D. 60 mph E. 70 mph"
```

### PubMedQA CSV
```csv
context,question
"The disease is caused by...", "Is it treatable?"
```

### GPQA CSV
```csv
problem,options
"Which planet is the largest?", "A. Earth B. Mars C. Jupiter D. Venus"
```

### HumanEval JSONL
```jsonl
{"task_id": "HumanEval/0", "prompt": "def add(a, b):\n    \"\"\"Add two numbers and return the sum.\"\"\"\n", "code_context": ""}
{"task_id": "HumanEval/1", "prompt": "def is_even(n):\n    \"\"\"Return True if n is even, otherwise False.\"\"\"\n", "code_context": ""}
```

## 📝 Output Files

- **CSV logging** (MathQA, PubMedQA, GPQA): Contains step-by-step agent outputs
- **JSONL output** (HumanEval): Each line contains `{"task_id": "...", "completion": "..."}`

## 🧪 Evaluation

### QA Tasks (MathQA, PubMedQA, GPQA)

```bash
python evaluate.py --pred pubmedqa_output.csv --label pubmedqa_labels.csv --task pubmedqa
```

**Supported tasks:**
- `mathqa` → uses `Correct Answer` column
- `gpqa` → uses `correct_answer` column  
- `pubmedqa` → uses `final decision` column

**Example output:**
```
✅ Task: pubmedqa
Total examples: 500
Correct answers: 421
Accuracy: 84.20%
```

### HumanEval Evaluation

#### 1. Download HumanEval Dataset

```bash
# Clone the repository
git clone https://github.com/openai/human-eval.git
cd human-eval

# Optional: Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Install requirements
pip install -e .
```

#### 2. Prepare Dataset

```bash
# Unzip dataset to your project folder
gunzip human-eval/data/HumanEval.jsonl.gz -c > HumanEval.jsonl
```

#### 3. Run Pipeline

```bash
python main.py --hierarchical \
  --data human_eval \
  --jsonl_path HumanEval.jsonl \
  --log_path samples_output.jsonl
```

#### 4. Evaluate Results

```bash
evaluate_functional_correctness samples_output.jsonl
```

**Example output:**
```
📊 Calculating pass@k...
pass@1: 0.1585
pass@10: 0.2317
pass@100: 0.3475
```

## 🔧 Important Notes

- ⚠️ **Backend must be running** before calling `main.py`
- For TextGen WebUI, provide the correct port when prompted
- For Ollama, ensure the model is downloaded and running
- Keep filenames lowercase to match imports (`logger.py`, `utils.py`)
- For HumanEval, use `["code_generation"]` as structure to avoid mixing reasoning into code
- You can add new agents in `subagents.py` and register them in `AGENT_FUNCTIONS`

## 🤝 Contributing

To add new agents:
1. Define the agent function in `subagents.py`
2. Register it in the `AGENT_FUNCTIONS` dictionary
3. Update the available agent lists in this README
