# Multi-Agent QA System: Command-Line Manual

This tool allows you to run multi-agent reasoning pipelines over datasets such as MathQA or PubMedQA using static (hierarchical) or dynamic (supervisor) agent orchestration.

---

## ✅ Setup

Make sure your directory includes:
- `main.py`
- `architecture.py`
- `subagents.py`
- `utils.py`
- `evaluate.py`
- Your model backend (e.g. TextGen WebUI or Ollama running)

---

## 🚀 Basic Usage

### Run Hierarchical Pipeline (MathQA-style) you can also chooes subagents for hierarchical just make sure the last is answering
  ["problem_understanding", "mathematical_formulation", "computation", "answering"],
```bash
python main.py --hierarchical --csv mathqa.csv --data mathqa
```
 ### Choices to use as subagents for MATHQA
  ["problem_understanding", "mathematical_formulation", "computation", "answering"], use as many or as little as you want for mathqa, also make sure the last is answering

```bash
python main.py --supervisor --csv mathqa.csv --data mathqa --structure problem_understanding,mathematical_formulation,computation,anwering
```


### Run Supervisor Pipeline (PubMedQA-style) use as many or as little as you want for pubmedqa from the list  [question_understanding,context_analysis,reasoning,answering]

```bash
python main.py --supervisor --csv pubmedqa.csv --data pubmedqa --structure question_understanding,context_analysis,reasoning,answering
```
## Run any kind for GPQA. use as many or as little as you want for gpqa from the list [question_understanding,knowledge_grounding,option_elimination,answering] we use the same data mode as mathqa

```bash
python main.py --supervisor --csv gpqa.csv --data mathqa --structure question_understanding,knowledge_grounding,option_elimination,answering
```

## if you want to log your result, please add --log_path output.csv
if you want to test run, try this
```bash
python main.py --hierarchical --csv mathqa_tets_5.csv --data mathqa --log_path output.csv
```

then you will get the outoput of the first five mathqa decisions in output.csv

---


## 🧠 Key Arguments


| Argument         | Type   | Description                                           |
| ---------------- | ------ | ----------------------------------------------------- |
| `--hierarchical` | flag   | Use static agent order                                |
| `--supervisor`   | flag   | Use dynamic/supervisor decisions                      |
| `--csv`          | string | Path to input CSV file                                |
| `--jsonl_path`   | string | Path to HumanEval JSONL file                          |
| `--data`         | string | One of: `mathqa`, `pubmedqa`, `human_eval`, or `auto` |
| `--structure`    | string | Comma-separated agent names                           |
| `--max_steps`    | int    | (Supervisor only) Max decision steps (default: 5)     |
| `--log_path`     | string | Path to save conversation trace (CSV/JSONL)           |
| `--verbose`      | flag   | Show detailed outputs                                 |

---

## 📦 Example Inputs

### MathQA CSV Format
```csv
problem,options
"If a car travels 60 miles...", "A. 30 B. 40 C. 50 D. 60 E. 70"
```

### PubMedQA CSV Format
```csv
context,question,options
"The disease is caused by...", "Is it treatable?", "A. Yes B. No C. Maybe"
```


### GPQA CSV Format
```csv
problem,options
"Which planet is the largest?", "A. Earth B. Mars C. Jupiter D. Venus"
```

---

## 📝 Output Files

- `log_path` logs full transcript of each agent call per step.

---

## 🧪 Advanced

- You can pass any subset of agents to `--structure` for custom flows.
- Add your own agent to `subagents.py` and register it in `AGENT_FUNCTIONS`.

---

## 🔧 Notes

- Backend must be running before calling `main.py`
- For TextGenWebUI, provide the correct port during CLI prompt
- For Ollama, ensure your model is downloaded

---
## NOW that you have produced a file output, you need to evaluate and print out the accuracy

### 🧪 Evaluation

After generating an output file, you can calculate its accuracy.
## 1. Evaluating QA Tasks (MathQA, PubMedQA, GPQA)


```bash
python evaluate.py --pred pubmedqa_output.csv --label pubmedqa.csv --task pubmedqa
```

Supported task types:
- `mathqa`: uses the `Correct Answer` column
- `gpqa`: uses the `correct_answer` column
- `pubmedqa`: uses the `final decision` column

---

### Output

```bash
✅ Task: pubmedqa
Total examples: 500
Correct answers: 421
Accuracy: 84.20%
```

The evaluator uses a robust parsing function to extract and normalize answers from model output, ensuring accurate scoring even with format noise.

---

## 2. Evaluating HumanEval

Use the official human-eval library script. Do not use evaluate.py.

```bash
evaluate_functional_correctness samples_output.jsonl
```

Example Output (Pass@k scores):

```
Calculating pass@k...
pass@1: 0.1585
pass@10: 0.2317
pass@100: 0.3475
```