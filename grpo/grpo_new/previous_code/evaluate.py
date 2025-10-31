import re

def extract_answer(last_column):

    match = re.search(r'Answer: (.*)', last_column)
    if match:
        answer = match.group(1).strip().lower()
        if answer:
            if "a" in answer:
                return "a"
            elif "b" in answer:
                return "b"
            elif "c" in answer:
                return "c"
            elif "d" in answer:
                return "d"
            elif "e" in answer:
                return "e"
 
    for opt in ["a)", "b)", "c)", "d)", "e)"]:
        if opt in last_column.lower():
            return opt[0]
    for opt in ["a )", "b )", "c )", "d )", "e )"]:
        if opt in last_column.lower():
            return opt[0]
    if last_column.lower().startswith("a"):
        return "a"
    elif last_column.lower().startswith("b"):
        return "b"
    elif last_column.lower().startswith("c"):
        return "c"
    elif last_column.lower().startswith("d"):
        return "d"
    elif last_column.lower().startswith("e"):
        return "e"
    elif last_column.lower() == "a":
        return "a"
    elif last_column.lower() == "b":
        return "b"
    elif last_column.lower() == "c":
        return "c"
    elif last_column.lower() == "d":
        return "d"
    elif last_column.lower() == "e":
        return "e"
    elif "A" in last_column.lower():
        return "a"
    elif "B" in last_column.lower():
        return "b"
    elif "C" in last_column.lower():
        return "c"
    elif "D" in last_column.lower():
        return "d"
    elif "E" in last_column.lower():
        return "e"
    elif last_column.endswith(" a."):
        return "a"
    elif last_column.endswith(" b."):
        return "b"
    elif last_column.endswith(" c."):
        return "c"
    elif last_column.endswith(" d."):
        return "d"
    elif last_column.endswith(" e."):
        return "e"

import csv
import re

def extract_last_column_values(csv_path):
    last_column_values = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            if row:
                last_column_values.append(row[-1].strip())
    return last_column_values

def extract_ground_truth(csv_path, column_name):
    gt_values = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_values.append(row.get(column_name, "").strip().lower())
    return gt_values

def extract_answer(text):
    match = re.search(r'Answer: (.*)', text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip().lower()
        if answer:
            if "a" in answer: return "a"
            if "b" in answer: return "b"
            if "c" in answer: return "c"
            if "d" in answer: return "d"
            if "e" in answer: return "e"

    for opt in ["a)", "b)", "c)", "d)", "e)"]:
        if opt in text.lower(): return opt[0]
    for opt in ["a )", "b )", "c )", "d )", "e )"]:
        if opt in text.lower(): return opt[0]

    lower_text = text.lower()
    if lower_text.startswith(("a", "b", "c", "d", "e")):
        return lower_text[0]
    if lower_text in ["a", "b", "c", "d", "e"]:
        return lower_text

    for letter in ["a", "b", "c", "d", "e"]:
        if f" {letter}." in lower_text:
            return letter

    return ""  # fallback
def extract_answer_pubmedqa(last_value):
    """For PubMedQA: extract yes/no/maybe"""
    lower = last_value.lower()
    if "yes" in lower:
        return "yes"
    elif "no" in lower:
        return "no"
    elif "maybe" in lower:
        return "maybe"
    return ""
def evaluate_accuracy(predictions, ground_truths):
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(ground_truths)} ground truths.")
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    total = len(ground_truths)
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total

def evaluate(prediction_csv_path, ground_truth_csv_path, task_type):
    label_map = {
        "mathqa": "Correct Answer",
        "gpqa": "correct_answer",
        "pubmedqa": "final decision"
    }
    if task_type not in label_map:
        raise ValueError(f"Unsupported task type '{task_type}'. Use one of: {list(label_map.keys())}")

    # Extract
    raw_predictions = extract_last_column_values(prediction_csv_path)
    if task_type == "pubmedqa":
        parsed_predictions = [extract_answer_pubmedqa(p) for p in raw_predictions]
    else:
        parsed_predictions = [extract_answer(p) for p in raw_predictions]

    #parsed_predictions = [extract_answer(p) for p in raw_predictions]
    ground_truths = extract_ground_truth(ground_truth_csv_path, label_map[task_type])

    # Evaluate
    accuracy, correct, total = evaluate_accuracy(parsed_predictions, ground_truths)
    print(f"✅ Task: {task_type}")
    print(f"Total examples: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True, help="Prediction CSV path")
    parser.add_argument('--label', required=True, help="Ground truth CSV path")
    parser.add_argument('--task', required=True, choices=["mathqa", "gpqa", "pubmedqa"])
    args = parser.parse_args()

    evaluate(args.pred, args.label, args.task)
