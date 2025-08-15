import argparse
import csv
import json
from architecture import Hierarchical, Supervisor
import subagents
from utils import Textgenwebui, Ollama
from logger import log_conversation_trace_supervisor, log_conversation_trace_hierarchical
from tqdm import tqdm


def select_model_backend():
    choice = input("Choose backend (1 = Textgen-webui, 2 = Ollama): ").strip()
    if choice == "1":
        port = input("Enter port for text-generation-webui (e.g. 5000): ").strip()
        return Textgenwebui(port=int(port))
    elif choice == "2":
        model = input("Enter Ollama model name (e.g. qwen2.5:7b): ").strip()
        return Ollama(model=model)
    else:
        print("❌ Invalid choice.")
        exit(1)


def run_single_pipeline(state, mode, structure, max_steps, verbose=False):
    if mode == "hierarchical":
        pipeline = Hierarchical(state, structure=structure, verbose=verbose)
        return pipeline.call_hierarchical()
    elif mode == "supervisor":
        pipeline = Supervisor(state, structure=structure, max_steps=max_steps, verbose=verbose)
        return pipeline.call_supervisor()
    else:
        raise ValueError("Invalid mode")


def run_batch(csv_path=None, mode="hierarchical", structure=None, max_steps=5,
              log_path=None, data="auto", verbose=False, jsonl_path=None):
    results = []

    # --- HumanEval dataset special handling ---
    if data == "human_eval":
        if not jsonl_path:
            raise ValueError("For HumanEval data, please provide --jsonl_path to load the dataset.")

        with open(jsonl_path, "r", encoding="utf-8") as jf:
            problems = [json.loads(line) for line in jf]

        for idx, row in enumerate(tqdm(problems, desc="Processing HumanEval tasks", unit="task", disable=verbose), start=1):
            if verbose:
                print(f"\n========== Running HumanEval task {idx}: {row['task_id']} ==========")

            state = {
                "problem": row.get("prompt", ""),
                "code_context": row.get("context", ""),
                "task_id": row.get("task_id", f"task_{idx}")
            }

            output = run_single_pipeline(state, mode, structure, max_steps, verbose=verbose)

            # Extract final output
            if isinstance(output, list):
                if isinstance(output[-1], tuple):
                    final_output = output[-1][1]
                else:
                    final_output = output[-1]
            else:
                final_output = str(output)

            result_obj = {
                "task_id": state["task_id"],
                "completion": final_output if final_output else "def placeholder_solution():\n    return None"
            }
            with open(log_path or "samples.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(result_obj, ensure_ascii=False) + "\n")

        print("\n✅ HumanEval results saved. You can now run:")
        print(f"evaluate_functional_correctness {log_path or 'samples.jsonl'}")
        return

    # --- Normal CSV-based dataset handling ---
    if not csv_path:
        raise ValueError("CSV path is required for non-HumanEval datasets.")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
        for idx, row in enumerate(tqdm(reader, desc="Processing examples", unit="example", disable=verbose), start=1):
            if verbose:
                print(f"\n========== Running example number: {idx} ==========")

            if data == "mathqa":
                state = {
                    "problem": row.get("problem", ""),
                    "options": row.get("options", "")
                }
            elif data == "pubmedqa":
                state = {
                    "context": row.get("context", ""),
                    "problem": row.get("question", "")
                }
            else:
                state = row  # fallback

            output = run_single_pipeline(state, mode, structure, max_steps, verbose=verbose)

            if mode == "supervisor" and log_path:
                log_conversation_trace_supervisor(idx, output, csv_path=log_path)
            if mode == "hierarchical" and log_path:
                log_conversation_trace_hierarchical(idx, output, structure or [
                    "problem_understanding", "mathematical_formulation", "computation", "answering"
                ], csv_path=log_path)


def main():
    parser = argparse.ArgumentParser(description="Run multi-agent pipelines.")
    parser.add_argument('--verbose', action='store_true', help='Print detailed step outputs')
    parser.add_argument('--hierarchical', action='store_true', help='Run static hierarchical agent pipeline')
    parser.add_argument('--supervisor', action='store_true', help='Run supervisor-driven agent pipeline')
    parser.add_argument('--structure', type=str, help='Comma-separated list of agents (optional)')
    parser.add_argument('--max_steps', type=int, default=5, help='Max steps for supervisor mode')
    parser.add_argument('--csv', type=str, help='Path to input CSV file (e.g. gpqa_test.csv)')
    parser.add_argument('--log_path', type=str, help='Path to save conversation traces or HumanEval JSONL results', default=None)
    parser.add_argument('--data', type=str, default="auto", choices=["auto", "mathqa", "pubmedqa", "human_eval"],
                        help='Data format: mathqa (problem+options), pubmedqa (context+question), human_eval (prompt+code_context), or auto')
    parser.add_argument('--jsonl_path', type=str, help='Path to HumanEval jsonl file')

    args = parser.parse_args()

    # Ensure mode is selected
    if not args.hierarchical and not args.supervisor:
        print("❌ Please specify either --hierarchical or --supervisor.")
        return

    # Select model backend
    backend = select_model_backend()
    subagents.call_model = backend.call_model

    # Determine structure if not provided
    structure = [s.strip() for s in args.structure.split(',')] if args.structure else None
    if structure is None:
        if args.data == "human_eval":
            structure = ["code_generation"]  # Simplified
        elif args.data == "mathqa":
            structure = ["problem_understanding", "mathematical_formulation", "computation", "answering"]
        elif args.data == "pubmedqa":
            structure = ["question_understanding", "context_analysis", "reasoning", "answering"]

    mode = "hierarchical" if args.hierarchical else "supervisor"

    # Run either batch or single example
    if args.csv or args.jsonl_path:
        run_batch(args.csv, mode, structure, args.max_steps, log_path=args.log_path,
                  data=args.data, verbose=args.verbose, jsonl_path=args.jsonl_path)
    else:
        # Example one-off input
        state = {
            "problem": "If a car travels 60 miles in 1.5 hours, what is its average speed?",
            "options": "A. 30 mph B. 40 mph C. 50 mph D. 60 mph E. 70 mph"
        }
        outputs = run_single_pipeline(state, mode, structure, args.max_steps, verbose=args.verbose)
        print("\n📋 Final Output:")
        for step in outputs:
            if isinstance(step, tuple):
                print(f"[{step[0]}]: {step[1]}")
            else:
                print(step)


if __name__ == "__main__":
    main()
