# main.py

import argparse
import csv
from architecture import Hierarchical, Supervisor
import subagents
from utils import Textgenwebui, Ollama
from logger import log_conversation_trace_supervisor, log_conversation_trace_hieratical
import os
from datetime import datetime
import time
import datetime



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

def run_single_pipeline(state, mode, structure, max_steps):
    if mode == "hieratical":
        pipeline = Hierarchical(state, structure=structure)
        return pipeline.call_hierarchical()
    elif mode == "supervisor":
        pipeline = Supervisor(state, structure=structure, max_steps=max_steps)
        return pipeline.call_supervisor()
    else:
        raise ValueError("Invalid mode")

def run_batch(csv_path, mode, structure, max_steps, log_path=None, data = "auto"):
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            print(f"\n========== Running example number: {idx} ==========")
            if data == "mathqa":
                state = {
                    "problem": row["Problem"],
                    "options": row["Options"]
                }
            elif data == "gpqa":
                state = {
                    "problem": row["problem"],
                    "options": row["options"]
                }
            elif data == "pubmedqa":
                state = {
                    "context": row["context"],
                    "problem": row["question"]
                }
            output = run_single_pipeline(state, mode, structure, max_steps)
            if mode == "supervisor" and log_path:
                log_conversation_trace_supervisor(idx, output, csv_path=log_path)
            if mode == "hieratical" and log_path:
                log_conversation_trace_hieratical(idx, output, structure or [
                    "problem_understanding", "mathematical_formulation", "computation", "answering"
                ], csv_path=log_path)



def main():
    parser = argparse.ArgumentParser(description="Run multi-agent pipelines.")
    parser.add_argument('--hieratical', action='store_true', help='Run static hierarchical agent pipeline')
    parser.add_argument('--supervisor', action='store_true', help='Run supervisor-driven agent pipeline')
    parser.add_argument('--structure', type=str, help='Comma-separated list of agents (optional)')
    parser.add_argument('--max_steps', type=int, default=5, help='Max steps for supervisor mode')
    parser.add_argument('--csv', type=str, help='Path to input CSV file (e.g. gpqa_test.csv)')
    parser.add_argument('--log_path', type=str, help='Path to save conversation traces (supervisor only)', default=None)
    parser.add_argument('--data', type=str, default="auto", choices=["auto", "mathqa", "pubmedqa", "gpqa"],
                    help='Data format: mathqa (problem+options), pubmedqa (context+question), or auto')


    args = parser.parse_args()

    if not args.hieratical and not args.supervisor:
        print("❌ Please specify either --hieratical or --supervisor.")
        return

    backend = select_model_backend()
    subagents.call_model = backend.call_model

    structure = [s.strip() for s in args.structure.split(',')] if args.structure else None
    mode = "hieratical" if args.hieratical else "supervisor"

    if args.csv:
        run_batch(args.csv, mode, structure, args.max_steps, log_path=args.log_path, data=args.data)

    else:
        # Example one-off input
        state = {
            "problem": "If a car travels 60 miles in 1.5 hours, what is its average speed?",
            "options": "A. 30 mph B. 40 mph C. 50 mph D. 60 mph E. 70 mph"
        }
        outputs = run_single_pipeline(state, mode, structure, args.max_steps)
        print("\n📋 Final Output:")
        for step in outputs:
            if isinstance(step, tuple):
                print(f"[{step[0]}]: {step[1]}")
            else:
                print(step)

if __name__ == "__main__":
    main()
