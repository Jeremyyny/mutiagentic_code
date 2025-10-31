
import os
import csv
from datetime import datetime
def log_conversation_trace_supervisor(count, supervisor_call_pairs, csv_path="conversation_trace_qwen_mathqa.csv"):
    """
    supervisor_call_pairs: list of tuples (supervisor_response, agent_call_response)
    """
    row = {"timestamp": datetime.now().isoformat(), "count": count}
    fieldnames = ["timestamp", "count"]
    
    for i, (supervisor, call) in enumerate(supervisor_call_pairs, 1):
        fieldnames.extend([f"supervisor_{i}", f"call_{i}"])
        row[f"supervisor_{i}"] = supervisor
        row[f"call_{i}"] = call

    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_conversation_trace_hieratical(count, hieratical_row, agent_names, csv_path="conversation_trace_qwen_mathqa.csv"):
    """
    Logs the output of a hierarchical pipeline.

    Parameters:
    - count: Index of the current example
    - hieratical_row: List of agent outputs in the same order as agent_names
    - agent_names: List of subagent names (e.g. ["problem_understanding", "mathematical_formulation", ...])
    - csv_path: Output CSV path
    """
    assert len(agent_names) == len(hieratical_row), "Mismatch between agents and outputs"

    row = {
        "timestamp": datetime.now().isoformat(),
        "count": count
    }

    for agent, output in zip(agent_names, hieratical_row):
        row[agent] = output

    # Construct fieldnames
    fieldnames = ["timestamp", "count"] + agent_names

    # Write or append to CSV
    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)