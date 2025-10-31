# =============================
# subagents.py (PubMedQA specialized enhanced version)
# =============================

AGENT_PROMPT_FUNCTIONS = {}

def register_prompt_function(name):
    def decorator(func):
        AGENT_PROMPT_FUNCTIONS[name] = func
        return func
    return decorator


def clean_history_content(content: str) -> str:
    if content.startswith("[") and "]:" in content:
        return content.split("]:", 1)[1].strip()
    return content


@register_prompt_function("problem_understanding")
def problem_understanding_prompt(state, history):
    problem = state.get("problem", "")
    context = state.get("context", "")
    prompt = (
        "You are a medical question parsing assistant.\n"
        "Task: Extract ONLY the essentials in 4 bullets: "
        "(1) condition/intervention, (2) population, (3) endpoint, (4) answer type (yes/no/maybe).\n"
        "Be concise (<120 words).\n\n"
        f"Question: {problem}\n\nContext: {context}\n\nBullets:"
    )
    return [{"role": "user", "content": prompt}]


@register_prompt_function("reasoning")
def reasoning_prompt(state, history):
    problem = state.get("problem", "")
    context = state.get("context", "")

    messages = []
    messages.append({
        "role": "user",
        "content": (
            "You are a clinical reasoning assistant. Analyze the question with brief steps: "
            "(a) key evidence from context, (b) pro vs con, (c) provisional stance.\n"
            "Keep it under 150 words.\n\n"
            f"Question: {problem}\n\nContext: {context}\n\nStart:"
        )
    })
    for h in history[-1:]:
        content = clean_history_content(h.get('content', ''))
        if content:
            messages.append({"role": "assistant", "content": content})
    messages.append({"role": "user", "content": "Give a 1-sentence provisional stance at the end starting with 'Stance: '"})
    return messages


@register_prompt_function("computation")
def computation_prompt(state, history):
    problem = state.get("problem", "")
    context = state.get("context", "")

    messages = []
    messages.append({
        "role": "user",
        "content": (
            "You are a data extraction assistant. List ONLY numeric facts (N, %, sensitivity/specificity, risk ratios, CIs).\n"
            "If not available, say 'No numeric evidence.'\n"
            "Finish with ONE short summary sentence starting with 'Summary: ' (<=20 words).\n\n"
            f"Question: {problem}\n\nContext: {context}\n\nNumbers:"
        )
    })
    for h in history[-1:]:
        content = clean_history_content(h.get('content', ''))
        if content:
            messages.append({"role": "assistant", "content": content})
    return messages


@register_prompt_function("answering")
def answering_prompt(state, history):
    problem = state.get("problem", "")
    context = state.get("context", "")

    recent_history = history[-2:]  # Keep only last two rounds

    messages = [
        {
            "role": "system",
            "content": (
                "You are the FINAL decision module for PubMedQA.\n"
                "You MUST output EXACTLY one of: 'yes', 'no', or 'maybe'.\n"
                "Do NOT output any explanation, punctuation, or extra words.\n"
                "Valid outputs:\n- yes\n- no\n- maybe\n"
                "Any other output is INVALID and will be scored as incorrect."
            )
        },
        {
            "role": "user",
            "content": f"Question: {problem}\n\nContext: {context}\n\nFinal one-word answer:"
        }
    ]

    for h in recent_history:
        content = clean_history_content(h.get('content', ''))
        if content:
            messages.append({"role": "assistant", "content": content})

    return messages