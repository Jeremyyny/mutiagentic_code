# =============================
# subagents_light_medical.py
# 适用于 7B bf16，多GPU
# =============================

AGENT_PROMPT_FUNCTIONS = {}


def register_prompt_function(name):
    def decorator(func):
        AGENT_PROMPT_FUNCTIONS[name] = func
        return func
    return decorator


def _clean(content):
    if content.startswith("[") and "]:" in content:
        return content.split("]:", 1)[1].strip()
    return content.strip()


def _format_history(history, max_rounds=2):
    if not history:
        return ""
    h = history[-max_rounds:]
    txt = "\n\nPrevious steps:\n"
    for i, s in enumerate(h, 1):
        c = _clean(s.get("content", ""))
        if c:
            c = c[:200] + "..." if len(c) > 200 else c
            txt += f"{i}. {c}\n"
    return txt


@register_prompt_function("problem_understanding")
def problem_understanding_prompt(state, history):
    problem = state.get("problem", "")
    context = state.get("context", "")
    return [
        {
            "role": "user",
            "content": (
                f"You are a medical analyst. Parse the research question.\n\n"
                f"Question: {problem}\nContext: {context}\n\n"
                "Explain:\n1. Condition/intervention\n2. Population\n3. Endpoint/outcome\n4. Question type\n"
                "Keep under 150 words."
            ),
        }
    ]


@register_prompt_function("reasoning")
def reasoning_prompt(state, history):
    problem = state.get("problem", "")
    context = state.get("context", "")
    hist = _format_history(history)
    return [
        {
            "role": "user",
            "content": (
                f"You are a clinical reasoning specialist.\n"
                f"Question: {problem}\nContext: {context}\n{hist}\n\n"
                "Write a concise reasoning (150–200 words) including evidence, pros/cons, and preliminary conclusion."
            ),
        }
    ]


@register_prompt_function("computation")
def computation_prompt(state, history):
    problem = state.get("problem", "")
    context = state.get("context", "")
    hist = _format_history(history)
    return [
        {
            "role": "user",
            "content": (
                f"You are a biostatistics expert.\n"
                f"Question: {problem}\nContext: {context}\n{hist}\n\n"
                "List quantitative evidence (sample size, OR/RR, p-values, etc) and summarize in 1–2 lines."
            ),
        }
    ]


@register_prompt_function("answering")
def answering_prompt(state, history):
    problem = state.get("problem", "")
    context = state.get("context", "")
    hist = _format_history(history, 3)
    return [
        {
            "role": "system",
            "content": "You are a medical decision module. Answer strictly yes/no/maybe.",
        },
        {
            "role": "user",
            "content": (
                f"Question: {problem}\nContext: {context}\n{hist}\n\n"
                "Final answer in one word (yes/no/maybe) followed by optional 1-line justification."
            ),
        },
    ]