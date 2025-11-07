# =============================
# subagents_improved.py (7B-optimized version)
# Key improvements:
# 1. Richer prompts for 7B capability
# 2. Better history utilization
# 3. More flexible output format
# =============================

AGENT_PROMPT_FUNCTIONS = {}

def register_prompt_function(name):
    def decorator(func):
        AGENT_PROMPT_FUNCTIONS[name] = func
        return func
    return decorator


def clean_history_content(content: str) -> str:
    """Remove agent prefix from history"""
    if content.startswith("[") and "]:" in content:
        return content.split("]:", 1)[1].strip()
    return content


def format_history(history, max_rounds=3):
    """Format history for context inclusion"""
    if not history:
        return ""
    
    recent_history = history[-max_rounds:]
    formatted = "\n\nPrevious analysis:\n"
    for i, h in enumerate(recent_history, 1):
        content = clean_history_content(h.get('content', ''))
        if content:
            formatted += f"{i}. {content[:200]}...\n" if len(content) > 200 else f"{i}. {content}\n"
    return formatted


@register_prompt_function("problem_understanding")
def problem_understanding_prompt(state, history):
    problem = state.get("problem", "")
    context = state.get("context", "")
    
    prompt = f"""You are a medical question analysis expert. Your task is to carefully parse and understand the research question.

Question: {problem}

Context: {context}

Please provide a structured analysis covering:
1. **Condition/Intervention**: What treatment, procedure, or exposure is being examined?
2. **Population**: Who are the subjects (age, condition, demographics)?
3. **Endpoint/Outcome**: What health outcome or measurement is being evaluated?
4. **Question Type**: Is this asking for yes/no/maybe regarding efficacy, safety, or association?

Be thorough but concise (aim for 150-200 words). Focus on clinically relevant details that will guide subsequent reasoning."""

    return [{"role": "user", "content": prompt}]


@register_prompt_function("reasoning")
def reasoning_prompt(state, history):
    problem = state.get("problem", "")
    context = state.get("context", "")
    history_text = format_history(history, max_rounds=2)
    
    prompt = f"""You are a clinical reasoning specialist. Analyze the evidence systematically.

Question: {problem}

Context: {context}
{history_text}

Provide a structured reasoning process:

**A. Key Evidence Analysis:**
- What are the main findings from the context?
- What is the quality/strength of evidence (sample size, study design, statistical significance)?

**B. Pros and Cons:**
- Evidence supporting a positive answer
- Evidence supporting a negative answer or uncertainty
- Any limitations or confounding factors

**C. Preliminary Conclusion:**
- Based on the evidence, what stance seems most justified?
- State your preliminary answer with confidence level

Aim for 200-300 words. Be analytical and evidence-based."""

    messages = [{"role": "user", "content": prompt}]
    
    # Include last agent's output for continuity
    if history:
        last_content = clean_history_content(history[-1].get('content', ''))
        if last_content:
            messages.insert(1, {"role": "assistant", "content": last_content})
    
    return messages


@register_prompt_function("computation")
def computation_prompt(state, history):
    problem = state.get("problem", "")
    context = state.get("context", "")
    history_text = format_history(history, max_rounds=2)
    
    prompt = f"""You are a biostatistics analyst. Extract and interpret quantitative evidence.

Question: {problem}

Context: {context}
{history_text}

Identify and analyze numerical evidence:

**Quantitative Findings:**
- Sample sizes (N, n)
- Effect measures (RR, OR, HR, mean differences)
- Statistical significance (p-values, confidence intervals)
- Sensitivity, specificity, accuracy metrics
- Percentages and proportions

**Interpretation:**
- Are the findings statistically significant?
- Are the effect sizes clinically meaningful?
- Is the sample size adequate?

**Summary Statement:**
Conclude with one sentence summarizing the numerical evidence's implication for answering the question.

If no substantial numerical data exists, state "Limited quantitative evidence available" and explain why qualitative analysis is sufficient.

Aim for 150-250 words."""

    messages = [{"role": "user", "content": prompt}]
    
    if history:
        last_content = clean_history_content(history[-1].get('content', ''))
        if last_content:
            messages.insert(1, {"role": "assistant", "content": last_content})
    
    return messages


@register_prompt_function("answering")
def answering_prompt(state, history):
    problem = state.get("problem", "")
    context = state.get("context", "")
    history_text = format_history(history, max_rounds=4)  # Include full context
    
    prompt = f"""You are the final decision module for a medical QA system. Synthesize all previous analysis to provide the definitive answer.

Question: {problem}

Context: {context}
{history_text}

Based on the comprehensive analysis above, provide your final answer.

**Instructions:**
- Your answer MUST be exactly one word: 'yes', 'no', or 'maybe'
- 'yes' = strong evidence supports the claim
- 'no' = strong evidence refutes the claim  
- 'maybe' = evidence is mixed, insufficient, or inconclusive

**Response format:**
First line: ONLY the answer word (yes/no/maybe)
Second line onwards (optional): Brief 1-sentence justification if needed for clarity

Example valid responses:
"yes"
or
"maybe
The evidence shows conflicting results across different populations."

Critical: The FIRST word of your response determines your answer. Make it count."""

    messages = [
        {
            "role": "system",
            "content": "You are a medical decision support system. Provide evidence-based answers in the required format."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    return messages