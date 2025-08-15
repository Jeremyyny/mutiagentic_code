call_model = None

def clean_code_for_humaneval(code_string: str) -> str:
    """
    Cleans the model's output to be pure Python code, ready for HumanEval.
    - Removes markdown fences.
    - Strips introductory text before the function definition.
    """
    # Remove markdown fences
    if "```python" in code_string:
        code_string = code_string.split("```python")[1]
    if "```" in code_string:
        code_string = code_string.split("```")[0]
    
    code_string = code_string.strip()

    # Find the start of the function definition
    lines = code_string.splitlines()
    def_line_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            def_line_index = i
            break
    
    # If a 'def' line is found, return everything from that line onwards
    if def_line_index != -1:
        return "\n".join(lines[def_line_index:])
    
    # If no 'def' is found (unlikely), return the cleaned string as is
    return code_string

def problem_understanding_agent(state, history):
    prompt = f"""You are the problem_understanding agent.

Your job is to understand the math problem below and extract the relevant components.

Problem:
{state['problem']}
"""
    extended_history = history + [{"role": "user", "content": prompt}]
    return call_model(extended_history)


def mathematical_formulation_agent(state, history):
    prompt = f"""You are the mathematical_formulation agent.

Based on the given problem, translate it into a solvable mathematical equation or formula.

Problem:
{state['problem']}
"""
    extended_history = history + [{"role": "user", "content": prompt}]
    return call_model(extended_history)


def computation_agent(state, history):
    prompt = f"""You are the computation agent.

Solve the following math problem using the formulated equation or method.
Focus on performing the necessary computations to arrive at the correct final answer.

Problem:
{state['problem']}

"""
    extended_history = history + [{"role": "user", "content": prompt}]
    return call_model(extended_history)

def answering_agent(state, history):
    context = state.get('context', None)
    if context is not None:
        prompt = f"""You are the answering agent.

Based on all the prior reasoning and context, provide a clear final answer to the question.
Answer strictly in the form of: Yes, No, or Maybe

Question:
{state['problem']}
Context:
{state['context']}

Provide your answer in one word.
"""
    else:
        prompt = f"""You are the answering agent.

Based on all the prior reasoning and context, provide the final answer to the math problem.
Answer strictly in the format : \nAnswer: <A/B/C/D/E>\n. ONLY give the final letter answer, no explanation.

Problem:
{state['problem']}

Options:
{state['options']} 

"""
    extended_history = history + [{"role": "user", "content": prompt}]
    return call_model(extended_history)
    

def knowledge_grounding_agent(state, history):
    prompt = f"""You are the knowledge_grounding agent.

Your job is to verify that the reasoning steps and retrieved context are consistent with established physical laws and principles.
Highlight any inconsistencies or confirm alignment with known physics.

Question:
{state['problem']}
Options:
{state['options']}
"""
    extended_history = history + [{"role": "user", "content": prompt}]
    # history.append({"role": "user", "content": prompt})
    return call_model(extended_history)



def option_elimination_agent(state, history):
    prompt = f"""You are the option_elimination agent.

Based on reasoning and known physics, eliminate implausible or incorrect options. Provide brief justifications.

Question:
{state['problem']}
Options:
{state['options']}
"""
    extended_history = history + [{"role": "user", "content": prompt}]
    return call_model(extended_history)



def question_understanding_agent(state, history):
    prompt = f"""You are the question_understanding agent.

Your job is to understand the question below and extract what is being asked.

Question:
{state['problem']}
"""
    extended_history = history + [{"role": "user", "content": prompt}]
    return call_model(extended_history)

def code_generation_agent(state, history):
    # Step 1: Build a strict, proven prompt for code generation.
    prompt_instruction = (
        "You are an expert Python programmer. Your sole task is to complete the following Python function.\n"
        "Return ONLY the raw, executable Python code for the function definition.\n"
        "Do NOT provide any explanations, introductory text, example usage, or markdown code fences like ```python.\n"
        "Your entire response must start directly with the 'def' line of the function.\n\n"
        "Complete this function:\n"
        f"---BEGIN FUNCTION---\n{state['problem']}\n---END FUNCTION---"
    )
    
    # Step 2: Call the model with this prompt.
    raw_completion = call_model(history + [{"role": "user", "content": prompt_instruction}])
    
    # Step 3: Clean the model's output using our robust cleaning function.
    cleaned_completion = clean_code_for_humaneval(raw_completion)
    
    return cleaned_completion

def context_analysis_agent(state, history):
    prompt = f"""You are the context_analysis agent.

Analyze the following context in relation to the question.

Context:
{state['context']}
Question:
{state['problem']}
"""
    extended_history = history + [{"role": "user", "content": prompt}]
    return call_model(extended_history)


def reasoning_agent(state, history):
    prompt = f"""You are the reasoning agent.

Using the question and context, reason out the answer in a logical way.

Context:
{state['context']}
Question:
{state['problem']}
"""
    extended_history = history + [{"role": "user", "content": prompt}]
    return call_model(extended_history)




AGENT_FUNCTIONS = {
    "knowledge_grounding": knowledge_grounding_agent,
    "option_elimination": option_elimination_agent,
    "answering": answering_agent,
    "problem_understanding": problem_understanding_agent,
    "mathematical_formulation": mathematical_formulation_agent,
    "computation": computation_agent,
    "question_understanding": question_understanding_agent,
    "context_analysis": context_analysis_agent,
    "reasoning": reasoning_agent,
    "code_generation": code_generation_agent, 
}

