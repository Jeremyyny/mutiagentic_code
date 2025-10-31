import subagents
import re

import utils

class Hierarchical:
    def __init__(self, state, structure=None):
        if structure is None:
            structure = ["problem_understanding", "mathematical_formulation", "computation", "answering"]
        self.state = state
        self.structure = structure
        self.history = []
        self.transcript = []

    def build_hierarchical(self):
        row = []
        for agent_name in self.structure:
            agent_fn = subagents.AGENT_FUNCTIONS.get(agent_name)
            if not agent_fn:
                print(f"[Warning] Agent '{agent_name}' not found.")
                continue
            print(f"\n--- {agent_name} ---")
            output = agent_fn(self.state, self.history)
            print(f"{agent_name}: {output}")
            self.history.append({"role": "assistant", "content": f"[{agent_name}]: {output}"})
            self.transcript.append((agent_name, output))
            row.append(output)
        return row

    def call_hierarchical(self):
        return self.build_hierarchical()

class Supervisor:
    def __init__(self, state, structure=None, max_steps=5):
        """
        Supervisor dynamically decides the next agent to call.
        `structure` restricts allowable agents to choose from.
        """
        if structure is None:
            structure = ["problem_understanding", "mathematical_formulation", "computation", "answering"]
        self.state = state
        self.structure = structure
        self.max_steps = max_steps
        self.history = []  # Store agent output history, passed to the next agent
        self.supervisor_history = []  # Store supervisor decision history
        self.transcript = []

    def call_supervisor(self):
        """
        Run a dynamic decision loop where the supervisor picks agents iteratively.
        """
        chosen_agent = None
        for i in range(1, self.max_steps + 1):
            print(f"\n--- Supervisor {i} ---")

            # Step 1: Ask which agent to call
            allowed_agents = "\n- " + "\n- ".join(self.structure)
            problem = self.state["problem"]
            
            # Build supervisor prompt including previous agent outputs
            context_info = ""
            if self.history:
                context_info = f"\n\nPrevious agent outputs:\n"
                for msg in self.history:
                    context_info += f"{msg['content']}\n"
            
            prompt = f"""Given the problem: {problem}{context_info}

Please choose ONE next agent to call from the following:{allowed_agents}

Reply STRICTLY in the form:
Agent: <name>
Then explain why."""

            # Create a separate history for the supervisor
            supervisor_prompt = self.supervisor_history.copy()
            supervisor_prompt.append({"role": "user", "content": prompt})
            
            supervisor_msg = subagents.call_model(supervisor_prompt)
            print(supervisor_msg)
            
            # Update supervisor history
            self.supervisor_history.append({"role": "user", "content": prompt})
            self.supervisor_history.append({"role": "assistant", "content": supervisor_msg})
            
            # Step 2: Extract agent name
            match = re.search(r"Agent:\s*(\w+)", supervisor_msg, re.IGNORECASE)
            chosen_agent = None
            if match:
                chosen_agent = match.group(1).lower()
            else:
                for name in self.structure:
                    if name in supervisor_msg.lower():
                        chosen_agent = name
                        break

            if chosen_agent not in subagents.AGENT_FUNCTIONS or chosen_agent not in self.structure:
                print(f"❌ Invalid or unauthorized agent '{chosen_agent}' specified. Defaulting to 'answering'.")
                chosen_agent = "answering"

            print(f"✅ Agent chosen: {chosen_agent}")
            
            # Step 3: Call the chosen agent with the cleaned agent history
            result = subagents.AGENT_FUNCTIONS[chosen_agent](self.state, self.history)
            self.transcript.append((chosen_agent, result))
            print(f"[{chosen_agent} → result]: {result}")

            # Step 4: Update agent history for the next agent to use
            self.history.append({"role": "assistant", "content": f"[{chosen_agent}]: {result}"})

            if chosen_agent == "answering":
                print("[Finished] Final answer reached.")
                return self.transcript

        # After max steps, force the final answering agent
        if chosen_agent != "answering":
            print("⚠️ Max steps reached. Forcing final 'answering' agent call.")
            result = subagents.AGENT_FUNCTIONS["answering"](self.state, self.history)
            self.transcript.append(("forced_answering", result))
            print(f"[answering → result]: {result}")

        return self.transcript
