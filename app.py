# ============================================================
# CELL 1 - INSTALL REQUIRED PACKAGE
# ============================================================

!pip install -qU langchain-google-genai langgraph


# ============================================================
# CELL 2 - IMPORTS
# ============================================================

import sys
import io
import traceback

from typing import TypedDict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END

from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# 1. LLM INITIALIZATION
# ============================================================

import google.generativeai as genai
from google.colab import userdata


# Retrieve API key from Google Colab Secrets
try:

    api_key = userdata.get("GEMINI_API_KEY")

    genai.configure(api_key=api_key)

    print("API Key configured successfully.")

except userdata.SecretNotFoundError:

    print("ERROR: GEMINI_API_KEY not found in Colab Secrets.")

    api_key = None


# Initialize Gemini
if api_key:

    llm_flash = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        google_api_key=api_key
    )

    llm = llm_flash

else:

    llm_flash = None
    llm = None


# ============================================================
# 2. STATE DEFINITION
# ============================================================

class CrewState(TypedDict):

    messages: List[BaseMessage]

    # Current location of workflow
    current_step: Optional[str]

    # Next location of workflow
    next_step: Optional[str]

    # Generated Python code
    code: Optional[str]

    # Testing report
    report: Optional[str]


# ============================================================
# 3. HELPER FUNCTION
# ============================================================

def get_text_from_response(response):

    """
    Safely extracts text from Gemini/LangChain response.
    """

    content = response.content

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):

                text_parts.append(
                    item.get("text", "")
                )

            else:

                text_parts.append(
                    str(item)
                )

        return "\n".join(text_parts)

    return str(content)


# ============================================================
# 4. DISPLAY WORKFLOW STATUS
# ============================================================

def display_workflow_status(current_step, next_step):

    steps = [
        "task_input",
        "developer",
        "tester",
        "manager_decision",
        "archiver"
    ]

    names = {
        "task_input": "TASK INPUT",
        "developer": "DEVELOPER",
        "tester": "TESTER",
        "manager_decision": "MANAGER",
        "archiver": "ARCHIVER"
    }

    print("\n")
    print("=" * 60)
    print("              REAL-TIME AI CODING WORKFLOW")
    print("=" * 60)

    for step in steps:

        if step == current_step:

            symbol = "[▶]"

        elif step == next_step:

            symbol = "[→]"

        else:

            symbol = "[ ]"

        print(f"{symbol} {names[step]}")

    print("-" * 60)

    print(
        f"Current Step : "
        f"{names.get(current_step, current_step)}"
    )

    print(
        f"Next Step    : "
        f"{names.get(next_step, next_step)}"
    )

    print("=" * 60)


# ============================================================
# 5. TOOLS
# ============================================================


# ------------------------------------------------------------
# TOOL 1 - RUN PYTHON CODE
# ------------------------------------------------------------

@tool
def run_python_code(code: str) -> str:

    """
    Execute Python code and return standard output
    or an error trace.
    """

    if not isinstance(code, str):

        code = str(code)

    # Remove markdown code fences
    clean_code = (
        code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout

    new_stdout = io.StringIO()

    sys.stdout = new_stdout

    try:

        local_scope = {}

        exec(
            clean_code,
            {},
            local_scope
        )

        result = new_stdout.getvalue()

    except Exception:

        result = (
            "Execution Error:\n"
            + traceback.format_exc()
        )

    finally:

        sys.stdout = old_stdout

    if result.strip():

        return result.strip()

    return "Success (no terminal output)"


# ------------------------------------------------------------
# TOOL 2 - GENERATE TEST CASES
# ------------------------------------------------------------

@tool
def generate_test_cases(task_description: str) -> str:

    """
    Generate 3 to 5 specific test scenarios
    for the given coding task.
    """

    if llm is None:

        return "LLM is not initialized."

    prompt = f"""

You are a Senior QA Engineer.

Generate 3 to 5 highly specific test scenarios
for the following Python coding task:

"{task_description}"

Include:

1. Standard test cases
2. Edge cases
3. Boundary cases where appropriate

Return the result as a numbered list.

"""

    response = llm.invoke(prompt)

    return get_text_from_response(response)


# ============================================================
# 6. GRAPH NODES
# ============================================================


# ------------------------------------------------------------
# NODE 1 - TASK INPUT
# ------------------------------------------------------------

def task_input_node(state: CrewState):

    display_workflow_status(
        "task_input",
        "developer"
    )

    print("\n--- NEW TASK INITIALIZATION ---")

    user_task = input(
        "\nEnter the coding task "
        "(or type 'exit' to quit): "
    ).strip()

    # User wants to exit
    if user_task.lower() == "exit":

        return {

            "current_step": "task_input",

            "next_step": "exit"

        }

    # New task
    return {

        "messages": [
            HumanMessage(
                content=user_task
            )
        ],

        "current_step": "task_input",

        "next_step": "developer"

    }


# ------------------------------------------------------------
# NODE 2 - DEVELOPER
# ------------------------------------------------------------

def real_time_developer(state: CrewState):

    display_workflow_status(
        "developer",
        "tester"
    )

    print(
        "\n[Developer] "
        "Writing dynamic code using LLM..."
    )

    # Get latest task
    task = state["messages"][-1].content

    # Developer prompt
    dev_prompt = f"""

Write a clean Python script to solve this coding task:

{task}

Requirements:

- Return ONLY Python code.
- Do NOT include explanation.
- Do NOT include markdown.
- Do NOT include ```python.
- Make the code executable.
- Use simple and clean Python.

"""

    # Check LLM
    if llm_flash is None:

        raise ValueError(
            "LLM is not initialized. "
            "Check GEMINI_API_KEY."
        )

    # Ask Gemini
    response = llm_flash.invoke(
        dev_prompt
    )

    # Extract response
    code_str = get_text_from_response(
        response
    )

    # Clean markdown if Gemini accidentally adds it
    code_str = (
        code_str
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    print("\n" + "-" * 60)

    print("GENERATED PYTHON CODE")

    print("-" * 60)

    print(code_str)

    print("-" * 60)

    # Return updated state
    return {

        "code": code_str,

        "current_step": "developer",

        "next_step": "tester"

    }


# ------------------------------------------------------------
# NODE 3 - TESTER
# ------------------------------------------------------------

def real_time_tester(state: CrewState):

    display_workflow_status(
        "tester",
        "manager_decision"
    )

    print(
        "\n[Tester] "
        "Generating dynamic tests and executing code..."
    )

    # Get task
    task = state["messages"][-1].content

    # --------------------------------------------------------
    # Generate test cases
    # --------------------------------------------------------

    print(
        "\n[Tester] Generating test scenarios..."
    )

    test_cases = generate_test_cases.invoke(
        task
    )

    cases_str = str(test_cases)

    # --------------------------------------------------------
    # Execute developer code
    # --------------------------------------------------------

    print(
        "\n[Tester] Executing generated code..."
    )

    execution_result = run_python_code.invoke(
        {
            "code": state["code"]
        }
    )

    # --------------------------------------------------------
    # Generate report
    # --------------------------------------------------------

    report = f"""

### EXECUTION OUTPUT

{execution_result}


### TEST SCENARIOS EVALUATED

{cases_str}

"""

    print("\n" + "=" * 60)

    print("TEST REPORT")

    print("=" * 60)

    print(report)

    print("=" * 60)

    # Return updated state
    return {

        "report": report,

        "current_step": "tester",

        "next_step": "manager_decision"

    }


# ------------------------------------------------------------
# NODE 4 - MANAGER
# ------------------------------------------------------------

def manager_decision_node(state: CrewState):

    display_workflow_status(
        "manager_decision",
        None
    )

    print(
        "\n--- MANAGER DASHBOARD : TEST REPORT ---"
    )

    print(
        state.get(
            "report",
            "No report available."
        )
    )

    print("=" * 60)

    print(
        "\nManager Options:"
    )

    print(
        "1. store   -> Store the task and exit"
    )

    print(
        "2. another -> Create another coding task"
    )

    user_input = input(
        "\nCommand (store / another): "
    ).lower().strip()

    # --------------------------------------------------------
    # Store task
    # --------------------------------------------------------

    if user_input == "store":

        return {

            "current_step": "manager_decision",

            "next_step": "archiver"

        }

    # --------------------------------------------------------
    # New task
    # --------------------------------------------------------

    else:

        return {

            "current_step": "manager_decision",

            "next_step": "task_input"

        }


# ------------------------------------------------------------
# NODE 5 - ARCHIVER
# ------------------------------------------------------------

def archiver_node(state: CrewState):

    display_workflow_status(
        "archiver",
        "exit"
    )

    print(
        "\n[Archiver] "
        "Task stored successfully."
    )

    print(
        "[Archiver] "
        "Closing workflow..."
    )

    return {

        "current_step": "archiver",

        "next_step": "exit"

    }


# ============================================================
# 7. GRAPH CONSTRUCTION
# ============================================================

rt_workflow = StateGraph(
    CrewState
)


# ------------------------------------------------------------
# Add nodes
# ------------------------------------------------------------

rt_workflow.add_node(
    "task_input",
    task_input_node
)

rt_workflow.add_node(
    "developer",
    real_time_developer
)

rt_workflow.add_node(
    "tester",
    real_time_tester
)

rt_workflow.add_node(
    "manager_decision",
    manager_decision_node
)

rt_workflow.add_node(
    "archiver",
    archiver_node
)


# ============================================================
# 8. START → TASK INPUT
# ============================================================

rt_workflow.add_edge(
    START,
    "task_input"
)


# ============================================================
# 9. ROUTING AFTER TASK INPUT
# ============================================================

def route_from_input(state: CrewState):

    next_step = state.get(
        "next_step"
    )

    if next_step == "exit":

        return END

    return "developer"


rt_workflow.add_conditional_edges(
    "task_input",
    route_from_input
)


# ============================================================
# 10. DEVELOPER → TESTER
# ============================================================

rt_workflow.add_edge(
    "developer",
    "tester"
)


# ============================================================
# 11. TESTER → MANAGER
# ============================================================

rt_workflow.add_edge(
    "tester",
    "manager_decision"
)


# ============================================================
# 12. ROUTING AFTER MANAGER
# ============================================================

def route_from_decision(state: CrewState):

    next_step = state.get(
        "next_step"
    )

    if next_step == "archiver":

        return "archiver"

    return "task_input"


rt_workflow.add_conditional_edges(
    "manager_decision",
    route_from_decision
)


# ============================================================
# 13. ARCHIVER → END
# ============================================================

rt_workflow.add_edge(
    "archiver",
    END
)


# ============================================================
# 14. COMPILE GRAPH
# ============================================================

rt_app = rt_workflow.compile()


print("\n")
print("=" * 60)
print(
    "Interactive pipeline compiled "
    "and ready for live execution."
)
print("=" * 60)


# ============================================================
# 15. EXECUTION LOOP
# ============================================================

if __name__ == "__main__":

    try:

        # Initial state
        initial_state = {

            "messages": [],

            "current_step": "task_input",

            "next_step": "developer",

            "code": None,

            "report": None

        }

        # Start workflow
        rt_app.invoke(
            initial_state,
            config={
                "recursion_limit": 50
            }
        )

    except KeyboardInterrupt:

        print(
            "\n\nWorkflow stopped by user."
        )

    except Exception as e:

        print(
            "\nAn error occurred:"
        )

        print(e)

        traceback.print_exc()