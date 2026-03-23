import pandas as pd
from typing import Optional, List

def build_prompt(user_input: str, matched_rows: Optional[pd.DataFrame] = None) -> str:
    """
    Constructs a highly structured prompt for the LLM, enforcing the persona and constraints.
    The goal is to clearly separate system instructions, data, and the query to guide the LLM.
    """
    
    # 1. SYSTEM INSTRUCTION (Persona and Strict Rules)
    system_instruction = (
        "You are HealthMate, a concise, professional, and friendly health advisor. "
        "Your task is to provide personalized, actionable health advice.\n"
        "STRICTLY follow these rules:\n"
        "1. Base your advice PRIMARILY on the CONTEXTUAL DATA provided below and the user's input.\n"
        "2. Do not use conversational fillers like 'Absolutely!', 'Sure, I can help with that!', or 'I appreciate it.'\n"
        "3. OUTPUT ONLY THE ASSISTANT'S REPLY. Do not generate text for 'USER:', 'ASSISTANT:', or any follow-up dialogue.\n"
        "4. Answer DIRECTLY in 2-4 sentences only.\n"
    )

    # 2. CONTEXT DATA INTEGRATION
    context_data = ""
    if matched_rows is not None and not matched_rows.empty:
        context_data += "\n--- CONTEXTUAL DATA FROM HEALTH RECORDS ---\n"
        
        # Convert the DataFrame to a simple string representation for the LLM
        # to_string is the clearest way to present tabular data to an LLM
        context_data += matched_rows.to_string(index=False, header=True)
        
        context_data += "\n-------------------------------------------\n"
    else:
        context_data += "\n[No specific EHR or wearable data matches found for context.]\n"

    # 3. FINAL PROMPT ASSEMBLY
    final_prompt = (
        f"{system_instruction}"
        f"{context_data}\n"
        f"USER: {user_input}\n"
        f"ASSISTANT: "
    )

    return final_prompt