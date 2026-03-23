import os
import streamlit as st
import requests
import json
import time 
from utils.prompt_builder import build_prompt
from utils.data_lookup import search_relevant_facts
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from dotenv import load_dotenv

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except Exception:
    Llama = Any  # type: ignore
    LLAMA_CPP_AVAILABLE = False

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"


# --- MISTRAL GPU MODEL CONFIGURATION ---
MISTRAL_MODEL_PATH = "model/mistral-7b-instruct-v0.2.Q5_K_M.gguf" 
LLM_KWARGS = {'n_ctx': 4096, 'n_threads': 4, 'n_gpu_layers': 999, 'verbose': True}
# --- END CONFIGURATION ---

# --- UTILITY FUNCTION DEFINITIONS ---

def clean_input(text: str) -> str:
    """Simple text cleaning to ensure continuity."""
    if not isinstance(text, str):
        return ""
    return text.lower().strip()

# --- DUAL MODEL INITIALIZATION (FIXED) ---
@st.cache_resource
def load_mistral_model_gpu() -> Optional[Any]:
    """
    Loads Mistral onto GPU using llama_cpp.
    Correctly checks for file existence before attempting to load.
    """
    if not LLAMA_CPP_AVAILABLE:
        st.error("Mistral backend unavailable: 'llama_cpp' is not installed in this environment.")
        return None

    # 1. Check if the file path exists.
    if not os.path.exists(MISTRAL_MODEL_PATH):
        st.error(f"Mistral Model Missing! File expected at '{MISTRAL_MODEL_PATH}' was not found.")
        # If the file is missing, we MUST return None and halt loading.
        return None 
        
    # 2. Attempt to load the model (File is confirmed to exist)
    try:
        print(f"--- ATTEMPTING TO LOAD MISTRAL GGUF ON RTX 3050 ---")
        # Load the Mistral GGUF model, offloading all layers to the RTX 3050 (6GB)
        return Llama(model_path=MISTRAL_MODEL_PATH, **LLM_KWARGS) 
        
    except Exception as e:
        # If loading fails (due to GPU link/CUDA error), catch the exception and return None.
        st.error(f"Mistral GPU Loading FAILED. Check console for CUDA/cuBLAS errors. Error: {e}")
        return None
# --- END DUAL MODEL INITIALIZATION ---


# --- GEMINI API FUNCTION ---
def generate_gemini_response(prompt: str) -> Tuple[str, int, float]:
    """Generates response from Gemini 1.5 Flash model."""
    system_instruction = (
        "You are HealthMate, a concise, professional, and friendly health advisor. "
        "Answer based on health data in 2–4 sentences. Avoid filler or conversational tone."
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": system_instruction + "\n\n" + prompt}]
            }
        ]
    }

    start_time = time.time()
    try:
        response = requests.post(
            GEMINI_API_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        result = response.json()

        text = (
            result.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )

        end_time = time.time()
        latency = end_time - start_time
        token_count = len(text.split())

        return text, token_count, latency

    except Exception as e:
        return f"⚠️ Gemini API error: {str(e)}", 0, 0


# --- STREAMLIT APP LOGIC & UI ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "metrics" not in st.session_state:
    st.session_state.metrics = []
if 'current_model' not in st.session_state:
    st.session_state.current_model = 'GEMINI (API)'

st.set_page_config(page_title="HealthMate - Comparison Testbed", layout="centered")
st.markdown("## 🧠 HealthMate - Model Comparison Testbed")
st.markdown("---")

# --- MODEL AND DATA SELECTION UI ---
col1, col2 = st.columns(2)

with col1:
    model_options = ['GEMINI (API)']
    if LLAMA_CPP_AVAILABLE and os.path.exists(MISTRAL_MODEL_PATH):
        model_options.append('MISTRAL (GPU)')

    model_selection = st.radio(
        "Select Model for Testing:",
        model_options,
        key='model_selector'
    )
    st.session_state.current_model = model_selection

if 'MISTRAL (GPU)' not in model_options:
    st.info("Mistral mode is hidden because local model/runtime is unavailable in this environment.")
    
with col2:
    data_type = st.radio(
        "Select Data Type:",
        ('EHR', 'WEARABLE'),
        key='data_type_selector'
    )

if data_type == 'EHR':
    dataset_options = ('PIMA', 'MIMIC')
else:
    dataset_options = ('WEARABLE_PERF', 'WEARABLE_SPORTS') 

st.session_state.current_dataset = st.radio(
    f"Select {data_type} Dataset (D1 or D2):",
    dataset_options,
    key='dataset_detail_selector',
    horizontal=True
)

st.caption(f"Active Scenario: **{st.session_state.current_model}** on **{st.session_state.current_dataset}** Data")
st.markdown("---")
# --- END MODEL AND DATA SELECTION UI ---


# Technical Metrics Display
with st.sidebar:
    st.header("⚙️ Performance Metrics")
    if st.session_state.metrics:
        latencies = [m['latency'] for m in st.session_state.metrics]
        tps_values = [m['tps'] for m in st.session_state.metrics]
        
        avg_latency = sum(latencies) / len(latencies)
        avg_tps = sum(tps_values) / len(tps_values)
        
        st.markdown(f"**Total Interactions:** {len(st.session_state.metrics)}")
        st.metric(label="Avg Latency", value=f"{avg_latency:.2f} s")
        st.metric(label="Avg Tokens/s", value=f"{avg_tps:.2f} t/s")
    else:
        st.info("Ask a query to start calculating metrics.")
        
    st.header("ℹ️ Tips")
    st.info("Remember to re-run your 10 standard queries for each of the 8 scenarios.")


# User Input and Chat Handling
user_input = st.chat_input("Talk to your health assistant...")
if user_input:
    st.chat_message("user").markdown(user_input)
    
    # 1. Clean Input & Greetings Check
    cleaned_input = clean_input(user_input)
    greetings = ["hi", "hello", "hey"]
    is_simple_greeting = any(word == cleaned_input for word in greetings)
    
    if is_simple_greeting:
        bot_reply = "👋 Hi there! I'm HealthMate. How can I support your wellness journey today?"
        
    else:
        # 2. Search for relevant facts (Pass the specific dataset name)
        try:
            matched_rows = search_relevant_facts(user_input, dataset_type=st.session_state.current_dataset) 
        except Exception as e:
            matched_rows = pd.DataFrame()
            print(f"Error during data lookup: {e}") 

        # 3. Build the prompt
        prompt = build_prompt(user_input, matched_rows)
        
        # 4. EXECUTE MODEL BASED ON SELECTION
        try:
            if st.session_state.current_model == 'GEMINI (API)':
                bot_reply, generated_tokens, inference_latency = generate_gemini_response(prompt)
                
            elif st.session_state.current_model == 'MISTRAL (GPU)':
                mistral_llm_instance = load_mistral_model_gpu()
                
                if mistral_llm_instance is None:
                    bot_reply = "⚠️ MISTRAL ERROR: Local GPU model failed to initialize. Check console."
                    generated_tokens = 0
                    inference_latency = 0
                    
                else:
                    start_time = time.time()
                    response = mistral_llm_instance(prompt, max_tokens=350, stop=["\nUSER:", "USER:", "</s>"], echo=False) 
                    end_time = time.time()
                    
                    # Metric Extraction (Local Model)
                    choice = response["choices"][0]
                    bot_reply = choice.get("text", "").strip()
                    inference_latency = end_time - start_time
                    generated_tokens = len(choice.get("tokens", []))
                    if generated_tokens == 0 and bot_reply:
                        generated_tokens = len(mistral_llm_instance.tokenize(bot_reply.encode('utf-8')))

            else:
                bot_reply = "Error: No model selected."
                generated_tokens = 0
                inference_latency = 0
            
            # 5. Log metrics and print to console
            if generated_tokens > 0 and inference_latency > 0:
                tps = generated_tokens / inference_latency
                st.session_state.metrics.append({'latency': inference_latency, 'tokens': generated_tokens, 'tps': tps})

                print(f"\n--- TECHNICAL METRICS ({st.session_state.current_model} on {st.session_state.current_dataset}) ---")
                print(f"| Inference Latency: {inference_latency:.2f} seconds")
                print(f"| Generated Tokens: {generated_tokens}")
                print(f"| Tokens Per Second (t/s): {tps:.2f}")
                print("--------------------------------------\n")
            else:
                if "API ERROR" not in bot_reply and "Error:" not in bot_reply:
                    bot_reply = f"⚠️ Could not generate a response. Model may have failed to run or crashed."
        
        except Exception as e:
            bot_reply = f"⚠️ Critical Error during generation: {e}. Check console."


    st.chat_message("assistant").markdown(bot_reply)
    st.session_state.chat_history.append((user_input, bot_reply))