import pandas as pd
import os
from fuzzywuzzy import fuzz
import streamlit as st
from typing import Optional, List, Any

# --- Configuration (UPDATED FOR ALL 4 DATASETS) ---
PIMA_PATH = 'data/pima_diabetes.csv'
MIMIC_PATH = 'data/mimic_demo/'
WEARABLE_PERF_PATH = 'data/wearable_perf.csv'      
WEARABLE_SPORTS_PATH = 'data/wearable_sports.csv'  

# --- Global DataFrames (Will be loaded via functions) ---
PIMA_DF: Optional[pd.DataFrame] = None
MIMIC_DF_CONTEXT: Optional[pd.DataFrame] = None 
PERF_DF: Optional[pd.DataFrame] = None
SPORTS_DF: Optional[pd.DataFrame] = None

# --- MIMIC Data Loading and Joining (CRITICALLY IMPORTANT FOR CACHING) ---
@st.cache_data
def load_mimic_data() -> pd.DataFrame:
    """Loads and joins key MIMIC tables into a single, denormalized DataFrame."""
    try:
        patients = pd.read_csv(os.path.join(MIMIC_PATH, 'patients.csv'))
        admissions = pd.read_csv(os.path.join(MIMIC_PATH, 'admissions.csv'))
        diagnoses = pd.read_csv(os.path.join(MIMIC_PATH, 'diagnoses_icd.csv'))
        
        diagnoses_simple = diagnoses.drop_duplicates(subset=['subject_id', 'icd_code'])

        df_merged = pd.merge(patients, admissions, on='subject_id', how='inner')
        
        df_diagnoses_agg = diagnoses_simple.groupby('subject_id')['icd_code'].apply(list).reset_index(name='ICD_DIAGNOSES')
        
        df_context = pd.merge(df_merged, df_diagnoses_agg, on='subject_id', how='left')
        df_context['Age'] = df_context['anchor_age'] # Create simple Age column
        
        print("MIMIC Demo data loaded and joined successfully.")
        return df_context

    except FileNotFoundError as e:
        print(f"Error loading MIMIC Demo files: {e}. Check if files are in {MIMIC_PATH}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred during MIMIC data processing: {e}")
        return pd.DataFrame()

# --- Main Search Logic for LLM Context (ALL DATASETS) ---

def search_relevant_facts(user_input: str, dataset_type: str = 'PIMA') -> pd.DataFrame:
    """Searches the appropriate dataset and returns a relevant context row."""
    global PIMA_DF, MIMIC_DF_CONTEXT, PERF_DF, SPORTS_DF
    
    # 1. EXTRACT AGE FOR FILTERING
    user_age = next((int(word) for word in user_input.split() if word.isdigit() and 10 <= int(word) <= 100), None)
    
    # 2. LOAD & SEARCH LOGIC
    if dataset_type == 'PIMA':
        if PIMA_DF is None:
            try:
                PIMA_DF = pd.read_csv(PIMA_PATH)
            except FileNotFoundError: return pd.DataFrame()
        
        df = PIMA_DF.copy()
        # [Existing PIMA data cleaning/search logic here]
        if user_age:
             df['Age_Diff'] = abs(df['Age'] - user_age)
             high_risk_matches = df[df['Outcome'] == 1].sort_values(by='Age_Diff').head(3)
             return high_risk_matches.drop(columns=['Age_Diff', 'Outcome'])
        return df.sample(1).drop(columns=['Outcome']) if not df.empty else pd.DataFrame()

    elif dataset_type == 'MIMIC':
        if MIMIC_DF_CONTEXT is None:
            MIMIC_DF_CONTEXT = load_mimic_data() 
        df = MIMIC_DF_CONTEXT
        if df.empty: return pd.DataFrame()
        
        if user_age:
             df = df.copy()
             df['Age_Diff'] = abs(df['Age'] - user_age)
             best_match = df.sort_values(by='Age_Diff').head(1)
             context_cols = ['subject_id', 'gender', 'Age', 'marital_status', 'hospital_expire_flag', 'ICD_DIAGNOSES']
             return best_match[context_cols].head(1)
             
        context_cols = ['subject_id', 'gender', 'Age', 'marital_status', 'hospital_expire_flag', 'ICD_DIAGNOSES']
        return df.sample(1)[context_cols].head(1)

    elif dataset_type == 'WEARABLE_PERF': 
        if PERF_DF is None:
            try:
                PERF_DF = pd.read_csv(WEARABLE_PERF_PATH)
            except FileNotFoundError: return pd.DataFrame()
        # Returns a random sample from the large device performance data
        return PERF_DF.sample(1) if not PERF_DF.empty else pd.DataFrame()
    
    elif dataset_type == 'WEARABLE_SPORTS': 
        if SPORTS_DF is None:
            try:
                SPORTS_DF = pd.read_csv(WEARABLE_SPORTS_PATH)
            except FileNotFoundError: return pd.DataFrame()
        # Returns a random sample from the sports monitoring dataset
        return SPORTS_DF.sample(1) if not SPORTS_DF.empty else pd.DataFrame()

    return pd.DataFrame()

# Load MIMIC data at startup using the cache decorator
MIMIC_DF_CONTEXT = load_mimic_data()