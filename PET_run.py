import torch
import pandas as pd
import numpy as np
import xgboost as xgb
from transformers import AutoTokenizer, EsmModel
from tqdm import tqdm
import gc
import sys
import os
# --- 1. CONFIGURATION ---
ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
FASTA_PATH = "sequences.fasta"            
XGB_MODEL_PATH = "PETase_model.json"
OUTPUT_FILE = "rank_PETase.csv"
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print("="*50)
print(" START IN-SILICO SCREENING (LOG BASE 10)")
print("="*50)
print(f"Using device: {DEVICE}")
# --- 2. FASTA FILE PARSER ---
def read_fasta(file_path):
    ids, seqs = [], []
    try:
        with open(file_path, 'r') as f:
            current_id, current_seq = "", []
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith(">"):
                    if current_id:
                        ids.append(current_id)
                        seqs.append("".join(current_seq))
                    current_id = line[1:]
                    current_seq = []
                else:
                    current_seq.append(line.replace(" ", "").replace("*", ""))
            if current_id:
                ids.append(current_id)
                seqs.append("".join(current_seq))
    except FileNotFoundError:
        print(f"ERROR: File '{file_path}' not found."); sys.exit(1)
    return ids, seqs
seq_ids, sequences = read_fasta(FASTA_PATH)
print(f"Found {len(sequences)} sequences.")
# --- 3. EMBEDDINGS EXTRACTION ---
tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
esm_model = EsmModel.from_pretrained(ESM_MODEL_NAME).to(DEVICE)
def get_embeddings(seq_list, batch_size=16):
    esm_model.eval()
    all_embeddings = []
    for i in tqdm(range(0, len(seq_list), batch_size), desc="ESM-2 Inference"):
        batch = seq_list[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE)
        with torch.no_grad():
            outputs = esm_model(**inputs)
            mask = inputs.attention_mask.unsqueeze(-1)
            mean_embeddings = torch.sum(outputs.last_hidden_state * mask, dim=1) / torch.sum(mask, dim=1)
            all_embeddings.append(mean_embeddings.cpu().numpy())
    return np.vstack(all_embeddings)
X_candidates = get_embeddings(sequences)
del esm_model, tokenizer
if DEVICE == "mps": torch.mps.empty_cache()
gc.collect()
# --- 4. PREDICTION ---
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(XGB_MODEL_PATH)
print("Computing activity (Seo et al. 2025 scale)...")
preds_log = xgb_model.predict(X_candidates)
# CRUCIAL FIX: Use base 10 for consistency with training
preds_orig = 10 ** preds_log 
# --- 5. SAVING ---
df_results = pd.DataFrame({
    'Protein_ID': seq_ids,
    'Predicted_Activity': preds_orig,
    'Log10_Activity': preds_log,
    'Sequence': sequences
})
df_sorted = df_results.sort_values(by='Predicted_Activity', ascending=False).reset_index(drop=True)
df_sorted.insert(0, 'Ranking', range(1, len(df_sorted) + 1))
df_sorted.to_csv(OUTPUT_FILE, index=False)
print("\nTOP 3 CANDIDATES:")
print(df_sorted[['Ranking', 'Protein_ID', 'Predicted_Activity']].head(3).to_string(index=False))
