import torch
import pandas as pd
import numpy as np
import xgboost as xgb
from transformers import AutoTokenizer, EsmModel
from tqdm import tqdm
import gc
import sys
import os

# --- 1. CONFIGURAZIONE ---
ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
FASTA_PATH = "sequence_for_ml.fasta"            
XGB_MODEL_PATH = "PETase_model.json"
OUTPUT_FILE = "rank_PETase.csv"

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print("="*50)
print(" AVVIO SCREENING IN-SILICO (LOG BASE 10)")
print("="*50)
print(f"Utilizzo device: {DEVICE}")

# --- 2. PARSER DEL FILE FASTA ---
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
        print(f"ERRORE: File '{file_path}' non trovato."); sys.exit(1)
    return ids, seqs

seq_ids, sequences = read_fasta(FASTA_PATH)
print(f"Trovate {len(sequences)} sequenze.")

# --- 3. ESTRAZIONE EMBEDDINGS ---
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

# --- 4. PREDIZIONE ---
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(XGB_MODEL_PATH)

print("Calcolo attività (Scala Seo et al. 2025)...")
preds_log = xgb_model.predict(X_candidates)

# CORREZIONE CRUCIALE: Usiamo base 10 per coerenza con il training
preds_orig = 10 ** preds_log 

# --- 5. SALVATAGGIO ---
df_results = pd.DataFrame({
    'Protein_ID': seq_ids,
    'Attivita_Predetta': preds_orig,
    'Log10_Activity': preds_log,
    'Sequenza': sequences
})

df_sorted = df_results.sort_values(by='Attivita_Predetta', ascending=False).reset_index(drop=True)
df_sorted.insert(0, 'Ranking', range(1, len(df_sorted) + 1))
df_sorted.to_csv(OUTPUT_FILE, index=False)

print("\nTOP 3 CANDIDATI:")
print(df_sorted[['Ranking', 'Protein_ID', 'Attivita_Predetta']].head(3).to_string(index=False))
