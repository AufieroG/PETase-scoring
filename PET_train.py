import os
import warnings
warnings.filterwarnings('ignore') # Silence UMAP and Transformers warnings

import torch
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from transformers import AutoTokenizer, EsmModel
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from scipy.stats import pearsonr
from tqdm import tqdm
import gc
import sys

# --- 1. CONFIGURATION ---
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DATA_PATH = "dataset.csv" 

# Optimization for Mac
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"--- START PIPELINE ---")
print(f"Using device: {DEVICE}")

# --- 2. Loading data and transformation ---

print("\nReading dataset...")
df = pd.read_csv(DATA_PATH)
sequences = df['Sequence'].tolist()

# Using logarithm Base 10
y = df['Consensus_Log_Activity'].values

# For graphs/tests we use the linear value calculated in R
y_orig_full = df['Consensus_Linear_Activity'].values



# --- 3. EMBEDDINGS (ESM-2) ---
print(f"Loading {MODEL_NAME}...")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = EsmModel.from_pretrained(MODEL_NAME).to(DEVICE)

def get_embeddings(seq_list, batch_size=16):
    model.eval()
    all_embeddings = []
    for i in tqdm(range(0, len(seq_list), batch_size), desc="Generation Embeddings"):
        batch = seq_list[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            mask = inputs.attention_mask.unsqueeze(-1)
            mean_embeddings = torch.sum(outputs.last_hidden_state * mask, dim=1) / torch.sum(mask, dim=1)
            all_embeddings.append(mean_embeddings.cpu().numpy())
    return np.vstack(all_embeddings)

X = get_embeddings(sequences)

print("Free memory ESM-2...")
del model
del tokenizer
if DEVICE == "mps": 
    torch.mps.empty_cache()
elif DEVICE == "cuda":
    torch.cuda.empty_cache()
gc.collect()

# --- 4. Calculate UMAP (entire dataset) ---
print("\nUMAP Dimensional Reduction Calculation (2D)...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X)

# --- 5. Split 90/10 ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42
)
print(f"\nDataset split: {len(X_train)} training (90%), {len(X_test)} test (10%)")


# =====================================================================
# 1: HYPERPARAMETER SEARCH (10-Fold CV on 90%)
# =====================================================================
print("\n=== PHASE 1: Optimal hyperparameter search ===")

def pearson_scorer(y_true, y_pred):
    if np.std(y_pred) < 1e-6 or np.std(y_true) < 1e-6:
        return 0.0
    r, _ = pearsonr(y_true, y_pred)
    return r

pearson_score = make_scorer(pearson_scorer, greater_is_better=True)

param_distributions = {
    'n_estimators': [1000, 1500, 2000, 3000],
    'max_depth': [4, 5, 6, 7],
    'learning_rate': [0.005, 0.01, 0.02, 0.05],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8]
}

xgb_base = xgb.XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist',
    n_jobs=-1,
    random_state=42
)

# Use 10 folds for a fast search
cv_search = KFold(n_splits=10, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_distributions,
    n_iter=100, 
    scoring=pearson_score,
    cv=cv_search,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

best_params = random_search.best_params_
best_params['tree_method'] = 'hist'
best_params['n_jobs'] = -1
best_params['random_state'] = 42

print("\nOptimal parameters found:")
for k, v in best_params.items():
    print(f"  {k}: {v}")


# =====================================================================
# 2: MODEL STRESS TEST (CV 10-Fold considering 90%)
# =====================================================================
print(f"\n=== 2: Stability Test (10-Fold CV) ===")
kf_10 = KFold(n_splits=10, shuffle=True, random_state=42)
cv_r2 = []
cv_pearson = []

for fold, (t_idx, v_idx) in enumerate(kf_10.split(X_train)):
    xt, xv = X_train[t_idx], X_train[v_idx]
    yt, yv = y_train[t_idx], y_train[v_idx]
    
    m = xgb.XGBRegressor(**best_params)
    m.fit(xt, yt)
    p = m.predict(xv)
    
    # Convert to real scale
    yv_orig = 10 ** yv
    p_orig = 10 ** p 
    
    r2 = r2_score(yv_orig, p_orig)
    pearson_val, _ = pearsonr(yv_orig, p_orig)
    
    cv_r2.append(r2)
    cv_pearson.append(pearson_val)
    print(f"Fold {fold+1:02d}: R2 = {r2:.4f} | Pearson = {pearson_val:.4f}")

print("\n----------------------------------------")
print(" REPORT (10-Fold)")
print("----------------------------------------")
print(f"R2 mean      : {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")
print(f"Pearson mean : {np.mean(cv_pearson):.4f} ± {np.std(cv_pearson):.4f}")
print("----------------------------------------")


# =====================================================================
# 3: FINAL RATING OUT OF 10%
# =====================================================================
print("\n=== 3: FINAL RATING OUT OF 10% Test")
val_model = xgb.XGBRegressor(**best_params)
val_model.fit(X_train, y_train)

preds_log = val_model.predict(X_test)

# Conversion to linear scale
preds_orig = 10 ** preds_log
y_test_orig = 10 ** y_test

r2_test = r2_score(y_test_orig, preds_orig)

rmse_test = np.sqrt(mean_squared_error(y_test_orig, preds_orig))
pearson_r, _ = pearsonr(y_test_orig, preds_orig)

print("\n========================================")
print("TEST SET RESULTS (10%)")
print("========================================")
print(f"R2: {r2_test:.4f}")
print(f"RMSE: {rmse_test:.2f}")
print(f"Pearson (Lineare): {pearson_r:.4f}")
print("========================================")


# =====================================================================
# 4: PLOT, 100% TRAINING AND SAVE
# =====================================================================
print("\nGraph Generation...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot A: Prediction vs. Reality
sns.scatterplot(x=y_test_orig, y=preds_orig, alpha=0.7, color='teal', edgecolor='w', s=80, ax=ax1)
lims = [0, max(y_test_orig.max(), preds_orig.max())]
ax1.plot(lims, lims, '--r', alpha=0.75, linewidth=2, label='Perfect Prediction')

ax1.set_xscale('symlog', linthresh=10) 
ax1.set_yscale('symlog', linthresh=10)
ax1.set_title(f"Test Set (10%)\nPearson r: {pearson_r:.3f} | R2: {r2_test:.3f}", fontsize=14)
ax1.set_xlabel("Real Activity (Observed)", fontsize=12)
ax1.set_ylabel("Predicted activity", fontsize=12)
ax1.legend()
ax1.grid(True, which="both", ls="--", alpha=0.3)

# Plot B: UMAP
scatter = ax2.scatter(X_umap[:, 0], X_umap[:, 1], c=y_orig_full, cmap='viridis', s=40, alpha=0.8)
fig.colorbar(scatter, ax=ax2, label='Real Activity (Consensus)')
ax2.set_title("Embeddings Space (UMAP)\nBiochemical signatures of PETases", fontsize=14)
ax2.set_xlabel("UMAP 1")
ax2.set_ylabel("UMAP 2")

plt.tight_layout()
plt.savefig("final_complete_pipeline.png", dpi=300)
print("Plot saved: 'final_complete_pipeline.png'")

print("\n=== PHASE 4: Training model (100%) ===")
final_model = xgb.XGBRegressor(**best_params)
final_model.fit(X, y)

final_model.save_model("PETase_model.json")
print("Saved model: 'PETase_model.json'")
print("\nPipeline completed!")

gc.collect()
sys.exit(0)
