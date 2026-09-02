# PETase-scoring

A machine learning framework using ESM-2 to predict the efficacy of putative PET-degrading enzymes.

Protein sequences are embedded with ESM-2 and scored by an XGBoost regressor trained on measured PETase activity. A pre-trained model (`PETase_model.json`) is included, so you can score your own sequences without retraining.

## Requirements

```bash
pip install torch transformers xgboost scikit-learn umap-learn pandas numpy scipy matplotlib seaborn tqdm
```

The scripts automatically use MPS (Apple Silicon), CUDA, or CPU, depending on what is available.

## Usage

### Score your sequences

Replace 'sequences.fasta' with your own sequences in FASTA format, or edit 'FASTA_PATH' at the top of 'PET_run.py' - then run:

```bash
python PET_run.py
```

This produces 'rank_PETase.csv', with the candidates ranked by predicted activity.

### Retrain the model (optional)

To rebuild the model from your own training data, replace `dataset.csv` — keep the filename, or edit `DATA_PATH` at the top of `PET_train.py` — then run:

```bash
python PET_train.py
```

This regenerates `PETase_model.json` and a diagnostic plot.

`dataset.csv` must contain these columns:

- `Sequence` — amino-acid sequences (input to ESM-2)
- `Consensus_Log_Activity` — activity on a log10 scale (the regression target)
- `Consensus_Linear_Activity` — activity on a linear scale (used only to colour the UMAP plot)
