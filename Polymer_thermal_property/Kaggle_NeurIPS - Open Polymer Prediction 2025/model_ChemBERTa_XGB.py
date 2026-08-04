import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/scratch16/rherna21/xingfei/MLAI/Kaggle_polymer'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
from sklearn.model_selection import train_test_split

csv_path = '/scratch16/rherna21/xingfei/MLAI/Kaggle_polymer/neurips-open-polymer-prediction-2025/train.csv'
train_df = pd.read_csv(csv_path)

# 1. split off 20% for dev_test
temp_df, dev_test = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,  # for reproducibility
    shuffle=True
)

# 2. split the remaining 80% into 75% train / 25% valid → 0.6 / 0.2 overall
dev_train, dev_val = train_test_split(
    temp_df,
    test_size=0.25,  # 0.25 * 0.8 = 0.2 of the original
    random_state=42,
    shuffle=True
)

# Verify sizes
print(f"Total rows:   {len(train_df)}")
print(f"Dev train:    {len(dev_train)} ({len(dev_train)/len(train_df):.2%})")
print(f"Dev valid:    {len(dev_val)} ({len(dev_val)/len(train_df):.2%})")
print(f"Dev test:     {len(dev_test)} ({len(dev_test)/len(train_df):.2%})")
print(f"Polymer example:{dev_train['SMILES'].to_list()[:3]}")
print(f"Columns:{dev_train.columns}")

#from tqdm.notebook import tqdm as notebook_tqdm
#import tqdm
#tqdm.tqdm = notebook_tqdm
#tqdm.trange = notebook_tqdm

import tqdm

# Monkey-patch: replace notebook tqdm with console tqdm
tqdm.notebook = type("tqdm_notebook", (), {})()   # fake module-like object
tqdm.notebook.tqdm = tqdm.tqdm
tqdm.notebook.trange = tqdm.trange


from torch_molecule import SMILESTransformerMolecularPredictor
from torch_molecule.utils.search import ParameterType, ParameterSpec

from torch.utils.data import Dataset, DataLoader
import torch

class SMILESDataset(Dataset):
    def __init__(self, smiles_list, labels, tokenizer):
        self.smiles = smiles_list
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        s = self.smiles[idx]
        y = self.labels[idx]
        enc = self.tokenizer(s, padding='max_length', truncation=True,
                             max_length=128, return_tensors="pt")
        # remove batch dim
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        return enc, y

import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Load pretrained ChemBERTa
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
chemberta = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chemberta.to(device)  # Move model to device

def get_embedding(smiles_list, batch_size=32):
    embeddings = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = chemberta(**inputs)
            # CLS token embedding
            emb = outputs.last_hidden_state[:,0,:].cpu().numpy()
        embeddings.append(emb)
    return np.vstack(embeddings)

X_train_smiles = dev_train["SMILES"].tolist()
y_train = dev_train[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()

X_val_smiles = dev_val['SMILES'].to_list()
y_val = dev_val[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()

print(np.isnan(y_train).sum())          # total NaNs
print(np.isinf(y_train).sum())          # total infinities
print((y_train > 1e6).sum())            # too large values

X_train = get_embedding(X_train_smiles)
X_val = get_embedding(X_val_smiles)
print("Embedding shape:", X_train.shape)  # (n_samples, hidden_dim=768)

from xgboost import XGBRegressor

# Prepare inputs
test_path = '/scratch16/rherna21/xingfei/MLAI/Kaggle_polymer/neurips-open-polymer-prediction-2025/test.csv'
test_df = pd.read_csv(test_path)
X_test_smiles = test_df.SMILES.tolist()
X_test = get_embedding(X_test_smiles)

targets = ["Tg", "FFV", "Tc", "Density", "Rg"]
preds = []

for i, target in enumerate(targets):
    y = y_train[:, i]
    mask = ~np.isnan(y)                  # remove NaNs in training only
    X_train_target = X_train[mask]
    y_train_target = y[mask]

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="gpu_hist" if torch.cuda.is_available() else "hist"
    )
    model.fit(X_train_target, y_train_target)

    y_pred = model.predict(X_test)       # predict on all test samples
    preds.append(y_pred)                 # append to list

y_pred_final = np.column_stack(preds)  # shape = (n_test, 5)

#test_df = pd.read_csv(test_path)
#test_X = test_df.SMILES.tolist()  # <-- convert Series to list of strings
#model_ChemBERTa = joblib.load("model_ChemBERTa.pkl")
#print("Model loaded!")

# --- Step 4: Predict on new SMILES text ---
# Example: text_X is your new SMILES input data
# Make sure it is preprocessed the same way as training!
#test_Y = model_ChemBERTa.predict(test_X)
#y_pred = test_Y['prediction']  # shape: (n_samples, n_targets)
#y_pred = np.array(y_pred)
#print("Predictions:", y_pred)

# Example: save predictions to a CSV with SMILES + predicted targets
pred_df = pd.DataFrame(y_pred_final, columns=['Tg', 'FFV', 'Tc', 'Density', 'Rg'])
pred_df.insert(0, 'SMILES', X_test_smiles)  # add SMILES column at front

# Save to CSV
pred_df.to_csv('/scratch16/rherna21/xingfei/MLAI/Kaggle_polymer/neurips-open-polymer-prediction-2025/test_predictions_ChemBERTa_XGB.csv', index=False)
print("Predictions saved as predictions_ChemBERTa_XGB.csv")
