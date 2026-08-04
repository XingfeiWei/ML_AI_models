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

class ChemBERTaRegressor(nn.Module):
    def __init__(self, base_model, hidden_size=768, n_tasks=5):
        super().__init__()
        self.base = base_model
        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.Linear(hidden_size, n_tasks)

    def forward(self, enc):
        outputs = self.base(**enc)
        pooled = outputs.last_hidden_state.mean(dim=1)  # mean pooling
        pooled = self.dropout(pooled)
        return self.regressor(pooled)

class ChemBERTaLSTMRegressor(nn.Module):
    def __init__(self, base_model, hidden_size=768, lstm_dim=256, n_layers=2, n_tasks=5, dropout=0.2):
        super().__init__()
        self.base = base_model
        self.lstm = nn.LSTM(hidden_size, lstm_dim, num_layers=n_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_dim * 2, n_tasks)  # *2 for bidirectional

    def forward(self, enc):
        # Step 1: ChemBERTa embeddings
        outputs = self.base(**enc)
        seq_emb = outputs.last_hidden_state   # (batch, seq_len, hidden_size)

        # Step 2: LSTM on embeddings
        lstm_out, _ = self.lstm(seq_emb)      # (batch, seq_len, lstm_dim*2)

        # Step 3: Pooling (mean or last hidden)
        pooled = lstm_out.mean(dim=1)         # (batch, lstm_dim*2)

        # Step 4: Regression head
        pooled = self.dropout(pooled)
        return self.fc(pooled)

print("Model initialized successfully")

X_train = dev_train['SMILES'].to_list()
y_train = dev_train[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()
mask = ~np.isnan(y_train)  # True where label is valid

X_val = dev_val['SMILES'].to_list()
y_val = dev_val[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()
mask = ~np.isnan(y_val)  # True where label is valid

# Create datasets
train_dataset = SMILESDataset(X_train, y_train, tokenizer)
val_dataset   = SMILESDataset(X_val, y_val, tokenizer)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = ChemBERTaRegressor(chemberta, hidden_size=768, n_tasks=5).to(device)
model = ChemBERTaLSTMRegressor(chemberta, hidden_size=768, lstm_dim=256, n_tasks=5).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
def masked_mse_loss(preds, targets):
    # preds, targets: (batch, n_tasks)
    mask = ~torch.isnan(targets)              # True where label is valid
    diff = preds[mask] - targets[mask]
    return (diff ** 2).mean()
criterion = masked_mse_loss

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for enc, y in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(enc)
        loss = criterion(preds, y)   # masked loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for enc, y in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            y = y.to(device)
            preds = model(enc)
            loss = criterion(preds, y)   # masked loss
            total_loss += loss.item()
    return total_loss / len(loader)

for epoch in range(500):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss = eval_epoch(model, val_loader, criterion)
    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

import pandas as pd
import numpy as np
import joblib

#joblib.dump(model, "model_ChemBERTa.pkl")
# Save model + tokenizer
#model.save_pretrained("model_ChemBERTa")
#tokenizer.save_pretrained("model_ChemBERTa")
torch.save(model.state_dict(), "model_ChemBERTa.pth")
tokenizer.save_pretrained("chemberta_tokenizer")
print("Model saved as model_ChemBERTa")

# ===============================
# Evaluate best model
# ===============================
# Load tokenizer + model
#tokenizer = AutoTokenizer.from_pretrained("model_ChemBERTa")
model = ChemBERTaLSTMRegressor(chemberta, hidden_size=768, lstm_dim=256, n_tasks=5).to(device)
model.load_state_dict(torch.load("model_ChemBERTa.pth", map_location=device))
tokenizer = AutoTokenizer.from_pretrained("chemberta_tokenizer")
model.eval()

test_path = '/scratch16/rherna21/xingfei/MLAI/Kaggle_polymer/neurips-open-polymer-prediction-2025/test.csv'

# Prepare inputs
test_df = pd.read_csv(test_path)
test_X = test_df.SMILES.tolist()
# 1. Tokenize your SMILES strings
inputs = tokenizer(
    test_X,                     # list of SMILES strings
    padding=True,
    truncation=True,
    return_tensors="pt"
).to(device)

# 2. Forward pass through your model
with torch.no_grad():
    preds = model(inputs)       # <-- works because forward(self, enc) expects a dict
    y_pred = preds.cpu().numpy()  # convert to NumPy if saving to CSV

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
pred_df = pd.DataFrame(y_pred, columns=['Tg', 'FFV', 'Tc', 'Density', 'Rg'])
pred_df.insert(0, 'SMILES', test_X)  # add SMILES column at front

# Save to CSV
pred_df.to_csv('/scratch16/rherna21/xingfei/MLAI/Kaggle_polymer/neurips-open-polymer-prediction-2025/test_predictions_ChemBERTa.csv', index=False)
print("Predictions saved as predictions_ChemBERTa.csv")
