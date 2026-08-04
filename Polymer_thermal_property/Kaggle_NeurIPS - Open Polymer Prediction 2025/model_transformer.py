import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/home/xwei20/Kaggle_Polymer'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
from sklearn.model_selection import train_test_split

csv_path = '/home/xwei20/Kaggle_Polymer/neurips-open-polymer-prediction-2025/train.csv'
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

search_parameters = {
    "num_layers": ParameterSpec(ParameterType.INTEGER, (2, 4)),
    "n_heads": ParameterSpec(ParameterType.INTEGER, (4, 8)),
    "hidden_size": ParameterSpec(ParameterType.INTEGER, (64, 256)),
    "dropout": ParameterSpec(ParameterType.FLOAT, (0.0, 0.5)),
    "learning_rate": ParameterSpec(ParameterType.LOG_FLOAT, (1e-4, 1e-3))
}

Transformer = SMILESTransformerMolecularPredictor(
    task_type="regression",
    num_task=5,
    batch_size=192,
    epochs=200,
    verbose=True
)

print("Model initialized successfully")
X_train = dev_train['SMILES'].to_list()
y_train = dev_train[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()
X_val = dev_val['SMILES'].to_list()
y_val = dev_val[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()
Transformer.autofit(
    X_train = X_train,
    y_train = y_train,
    X_val = X_val,
    y_val = y_val,
    search_parameters=search_parameters,
    n_trials = 10 # number of times searching the best hyper-parameters
)

import pandas as pd
import numpy as np
import joblib

joblib.dump(Transformer, "model_Transformer.pkl")
print("Model saved as model_Transformer.pkl")

# ===============================
# Evaluate best model
# ===============================
test_path = '/home/xwei20/Kaggle_Polymer/neurips-open-polymer-prediction-2025/test.csv'
test_df = pd.read_csv(test_path)
test_X = test_df.SMILES.tolist()  # <-- convert Series to list of strings

model_Transformer = joblib.load("model_Transformer.pkl")
print("Model loaded!")

# --- Step 4: Predict on new SMILES text ---
# Example: text_X is your new SMILES input data
# Make sure it is preprocessed the same way as training!
test_Y = model_Transformer.predict(test_X)
y_pred = test_Y['prediction']  # shape: (n_samples, n_targets)
y_pred = np.array(y_pred)
print("Predictions:", y_pred)

# Example: save predictions to a CSV with SMILES + predicted targets
pred_df = pd.DataFrame(y_pred, columns=['Tg', 'FFV', 'Tc', 'Density', 'Rg'])
pred_df.insert(0, 'SMILES', test_X)  # add SMILES column at front

# Save to CSV
pred_df.to_csv('/home/xwei20/Kaggle_Polymer/neurips-open-polymer-prediction-2025/test_predictions_Transformer.csv', index=False)
print("Predictions saved as predictions_Transformer.csv")
