import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
import numpy as np
from collections import defaultdict

# Load the validation file
val_df = pd.read_csv('../data/validation_labels.csv')  # Assuming this is the file name

# Extract base mol_name and res_index
#val_df['mol_base'] = val_df['ID'].apply(lambda x: '_'.join(x.split('_')[:2]))
#val_df['res_index'] = val_df['ID'].apply(lambda x: int(x.split('_')[-1]))
val_df['mol_base'] = val_df['ID'].apply(lambda x: x.split('_')[0])

# Initialize output list
output_rows = []
global_mol_id = 10001  # Use a distinct ID range for validation set

for mol_base, group in val_df.groupby('mol_base'):
    group = group.sort_values('resid').reset_index(drop=True)

    current_segment = []
    prev_resid = None

    for _, row in group.iterrows():
        # Consider -1e+18 as missing (common in your data)
        missing_coords = any([
            np.isclose(row['x_1'], -1e+18),
            np.isclose(row['y_1'], -1e+18),
            np.isclose(row['z_1'], -1e+18)
        ])
        resid_break = (prev_resid is not None and row['resid'] != prev_resid + 1)

        if (resid_break or missing_coords) and current_segment:
            for r in current_segment:
                r['mol_name'] = mol_base
                r['mol_id'] = global_mol_id
                output_rows.append(r)
            global_mol_id += 1
            current_segment = []

        if not missing_coords:
            current_segment.append(row.to_dict())

        prev_resid = row['resid']

    if current_segment:
        for r in current_segment:
            r['mol_name'] = mol_base
            r['mol_id'] = global_mol_id
            output_rows.append(r)
        global_mol_id += 1

# Create final validation DataFrame
final_df = pd.DataFrame(output_rows)
final_df = final_df[['mol_name', 'mol_id', 'resname', 'resid', 'x_1', 'y_1', 'z_1']]

# Save
final_df.to_csv('RNA_val_data.csv', index=False)

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    
def pad_to_match(tensor_to_pad, target_tensor):
    diff = target_tensor.size(2) - tensor_to_pad.size(2)
    if diff > 0:
        pad_left = diff // 2
        pad_right = diff - pad_left
        tensor_to_pad = F.pad(tensor_to_pad, (pad_left, pad_right))
    elif diff < 0:
        # Optional: If tensor_to_pad is longer, crop it
        crop_left = (-diff) // 2
        crop_right = crop_left + target_tensor.size(2)
        tensor_to_pad = tensor_to_pad[:, :, crop_left:crop_right]
    return tensor_to_pad

class UNet1D(nn.Module):
    def __init__(self, in_channels=4, base_channels=32, out_channels=3):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock1D(in_channels, base_channels)         # [B, 32, L]
        self.pool1 = nn.MaxPool1d(2)                                 # [B, 32, L/2]
        self.enc2 = ConvBlock1D(base_channels, base_channels*2)     # [B, 64, L/2]
        self.pool2 = nn.MaxPool1d(2)                                 # [B, 64, L/4]
        self.enc3 = ConvBlock1D(base_channels*2, base_channels*4)   # [B, 128, L/4]

        # Decoder (upsampling with kernel=2, stride=2 to double length)
        self.up2 = nn.ConvTranspose1d(base_channels*4, base_channels*2, kernel_size=2, stride=2)  # [B, 64, L/2]
        self.dec2 = ConvBlock1D(base_channels*4, base_channels*2)

        self.up1 = nn.ConvTranspose1d(base_channels*2, base_channels, kernel_size=2, stride=2)     # [B, 32, L]
        self.dec1 = ConvBlock1D(base_channels*2, base_channels)

        # Output: map to 3D coordinates per position with padding to keep length
        self.final = nn.Conv1d(base_channels, out_channels, kernel_size=9, padding=4)  # [B, 3, L]

    def forward(self, x):
        e1 = self.enc1(x)                     # [B, 32, L]
        e2 = self.enc2(self.pool1(e1))       # [B, 64, L/2]
        e3 = self.enc3(self.pool2(e2))       # [B, 128, L/4]

        d2 = self.up2(e3)                    # [B, 64, L/2]
        d2 = pad_to_match(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)                   # [B, 32, L]
        d1 = pad_to_match(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        #print(f"d2 shape: {d2.shape}, e2 shape: {e2.shape}")
        #print(f"d1 shape: {d1.shape}, e1 shape: {e1.shape}")

        return self.final(d1)               # [B, 3, L]
        #return self.final(d1).mean(dim=2)  # [B, 3]


val_df = pd.read_csv('RNA_val_data.csv')


import pandas as pd
import random
import torch
from torch.nn import functional as F
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Define your dataset class
class RNADataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.data = []
        self.labels = []
        mapping = {'A': [1, 0, 0, 0],
                   'U': [0, 1, 0, 0],
                   'C': [0, 0, 1, 0],
                   'G': [0, 0, 0, 1]}
        
        for mol_id, group in df.groupby('mol_id'):
            group = group.sort_values('resid').reset_index(drop=True)
            seq = group['resname'].values
            coords = group[['x_1', 'y_1', 'z_1']].values
            
            encoded_seq = torch.tensor([mapping.get(nuc, [0,0,0,0]) for nuc in seq]).T.float()  # shape: [4, seq_len]
            coords_seq = torch.tensor(coords).T.float()  # shape: [3, seq_len]
            
            self.data.append(encoded_seq)
            self.labels.append(coords_seq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Load your validation dataframe
val_df = pd.read_csv('RNA_val_data.csv')

# Create dataset and dataloader
dataset = RNADataset(val_df)
loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=lambda x: (
    torch.nn.utils.rnn.pad_sequence([item[0].T for item in x], batch_first=True).permute(0, 2, 1),
    torch.nn.utils.rnn.pad_sequence([item[1].T for item in x], batch_first=True).permute(0, 2, 1)
))

# Now you can use 'loader' in your validation loop

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet1D().to(device)
model.load_state_dict(torch.load('RNA_unet2.pt', map_location=device))
model.eval()

def tm_score_approx(pred, target, d0=1.24):
    dists = torch.norm(pred - target, dim=1)  # (B, L)
    score = 1 / (1 + (dists / d0) ** 2)
    return score.mean()

total_tm = 0
total_loss = 0
criterion = torch.nn.MSELoss()

with torch.no_grad():
    for inputs, targets in loader:  # use your DataLoader here
        inputs = inputs.to(device)     # [B, 4, L]
        targets = targets.to(device)   # [B, 3, L]

        outputs = model(inputs)        # [B, 3, L]
        loss = criterion(outputs, targets)

        tm = tm_score_approx(outputs.permute(0, 2, 1), targets.permute(0, 2, 1))  # (B, L, 3)

        total_loss += loss.item()
        total_tm += tm.item()

avg_loss = total_loss / len(loader)
avg_tm = total_tm / len(loader)

print(f"Validation Loss: {avg_loss:.4f}")
print(f"Approximate TM-score: {avg_tm:.4f}")

