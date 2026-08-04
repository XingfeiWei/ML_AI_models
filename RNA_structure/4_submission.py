import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load the sequence data
val_seq_df = pd.read_csv('../data/sample_submission.csv')  # Assuming you've saved the above to a file

# Sort by resid just in case
#val_seq_df = val_seq_df.sort_values('resid')

# Extract sequence
sequence = val_seq_df['resname'].tolist()  # e.g., ['G', 'G', ..., 'C']


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


def prepare_sequence(seq, window_size=5):
    mapping = {'A': [1, 0, 0, 0], 'U': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    pad = [0, 0, 0, 0]
    input_data = []

    for i in range(len(seq)):
        context = []
        for j in range(i - window_size // 2, i + window_size // 2 + 1):
            if 0 <= j < len(seq):
                base = seq[j]
                one_hot = mapping.get(base, random.choice(list(mapping.values())))
                context.append(one_hot)
            else:
                context.append(pad)
        input_data.append(torch.tensor(context).T.float())  # shape: [4, window_size]

    return input_data

window_size = 5
inputs = prepare_sequence(sequence, window_size=window_size)
X_val = torch.stack(inputs).to('cuda' if torch.cuda.is_available() else 'cpu')  # [N, 4, window_size]


# === Define your model class here ===
model_classes = {
    'model1': UNet1D,
    'model2': UNet1D,
    'model3': UNet1D,
    'model4': UNet1D,
    'model5': UNet1D
}

model_names = ['model1', 'model2', 'model3', 'model4', 'model5']  # your model keys
# === Inputs: validation data (X_val), metadata (val_seq_df), model paths ===
model_paths = [
    'RNA_unet2.pt',
    'RNA_unet2.pt',
    'RNA_unet2.pt',
    'RNA_unet2.pt',
    'RNA_unet2.pt'
]

# Initialize output DataFrame with metadata
merged_df = val_seq_df[['ID', 'resname', 'resid']].copy()

# Predict with each model
for i, (name, path) in enumerate(zip(model_names, model_paths), 1):
    print(f"Loading model {i}: {path}")
    
    model = UNet1D()
    model = model_classes[name]()
    model.load_state_dict(torch.load(path))  # Or 'cuda'
    model.eval()
    model.to(X_val.device)

    with torch.no_grad():
        Y_pred = model(X_val)  # shape: [N, 3]

    # Store prediction in DataFrame
    merged_df[f'x_{i}'] = Y_pred[:, 0].cpu().numpy()
    merged_df[f'y_{i}'] = Y_pred[:, 1].cpu().numpy()
    merged_df[f'z_{i}'] = Y_pred[:, 2].cpu().numpy()

# === Save result ===
merged_df.to_csv('sample_submission_predicted.csv', index=False)
print("Saved: RNA_predicted.csv")
