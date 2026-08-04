import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader

class Residue3DPredictor(nn.Module):
    def __init__(self, window_size=5, num_classes=4):
        super().__init__()
        self.window_size = window_size

        # First convolution layer (4 channels → 32 channels)
        self.initial_conv = nn.Conv1d(num_classes, 32, kernel_size=9, padding=4)

        # 10 convolutional layers (32 → 32), with kernel_size=9, padding=4
        self.middle_convs = nn.Sequential(
            *[nn.Sequential(
                nn.Conv1d(32, 32, kernel_size=9, padding=4),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(10)]
        )

        # Final convolution (32 → 128)
        self.final_conv = nn.Conv1d(32, 128, kernel_size=9, padding=4)

        # Pool to fixed window size
        self.pool = nn.AdaptiveAvgPool1d(output_size=window_size)

        # Fully connected to 3D coordinates
        self.fc = nn.Linear(128 * window_size, 3)

    def forward(self, x):
        x = F.relu(self.initial_conv(x))     # [B, 32, W]
        x = self.middle_convs(x)             # [B, 32, W]
        x = F.relu(self.final_conv(x))       # [B, 128, W]
        x = self.pool(x)                     # [B, 128, window_size]
        x = x.view(x.size(0), -1)            # [B, 128 * window_size]
        return self.fc(x)

class RNADataset(torch.utils.data.Dataset):
    def __init__(self, df, window_size=5):
        self.window_size = window_size
        self.data = []
        self.labels = []
        seq = df['resname'].values
        coords = df[['x_1', 'y_1', 'z_1']].values

        mapping = {'A': [1,0,0,0], 'U': [0,1,0,0], 'C': [0,0,1,0], 'G': [0,0,0,1]}
        pad = [0, 0, 0, 0]

        for i in range(len(seq)):
            context = []
            for j in range(i - window_size//2, i + window_size//2 + 1):
                if 0 <= j < len(seq):
                    context.append(mapping[seq[j]])
                else:
                    context.append(pad)
            self.data.append(torch.tensor(context).T)  # shape: [4, window_size]
            self.labels.append(torch.tensor(coords[i]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].float(), self.labels[idx].float()


# Load the data
df = pd.read_csv('RNA_combined_data.csv')

# Your RNADataset class (assumed already defined)
class RNADataset(torch.utils.data.Dataset):
    def __init__(self, df, window_size=5):
        self.window_size = window_size
        self.data = []
        self.labels = []

        seq = df['resname'].values
        coords = df[['x_1', 'y_1', 'z_1']].values
        mapping = {'A': [1, 0, 0, 0], 'U': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
        pad = [0, 0, 0, 0]

        for i in range(len(seq)):
            context = []
            for j in range(i - window_size//2, i + window_size//2 + 1):
                if 0 <= j < len(seq):
                    context.append(mapping.get(seq[j], pad))
                else:
                    context.append(pad)
            self.data.append(torch.tensor(context).T)  # shape: [4, window_size]
            self.labels.append(torch.tensor(coords[i]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].float(), self.labels[idx].float()

# --- Build dataset from multiple molecules ---

all_data = []

for mol_id, group in df.groupby('mol_id'):
    group = group.sort_values('resid').reset_index(drop=True)
    dataset = RNADataset(group, window_size=5)
    all_data.extend([dataset[i] for i in range(len(dataset))])

# Create a DataLoader
loader = DataLoader(all_data, batch_size=32, shuffle=True)

model = Residue3DPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
def tm_score_approx(pred, target, d0=1.24):
    """
    pred, target: (B, N, 3) coordinates
    d0: normalization constant
    """
    dists = torch.norm(pred - target, dim=-1)  # (B, N)
    score = 1 / (1 + (dists / d0) ** 2)
    return score.mean()  # average over residues and batch

for epoch in range(2000):
    total_loss = 0
    total_tm = 0
    model.train()

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)  # (B, N, features), (B, N, 3)

        optimizer.zero_grad()
        outputs = model(inputs)  # (B, N, 3)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Surrogate TM-score
        tm = tm_score_approx(outputs, targets)

        total_loss += loss.item()
        total_tm += tm.item()

    avg_loss = total_loss / len(loader)
    avg_tm = total_tm / len(loader)

    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, TM-score ≈ {avg_tm:.4f}")

# Save the model
torch.save(model.state_dict(), 'RNA_amodel.pt')
# Initialize and load the model
model = Residue3DPredictor()
model.load_state_dict(torch.load('RNA_amodel.pt'))
model.eval()  # Set to evaluation mode

