import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


SEQ_LEN = 10
DATA_DIR = "videos"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PoseDataset(Dataset):
    def __init__(self, files):
        self.samples = []
        for file in files:
            pose = np.load(file)
            pose = self.clean_sequence(pose)
            PRED_LEN =15

            for i in range(len(pose) - SEQ_LEN - PRED_LEN):
                x = pose[i:i+SEQ_LEN]
                y = pose[i+SEQ_LEN:i+SEQ_LEN+ PRED_LEN]
                self.samples.append((x, y))


    def clean_sequence(self, pose):
        valid_frames = []
        for frame in pose:
            if not np.all(frame == 0):
                valid_frames.append(frame)
        return np.array(valid_frames)


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )


class PosePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=66,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(128, 66 * 15)


    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out.view(-1, 15, 33, 2)


files = [
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if "pose" in f
]

dataset = PoseDataset(files)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset)) 
test_size = len(dataset) - train_size - val_size


train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=32
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 32
)


model = PosePredictor().to(DEVICE)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)
loss_fn = nn.MSELoss()


for epoch in range(20):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    model.eval()
    val_loss = 0
    test_loss = 0


    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)

            loss = loss_fn(pred, y)

            val_loss += loss.item()


    print(
        f"Epoch {epoch} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}"
    )

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
    print(f"Final Test Loss: {test_loss:.4f}")
torch.save(model.state_dict(), "pose_baseline_model.pt")