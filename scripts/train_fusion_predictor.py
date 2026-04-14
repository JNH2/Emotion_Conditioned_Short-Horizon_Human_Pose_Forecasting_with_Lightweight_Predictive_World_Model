import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


SEQ_LEN = 10
DATA_DIR = "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FusionDataset(Dataset):

    def __init__(self, pose_files, emotion_files):

        self.samples = []


        for p_file, e_file in zip(pose_files, emotion_files):

            pose = np.load(p_file)
            emo = np.load(e_file)


            for i in range(len(pose) - SEQ_LEN):

                pose_seq = pose[i:i+SEQ_LEN]
                emo_seq = emo[i:i+SEQ_LEN]

                target = pose[i+SEQ_LEN]

                x = np.concatenate(
                    [
                        pose_seq.reshape(SEQ_LEN, -1),
                        emo_seq
                    ],
                    axis=1
                )

                self.samples.append((x, target))


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, idx):

        x, y = self.samples[idx]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )


class FusionPredictor(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=86,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(128, 66)


    def forward(self, x):

        _, (h, _) = self.lstm(x)

        out = self.fc(h[-1])

        return out.view(-1, 33, 2)


pose_files = sorted([
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if "pose" in f
])


emotion_files = sorted([
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if "emotion" in f
])


dataset = FusionDataset(
    pose_files,
    emotion_files
)


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size


train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
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


model = FusionPredictor().to(DEVICE)


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