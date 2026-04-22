import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


SEQ_LEN = 10
DATA_DIR = "videos"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FusionDataset(Dataset):
    def __init__(self, pose_files, emotion_files):

        self.samples = []
        all_poses = []
        all_emos = []

        for p_file, e_file in zip(pose_files, emotion_files):
            pose = np.load(p_file)
            emo = np.load(e_file)

            pose, emo = self.clean_sequence(pose, emo)
            all_poses.append(pose.reshape(-1, 66))
            all_emos.append(emo)
        
        self.p_mean = np.mean(np.concatenate(all_poses), axis = 0)
        self.p_std = np.std(np.concatenate(all_poses), axis = 0) + 1e-6
        self.e_mean = np.mean(np.concatenate(all_emos), axis = 0)
        self.e_std = np.std(np.concatenate(all_emos), axis = 0) + 1e-6

        for p_file, e_file in zip(pose_files, emotion_files):
            pose, emo = self.clean_sequence(np.load(p_file), np.load(e_file))
            pose = (pose.reshape(-1, 66) - self.p_mean) / self.p_std
            emo = (emo - self.e_mean) / self.e_std                        
            
            PRED_LEN = 15
            for i in range(len(pose) - SEQ_LEN - PRED_LEN):
                pose_seq = pose[i : i + SEQ_LEN]
                emo_seq = emo[i : i + SEQ_LEN]

                target = pose[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN].reshape(15, 33, 2)

                x = np.concatenate(
                    [
                        pose_seq,
                        emo_seq
                    ],
                    axis=1
                )

                self.samples.append((x, target))

    def clean_sequence(self, pose, emo):
        valid_pose = []
        valid_emo = []

        for p, e in zip(pose, emo):
            if not np.all(p == 0):
                valid_pose.append(p)
                valid_emo.append(e)
        return np.array(valid_pose), np.array(valid_emo)




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
        self.emotion_gate = nn.Parameter(torch.tensor(0.1))

        self.lstm = nn.LSTM(
            input_size=86,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(128, 66 * 15)


    def forward(self, x):
        pose_part = x[:, :, :66]
        emo_part = x[:, :, 66:]

        x = torch.cat([pose_part, self.emotion_gate * emo_part], dim = 2)

        _, (h, _) = self.lstm(x)

        out = self.fc(h[-1])

        return out.view(-1, 15, 33, 2)


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


train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size


train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size,test_size]
)


train_loader = DataLoader(
    train_dataset,
    batch_size = 32,
    shuffle = True
)


val_loader = DataLoader(
    val_dataset,
    batch_size = 32
)

test_loader = DataLoader (
    test_dataset,
    batch_size = 32
)


model = FusionPredictor().to(DEVICE)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


loss_fn = nn.MSELoss()
train_losses = []
val_losses = []
gate_values = []

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
        gate_values.append(model.emotion_gate.item())


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
    print(
        f"Finl Test Loss {test_loss:.4f}"
    )
    train_losses.append(train_loss)
    val_losses.append(val_loss)

import matplotlib.pyplot as plt

plt.plot(train_losses)
plt.plot(val_losses)

plt.legend(["train", "val"])

plt.title("Loss Curve")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.savefig("loss_curve.png")
torch.save(model.state_dict(), "fusion_model.pt")
print("Learned emotion gate:", model.emotion_gate.item())
plt.plot(gate_values)
plt.title("Emotion gate evolution")
plt.xlabel("Epoch")
plt.ylabel("Gate value")
plt.savefig("emotion_gate_curve.png")
