import torch
import torch.nn as nn

from train_fusion_predictor import train_loader, val_loader, test_loader
from train_fusion_predictor import DEVICE


class WorldModelPredictor(nn.Module):

    def __init__(self):

        super().__init__()

        self.emotion_gate = nn.Parameter(torch.tensor(0.1))

        self.lstm = nn.LSTM(
            input_size=86,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        self.decoder = nn.Linear(128, 66)
        self.rollout_steps = 15


    def forward(self, x):
        batch_size = x.shape[0]
        pose_part = x[:, :, :66]
        emo_part = x[:, :, 66:]

        combined_input = torch.cat(
            [pose_part, self.emotion_gate * emo_part],
            dim=2
        )

        _, (h, c) = self.lstm(combined_input)
        last_emo = emo_part[:, -1:, :]
        current_input_step = x[:, -1:, :]
        outputs = []

        for _ in range(self.rollout_steps):
            out, (h, c) = self.lstm(current_input_step, (h, c))
            pose_pred = self.decoder(out)
            outputs.append(pose_pred)
            current_input_step = torch.cat(
                [pose_pred, self.emotion_gate * last_emo],
                dim=2
            )

        final_outputs = torch.cat(outputs, dim=1)
        return final_outputs.view(batch_size, 15, 33, 2)

model = WorldModelPredictor().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()


train_losses = []
val_losses = []


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
    test_loss = 0
    with torch.no_grad():
        for x,y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            test_loss += loss.item()


    print(
        f"Epoch {epoch} | Train {train_loss:.4f} | Val {val_loss:.4f} | Test{test_loss:.4f}"
    )

torch.save(model.state_dict(), "world_model_fusion.pt")
print("Saved trained world_model_fusion.pt")
print("Learned emotion gate:", model.emotion_gate.item())