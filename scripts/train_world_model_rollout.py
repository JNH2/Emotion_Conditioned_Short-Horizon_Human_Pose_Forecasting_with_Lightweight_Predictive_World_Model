
import torch
import torch.nn as nn

class WorldModelPredictor(nn.Module):

    def __init__(self):

        super().__init__()

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

        outputs = []

        _, (h, c) = self.lstm(x)

        input_step = x[:, -1, :]

        for _ in range(self.rollout_steps):

            input_step = input_step.unsqueeze(1)

            out, (h, c) = self.lstm(input_step, (h, c))

            pose_pred = self.decoder(out.squeeze(1))

            outputs.append(pose_pred)

            input_step = pose_pred

        outputs = torch.stack(outputs, dim=1)

        return outputs