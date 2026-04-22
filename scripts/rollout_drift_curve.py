import torch
import numpy as np
import matplotlib.pyplot as plt

from train_world_model_rollout import WorldModelPredictor
from train_fusion_predictor import dataset, test_loader, DEVICE


def mpjpe_denorm(pred, target, p_mean, p_std):

    p_mean = torch.tensor(p_mean).view(1,1,33,2).to(pred.device)
    p_std = torch.tensor(p_std).view(1,1,33,2).to(pred.device)

    pred = pred * p_std + p_mean
    target = target * p_std + p_mean

    return torch.norm(pred-target, dim=-1).mean(dim=(0,2))


model = WorldModelPredictor().to(DEVICE)

model.load_state_dict(
    torch.load("world_model_fusion.pt", map_location=DEVICE)
)

model.eval()

p_mean = dataset.p_mean.reshape(33,2)
p_std = dataset.p_std.reshape(33,2)

errors = torch.zeros(15)

count = 0


with torch.no_grad():

    for x,y in test_loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(x)

        err = mpjpe_denorm(pred,y,p_mean,p_std)

        errors += err.cpu()

        count+=1


errors/=count


plt.plot(errors.numpy())

plt.xlabel("Prediction Step")

plt.ylabel("MPJPE")

plt.title("Rollout Drift Curve")

plt.savefig("rollout_drift_curve.png")

print(errors)