import torch
import matplotlib.pyplot as plt
import os

#from train_fusion_predictor import FusionPredictor
#from train_fusion_predictor import test_loader
from train_world_model_rollout import WorldModelPredictor, test_loader
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = WorldModelPredictor().to(DEVICE)
model.load_state_dict(torch.load("world_model_fusion.pt"))

model.eval()

x, y = next(iter(test_loader))

x = x.to(DEVICE)
y = y.to(DEVICE)

pred = model(x)

pred = pred[0].cpu().detach().numpy()
gt = y[0].cpu().detach().numpy()

plt.plot(pred[:,0,0], label="pred")
plt.plot(gt[:,0,0], label="gt")

plt.legend()

#plt.title("Fusion Prediction")
#plt.savefig("fusion_prediction.png")

plt.title("Wold Model Rollout Prediction")
plt.savefig("rollout_prediction.png")