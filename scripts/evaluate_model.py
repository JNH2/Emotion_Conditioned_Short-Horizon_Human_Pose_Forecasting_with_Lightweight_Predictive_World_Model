import torch

from train_fusion_predictor import FusionPredictor
from train_fusion_predictor import test_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FusionPredictor().to(DEVICE)

model.load_state_dict(torch.load("fusion_model.pt"))


def mpjpe(pred, target):

    return torch.mean(
        torch.norm(pred - target, dim=-1)
    )


def evaluate(model, loader, device):

    model.eval()

    total_mpjpe = 0
    total_samples = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            total_mpjpe += mpjpe(pred, y).item()
            total_samples += 1

    return total_mpjpe / total_samples


score = evaluate(model, test_loader, DEVICE)

print("MPJPE:", score)