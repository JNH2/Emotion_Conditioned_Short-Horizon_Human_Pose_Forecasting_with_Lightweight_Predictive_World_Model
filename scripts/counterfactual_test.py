import torch
import numpy as np
import os
from train_fusion_predictor import FusionPredictor
from train_fusion_predictor import FusionDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def counterfactual_test(model, sample):

    x, _ = sample

    x = x.unsqueeze(0).to(DEVICE)

    original_pred = model(x)

    modified_x = x.clone()

    noise = torch.randn_like(modified_x) * 0.1

    modified_x += noise

    cf_pred = model(modified_x)

    diff = torch.norm(original_pred - cf_pred)

    print("Counterfactual difference:", diff.item())

pose_files = sorted([
    os.path.join("videos", f)
    for f in os.listdir("videos")
    if "pose" in f
])

emotion_files = sorted([
    os.path.join("videos", f)
    for f in os.listdir("videos")
    if "emotion" in f
])

dataset = FusionDataset(pose_files, emotion_files)
model = FusionPredictor().to(DEVICE)
model.load_state_dict(torch.load("fusion_model.pt"))

counterfactual_test(model, dataset[10])