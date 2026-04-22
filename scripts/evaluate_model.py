import torch
import numpy as np
from train_world_model_rollout import WorldModelPredictor
from train_fusion_predictor import dataset, test_loader, DEVICE

def mpjpe_denorm(pred, target, p_mean, p_std):
    p_mean_torch = torch.tensor(p_mean).view(1, 1, 33, 2).to(pred.device)
    p_std_torch = torch.tensor(p_std).view(1, 1, 33, 2).to(pred.device)
    
    # 反歸一化公式
    pred_raw = pred * p_std_torch + p_mean_torch
    target_raw = target * p_std_torch + p_mean_torch

    return torch.mean(torch.norm(pred_raw - target_raw, dim=-1))

def evaluate(model, loader, p_mean, p_std, device):
    model.eval()
    total_mpjpe = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            # World Model 執行自回歸滾動預測
            pred = model(x)
            
            error = mpjpe_denorm(pred, y, p_mean, p_std)
            total_mpjpe += error.item()
            total_samples += 1

    return total_mpjpe / total_samples

# --- 執行部分 ---

# 實例化模型並加載訓練好的世界模型權重
model = WorldModelPredictor().to(DEVICE)
model.load_state_dict(torch.load("world_model_fusion.pt")) 

# 提取歸一化參數
p_mean = dataset.p_mean.reshape(33, 2)
p_std = dataset.p_std.reshape(33, 2)

# 開始評估
score = evaluate(model, test_loader, p_mean, p_std, DEVICE)

print("-" * 30)
print(f"World Model Corrected MPJPE: {score:.4f}")
print(f"Learned Emotion Gate: {model.emotion_gate.item():.4f}")
print("-" * 30)

'''This paragraph is for train_fusion_predictor evaluate
import torch
import numpy as np
from train_fusion_predictor import model, test_loader, dataset, DEVICE

def mpjpe(pred, target, p_mean, p_std):
    p_mean_torch = torch.tensor(p_mean).view(1, 1, 33, 2).to(pred.device)
    p_std_torch = torch.tensor(p_std).view(1, 1, 33, 2).to(pred.device)
    
    pred_raw = pred * p_std_torch + p_mean_torch
    target_raw = target * p_std_torch + p_mean_torch

    return torch.mean(torch.norm(pred_raw - target_raw, dim=-1))

def evaluate(model, loader, p_mean, p_std, device):
    model.eval()
    total_mpjpe = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            
            error = mpjpe(pred, y, p_mean, p_std)
            total_mpjpe += error.item()
            total_samples += 1

    return total_mpjpe / total_samples

model.load_state_dict(torch.load("fusion_model.pt"))

p_mean = dataset.p_mean.reshape(33, 2)
p_std = dataset.p_std.reshape(33, 2)

score = evaluate(model, test_loader, p_mean, p_std, DEVICE)
print(f"Final Corrected MPJPE: {score:.4f}")
'''

'''This paragraph is for Train_baseline_model evaluate
from train_pose_baseline import PosePredictor as ModelClass 
model.load_state_dict(torch.load("pose_baseline_model.pt"))
def mpjpe_for_baseline(pred, target):
    return torch.mean(torch.norm(pred - target, dim=-1))
'''