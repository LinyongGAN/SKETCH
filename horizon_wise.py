import torch
from tqdm import tqdm
import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt


from utils.earth_computation import rad_to_deg
from utils.metrics import frechet_distance, curvature_calculation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def horizon_evaluation(pred_input, true_target, horizon_errors):
    
    for t in range(0, 144, 12):
        if t == 0:
            continue
        p_t_deg = rad_to_deg(pred_input[:, :t, :])
        gt_t_deg = rad_to_deg(true_target[:, :t, :])
        
        # 使用安全的 Haversine 计算
        # dist = torch_safe_haversine(p_t_deg, gt_t_deg)
        dist = frechet_distance(p_t_deg, gt_t_deg)
        horizon_errors[t] += torch.sum(dist).item()
    
    # 最后一个点
    p_t_deg = rad_to_deg(pred_input[:, :, :])
    gt_t_deg = rad_to_deg(true_target[:, :, :])
    
    # 使用安全的 Haversine 计算
    # dist = torch_safe_haversine(p_t_deg, gt_t_deg)
    dist = frechet_distance(p_t_deg, gt_t_deg)
    horizon_errors[-1] += torch.sum(dist).item()
