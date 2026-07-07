import torch
from tqdm import tqdm
import os
import sys
from models.model_minimind import MiniMindConfig
from models.model_minimind_final import MiniMindFinal
sys.path.append(os.path.abspath(os.path.dirname('__file__')))
from utils.process import create_folder_if_not_exists
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd
import random
import matplotlib.pyplot as plt
from utils.earth_computation import rad_to_deg, deg_to_rad, deg_to_vec, haversine_distance
from utils.metrics import frechet_distance, curvature_calculation
from utils.dataloader_public import create_public_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import numpy as np
from math import sqrt

seed = 42

random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def evaluate(data_path, model_sft_path, model_train_path, faiss_index_path, seq_len = 128, pred_len = 16):
    print('model initialization...')
    lm_config = MiniMindConfig(dropout=0.1)
    
    model = MiniMindFinal(lm_config, model_train_path, model_sft_path, faiss_index_path).to(device)
    model.eval()

    train_dataloader, test_dataloader = create_public_dataloader(
        data_path=data_path,
        seq_len=seq_len,
        pred_len=pred_len,
        batch_size=128,
        test_prop=0.2,
        random_state=seed
    )
    # 初始化指标累加器
    avg_loss = 0.0
    afd = 0.0  # 总弗雷歇距离
    acvt = 0.0  # 总曲率差异
    cnt = 0  # 样本计数
    for batch in tqdm(test_dataloader):
        input_ids = batch["input_ids"].to(device)  # [1, seq_len, 4] - 已经是弧度
        output_ids = batch["output_ids"].to(device)  # [1, seq_len, 4]
        
        cnt += input_ids.size(0)
        
        true_out = output_ids[:, seq_len-1:seq_len-1+pred_len, :2]  # [pred_len, 2]
        
        Y_in = input_ids[:, :seq_len, :]  # [1, seq_len, 4]
        
        predictions = []
        current_input = Y_in.clone()

        for i in range(pred_len):
        
            input_seq = current_input.clone()  # [1, seq_len, 4]
        
            with torch.no_grad():
                new_position, pred_vol_vec, displacement, pred_coord, pred_tgt = model(input_seq)
                
                last_point = new_position[:, -1, :].clone()  # [4]
                
                last_point[:, :2] = (displacement[:, -1, :] + current_input[:, -1, :2]).squeeze().clone()
            
            predictions.append(rad_to_deg(last_point[:, :2]).cpu().numpy())
            
            last_point = last_point[:, :4].clone()
            # print(current_input.shape, last_point.unsqueeze(1).shape)
            current_input = torch.cat((current_input, last_point.unsqueeze(1)), dim=1)  # [bs, seq_len+1, 4]
            
        predictions = np.array(predictions)  # [pred_len+1, 2]
        
        pred_input = current_input[:,-min(pred_len, current_input.shape[1]):, :2].squeeze()
        true_target = true_out[:, -min(pred_len, current_input.shape[1]):, :2]
        assert pred_input.shape == true_target.shape
        
        msep_per_traj = ((pred_input - true_target) ** 2).sum(dim=2).mean(dim=1)

        avg_loss += msep_per_traj.sum().item()
        
        afd += frechet_distance(rad_to_deg(pred_input), rad_to_deg(true_target)).sum().item()
        
        acvt += torch.mean((curvature_calculation(pred_input) - curvature_calculation(true_target)) ** 2).item()
        
        true_target = rad_to_deg(true_target)
        
    for batch in tqdm(train_dataloader):
        input_ids = batch["input_ids"].to(device)  # [1, seq_len, 4] - 已经是弧度
        output_ids = batch["output_ids"].to(device)  # [1, seq_len, 4]
        
        cnt += input_ids.size(0)
        
        true_out = output_ids[:, seq_len-1:seq_len-1+pred_len, :2]  # [pred_len, 2]
        
        Y_in = input_ids[:, :seq_len, :]  # [1, seq_len, 4]
        
        predictions = []
        current_input = Y_in.clone()

        for i in range(pred_len):
        
            input_seq = current_input.clone()  # [1, seq_len, 4]
        
            with torch.no_grad():
                new_position, pred_vol_vec, displacement, pred_coord, pred_tgt = model(input_seq)
                
                last_point = new_position[:, -1, :].clone()  # [4]
                
                last_point[:, :2] = (displacement[:, -1, :] + current_input[:, -1, :2]).squeeze().clone()
            
            predictions.append(rad_to_deg(last_point[:, :2]).cpu().numpy())
            
            last_point = last_point[:, :4].clone()
        
            current_input = torch.cat((current_input, last_point.unsqueeze(1)), dim=1)  # [bs, seq_len+1, 4]
            
            
        predictions = np.array(predictions)  # [pred_len+1, 2]
        
        pred_input = current_input[:,-min(pred_len, current_input.shape[1]):, :2].squeeze()
        
        true_target = true_out[:, -min(pred_len, current_input.shape[1]):, :2]
        assert pred_input.shape == true_target.shape
        
        msep_per_traj = ((pred_input - true_target) ** 2).sum(dim=2).mean(dim=1)

        avg_loss += msep_per_traj.sum().item()

        afd += frechet_distance(rad_to_deg(pred_input), rad_to_deg(true_target)).sum().item()

        curv_pred = curvature_calculation(pred_input)
        curv_true = curvature_calculation(true_target)

        curv_mse_per_traj = ((curv_pred - curv_true) ** 2).mean(dim=1)
        acvt += curv_mse_per_traj.sum().item()
                
        true_target = rad_to_deg(true_target)
        
    print(f"共计 {cnt}条轨迹")
    print(f"Average Loss: {avg_loss/cnt}")   
    print(f"Average Frechet Distance: {afd/cnt}")
    print(f"Average Curvature: {acvt/cnt}")

if __name__ == '__main__':
    
    evaluate(
            ['./data_1_13/210238000.csv',
            './data_1_13/210279000.csv',
            './data_1_13/356285000.csv',
            './data_1_13/414062000.csv',
            './data_1_13/414066000.csv',
            './data_1_13/636015239.csv'], 
             "./weights_pretrain/830_statedict_0.16575811230219328.pth",
             "./weights_sft_new/476_statedict_0.004620137336290959.pth",
             "./enrolled_trajectory.npy",
             seq_len=288, pred_len=144)