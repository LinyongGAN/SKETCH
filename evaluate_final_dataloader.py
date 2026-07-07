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
from utils.process import create_final_dataloader
from horizon_wise import horizon_evaluation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import numpy as np
from math import sqrt

seed = 42

random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def remove_cross_180_traj(input_ids):

    lon = input_ids[:, :, 1]
    dlon = torch.diff(lon)

    cross_180 = torch.any(torch.abs(dlon) > math.pi, dim=1)

    keep_mask = ~cross_180

    return keep_mask


def evaluate(data_path, model_sft_path, model_train_path, faiss_index_path, seq_len = 128, pred_len = 16):
    print('model initialization...')
    lm_config = MiniMindConfig(dropout=0.1)
    
    model = MiniMindFinal(lm_config, model_train_path, model_sft_path, faiss_index_path).to(device)
    model.eval()

    _, test_dataloader = create_final_dataloader(
        data_path=data_path,
        seq_len=seq_len,
        pred_len=pred_len,
        batch_size=128,
        test_prop=0.2,
        random_state=seed
    )
    
    avg_loss = 0.0
    afd = 0.0
    acvt = 0.0
    horizon_errors = np.zeros(pred_len)  # horizon evaluation
    cnt = 0  # # of samples
    for batch in tqdm(test_dataloader):
        input_ids = batch["input_ids"].to(device)  # [1, seq_len, 4] - 已经是弧度
        output_ids = batch["output_ids"].to(device)  # [1, seq_len, 4]
        target_ids = batch["target_ids"].to(device)  # [1, seq_len, 2] - 目的港经纬度（弧度）

        mask = remove_cross_180_traj(output_ids)

        input_ids = input_ids[mask]
        output_ids = output_ids[mask]
        target_ids = target_ids[mask]
        
        cnt += input_ids.size(0)
        
        # 真实未来轨迹（弧度）
        true_out = output_ids[:, seq_len-1:seq_len-1+pred_len, :2]  # [pred_len, 2]
        
        # 从数据中获取目的港坐标（取最后一个时间步的target）
        dest_lat, dest_lon = target_ids[:, -1, 0], target_ids[:, -1, 1]
        
        # 构建四通道输入
        Y_in = input_ids[:, :seq_len, :]  # [1, seq_len, 4]
        
        # 级联推理
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
        
        # input: rad, will transfer to deg inside the function
        horizon_evaluation(pred_input, true_target, horizon_errors)

        true_target = rad_to_deg(true_target)
        
        pred_tgt[0, -1, 0] *= (math.pi/2)
        pred_tgt[0, -1, 1] *= math.pi
        pred_tgt = rad_to_deg(pred_tgt)
        
        distance = haversine_distance(true_target[0][0], true_target[0][1], pred_tgt[0, -1, 0], pred_tgt[0, -1, 1])
        
    print(f"共有{cnt}条轨迹")
    print(f"Average Loss: {avg_loss/cnt}")   
    print(f"Average Frechet Distance: {afd/cnt}")
    print(f"Average Curvature: {acvt/cnt}")

    horizon_avg_errors = horizon_errors / cnt
    ade = np.mean(horizon_avg_errors[horizon_avg_errors != 0])
    fde = horizon_avg_errors[-1]
    non_zero = [(i, v) for i, v in enumerate(horizon_avg_errors) if v != 0]
    print(non_zero)
    print(f"ADE (平均位移误差): {ade:.4f} km")
    print(f"FDE (终点位移误差): {fde:.9f} km")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, pred_len + 1), horizon_avg_errors, marker='o', markersize=3, color='#1f77b4', label='Step-wise Error')
    plt.axhline(y=ade, color='r', linestyle='--', label=f'ADE: {ade:.2f} km')
    plt.title('Prediction Error Growth over Horizon')
    plt.xlabel('Horizon Step')
    plt.ylabel('Distance Error (km)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('horizon_evaluation_fixed.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    evaluate(
            "/path/to/dataset_final.csv" # [TODO] specify the dataset path
             "./weights_pretrain/830_statedict_0.16575811230219328.pth",
             "./weights_sft_new/476_statedict_0.004620137336290959.pth",
             "./enrolled_trajectory.npy",
             seq_len=288, pred_len=144)