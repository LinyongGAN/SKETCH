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
from utils.visualization import draw_single_traj_route
from utils.metrics import frechet_distance
from datetime import datetime, timedelta

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(data_path, model_sft_path, model_train_path, faiss_index_path, seq_len = 128, pred_len = 16):
    print('model initialization...')
    lm_config = MiniMindConfig(dropout=0.1)
    
    model = MiniMindFinal(lm_config, model_train_path, model_sft_path, faiss_index_path).to(device)
    model.eval()
    
    df = pd.read_csv(data_path)
    data_dict = dict(tuple(df.groupby("mmsi")))
    mmsi_list = list(data_dict.keys())
    
    total_len = seq_len + pred_len

    cnt = 0
    while cnt < 1:

        mmsi = random.choice(mmsi_list)
        coord = data_dict[mmsi][["lat", "lon", "sog", "cog"]].values # deg
        lat = deg_to_rad(coord[:,0]).astype(np.float32)
        lon = deg_to_rad(coord[:,1]).astype(np.float32)
        Sog = coord[:,2].astype(np.float32)
        Cog = deg_to_vec(coord[:,3]).astype(np.float32)
        
        s = random.randint(0, len(coord)-total_len)
        
        lat_window = lat[s:s+total_len, None]
        lon_window = lon[s:s+total_len, None]
        sog_window = Sog[s:s+total_len, None]
        cog_window = Cog[s:s+total_len]
        total_loc = np.concatenate((lat_window, lon_window, sog_window*cog_window), axis=1)
        
        while True:
        
            s = random.randint(0, len(coord)-total_len)
            lat_window = lat[s:s+total_len, None]
            lon_window = lon[s:s+total_len, None]
            sog_window = Sog[s:s+total_len, None]
            cog_window = Cog[s:s+total_len]
            total_loc = np.concatenate((lat_window, lon_window, sog_window*cog_window), axis=1)
            
            sog = data_dict[mmsi]["sog"].values # nmi/h
            cog = deg_to_rad(data_dict[mmsi]["cog"].values) # rad
            print(sog_window.mean())
            
            if 7 < sog_window.mean() < 19:  # 0-25节的合理速度范围
                break
            s = random.randint(0, len(coord)-total_len)
            
            lat_window = lat[s:s+total_len, None]
            lon_window = lon[s:s+total_len, None]
            sog_window = Sog[s:s+total_len, None]
            cog_window = Cog[s:s+total_len]
            
            total_loc = np.concatenate((lat_window, lon_window, sog_window*cog_window), axis=1)
            
            sog = data_dict[mmsi]["sog"].values # nmi/h
            cog = deg_to_rad(data_dict[mmsi]["cog"].values) # rad
            print(sog_window.mean())
        
        print("vessal mmsi:", mmsi)
        print("start index:", s)
        print("start lat, lon:", rad_to_deg(lat_window[0]), rad_to_deg(lon_window[0]))
        print("end lat, lon:", rad_to_deg(lat_window[-1]), rad_to_deg(lon_window[-1]))
        print("vessel average sog: ", sog_window.mean())
        
        total_points = total_len
        x_in_len = seq_len
        y_out_len = pred_len
        
        Y_in = torch.tensor(total_loc[:x_in_len], dtype=torch.float32).unsqueeze(0).to(device)  # [1, seq_len, 4]
        true_out = torch.tensor(total_loc[x_in_len:x_in_len+y_out_len], dtype=torch.float32).to(device)  # [pred_len, 4]
        
        predictions = [rad_to_deg(true_out[0, :2].cpu()).numpy()]
        current_input = Y_in.clone()  # rad
        
        pred_tgt = None
        for i in tqdm(range(y_out_len)):
            input_seq = current_input.clone()  # [1, seq_len, 4]
            
            with torch.no_grad():
                new_position, pred_vol_vec, displacement, pred_coord, pred_target = model(input_seq)
                if pred_tgt is None:
                    pred_tgt = pred_target.clone()
                
                last_point = new_position[0, -1, :].clone()  # [4]
                
                last_point[:2] = (displacement[0, -1, :] + current_input[0, -1, :2]).squeeze().clone()
            
            predictions.append(rad_to_deg(last_point[:2]).cpu().numpy())
            
            last_point = last_point[:4].clone()
            current_input = torch.cat((current_input, last_point.unsqueeze(0).unsqueeze(0)), dim=1)  # [1, seq_len+1, 4]
            
        predictions = np.array(predictions)  # [pred_len+1, 2]
        
        try:
            critria = torch.nn.MSELoss()
            pred_input = current_input[:,-pred_len:, :2].squeeze()
            true_target = true_out[:pred_len, :2]
            if pred_input.shape == true_target.shape:
                loss_pos = critria(pred_input, true_target)
                print(f"Position Loss: {loss_pos.item()}")
            
            if current_input.shape[1] >= pred_len and true_out.shape[0] >= pred_len:
                pred_vel = current_input[:,-pred_len:, 2:].squeeze()
                true_vel = true_out[:pred_len, 2:]
                if pred_vel.shape == true_vel.shape:
                    loss_vel = critria(pred_vel, true_vel)
                    print(f"Velocity Loss: {loss_vel.item()}")
            
            frechet = frechet_distance(rad_to_deg(pred_input), rad_to_deg(true_target))
            print("frechet distance:", frechet)
            
            true_target = rad_to_deg(true_target)
            print(true_target[0])
            
        except Exception as e:
            print(f"Error calculating loss: {e}")

        if True:
            cnt += 1
            print("found a valid trajectory")
        
            # 可视化结果
            plt.figure(figsize=(10, 6))
            
            # Input sequence
            Y_in = rad_to_deg(Y_in)
            plt.scatter(Y_in[0, :, 1].cpu(), Y_in[0, :, 0].cpu(), c='blue', label='Input Sequence')
            
            # True output
            true_out = rad_to_deg(true_out)
            plt.scatter(true_out[:, 1].cpu(), true_out[:, 0].cpu(), c='green', label='True Trajectory')
            
            # 预测轨迹
            plt.plot(predictions[:, 1], predictions[:, 0], 'r--', label='Predicted Trajectory')
            plt.scatter(predictions[:, 1], predictions[:, 0], c='red', marker='x')
            
            plt.title("Trajectory Prediction")
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"trajectory_prediction_{cnt}.png")
            plt.show()
            
            base_time = datetime.now()
            time_interval = timedelta(minutes=5)
            
            input_data = []
            for i in range(Y_in.shape[1]):
                input_data.append({
                    'lat': Y_in[0, i, 0].cpu().item(),
                    'lon': Y_in[0, i, 1].cpu().item(),
                    'postime': base_time + i * time_interval,
                    'type': 0,  # 输入序列显示为蓝色
                    'mmsi': mmsi
                })
            
            # 创建预测序列数据框（type=1）
            # 注意：predictions的第一个点是true_out的第一个点，我们从第二个点开始
            pred_data = []
            for i in range(1, len(predictions)):
                pred_data.append({
                    'lat': predictions[i, 0],
                    'lon': predictions[i, 1],
                    'postime': base_time + (Y_in.shape[1] + i - 1) * time_interval,
                    'type': 1,  # 预测序列显示为红色
                    'mmsi': mmsi
                })
            
            # 创建真实值序列数据框（type=2）
            true_data = []
            for i in range(len(true_out)):
                true_data.append({
                    'lat': true_out[i, 0].cpu().item(),
                    'lon': true_out[i, 1].cpu().item(),
                    'postime': base_time + (Y_in.shape[1] + i) * time_interval,
                    'type': 2,  # 真实值序列显示为绿色
                    'mmsi': mmsi
                })
            
            # 合并数据
            map_df = pd.DataFrame(input_data + pred_data + true_data)
            
            # 生成HTML地图文件
            html_output_path = f"trajectory_map_{cnt}.html"
            draw_single_traj_route(map_df, html_output_path)
            print(f"地图已保存到: {html_output_path}")

if __name__ == '__main__':
    evaluate(
            "./data_1_13/356285000.csv",
             "./weights_pretrain/830_statedict_0.16575811230219328.pth",
             "./weights_sft_new/476_statedict_0.004620137336290959.pth",
             "./enrolled_trajectory.npy",
             seq_len=288, pred_len=144)