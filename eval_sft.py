import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.nn import functional as F
from models.model_minimind import MiniMindConfig
from models.model_minimind_sft import MiniMindSFT
from utils.process import create_dataloader, load_enrolled_data
from utils.earth_computation import haversine_distance

sys.path.append(os.path.abspath(os.path.dirname('__file__')))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_target_ids(hidden_norm, enrolled_hidden_norm, enrolled_labels, top_k=5):
    """
    计算当前批次隐藏状态与已注册隐藏状态的余弦相似度，并返回预测的目标ID
    
    Args:
        hidden_norm: 当前批次归一化隐藏状态 [batch_size, hidden_size]
        enrolled_hidden_norm: 已注册的归一化隐藏状态 [n_enrolled, hidden_size]
        enrolled_labels: 已注册的标签 [n_enrolled, ...]
        top_k: 返回前k个最相似的样本
    
    Returns:
        dict: 包含相似度计算结果和预测信息的字典
    """
    
    cosine_similarity_matrix = torch.mm(hidden_norm, enrolled_hidden_norm.T)
    
    top_similarities, top_indices = torch.topk(cosine_similarity_matrix, k=top_k, dim=1)
    
    max_similarities, max_indices = torch.max(cosine_similarity_matrix, dim=1)
    
    predicted_labels = enrolled_labels[max_indices]
    
    result = {
        'cosine_similarity_matrix': cosine_similarity_matrix,  # [batch_size, n_enrolled]
        'top_similarities': top_similarities,  # [batch_size, top_k]
        'top_indices': top_indices,  # [batch_size, top_k]
        'max_similarities': max_similarities,  # [batch_size]
        'max_indices': max_indices,  # [batch_size]
        'predicted_labels': predicted_labels  # [batch_size, ...]
    }
    
    return result

if __name__ == "__main__":
    test_data_path = "/path/to/dataset.csv" # [TODO]
    enrolled_data_path = "./enrolled_trajectory.npy"
    pretrain_path = "./weights_pretrain/830_statedict_0.16575811230219328.pth"
    sft_path = "./weights_sft_new/476_statedict_0.004620137336290959.pth"
    # dataloader for tested traj
    _, dataloader = create_dataloader(data_path = test_data_path)
    # dataloader for enrolled traj
    enrolled_dataset = load_enrolled_data(data_path = enrolled_data_path)
    enrolled_traj = torch.from_numpy(enrolled_dataset[:,:,:4]).float().to(device)
    enrolled_labels = torch.from_numpy(enrolled_dataset[:,-1,4:]).float().to(device)
    # calcualte the hidden states for faiss data
    model = MiniMindSFT(MiniMindConfig(), pretrain_path)
    model_dict = torch.load(sft_path)["model_state_dict"]
    model.load_state_dict(model_dict, strict=True)
    model.to(device)
    model.eval()
    with torch.no_grad():
        enrolled_hidden_states = model(enrolled_traj, part = "sft")
    enrolled_hidden_mean = torch.mean(enrolled_hidden_states, dim=1)
    enrolled_hidden_norm = F.normalize(enrolled_hidden_mean, p=2, dim=1)
    print(enrolled_hidden_norm.shape)
    # eval
    acc, cnt = 0, 0
    with torch.no_grad():
        for batch in tqdm(_):
            traj = batch["input_ids"][:,:288,:].to(device)
            dest_coord = batch["target_ids"][:,:288,:].to(device)

            hidden_states = model(traj, part = "sft")
            hidden_mean = torch.mean(hidden_states, dim=1)
            hidden_norm = F.normalize(hidden_mean, p=2, dim=1)
            result = get_target_ids(
                hidden_norm=hidden_norm,
                enrolled_hidden_norm=enrolled_hidden_norm,
                enrolled_labels=enrolled_labels,
                top_k=5
            )
            
            distance = haversine_distance(result['predicted_labels'][:, 0]*60, result['predicted_labels'][:, 1]*180, dest_coord[:, -1, 0]*60, dest_coord[:, -1, 1]*180)
            acc += (distance < 1).float().sum()
            cnt += distance.shape[0]
    
    print(f"Accuracy: {acc / cnt:.4f}")
