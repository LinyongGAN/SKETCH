from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset, random_split
from collections import Counter
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import os
from torch.autograd import Variable
import pandas as pd
import csv
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
from utils.earth_computation import deg_to_rad, rad_to_deg, deg_to_vec
import faiss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 创建成功")

def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask==0))
    return np_mask


def read_file(src_path):
    text_list = []
    with open(src_path, 'r', encoding='utf-8') as reader:
        for line in reader:
            text_list.append(line.strip().replace('\n',''))
    return text_list

def haversine(lat1, lon1, lat2, lon2):
    
    # Haversine
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # 地球半径（千米）
    r = 6371
    return c * r
 
def calculate_relative_position(current_lat, current_lon, dest_lat, dest_lon):
    """
    计算相对位置和方向
    
    参数:
    current_lat, current_lon: 当前位置纬度/经度（标量或数组）
    dest_lat, dest_lon: 目标位置纬度/经度（标量或数组）
    
    返回:
    direction: 从当前点到目标点的方位角（弧度），范围[-π, π]
    distance: 两点之间的距离（与输入形状相同的数组）
    """
    # 计算纬度差和经度差
    dlat = dest_lat - current_lat
    dlon = dest_lon - current_lon
    
    # 计算方向（方位角）
    direction = np.arctan2(dlon, dlat)  # 弧度制，范围[-π, π]
    
    # 计算距离
    distance = haversine(current_lat, current_lon, dest_lat, dest_lon)
    
    return direction, distance


class TemporalContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, temperature=0.07, similarity='cosine'):
        super(TemporalContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        self.threshold = 0.5
        self.similarity = similarity  # 'cosine' or 'euclidean'
        
    def forward(self, h1, h2, label):
        
        # 对时间维度取平均，得到序列级别的表示
        h1_avg = torch.mean(h1, dim=1)  # [batch_size, hidden_size]
        h2_avg = torch.mean(h2, dim=1)  # [batch_size, hidden_size]
        if self.similarity == 'cosine':
            # 计算余弦相似度
            h1_norm = F.normalize(h1_avg, p=2, dim=1)
            h2_norm = F.normalize(h2_avg, p=2, dim=1)
            similarity = torch.sum(h1_norm * h2_norm, dim=1)  # [batch_size]
            score = (similarity>self.threshold).to(torch.int32) ^ label.to(torch.int32)
            distance = 1 - similarity
        else:  # euclidean
            # 计算欧氏距离
            distance = F.pairwise_distance(h1_avg, h2_avg)
        
        # 对比损失计算
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(distance, 2) +
            label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        )
        
        return loss_contrastive, score

class TrajectoryRegularDataset(Dataset):
    def __init__(self, data_path: str, seq_len: int = 128, pred_len: int = 16, stride: int = 8):

        df = pd.read_csv(data_path)
        self.samples = []

        required_columns = ["mmsi", "lat", "lon", "sog", "cog", "coastline_distance"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"column {col} not exists")
            
        # 按照mmsi分组
        for mmsi, group in df.groupby("mmsi"):
            coord = group[["lat", "lon", "sog", "cog", "coastline_distance", "next_lat", "next_lon"]].values

            lat = deg_to_rad(coord[:,0]).astype(np.float32)
            lon = deg_to_rad(coord[:,1]).astype(np.float32)
            sog = coord[:,2].astype(np.float32)
            cog = deg_to_vec(coord[:,3]).astype(np.float32)
            dist = coord[:,4].astype(np.float32)
            
            target_lat = deg_to_rad(coord[:,5]).astype(np.float32)
            target_lon = deg_to_rad(coord[:,6]).astype(np.float32)
            
            N = len(coord)
            total_len = seq_len + pred_len

            # 滑动窗口取样
            for s in range(0, N-total_len+1, stride):
                lat_window = lat[s:s+total_len, None]
                lon_window = lon[s:s+total_len, None]
                sog_window = sog[s:s+total_len, None]
                cog_window = cog[s:s+total_len]
                dist_window = dist[s:s+total_len]

                # 1 拼接经纬度
                target_lat_window = target_lat[s:s+total_len, None]
                target_lon_window = target_lon[s:s+total_len, None]
                
                if sog_window.min() > 1.5 and dist_window.min() > 80000:
                    merging = np.concatenate((lat_window, lon_window, sog_window*cog_window), axis=1)
                    target_dim = np.concatenate((target_lat_window, target_lon_window), axis=1)
                    self.samples.append([merging[:-1, :], merging[1:, :], target_dim[:-1, :]])

        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, labels, target = self.samples[idx]
        return {
            "input_ids": torch.from_numpy(input_ids).to(torch.float32),
            "output_ids": torch.from_numpy(labels).to(torch.float32),
            "target_ids": torch.from_numpy(target).to(torch.float32)
        }

def create_dataloader(data_path: str,
                      world_size: int = 1,
                      rank: int = 0,
                      seq_len: int = 288,
                      pred_len: int = 144,
                      stride: int = 8,
                      test_prop: float = 0.2,
                      batch_size: int = 16,
                      random_state: int = 42):
    
    dataset = TrajectoryRegularDataset(data_path, seq_len, pred_len, stride)
    idxs = list(range(len(dataset)))
    
    train_idx, test_idx = train_test_split(
        idxs, test_size=test_prop, random_state=random_state
    )
    train_set = torch.utils.data.Subset(dataset, train_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)
    if world_size == 1:
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    else:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        
        train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4*world_size,
            pin_memory=True
        )

    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    if world_size == 1:
        return train_dataloader, test_dataloader
    else:
        return train_dataloader, train_sampler, test_dataloader

class TrajectoryRegularDataset_Contrastive(Dataset):
    def __init__(self, data_path: str, seq_len: int = 128, pred_len: int = 16, stride: int = 8):

        df = pd.read_csv(data_path)
        self.samples = {}
        num_sample = 0
        
        required_columns = ["mmsi", "lat", "lon", "sog", "cog", "coastline_distance"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"column {col} not exists")
            
        for mmsi, group in tqdm(df.groupby("mmsi"), desc="Loading data..."):
            coord = group[["lat", "lon", "sog", "cog", "coastline_distance", "next_code"]].values

            lat = deg_to_rad(coord[:,0]).astype(np.float32)
            lon = deg_to_rad(coord[:,1]).astype(np.float32)
            sog = coord[:,2].astype(np.float32)
            cog = deg_to_vec(np.asarray(coord[:,3], dtype=float)).astype(np.float32)
            dist = coord[:,4].astype(np.float32)
            
            next_code = coord[:,5]
            
            N = len(coord)
            
            for s in range(0, N-seq_len+1, stride):
                lat_window = lat[s:s+seq_len, None]
                lon_window = lon[s:s+seq_len, None]
                sog_window = sog[s:s+seq_len, None]
                cog_window = cog[s:s+seq_len]
                dist_window = dist[s:s+seq_len]
                next_code_window = next_code[s:s+seq_len]

                
                if sog_window.min() > 1.5 and dist_window.min() > 80000 and len(np.unique(next_code_window))==1: # sog > 1.5
                    if next_code_window[0] not in self.samples:
                        self.samples[next_code_window[0]] = []
                    merging = np.concatenate((lat_window, lon_window, sog_window*cog_window), axis=1)
                    self.samples[next_code_window[0]].append(merging)
                    num_sample += 1

        self.seq_len = seq_len
        self.pred_len = pred_len
        print(f"Total {num_sample} samples from {len(self.samples)} different targets.")
        self.pairs = self._make_pairs()

    def _make_pairs(self):
        print("######## dataset report ########")
        pairs = []
        cnt1, cnt2 = 0, 0
        stride1 = 15
        for target_dim in self.samples:
            trajs = self.samples[target_dim]
            for i in range(0, len(trajs), stride1):
                for j in range(i+1, len(trajs), stride1):
                    pairs.append((trajs[i], trajs[j], 0))
                    cnt1 += 1
        print(f"Same target pairs: {cnt1}")
        
        stride2 = 35
        for i, tar1 in enumerate(self.samples):
            for j, tar2 in enumerate(self.samples):
                if i < j:
                    traj1 = self.samples[tar1]
                    traj2 = self.samples[tar2]
                    for x in range(0, len(traj1), stride2):
                        for y in range(0, len(traj2), stride2):
                            pairs.append((traj1[x], traj2[y], 1))
                            cnt2 += 1

        print(f"Different target pairs: {cnt2}")
        print(f"Total pairs: {len(pairs)}")
        print("################################")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_ids1, input_ids2, label = self.pairs[idx]
        return {
            "input_ids1": torch.from_numpy(input_ids1).to(torch.float32),
            "input_ids2": torch.from_numpy(input_ids2).to(torch.float32),
            "label": torch.tensor(label, dtype=torch.float32)
        }

def create_balanced_dataloader(original_dataloader, subset_ratio=1.0, batch_size=None):
    """
    创建一个只使用原始数据一部分的DataLoader，用于均衡训练
    
    参数:
        original_dataloader: 原始的DataLoader对象
        subset_ratio: 使用的数据比例，范围[0, 1]
        batch_size: 可选，新的batch_size，如果为None则使用原始的batch_size
    
    返回:
        新的DataLoader对象，只包含随机选择的一部分数据
    """
    if subset_ratio >= 1.0:
        return original_dataloader
    
    # 确定要使用的批次数量
    subset_size = int(len(original_dataloader.dataset) * subset_ratio)
    
    # 创建随机采样器
    subset_indices = torch.randperm(len(original_dataloader.dataset))[:subset_size]
    subset_sampler = torch.utils.data.SubsetRandomSampler(subset_indices)
    
    # 确定batch_size
    if batch_size is None:
        batch_size = original_dataloader.batch_size
    
    # 创建新的DataLoader
    subset_dataloader = torch.utils.data.DataLoader(
        original_dataloader.dataset,
        batch_size=batch_size,
        sampler=subset_sampler,
        shuffle=False,  # 已经通过sampler实现了shuffle
        num_workers=original_dataloader.num_workers if hasattr(original_dataloader, 'num_workers') else 0,
        pin_memory=original_dataloader.pin_memory if hasattr(original_dataloader, 'pin_memory') else False
    )
    
    return subset_dataloader, subset_size

def create_Contrastive_dataloader(data_path: str,
                      world_size: int = 1,
                      rank: int = 0,
                      seq_len: int = 128,
                      pred_len: int = 16,
                      stride: int = 8,
                      test_prop: float = 0.2,
                      batch_size: int = 16,
                      random_state: int = 42):
    
    dataset = TrajectoryRegularDataset_Contrastive(data_path, seq_len, pred_len, stride)
    idxs = list(range(len(dataset)))
    
    train_idx, test_idx = train_test_split(idxs, test_size=test_prop, random_state=random_state)
    train_set = torch.utils.data.Subset(dataset, train_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)
    if world_size == 1:
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    else:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        
        train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4*world_size,
            pin_memory=True
        )
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    print("data loader created")
    if world_size == 1:
        return train_dataloader, test_dataloader
    else:
        return train_dataloader, train_sampler, test_dataloader

def load_enrolled_data(data_path, batch_size: int = 16):
    enrolled_data = np.load(data_path)
    return enrolled_data

# 第三阶段训练数据集
class TrajectoryRegularDataset_Final(Dataset):
    def __init__(self, data_path: str, seq_len: int = 288, pred_len: int = 144, stride: int = 7):
        
        df = pd.read_csv(data_path)
        self.samples = []
        
        required_columns = ["mmsi", "lat", "lon", "sog", "cog", "coastline_distance", "next_lat", "next_lon", "next_code"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"column {col} not exists")
        
        # 按照mmsi分组
        for mmsi, group in tqdm(df.groupby("mmsi"), desc="Loading final training data..."):
            coord = group[["lat", "lon", "sog", "cog", "coastline_distance", "next_lat", "next_lon", "next_code"]].values
            
            lat = deg_to_rad(coord[:,0]).astype(np.float32)
            lon = deg_to_rad(coord[:,1]).astype(np.float32)
            sog = coord[:,2].astype(np.float32)
            cog = deg_to_vec(np.asarray(coord[:,3], dtype=float)).astype(np.float32)
            dist = coord[:,4].astype(np.float32)
            
            # 目的港经纬度
            target_lat = deg_to_rad(coord[:,5]).astype(np.float32)
            target_lon = deg_to_rad(coord[:,6]).astype(np.float32)
            
            # 下一个关键节点编码
            next_code = coord[:,7]
            
            N = len(coord)
            total_len = seq_len + pred_len
            
            # 滑动窗口取样
            for s in range(0, N-total_len+1, stride):
                lat_window = lat[s:s+total_len, None]
                lon_window = lon[s:s+total_len, None]
                sog_window = sog[s:s+total_len, None]
                cog_window = cog[s:s+total_len]
                dist_window = dist[s:s+total_len]
                
                target_lat_window = target_lat[s:s+total_len, None]
                target_lon_window = target_lon[s:s+total_len, None]
                next_code_window = next_code[s:s+total_len]
                
                if 18 <= sog_window.mean() <= 19 and dist_window.min() > 80000:
                
                    merging = np.concatenate((lat_window, lon_window, sog_window*cog_window), axis=1)
                
                    target_dim = np.concatenate((target_lat_window, target_lon_window), axis=1)
                    
                    last_key_node = np.array([target_lat_window[-1], target_lon_window[-1]]).squeeze()
                    
                    self.samples.append([merging[:-1, :], merging[1:, :], target_dim[:-1, :], last_key_node])
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        print(f"Total final training samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_ids, labels, target, last_key_node = self.samples[idx]
        return {
            "input_ids": torch.from_numpy(input_ids).to(torch.float32),
            "output_ids": torch.from_numpy(labels).to(torch.float32),
            "target_ids": torch.from_numpy(target).to(torch.float32),
            "last_key_node": torch.from_numpy(last_key_node).to(torch.float32)
        }

# 第三阶段训练dataloader
def create_final_dataloader(data_path: str,
                      world_size: int = 1,
                      rank: int = 0,
                      seq_len: int = 288,
                      pred_len: int = 144,
                      stride: int = 7,
                      test_prop: float = 0.2,
                      batch_size: int = 16,
                      random_state: int = 42):
    
    dataset = TrajectoryRegularDataset_Final(data_path, seq_len, pred_len, stride)
    idxs = list(range(len(dataset)))
    
    # 划分训练集和测试集
    train_idx, test_idx = train_test_split(
        idxs, test_size=test_prop, random_state=random_state
    )
    train_set = torch.utils.data.Subset(dataset, train_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)
    
    if world_size == 1:
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    else:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4*world_size,
            pin_memory=True
        )
    
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    print("Final data loader created")
    
    if world_size == 1:
        return train_dataloader, test_dataloader
    else:
        return train_dataloader, train_sampler, test_dataloader
