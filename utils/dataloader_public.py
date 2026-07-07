from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import pandas as pd
from utils.earth_computation import deg_to_rad, deg_to_vec


class PublicDataset(Dataset):
    def __init__(self, data_path: str or list, seq_len: int = 288, pred_len: int = 144, stride: int = 7):
        
        self.samples = []
        
        if isinstance(data_path, str):
            data_paths = [data_path]
        else:
            data_paths = data_path
        
        required_columns = ["mmsi", "lat", "lon", "sog", "cog"]
        
        for file_path in data_paths:
            df = pd.read_csv(file_path)
            
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"column {col} not exists in file {file_path}")
            
            # group by mmsi
            for mmsi, group in tqdm(df.groupby("mmsi"), desc=f"Loading NOAA data from {file_path}..."):
                
                group = group.sort_values(by="postime")
                coord = group[["lat", "lon", "sog", "cog"]].values
                
                lat = deg_to_rad(coord[:, 0]).astype(np.float32)
                lon = deg_to_rad(coord[:, 1]).astype(np.float32)
                sog = coord[:, 2].astype(np.float32)
                cog = deg_to_vec(np.asarray(coord[:, 3], dtype=float)).astype(np.float32)
                
                N = len(coord)
                total_len = seq_len + pred_len
                
                for s in range(0, N - total_len + 1, stride):
                    lat_window = lat[s:s+total_len, None]
                    lon_window = lon[s:s+total_len, None]
                    sog_window = sog[s:s+total_len, None]
                    cog_window = cog[s:s+total_len]
                    
                    merging = np.concatenate((lat_window, lon_window, sog_window * cog_window), axis=1)
                    
                    self.samples.append([merging[:-1, :], merging[1:, :]])
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        print(f"Total public samples from {len(data_paths)} files: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_ids, labels = self.samples[idx]
        return {
            "input_ids": torch.from_numpy(input_ids).to(torch.float32),
            "output_ids": torch.from_numpy(labels).to(torch.float32),
        }


def create_public_dataloader(data_path: str or list,
                      world_size: int = 1,
                      rank: int = 0,
                      seq_len: int = 288,
                      pred_len: int = 144,
                      stride: int = 7,
                      test_prop: float = 0.2,
                      batch_size: int = 16,
                      random_state: int = 42):
    
    dataset = PublicDataset(data_path, seq_len, pred_len, stride)
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
    print("Public data loader created")
    
    if world_size == 1:
        return train_dataloader, test_dataloader
    else:
        return train_dataloader, train_sampler, test_dataloader
