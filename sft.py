import torch
from tqdm import tqdm
import os
import sys
from models.model_minimind import MiniMindConfig
from models.model_minimind_sft import MiniMindSFT
sys.path.append(os.path.abspath(os.path.dirname('__file__')))

import signal

def clear_cuda_memory():
    if torch.cuda.is_available():
        for _ in range(2):
            torch.cuda.empty_cache()
            dummy_tensor = torch.randn(1, device='cuda')
            del dummy_tensor
        
        torch.cuda.reset_max_memory_allocated()
        print(f"CUDA memory has been cleared. Current allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB, Reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    else:
        print("CUDA is not available.")

def setup_signal_handlers():
    def signal_handler(sig, frame):
        print(f'\n接收到信号 {sig}，正在清理CUDA内存...')
        clear_cuda_memory()
        os._exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

from utils.process import create_Contrastive_dataloader, create_folder_if_not_exists, TemporalContrastiveLoss, create_balanced_dataloader
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import os
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.autograd.set_detect_anomaly(True)

def train(data_path, prev_path, pretrain_path, learning_rate, batch_size, epoch_nums, weights_path, train_subset_ratio=1.0):

    records = pd.DataFrame([], columns=["epoch num", "train loss", "train accurance", "validate loss", "validate accurance"])

    train_dataloader, test_dataloader = create_Contrastive_dataloader(data_path = data_path, batch_size = batch_size, seq_len=288, pred_len=144, stride=7)
    print('model initialization...')
    lm_config = MiniMindConfig(dropout=0.1)
    model = MiniMindSFT(lm_config, pretrain_path, part = "sft").to(device)
    
    criterion = TemporalContrastiveLoss(margin=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=3e-5)
    
    prev_batch = 0
    best_loss = 968978846685320
    if prev_path:
        records = pd.read_csv("./records.csv")
        checkpoint = torch.load(prev_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        prev_batch = int(prev_path.split("_")[0].split("/")[-1])
        best_loss = float(prev_path.split("_")[-1][:-4])
        # optimizer.load_state_dict(checkpoint["optimizer_state"])
        # scheduler.load_state_dict(checkpoint["scheduler_state"])
        print(f"model {prev_path} loaded, train from {prev_batch} with prev loss {best_loss}")
    
    total_params = sum(p.numel() for p in tqdm(model.parameters(), desc='caculating trainable parameters...') if p.requires_grad)
    print(f'###### total trainable parameter: {total_params}({(total_params/1000000):.3f}M) ######')

    for epoch_cur in range(prev_batch, epoch_nums):

        running_loss = 0.0
        step = 0
        cum_score = 0.0
        cnt = 0
        current_accs = []
        current_losses = []
        model.train()
        
        if train_subset_ratio < 1.0:
            train_iter, subset_size = create_balanced_dataloader(
                train_dataloader, 
                subset_ratio=train_subset_ratio, 
                batch_size=batch_size
            )
            desc_suffix = f"(subset {subset_size}/{len(train_dataloader.dataset)})"
        else:
            train_iter = train_dataloader
            desc_suffix = ""
        
        pbar = tqdm(train_iter, desc=f"Epoch [{epoch_cur+1}/{epoch_nums}] | tra acc 0.0000 {desc_suffix}")
        for batch in pbar:
            X_coord1 = batch['input_ids1'].to(device)
            X_coord2 = batch['input_ids2'].to(device)
            labels = batch['label'].to(device)

            hidden1 = model(X_coord1, part="sft")  # [batch_size, seq_len, hidden_size]
            hidden2 = model(X_coord2, part="sft")  # [batch_size, seq_len, hidden_size]
            
            loss, score = criterion(hidden1, hidden2, labels)
            cum_score += score.sum().item()
            cnt += score.shape[0]
            
            optimizer.zero_grad()
            loss.backward()
            running_loss += loss.item()
            step += 1
            clip_grad_norm_(model.parameters(), max_norm=5.0)  # 梯度裁剪
            optimizer.step()
            scheduler.step()
            
            # 动态更新进度条描述
            current_acc = cum_score / cnt if cnt > 0 else 0
            pbar.set_description(f"Epoch [{epoch_cur+1}/{epoch_nums}] | tra acc {current_acc:.4f} | loss {running_loss/step:.4f}")
            current_accs.append(current_acc)
            current_losses.append(running_loss/step)
            
        running_loss /= step

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_step = 0
            val_cum_score = 0.0
            val_cnt = 0
            val_accs = []
            val_losses = []
            pbar = tqdm(test_dataloader, desc=f"validating | val acc 0.0000")
            for batch in pbar:
                
                X_coord1 = batch['input_ids1'].to(device)
                X_coord2 = batch['input_ids2'].to(device)
                labels = batch['label'].to(device)

                hidden1 = model(X_coord1, part = "sft")  # [batch_size, seq_len, hidden_size]
                hidden2 = model(X_coord2, part = "sft")  # [batch_size, seq_len, hidden_size]

                loss, score = criterion(hidden1, hidden2, labels)
                val_cum_score += score.sum().item()
                val_cnt += score.shape[0]
                val_loss += loss.item()
                val_step += 1

                current_acc = val_cum_score / val_cnt if val_cnt > 0 else 0
                pbar.set_description(f"validating | acc {current_acc:.4f} | loss {val_loss/val_step:.4f}")
                val_accs.append(current_acc)
                val_losses.append(val_loss/val_step)

            val_loss /= val_step
            val_acc = val_cum_score / val_cnt

        print(f'   Epoch {epoch_cur+1} training loss {running_loss} validating loss {val_loss} val acc {val_acc}')
        current_lrs = scheduler.get_last_lr()
        print("Current learning rates:", [f"{lr:.2e}" for lr in current_lrs])
        if val_loss < best_loss:
            create_folder_if_not_exists(weights_path)
            torch.save({'model_state_dict': model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),},
                        Path(weights_path) / f'{epoch_cur+1}_statedict_{running_loss}.pth')
            best_loss = val_loss
        records.loc[len(records)] = [epoch_cur+1, running_loss, current_accs[-1], val_loss, val_acc]
        records.to_csv("./records.csv", index=False)
    

if __name__ == '__main__':
    
    setup_signal_handlers()
    
    clear_cuda_memory()
    
    try:
        train(
            "/path/to/dataset_final.csv", # [TODO] specify the dataset path 
             None,
             "./weights_pretrain/830_statedict_0.16575811230219328.pth",
             1e-5,
             512, 1000, './weights_sft_new',
             train_subset_ratio=0.5)
    finally:
        clear_cuda_memory()