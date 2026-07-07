import torch
from tqdm import tqdm
import os
import sys
from models.model_minimind import MiniMindConfig, MiniMindForCausalLM
sys.path.append(os.path.abspath(os.path.dirname('__file__')))
from utils.process import create_dataloader, create_folder_if_not_exists
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import os
import numpy as np
from pathlib import Path
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)

def train(data_path, prev_path, learning_rate, batch_size, epoch_nums, weights_path):
    
    records = pd.DataFrame([], columns=["epoch num", "total loss", "validate loss",   "sog loss", "cog loss"])

    train_dataloader, test_dataloader = create_dataloader(data_path = data_path, batch_size = batch_size, seq_len=288, pred_len=144, stride=7)
    print('model initialization...')
    lm_config = MiniMindConfig(dropout=0.1)
    model = MiniMindForCausalLM(lm_config).to(device)
    
    loss_pos = torch.nn.MSELoss()
    loss_sog = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=3e-5)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=range(2, epoch_nums, 2), gamma=0.99)
    
    prev_batch = 0
    best_loss = 968978846685320
    if prev_path:
        records = pd.read_csv("./records.csv")
        checkpoint = torch.load(prev_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        prev_batch = int(prev_path.split("_")[0].split("/")[-1])
        best_loss = float(prev_path.split("_")[-1][:-4])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        print(f"model {prev_path} loaded, train from {prev_batch} with prev loss {best_loss}")

    total_params = sum(p.numel() for p in tqdm(model.parameters(), desc='caculating trainable parameters...') if p.requires_grad)
    print(f'###### total trainable parameter: {total_params}({(total_params/1000000):.3f}M) ######')
    
    for epoch_cur in range(prev_batch, epoch_nums):

        running_loss = 0.0
        step = 0
        loss1 = 0
        loss2 = 0

        model.train()
        for batch in tqdm(train_dataloader, desc=f"Epoch [{epoch_cur+1}/{epoch_nums}]"):
            
            X_coord = batch['input_ids'].to(device)
            output_ids = batch['output_ids'].to(device)
            target_ids = batch['target_ids'].to(device)

            Y_coord = output_ids[:,:,:2].clone().detach()
            gt_vol_vec = output_ids[:,:,2:].clone().detach()

            Y_pred, pred_vol_vec, displacement, logits = model(X_coord, target_ids, part = "train")
        
            loss = loss_pos(pred_vol_vec, gt_vol_vec)
        
            loss1 += loss_pos(Y_pred[:,:,:2], Y_coord).item()
            loss2 += loss_sog(pred_vol_vec, gt_vol_vec).item()
        
            optimizer.zero_grad()
            loss.backward()
            running_loss += loss.item()
            step += 1
            clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            
        running_loss /= step
        loss1 /= step
        loss2 /= step

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_step = 0
            for batch in tqdm(test_dataloader, desc="validation..."):
                X_coord = batch['input_ids'].to(device)
                output_ids = batch['output_ids'].to(device)
                target_ids = batch['target_ids'].to(device)

                Y_coord = output_ids[:,:,:2].clone().detach()
                gt_vol_vec = output_ids[:,:,2:].clone().detach()

                Y_pred, pred_vol_vec, displacement, logits = model(X_coord, target_ids, part = "inference")
                loss = loss_pos(pred_vol_vec, gt_vol_vec)
                val_loss += loss.item()
                val_step += 1
            val_loss /= val_step

        print(f'   Epoch {epoch_cur+1} training loss {running_loss} validating loss {val_loss} loss_pos {loss1}')
        current_lrs = scheduler.get_last_lr()
        print("Current learning rates:", [f"{lr:.2e}" for lr in current_lrs])
        if val_loss < best_loss:
            create_folder_if_not_exists(weights_path)
            torch.save({'model_state_dict': model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),},
                        Path(weights_path) / f'{epoch_cur+1}_statedict_{val_loss}.pth')
            best_loss = val_loss
        records.loc[len(records)] = [epoch_cur+1, running_loss, val_loss, loss1, loss2]
        if epoch_cur % 10 == 9:
            records.to_csv("./records.csv", index=False)
    

if __name__ == '__main__':
    train(
        "/path/to/dataset_final.csv", #[TODO] specify the dataset path 
         None,
         7e-5,
         256, 1000, './weight_pretrain')