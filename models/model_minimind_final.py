from torch import nn
import torch
import torch.nn.functional as F
import math
import faiss
import numpy as np
from tqdm import tqdm
from utils.earth_computation import UnitConversion
from models.model_minimind import MiniMindForCausalLM, MiniMindConfig
from models.model_minimind_sft import MiniMindSFT
from typing import Optional, Tuple, List, Union
from utils.process import load_enrolled_data
from utils.earth_computation import deg_to_rad, rad_to_deg, deg_to_vec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MiniMindFinal(nn.Module):
    def __init__(self, config: MiniMindConfig, train_model_path: str, sft_model_path: str, enrolled_data_path: str):
        super(MiniMindFinal, self).__init__()
        
        self.sft_model = MiniMindSFT(config, train_model_path, part = "gathering")
        
        sft_state_dict = torch.load(sft_model_path)['model_state_dict']
        
        adjusted_sft_dict = {}
        for k, v in sft_state_dict.items():
            if not k.startswith("model.") and not k.startswith("DestPredHead"):
                adjusted_key = "model." + k
                adjusted_sft_dict[adjusted_key] = v
            else:
                adjusted_sft_dict[k] = v
        
        self.sft_model.load_state_dict(adjusted_sft_dict, strict=False)
        
        self.sft_model = self.sft_model.to(device)
        
        self.enrolled_hidden_norm, self.enrolled_labels = self.derive_enrolled_hidden_states(enrolled_data_path)

    def derive_enrolled_hidden_states(self, enrolled_data_path):
        enrolled_dataset = load_enrolled_data(data_path = enrolled_data_path)
        enrolled_traj = torch.from_numpy(enrolled_dataset[:,:,:4]).float().to(device)
        enrolled_labels = torch.from_numpy(enrolled_dataset[:,-1,4:]).float().to(device)
        self.sft_model.eval()
        with torch.no_grad():
            enrolled_hidden_states = self.sft_model(enrolled_traj, part = "sft")
        enrolled_hidden_mean = torch.mean(enrolled_hidden_states, dim=1)
        enrolled_hidden_norm = F.normalize(enrolled_hidden_mean, p=2, dim=1)
        return enrolled_hidden_norm, enrolled_labels

    def get_target_ids(self, hidden_states, top_k=5, target_ids=None):
        if target_ids is not None:
            print("target_ids is given")
            target = target_ids.clone()
            target[:, 0] /= (math.pi/3)
            target[:, 1] /= math.pi
            return {"predicted_labels":target}
        bs, _, _ = hidden_states.shape
        # return {"predicted_labels": torch.empty(bs, 2, device=device).uniform_(-1, 1)}
        hidden_mean = torch.mean(hidden_states, dim=1)
        hidden_norm = F.normalize(hidden_mean, p=2, dim=1)

        # 计算余弦相似度矩阵
        cosine_similarity_matrix = torch.mm(hidden_norm, self.enrolled_hidden_norm.T)
        
        # 找到每个样本最相似的前k个enrolled样本
        top_similarities, top_indices = torch.topk(cosine_similarity_matrix, k=top_k, dim=1)
        
        # 找到每个样本最相似的单个样本
        max_similarities, max_indices = torch.max(cosine_similarity_matrix, dim=1)
        
        predicted_labels = self.enrolled_labels[max_indices]
        predicted_labels[:, 0] /= (math.pi/3)
        predicted_labels[:, 1] /= math.pi
        
        result = {
            'cosine_similarity_matrix': cosine_similarity_matrix,  # [batch_size, n_enrolled]
            'top_similarities': top_similarities,  # [batch_size, top_k]
            'top_indices': top_indices,  # [batch_size, top_k]
            'max_similarities': max_similarities,  # [batch_size]
            'max_indices': max_indices,  # [batch_size]
            'predicted_labels': predicted_labels  # [batch_size, ...]
        }
        return result

    def displacement_calculation(self, coord, vol_vec):
        """
        coord: [batch_size, seq_len, 2], (lat, lon) in rad
        vol_vec: [batch_size, seq_len, 2], (sog in kn, cog in rad)
        return: displacement: [batch_size, seq_len, 2], (dlat in rad, dlon in rad)
        """
        time_interval = 5.0/60.0 # h
        EARTH_RADIUS = 6371/1.852 # nmi
        dlat = vol_vec[:,:,0] * time_interval / EARTH_RADIUS
        tmp = (coord[:,:,0] + vol_vec[:,:,0]*time_interval/EARTH_RADIUS) # final latitude
        dlon = vol_vec[:,:,1]/(vol_vec[:,:,0]+1e-6) * torch.log(torch.abs((1/torch.cos(tmp) \
                    + torch.tan(tmp)) / (1/torch.cos(coord[:,:,0]) + torch.tan(coord[:,:,0]))))
        
        displacement = torch.stack((dlat, dlon), dim = -1)
        return displacement.clone()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                part: str = "train",
                **args):
        
        
        coord = input_ids[:,:,:2].clone()
        vol_vec = input_ids[:,:,2:4].clone()
        displacement = self.displacement_calculation(coord, vol_vec)  # [batch_size, seq_len, 2]
        std_displacement = displacement.clone()
        std_displacement[:,:,0] /= math.pi/3
        std_displacement[:,:,1] /= math.pi

        std_input_ids, _ = UnitConversion(input_ids, None, fr="rad", to="std")

        hidden_states1, _ = self.sft_model.model.model1(
            hidden_states=self.sft_model.model.dropout1(self.sft_model.model.encoder1(std_input_ids)),
            seq_length=std_input_ids.shape[1],
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
            **args
        )

        hidden_states5 = self.sft_model.dropout(self.sft_model.DestPredHead(hidden_states1)) # [batch_size, seq_len, hidden_size]
        h5 = hidden_states5[:, -288:, :].clone()
        std_dest_coord = self.get_target_ids(h5)["predicted_labels"].unsqueeze(1).expand(-1, hidden_states5.shape[1], -1) # std

        hidden_states2 = self.sft_model.model.dropout2(self.sft_model.model.encoder2(std_dest_coord))
        
        hidden_states3, _ = self.sft_model.model.model2(
            hidden_states=self.sft_model.model.dense(torch.concat((hidden_states1, hidden_states2), dim=-1)),
            seq_length=std_input_ids.shape[1],
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
            **args
        )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.sft_model.model.decoder1(hidden_states3[:, slice_indices, :])
        
        logits[:,:,:2] = (std_input_ids[:,:,:2] + std_displacement).clone()
        logits, _ = UnitConversion(logits, None, fr="std", to="rad")
        
        pred_coord = logits[:,:,:2].clone()
        pred_vol_vec = logits[:,:,2:4].clone()
        
        new_position = logits.clone()
        
        return new_position, pred_vol_vec, displacement, pred_coord, std_dest_coord