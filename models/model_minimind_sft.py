from torch import nn
import torch
from models.model_minimind import MiniMindForCausalLM, MiniMindConfig, MiniMindModel
from typing import Optional, Tuple, List, Union

class MiniMindSFT(nn.Module):
    def __init__(self, config: MiniMindConfig, pretrain_path: str, part = "train"):
        super(MiniMindSFT, self).__init__()
        self.model = MiniMindForCausalLM(config)
        self.DestPredHead = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.dropout = nn.Dropout(0.5)
        for param in self.model.parameters():
            param.requires_grad = not(part == "sft")
        for param in self.DestPredHead.parameters():
            param.requires_grad = True
        
        if pretrain_path != None:
            pretrained_dict = torch.load(pretrain_path)["model_state_dict"]
            
            adjusted_dict = {}
            for k, v in pretrained_dict.items():
                
                if k.startswith("sft_model."):
                    adjusted_key = k[10:]  # remove "sft_model."
                elif k.startswith("model."):
                    adjusted_key = k[6:]  # remove "model."
                else:
                    adjusted_key = k
                adjusted_dict[adjusted_key] = v
            
            filtered_dict = {k: v for k, v in adjusted_dict.items() if not k.startswith('DestPredHead')}
            
            final_dict = {}
            for k, v in filtered_dict.items():
                if k.startswith("model."):
                    final_key = k[6:]  # remove "model."
                else:
                    final_key = k
                final_dict[final_key] = v
            
            self.model.load_state_dict(final_dict, strict=True)
        
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                dest_coord: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                part = "train",
                **args):

        hidden_states1 = self.model(input_ids, dest_coord, part = part)

        hidden_states5 = self.dropout(self.DestPredHead(hidden_states1)) # [batch_size, seq_len, hidden_size]
        return hidden_states5 # [batch_size, seq_len, hidden_size]