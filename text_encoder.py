import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizerFast
from typing import List

class RobertaTextEncoder(nn.Module):
    def __init__(self, use_text_attention=True):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.output_dim = 768  # RoBERTa-base output dimension
        self.use_text_attention = use_text_attention

    def forward(self, texts: List[str]):
        # get offset mapping for attention
        tokenized = self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True
        )

        text_inputs = {
            key: value.to(next(self.parameters()).device)
            for key, value in tokenized.items()
            if key != "offset_mapping"  # Exclude offset_mapping from RoBERTa input
        }

        if self.use_text_attention:
            outputs = self.roberta(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"]
            )
            hidden_states = outputs["last_hidden_state"]  # (B, T, 768)
            attention_mask = text_inputs["attention_mask"]  # (B, T)
            offset_mapping = tokenized["offset_mapping"]  # (B, T, 2)   
            return hidden_states, attention_mask, offset_mapping
        else:
            outputs = self.roberta(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"]
            )["pooler_output"]  # (B, 768)
            return outputs

    def load_default_state_dict(self):
        pass
