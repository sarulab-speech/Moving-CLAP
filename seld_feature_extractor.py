import torch
import torch.nn as nn
import torch.nn.functional as F
from beats_seldnet_model import BEATsSELDnetModel

class SELDFeatureExtractorPretrained(nn.Module):
    def __init__(self, seldnet_path=None, freeze_encoder=True, 
                 pretrained_model_name='BEATs', feature_dim=256):
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.feature_dim = feature_dim
        self.pretrained_model_name = pretrained_model_name
        
        self.seld_pretrained = BEATsSELDnetModel(
            feature_dim=feature_dim,
            pretrained_model_name=pretrained_model_name,
            freeze_pretrained=freeze_encoder
        )
        
        # if checkpoint path is provided, load weights
        if seldnet_path:
            checkpoint = torch.load(seldnet_path, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # only load DOA encoder weights
            encoder_state = {k: v for k, v in state_dict.items() 
                           if k.startswith('feature_extractor_doa.') or k.startswith('encoder_doa.')}
            
            if encoder_state:
                self.seld_pretrained.load_state_dict(encoder_state, strict=False)
                print(f"Loaded SELD-Pretrained DOA encoder from {seldnet_path}")

        self.feature_extractor_doa = self.seld_pretrained.feature_extractor_doa
        self.encoder_doa = self.seld_pretrained.encoder_doa
        self.pretrained_encoder = self.seld_pretrained.pretrained_encoder
        
        # Output dimension
        self.output_dim = feature_dim + self.seld_pretrained.sed_features_dim
        
        # Freeze DOA encoder if specified
        if freeze_encoder:
            for param in self.feature_extractor_doa.parameters():
                param.requires_grad = False
            for param in self.encoder_doa.parameters():
                param.requires_grad = False
            print("🔒 DOA encoder frozen")
    
    def forward(self, x):
        # --- DOA Branch ---
        features_doa = self.feature_extractor_doa(x)  # (B, T, F, C)
        encoded_doa = self.encoder_doa(features_doa)  # (B, T, 256)
        
        # --- BEATs Branch ---
        x_mono = x.mean(dim=1)  # (B, T)
        
        enable_grad = self.pretrained_encoder.training and any(
            p.requires_grad for p in self.pretrained_encoder.parameters()
        )
        with torch.set_grad_enabled(enable_grad):
            mel = self.pretrained_encoder.mel_forward(x_mono)
            embeddings = self.pretrained_encoder.model(mel)  # (B, T_beats, 768)
            if embeddings.size(-2) > self.pretrained_encoder.seq_len:
                embeddings = F.adaptive_avg_pool1d(
                    embeddings.transpose(1, 2), self.pretrained_encoder.seq_len
                ).transpose(1, 2)
        
        # --- Feature Alignment ---
        target_time = encoded_doa.shape[1]
        
        if embeddings.shape[1] != target_time:
            embeddings = embeddings.transpose(1, 2)
            embeddings = F.interpolate(
                embeddings, size=target_time, mode='linear', align_corners=False
            )
            embeddings = embeddings.transpose(1, 2)
        
        # --- Fusion ---
        fused_features = torch.cat([encoded_doa, embeddings.detach()], dim=-1)
        
        return fused_features
    
    def load_default_state_dict(self):
        pass