import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
import os
from text_encoder import RobertaTextEncoder
from seld_feature_extractor import SELDFeatureExtractorPretrained

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, mel_dim=768, spatial_dim=256, hidden_dim=768, num_heads=8):
        super().__init__()

        # Multi-head attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            kdim=spatial_dim,
            vdim=spatial_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def _project(self, features, proj_layer):
        return proj_layer(features) if proj_layer is not None else features
    
    def forward(self, mel_features, spatial_features):
        mel_proj = mel_features
        spatial_proj = spatial_features
        
        query = mel_proj
        key = spatial_proj
        value = spatial_proj
        residual = mel_proj
        
        # Cross attention
        attn_out, _ = self.cross_attn(query=query, key=key, value=value)
        
        # Residual + Norm
        fused = self.norm1(residual + attn_out)
        
        # FFN
        ffn_out = self.ffn(fused)
        output = self.norm2(fused + ffn_out)
        
        return output

class PMA(nn.Module):
    def __init__(self, hidden_dim, num_queries=8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim))
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * num_queries, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_queries),
        )
    
    def forward(self, x):
        B, T, D = x.shape
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        
        attn_scores = torch.bmm(queries, x.transpose(1, 2)) / (D ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attended = torch.bmm(attn_weights, x)
        
        gate_input = attended.flatten(1)
        gate_logits = self.gate(gate_input)
        gate_weights = F.softmax(gate_logits, dim=-1).unsqueeze(-1)
        
        output = (attended * gate_weights).sum(dim=1)
        return output, attn_weights

class MovingCLAP(nn.Module):
    def __init__(self, 
                 joint_embed_shape=512,
                 seldnet_path=None,
                 freeze_spatial_encoder=True,
                 freeze_pretrained_encoders=True,
                 hidden_dim=768,
                 pretrained_model_name='BEATs',
                 fusion_type='cross_attn',
                 use_text_attention=True,
                 spatial_bias_value=5.0,
                 pooling_type='adaptive'):
        super().__init__()
        
        self.freeze_spatial_encoder = freeze_spatial_encoder
        self.freeze_pretrained_encoders = freeze_pretrained_encoders
        self.fusion_type = fusion_type
        self.pooling_type = pooling_type
        self.use_text_attention = use_text_attention
        self.spatial_bias_value = spatial_bias_value
        
        self.spatial_encoder = SELDFeatureExtractorPretrained(
            seldnet_path=seldnet_path,
            freeze_encoder=freeze_spatial_encoder,
            pretrained_model_name=pretrained_model_name,
        )
        spatial_dim = 256
        beats_dim = 768
        self.spatial_dim = spatial_dim
        self.resampler = None
        
        if fusion_type == 'none':
            self.fusion = None
            feature_dim = beats_dim + spatial_dim
            self.concat_proj = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            feature_dim = hidden_dim
        else:
            self.fusion = MultiHeadCrossAttention(
                mel_dim=beats_dim,
                spatial_dim=spatial_dim,
                hidden_dim=hidden_dim,
                num_heads=8,
            )
            feature_dim = hidden_dim
        
        if pooling_type == 'adaptive':
            self.adaptive_pool = PMA(
                hidden_dim=feature_dim,
                num_queries=8
            )
        else:
            self.adaptive_pool = None
        
        self.audio_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, joint_embed_shape),
            nn.LayerNorm(joint_embed_shape)
        )
        
        # Text encoder (RoBERTa)
        self.text_encoder = RobertaTextEncoder(use_text_attention=use_text_attention)
        text_input_dim = self.text_encoder.output_dim  # 768
        
        if use_text_attention:
            self.doa_phrases = [
                "right side", "left side", "front-left", "front-right", "front",
            ]
            self.text_attention = nn.Sequential(
                nn.Linear(text_input_dim, text_input_dim),
                nn.Tanh(),
                nn.Linear(text_input_dim, 1)
            )
        else:
            self.doa_phrases = None
            self.text_attention = None
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, joint_embed_shape),
            nn.LayerNorm(joint_embed_shape)
        )
        
        self.logit_scale_clap = nn.Parameter(torch.tensor(np.log(1/0.07)))
        self.logit_scale_spatial = nn.Parameter(torch.tensor(np.log(1/0.07)))
        self.logit_scale_reversal = nn.Parameter(torch.tensor(np.log(1/0.07)))
        self.logit_scale = self.logit_scale_clap
        
        if freeze_pretrained_encoders:
            self._freeze_pretrained_encoders()
    
    def _freeze_pretrained_encoders(self):        
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def unfreeze_beats_top_layers(self, num_layers=2):
        """Unfreeze top layers of BEATs for fine-tuning."""
        
        pretrained_encoder = self.spatial_encoder.pretrained_encoder        
        beats_model = pretrained_encoder.model
        if hasattr(beats_model, 'blocks'):
            blocks = beats_model.blocks
            total_blocks = len(blocks)
            unfrozen_count = 0
            for i in range(max(0, total_blocks - num_layers), total_blocks):
                for param in blocks[i].parameters():
                    param.requires_grad = True
                    unfrozen_count += 1
            print(f"   Total blocks: {total_blocks}, Unfrozen: layers {max(0, total_blocks - num_layers)}-{total_blocks-1}")
        elif hasattr(beats_model, 'encoder') and hasattr(beats_model.encoder, 'layers'):
            layers = beats_model.encoder.layers
            total_layers = len(layers)
            unfrozen_count = 0
            for i in range(max(0, total_layers - num_layers), total_layers):
                for param in layers[i].parameters():
                    param.requires_grad = True
                    unfrozen_count += 1
            print(f"   Total layers: {total_layers}, Unfrozen: layers {max(0, total_layers - num_layers)}-{total_layers-1}")
        elif hasattr(beats_model, 'beats'):
            inner_beats = beats_model.beats
            if hasattr(inner_beats, 'blocks'):
                blocks = inner_beats.blocks
                total_blocks = len(blocks)   
                unfrozen_count = 0
                for i in range(max(0, total_blocks - num_layers), total_blocks):
                    for param in blocks[i].parameters():
                        param.requires_grad = True
                        unfrozen_count += 1
                print(f"   Total blocks: {total_blocks}, Unfrozen: layers {max(0, total_blocks - num_layers)}-{total_blocks-1}")
            elif hasattr(inner_beats, 'encoder') and hasattr(inner_beats.encoder, 'layers'):
                layers = inner_beats.encoder.layers
                total_layers = len(layers)
                unfrozen_count = 0
                for i in range(max(0, total_layers - num_layers), total_layers):
                    for param in layers[i].parameters():
                        param.requires_grad = True
                        unfrozen_count += 1
                print(f"   Total layers: {total_layers}, Unfrozen: layers {max(0, total_layers - num_layers)}-{total_layers-1}")
            else:
                print(f"   Available in beats_model.beats: {dir(inner_beats)}")   
        else:
            print(f"   Available attributes: {dir(beats_model)}")
    
    def get_trainable_params_summary(self):
        total_params = 0
        trainable_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"   Total: {total_params:,}")
        print(f"   Trainable: {trainable_params:,} ({100.0 * trainable_params / total_params:.2f}%)")
        print(f"   Frozen: {total_params - trainable_params:,} ({100.0 * (total_params - trainable_params) / total_params:.2f}%)")
        
        return {'total': total_params, 'trainable': trainable_params}
    
    def load_default_state_dict(self):
        self.spatial_encoder.load_default_state_dict()
        self.text_encoder.load_default_state_dict()
    
    def encode_audio(self, audio, return_temporal=False):
        combined_features = self.spatial_encoder(audio)  # (B, T, 256+768)
        spatial_temporal = combined_features[..., :256]  # (B, T, 256)
        semantic_temporal = combined_features[..., 256:]  # (B, T, 768)
        
        if self.fusion is not None:
            fused_features = self.fusion(semantic_temporal, spatial_temporal)  # (B, T, hidden_dim)
        else:
            # fusion_type == 'none': Concat + Projection
            concat_features = torch.cat([semantic_temporal, spatial_temporal], dim=-1)
            fused_features = self.concat_proj(concat_features)
        # Pooling
        if self.adaptive_pool is not None:
            pooled_features, attn_weights = self.adaptive_pool(fused_features)  # (B, hidden_dim)
        else:
            pooled_features = fused_features.mean(dim=1)
            attn_weights = None
        
        audio_embed = self.audio_proj(pooled_features)  # (B, joint_embed_shape)
        audio_embed = F.normalize(audio_embed, dim=-1)
        
        if return_temporal:
            return audio_embed, fused_features, attn_weights
        else:
            return audio_embed

    def encode_text(self, text):
        if self.use_text_attention:
            hidden_states, attention_mask, offset_mapping = self.text_encoder(text) # (B, T, 768), (B, T), (B, T, 2)
            # Spatial-aware attention pooling
            pooled = self._pool_text_features(hidden_states, attention_mask, offset_mapping, text)
        else:
            pooled = self.text_encoder(text)  # (B, 768)

        # Projection
        text_embed = self.text_proj(pooled)
        text_embed = F.normalize(text_embed, dim=-1)
        return text_embed
    
    def _pool_text_features(self, hidden_states, attention_mask, offset_mapping, texts):
        # --- Soft Attention with Spatial Bias ---     
        attn_logits = self.text_attention(hidden_states).squeeze(-1)  # (B, T)
        
        attn_logits = attn_logits.masked_fill(attention_mask == 0, float('-inf'))
        
        spatial_mask = self._build_spatial_mask(texts, offset_mapping, hidden_states.shape[1])
        spatial_mask = spatial_mask.to(hidden_states.device)
        
        attn_logits = attn_logits + (spatial_mask.float() * self.spatial_bias_value)
        # Softmax & Weighted Sum
        attn_weights = torch.softmax(attn_logits, dim=-1)  # (B, T)
        pooled = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)  # (B, 768)
        
        return pooled
    
    def _build_spatial_mask(self, texts, offset_mapping, seq_len):
        """
        mask for spatial phrases
        """
        B = len(texts)
        mask = torch.zeros(B, seq_len, dtype=torch.bool)
        
        mask[:, 0] = True
        
        if self.doa_phrases is None:
            return mask
        
        offsets = offset_mapping.cpu().numpy()
        
        for i, text in enumerate(texts):
            lowered = text.lower()
            sample_offsets = offsets[i]
            
            for phrase in self.doa_phrases:
                phrase_lower = phrase.lower()
                start = 0
                while True:
                    idx = lowered.find(phrase_lower, start)
                    if idx == -1:
                        break
                    
                    end_idx = idx + len(phrase_lower)
                    
                    for token_idx, (tok_start, tok_end) in enumerate(sample_offsets):
                        if token_idx >= seq_len:
                            break
                        if tok_start == tok_end:  # Special tokens
                            continue
                        
                        if not (tok_end <= idx or tok_start >= end_idx):
                            mask[i, token_idx] = True     
                    start = idx + 1
        return mask
    
    def forward(self, audio, text=None, return_temporal=False):
        combined_features = self.spatial_encoder(audio)  # (B, T, 256+768)
        spatial_temporal = combined_features[..., :256]  # (B, T, 256)
        semantic_temporal = combined_features[..., 256:]  # (B, T, 768)
        if self.fusion is not None:
            fused_features = self.fusion(semantic_temporal, spatial_temporal)  # (B, T, hidden_dim)
        else:
            # fusion_type == 'none': Concat + Projection
            concat_features = torch.cat([semantic_temporal, spatial_temporal], dim=-1)
            fused_features = self.concat_proj(concat_features)
        # Pooling
        if self.adaptive_pool is not None:
            pooled_features, attn_weights = self.adaptive_pool(fused_features)  # (B, hidden_dim)
        else:
            pooled_features = fused_features.mean(dim=1)
            attn_weights = None
        
        audio_embed = self.audio_proj(pooled_features)  # (B, joint_embed_shape)
        audio_embed = F.normalize(audio_embed, dim=-1)
        
        if text is None:
            result = {"audio": audio_embed}
            if return_temporal:
                result["audio_temporal"] = fused_features
                result["attn_weights"] = attn_weights
            return result
        
        if self.use_text_attention:
            hidden_states, attention_mask, offset_mapping = self.text_encoder(text)
            pooled = self._pool_text_features(hidden_states, attention_mask, offset_mapping, text)
        else:
            pooled = self.text_encoder(text)
        text_embed = self.text_proj(pooled)
        text_embed = F.normalize(text_embed, dim=-1)
        
        result = {"audio": audio_embed, "text": text_embed}
        
        if return_temporal:
            result["audio_temporal"] = fused_features
            result["attn_weights"] = attn_weights
        
        return result

    def load_spatial_clap_pretrained(self, url=None):
        if os.path.exists("data/ckpt/l1proposed-spatial_contrastive-model_epoch_49.pt"):
            ckpt = torch.load("data/ckpt/l1proposed-spatial_contrastive-model_epoch_49.pt", map_location='cpu')
        elif url is not None:
            url = "https://huggingface.co/sarulab-speech/SpatialCLAP/resolve/main/ckpt/l1proposed-spatial_contrastive-model_epoch_49.pt"
            print(f"Loading SpatialCLAP pretrained weights from: {url[:50]}...")
            ckpt = torch.hub.load_state_dict_from_url(url, map_location='cpu', check_hash=True)
        else:
            raise ValueError("Either a valid local checkpoint path or a URL must be provided.")

        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        
        key_mapping = {
            'text_encoder.': 'text_encoder.',
        }
        
        current_state = self.state_dict()
        filtered_state = {}
        skipped_keys = []
        loaded_components = set()
        
        for key, value in state_dict.items():
            new_key = key
            
            for old_prefix, new_prefix in key_mapping.items():
                if key.startswith(old_prefix):
                    new_key = key.replace(old_prefix, new_prefix, 1)
                    break
            if new_key in current_state:
                if current_state[new_key].shape == value.shape:
                    filtered_state[new_key] = value
                    component = new_key.split('.')[0]
                    loaded_components.add(component)
                else:
                    skipped_keys.append(f"{key} → {new_key} (shape: {value.shape} vs {current_state[new_key].shape})")
            else:
                skipped_keys.append(f"{key} (not mapped)")
        self.load_state_dict(filtered_state, strict=False)