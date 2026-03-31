import torch
import torch.nn as nn
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
from feature_extractor import FeatureExtractor
from encoder import Encoder
from decoder import Decoder
pretrained_sed_path = os.path.join(current_dir, '..', 'PretrainedSED')
if pretrained_sed_path not in sys.path:
    sys.path.append(pretrained_sed_path)
try:
    from models.beats.BEATs_wrapper import BEATsWrapper
    from models.prediction_wrapper import PredictionsWrapper
except ImportError as e:
    print(f"PretrainedSED modules not found: {e}")

class BEATsSELDnetModel(nn.Module):
    def __init__(self,
                 input_ch=2,
                 n_fft=1024,
                 hop_length=512,
                 num_classes=344,
                 feature_dim=256,
                 use_temporal_decoder=True,
                 pretrained_model_name='BEATs',
                 freeze_pretrained=True):
        super(BEATsSELDnetModel, self).__init__()

        self.use_temporal_decoder = use_temporal_decoder
        
        # --- 1. DOA Branch ---
        self.feature_extractor_doa = FeatureExtractor(n_fft=n_fft, hop_length=hop_length)
        self.encoder_doa = Encoder(
            input_channels=self.feature_extractor_doa.input_ch * 2, 
            output_features=feature_dim,
            n_fft=n_fft
        )

        # --- 2. SED Branch ---
        self.pretrained_model_name = pretrained_model_name
        
        if pretrained_model_name == 'BEATs':
            base_model = BEATsWrapper()
            self.pretrained_encoder = PredictionsWrapper(
                base_model, 
                checkpoint="BEATs_strong_1", 
                head_type=None,
            )
            pretrained_out_dim = 768 
            self.sed_features_dim = 768
        else:
            raise NotImplementedError(f"Support for {pretrained_model_name} not implemented.")

        # freeze BEATs parameters
        if freeze_pretrained:
            for param in self.pretrained_encoder.parameters():
                param.requires_grad = False
            self.pretrained_encoder.eval()

        self.decoder_sed = Decoder(
            input_features=pretrained_out_dim,
            hidden_features=256,
            num_classes=num_classes,
        )

        self.doa_feature_dim = feature_dim
        
        self.decoder_doa = Decoder(
            input_features=self.doa_feature_dim + self.sed_features_dim,  # 256 + 768 = 1024
            hidden_features=512,
            num_classes=num_classes,
        )

    def forward(self, x):
        """
        x: (batch, channels, time) - Raw Waveform
        """
        # --- 1. DOA Branch (Feature Extraction) ---
        features_doa = self.feature_extractor_doa(x)
        encoded_doa = self.encoder_doa(features_doa, return_temporal=self.use_temporal_decoder)

        # --- 2. SED Branch (Feature Extraction) ---
        x_mono = x.mean(dim=1)
        
        enable_grad = self.pretrained_encoder.training and any(p.requires_grad for p in self.pretrained_encoder.parameters())
        with torch.set_grad_enabled(enable_grad):
            mel = self.pretrained_encoder.mel_forward(x_mono)
            embeddings = self.pretrained_encoder.model(mel)
            if embeddings.size(-2) > self.pretrained_encoder.seq_len:
                 print(f"Warning: SED embeddings length {embeddings.size(-2)} exceeds pretrained seq_len {self.pretrained_encoder.seq_len}. Applying adaptive avg pool.")
                 embeddings = torch.nn.functional.adaptive_avg_pool1d(
                     embeddings.transpose(1, 2), self.pretrained_encoder.seq_len
                 ).transpose(1, 2)

        # --- 3. Feature Alignment & Fusion ---
        target_time = encoded_doa.shape[1]
        
        if embeddings.shape[1] != target_time:
            embeddings = embeddings.transpose(1, 2)
            embeddings = torch.nn.functional.adaptive_avg_pool1d(
                embeddings, target_time
            )
            embeddings = embeddings.transpose(1, 2)

        fusion_features = torch.cat([encoded_doa, embeddings.detach()], dim=-1) 

        # --- 4. Decoding ---
        sed_output, _ = self.decoder_sed(embeddings)
        _, doa_output = self.decoder_doa(fusion_features)

        return sed_output, doa_output
