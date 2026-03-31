import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, input_features=256, hidden_features=256, num_classes=344):
        super(Decoder, self).__init__()
        self.num_classes = num_classes

        self.gru_sed = nn.GRU(
            input_size=input_features,
            hidden_size=hidden_features // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        self.gru_doa = nn.GRU(
            input_size=input_features,
            hidden_size=hidden_features // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # SED: Logits
        self.fc_sed = nn.Linear(hidden_features, num_classes)
        # DOA: (cos θ, sin θ)
        self.fc_doa_cos = nn.Linear(hidden_features, num_classes)  # cos(θ)
        self.fc_doa_sin = nn.Linear(hidden_features, num_classes)  # sin(θ)

    def forward(self, x):
        assert x.dim() == 3
        batch_size, time_steps, input_features = x.size()
        
        sed_features, _ = self.gru_sed(x)  # (batch, time_steps, hidden_features)
        sed_logits = self.fc_sed(sed_features)  # (batch, time_steps, num_classes) - Logits
        
        doa_features, _ = self.gru_doa(x)  # (batch, time_steps, hidden_features)
        
        doa_cos = torch.tanh(self.fc_doa_cos(doa_features))  # (batch, time_steps, num_classes)
        doa_sin = torch.tanh(self.fc_doa_sin(doa_features))  # (batch, time_steps, num_classes)
        
        norm = torch.sqrt(doa_cos**2 + doa_sin**2 + 1e-8)
        doa_cos = doa_cos / norm
        doa_sin = doa_sin / norm
        
        # (batch, time_steps, num_classes, 2) 
        doa_output = torch.stack([doa_cos, doa_sin], dim=-1)
        
        return sed_logits, doa_output
