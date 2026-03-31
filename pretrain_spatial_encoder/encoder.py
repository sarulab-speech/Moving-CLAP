import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_channels=4, cnn_channels=64, middle_features=128, output_features=256, n_fft=1024):
        super(Encoder, self).__init__()
        assert (output_features % 2) == 0
        self.output_features = output_features

        self.cnn_block = nn.Sequential(
            nn.Conv2d(input_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cnn_channels),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cnn_channels),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cnn_channels),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        self.freq_after_cnn = (n_fft // 2) // (2 ** 3)
        self.middle_linear = nn.Linear(cnn_channels * self.freq_after_cnn, middle_features)

        self.gru = nn.GRU(
            input_size=middle_features,
            hidden_size=output_features // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
    def forward(self, x, return_temporal=True):
        batch_size, time_frames, freq_bins, channels = x.shape
        x = x.permute(0, 3, 1, 2)
        x = self.cnn_block(x)
        batch_size, cnn_channels, time_steps, freq_bins_reduced = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, time_steps, -1)
        x = self.middle_linear(x)
        x, _ = self.gru(x)
        
        if return_temporal:
            return x
        else:
            return torch.mean(x, dim=1)
