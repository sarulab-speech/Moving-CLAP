import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import random
import torchaudio
import os
import copy

from util import get_event_label

class AudioCapsDataset(Dataset):
    def __init__(self, config, sample_rate=16000):
        """
        csv_path: CSVファイルパス
        config: dict形式 (例: {"wav_dir": "data/audiocaps/wav"})
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.wav_dir = config['wav_dir']

        # CSV読み込み＆リスト化
        df = pd.read_csv(config['csv_path'])
        self.meta_list = df.to_dict(orient='records')  # 各行がdict型になる

        # Event lalbe を付与
        event_label = get_event_label()
        for i, meta in enumerate(self.meta_list):
            aid = int(meta["audiocap_id"])
            self.meta_list[i]["event"] = event_label[aid]

    def __len__(self):
        return len(self.meta_list)

    def _load_audio(self, meta):
        youtube_id = meta['youtube_id']
        start_time = meta['start_time']
        filename = f"{youtube_id}_{start_time}.wav"
        wav_path = os.path.join(self.wav_dir, filename)
        waveform, sr = torchaudio.load(wav_path)  # (channels, time)

        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)

        mono_waveform = waveform[0]  # 1chだけ使う (shape: (time,))
        return mono_waveform  # (time,)

        
    def __getitem__(self, idx):
        meta = copy.deepcopy(self.meta_list[idx])
        waveform = self._load_audio(meta)

        return (waveform, meta)