import torch
import torch.nn.functional as F
import torch.fft
import copy
import numpy as np
import pickle
import os


def conv_rir_moving(dry_sources, rir_sequences, ch_num, device):
    """Apply RIR convolution for moving sources."""
    batch_size = len(dry_sources)
    assert len(rir_sequences) == batch_size
    
    results = []
    
    for i, (waveform, rir_seq) in enumerate(zip(dry_sources, rir_sequences)):
        waveform = waveform.to(device)
        
        result = conv_moving_source_fft(waveform, rir_seq, ch_num, device)
        results.append(result)
    
    max_len = max(r.shape[-1] for r in results)
    padded_results = []
    
    for result in results:
        if result.shape[-1] < max_len:
            pad = max_len - result.shape[-1]
            result = F.pad(result, (0, pad))
        padded_results.append(result)
    
    return torch.stack(padded_results, dim=0)

def get_frame_center_time(frame_idx, hop_size, frame_size, fs):
    return (frame_idx * hop_size + frame_size / 2) / fs

def next_power_of_2(n):
    return 1 << (n - 1).bit_length()

def conv_moving_source_fft(waveform, rir_sequence, ch_num, device, 
                           frame_len=2048, hop_len=1024, fs=16000):
    """RIR convolution using FFT for moving sources."""
    waveform = waveform.to(device)
    nb_steps = len(rir_sequence)
    
    rir_len = rir_sequence[0].shape[-1]
    
    conv_result_len = frame_len + rir_len - 1
    n_fft = next_power_of_2(conv_result_len)
    
    rir_fft_seq = []
    for rir in rir_sequence:
        rir = rir.to(device)
        rir_fft = torch.fft.rfft(rir, n=n_fft, dim=-1)
        rir_fft_seq.append(rir_fft)
    
    rir_fft_tensor = torch.stack(rir_fft_seq) 

    output_len = len(waveform) + rir_len - 1
    convolved = torch.zeros(ch_num, output_len, device=device)
    
    fade_window = torch.bartlett_window(frame_len, device=device)
    num_frames = (len(waveform) - frame_len) // hop_len + 1
    
    for i in range(num_frames):
        start = i * hop_len
        end = start + frame_len
        
        current_time = get_frame_center_time(i, hop_len, frame_len, fs)
        audio_duration = len(waveform) / fs
        progress = min(current_time / audio_duration, 1.0)
        rir_idx = int(progress * (nb_steps - 1))
        rir_idx = min(rir_idx, nb_steps - 1)
        
        current_rir_fft = rir_fft_tensor[rir_idx]
        
        frame = waveform[start:end] * fade_window
        
        frame_fft = torch.fft.rfft(frame, n=n_fft)
        
        convolved_fft = current_rir_fft * frame_fft.unsqueeze(0)
        
        conv_time_batch = torch.fft.irfft(convolved_fft, n=n_fft, dim=-1)
        
        conv_time_batch = conv_time_batch[:, :conv_result_len]
        
        output_end = min(start + conv_result_len, output_len)
        valid_len = output_end - start
        
        convolved[:, start:output_end] += conv_time_batch[:, :valid_len]
    
    return convolved

class RIRDataLoader:
    def __init__(self, rir_base_dir, split):
        """
        Args:
            rir_base_dir: RIR base directory
            split: 'train', 'val', 'test'
        """
        self.split = split
        
        rir_dir = os.path.join(rir_base_dir, split)
        assert os.path.exists(rir_dir), f"RIR directory not found: {rir_dir}"
        
        self.rir_dir = os.path.join(rir_dir, 'rir')
        self.label_dir = os.path.join(rir_dir, 'labels')
        
        metadata_path = os.path.join(rir_dir, 'metadata.pkl')
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.num_samples = len(self.metadata)
        
        self.room_condition_groups = {}
        for idx, meta in enumerate(self.metadata):
            room_id = meta['room_condition_id']
            if room_id not in self.room_condition_groups:
                self.room_condition_groups[room_id] = []
            self.room_condition_groups[room_id].append(idx)
        
        self.room_condition_ids = list(self.room_condition_groups.keys())
        print(f"Loaded {len(self.room_condition_ids)} room conditions with {self.num_samples} total RIRs")
        self.sequential_idx = np.random.randint(0, self.num_samples)
        
    def get_rir_by_index(self, idx):
        """Get RIR and metadata by index."""
        rir_path = os.path.join(self.rir_dir, f'{idx:06d}.npy')
        label_path = os.path.join(self.label_dir, f'{idx:06d}.pkl')
        
        rir = np.load(rir_path)  # (num_segments, 2, rir_len)
        with open(label_path, 'rb') as f:
            labels = pickle.load(f)
        
        meta = self.metadata[idx]
        
        return rir, labels, meta
    
    def get_random_rir_pair_same_room(self, filter_func1=None, filter_func2=None, 
                                      zone_check=None):
        """Get random RIR pair from same room condition."""
        max_attempts = 1000
        
        for _ in range(max_attempts):
            room_id = np.random.choice(self.room_condition_ids)
            available_indices = self.room_condition_groups[room_id]
            
            if len(available_indices) < 2:
                continue
            valid_indices1 = [idx for idx in available_indices 
                            if filter_func1 is None or filter_func1(self.metadata[idx])]
            
            if not valid_indices1:
                continue
            
            idx1 = np.random.choice(valid_indices1)
            meta1 = self.metadata[idx1]
            
            def combined_filter2(meta):
                if filter_func2 is not None and not filter_func2(meta):
                    return False
                if zone_check == 'same_zone' or zone_check == 'same_start_zone':
                    if meta['start_zone'] == meta1['start_zone']:
                        return False
                
                return True
            
            valid_indices2 = [idx for idx in available_indices 
                            if idx != idx1 and combined_filter2(self.metadata[idx])]
            
            if not valid_indices2:
                continue
            
            idx2 = np.random.choice(valid_indices2)
            meta2 = self.metadata[idx2]
            
            rir1, labels1, _ = self.get_rir_by_index(idx1)
            rir2, labels2, _ = self.get_rir_by_index(idx2)
            return (rir1, labels1, meta1, idx1), (rir2, labels2, meta2, idx2)
        print(f"Warning: Could not find RIR pair satisfying constraints after {max_attempts} attempts")
        return None, None
    
    def get_filtered_rir(self, filter_func, room_condition_id=None):
        """Get RIR satisfying filter conditions."""
        max_attempts = 1000
        
        if room_condition_id is not None:
            available_indices = self.room_condition_groups.get(room_condition_id, [])
            if not available_indices:
                print(f"Warning: No RIRs found for room_condition_id={room_condition_id}")
                return None, None, None, None
        else:
            available_indices = list(range(self.num_samples))
        
        for _ in range(max_attempts):
            idx = np.random.choice(available_indices)
            meta = self.metadata[idx]
            
            if filter_func is None or filter_func(meta):
                rir, labels, _ = self.get_rir_by_index(idx)
                return rir, labels, meta, idx
        
        print(f"Warning: Could not find RIR satisfying filter after {max_attempts} attempts")
        return None, None, None, None


class SpatialAudioMixer:
    """Spatial audio mixer for processing and metadata generation."""
    
    def __init__(self, device='cuda', ch_num=2, fs=16000):
        self.device = device
        self.ch_num = ch_num
        self.fs = fs
        self.frame_size = 2048
        self.hop_size = 1024
    
    def apply_rir_to_waveform(self, waveform, rir, labels, rir_meta):
        """Apply RIR to single source waveform."""
        if rir.ndim == 3:
            rir_sequence = [torch.from_numpy(rir[i]).float() for i in range(rir.shape[0])]
        else:
            rir_sequence = [torch.from_numpy(rir).float()]
        
        waveform_tensor = torch.from_numpy(waveform).float() if isinstance(waveform, np.ndarray) else waveform
        convolved = conv_rir_moving([waveform_tensor], [rir_sequence], self.ch_num, self.device)[0]
        metadata = self._create_metadata(waveform_tensor, labels, rir_meta)
        
        return convolved, metadata
    
    def mix_two_sources(self, audio1, audio2, meta1, meta2):
        """Mix two audio sources."""
        max_len = max(audio1.shape[-1], audio2.shape[-1])
        
        if audio1.shape[-1] < max_len:
            audio1 = F.pad(audio1, (0, max_len - audio1.shape[-1]))
        if audio2.shape[-1] < max_len:
            audio2 = F.pad(audio2, (0, max_len - audio2.shape[-1]))
        
        mixed = (audio1 + audio2) / 2
        mixed_meta = self._create_mixed_metadata(meta1, meta2, max_len)
        
        return mixed, mixed_meta
    
    def _create_metadata(self, waveform, rir_labels, rir_meta):
        """Create metadata for single source."""
        audio_len = len(waveform)
        num_frames = (audio_len - self.frame_size) // self.hop_size + 1
        
        frame_times = np.array([
            get_frame_center_time(i, self.hop_size, self.frame_size, self.fs)
            for i in range(num_frames)
        ])
        
        rir_doas = np.array([label['doa'] for label in rir_labels])
        audio_duration = audio_len / self.fs
        rir_times = np.linspace(0, audio_duration, rir_meta.get('num_segments', len(rir_labels)))
        frame_doas = np.interp(frame_times, rir_times, rir_doas, 
                               left=rir_doas[0], right=rir_doas[-1]).astype(np.float32)
        start_doa = rir_labels[0]['doa']
        end_doa = rir_labels[-1]['doa']
        
        metadata = {
            'frame_doas': frame_doas,
            'frame_times': frame_times.astype(np.float32),
            'num_frames': num_frames,
            'room_size': rir_meta.get('room_size', None),
            'doa': (start_doa + end_doa) / 2,
            'start_doa': start_doa,
            'end_doa': end_doa,
            'start_zone': rir_meta.get('start_zone', None),
            'end_zone': rir_meta.get('end_zone', None),
            'trajectory_positions': rir_meta.get('trajectory_positions', []),
            'is_stationary': rir_meta.get('is_stationary', None),
            'trajectory_type': rir_meta.get('trajectory_type', 'unknown'),
            'doa_change': abs(end_doa - start_doa)
        }
        
        return metadata
    
    def _create_mixed_metadata(self, meta1, meta2, audio_len):
        """Create metadata for mixed sources."""
        num_frames = (audio_len - self.frame_size) // self.hop_size + 1
        
        frame_times = np.array([
            (i * self.hop_size + self.frame_size / 2) / self.fs
            for i in range(num_frames)
        ])
        
        doas1 = meta1.get('frame_doas', np.full(num_frames, meta1.get('doa', 0.0)))
        doas2 = meta2.get('frame_doas', np.full(num_frames, meta2.get('doa', 0.0)))
        
        if len(doas1) != num_frames:
            orig_times_1 = np.linspace(0, audio_len / self.fs, len(doas1))
            doas1 = np.interp(frame_times, orig_times_1, doas1, 
                             left=doas1[0], right=doas1[-1])
        if len(doas2) != num_frames:
            orig_times_2 = np.linspace(0, audio_len / self.fs, len(doas2))
            doas2 = np.interp(frame_times, orig_times_2, doas2,
                             left=doas2[0], right=doas2[-1])
        frame_doas = np.stack([doas1[:num_frames], doas2[:num_frames]], axis=1)
        frame_n_sources = np.full(num_frames, 2, dtype=np.int32)
        
        return {
            'frame_doas': frame_doas.astype(np.float32),
            'frame_n_sources': frame_n_sources,
            'frame_times': frame_times.astype(np.float32),
            'num_frames': num_frames,
            'room_size': meta1.get('room_size', None),
            'mixed': True,
            'n_sources': 2,
            'source_metas': [meta1, meta2],
            'trajectory_positions': [
                meta1.get('trajectory_positions', []),
                meta2.get('trajectory_positions', [])
            ],
            'is_stationary': [meta1.get('is_stationary', None), meta2.get('is_stationary', None)],
            'source_types': [
                "stationary" if meta1.get('is_stationary', False) else "moving",
                "stationary" if meta2.get('is_stationary', False) else "moving"
            ],
            'doa': (np.mean(doas1) + np.mean(doas2)) / 2,
            'trajectory_types': [
                meta1.get('trajectory_type', 'unknown'),
                meta2.get('trajectory_type', 'unknown')
            ]
        }


class SpatialCollate:
    def __init__(self, rir_base_dir, split, source_type='stationary', trajectory_type=None,
                 enable_mixing=False, zone_check=None, device='cuda'):
        """
        Args:
            rir_base_dir: RIR directory
            split: 'train', 'val', 'test'
            source_type: 'stationary', 'moving', or 'mixed_stat_mov'
            trajectory_type: Filter by 'linear' or 'circular'
            enable_mixing: Enable mixed source generation
            zone_check: Zone constraint type
            device: Computation device
        """
        self.source_type = source_type
        self.trajectory_type = trajectory_type
        self.enable_mixing = enable_mixing
        self.zone_check = zone_check
        self.rir_loader = RIRDataLoader(rir_base_dir, split)
        self.mixer = SpatialAudioMixer(device=device)
    
    def __call__(self, batch):
        """Process batch for single or mixed source augmentation."""
        if not self.enable_mixing:
            return self._process_single_sources(batch)
        else:
            return self._process_mixed_sources(batch)
    
    def _process_single_sources(self, batch):
        """Process single sources."""
        out_wavs = []
        out_metas = []
        
        filter_func = self._build_filter_func()
        
        for waveform, caption_meta in batch:
            rir, labels, rir_meta, rir_idx = self.rir_loader.get_filtered_rir(filter_func)
            
            convolved, spatial_meta = self.mixer.apply_rir_to_waveform(waveform, rir, labels, rir_meta)
            spatial_meta['rir_idx'] = rir_idx
            
            meta = {**caption_meta, **spatial_meta}
            meta['augmentation_type'] = self.source_type
            
            out_wavs.append(convolved)
            out_metas.append(meta)
        
        wavs_tensor = torch.stack(out_wavs, dim=0)
        return wavs_tensor, out_metas
    
    def _build_filter_func(self):
        """Build filter function based on source_type and trajectory_type."""
        def filter_func(meta):
            if self.source_type == 'stationary':
                if not meta['is_stationary']:
                    return False
            elif self.source_type == 'moving':
                if meta['is_stationary']:
                    return False
            
            if self.trajectory_type is not None:
                if meta['trajectory_type'] != self.trajectory_type:
                    return False
            
            return True
        
        return filter_func
    
    def _process_mixed_sources(self, batch):
        """Process mixed sources (pair-wise)."""
        out_wavs = []
        out_metas = []
        
        if self.source_type == 'mixed_stat_mov':
            filter_func1 = lambda meta: meta['is_stationary']
            filter_func2 = lambda meta: not meta['is_stationary']
            if self.trajectory_type is not None:
                original_filter2 = filter_func2
                filter_func2 = lambda meta: original_filter2(meta) and meta['trajectory_type'] == self.trajectory_type
        else:
            base_filter = self._build_filter_func()
            filter_func1 = base_filter
            filter_func2 = base_filter
        
        n = len(batch)
        
        for i in range(0, n, 2):
            if i + 1 >= n:
                print("Warning: Odd number of samples in batch, skipping last sample for mixing")
                break
            
            waveform1, caption_meta1 = batch[i]
            waveform2, caption_meta2 = batch[i + 1]
            
            result = self.rir_loader.get_random_rir_pair_same_room(
                filter_func1=filter_func1,
                filter_func2=filter_func2,
                zone_check=self.zone_check
            )
            
            if result is None:
                print("Warning: Failed to get RIR pair, skipping this pair")
                continue
            
            (rir1, labels1, meta1, idx1), (rir2, labels2, meta2, idx2) = result
            
            convolved1, spatial_meta1 = self.mixer.apply_rir_to_waveform(waveform1, rir1, labels1, meta1)
            spatial_meta1['rir_idx'] = idx1
            spatial_meta1['room_condition_id'] = meta1['room_condition_id']
            meta1_full = {**caption_meta1, **spatial_meta1}
            
            convolved2, spatial_meta2 = self.mixer.apply_rir_to_waveform(waveform2, rir2, labels2, meta2)
            spatial_meta2['rir_idx'] = idx2
            spatial_meta2['room_condition_id'] = meta2['room_condition_id']
            meta2_full = {**caption_meta2, **spatial_meta2}
            
            mixed_audio, mixed_meta = self.mixer.mix_two_sources(
                convolved1, convolved2, meta1_full, meta2_full
            )
            
            mixed_meta['augmentation_type'] = self._get_augmentation_type()
            mixed_meta['caption'] = meta1_full.get('caption', '')
            
            out_wavs.append(mixed_audio)
            out_metas.append(mixed_meta)
        
        wavs_tensor = torch.stack(out_wavs, dim=0)
        return wavs_tensor, out_metas
    
    def _get_augmentation_type(self):
        """Determine augmentation type."""
        if self.source_type == 'mixed_stat_mov':
            return 'mixed_stat_mov'
        elif self.source_type == 'stationary':
            return 'mixed_stat_stat'
        elif self.source_type == 'moving':
            return 'mixed_mov_mov'
        else:
            return 'mixed'
