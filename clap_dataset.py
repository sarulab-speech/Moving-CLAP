import os
import pickle
import numpy as np
import torch
import torch.utils.data as data
from collections import defaultdict, deque

class BalancedAugmentationBatchSampler(data.Sampler):
    """
    - sampler that creates batches with balanced augmentation_type distribution
    - ensures no samples with the same youtube_id are in the same batch
    """
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.aug_type_to_indices = defaultdict(list)
        self.index_to_audio_ids = {} 
        
        if hasattr(dataset, 'indices'):
            metas = dataset.full_dataset.metas
            for local_idx, real_idx in enumerate(dataset.indices):
                meta = metas[real_idx]
                aug_type = dataset._get_augmentation_type(meta)
                self.aug_type_to_indices[aug_type].append(local_idx)
                
                audio_ids = self._extract_audio_ids(meta, real_idx)
                self.index_to_audio_ids[local_idx] = audio_ids
        else:
            raise ValueError("BalancedAugmentationBatchSampler requires SpatialDataset")
        
        self.aug_types = list(self.aug_type_to_indices.keys())
        self.total_samples = sum(len(v) for v in self.aug_type_to_indices.values())
        
        self.aug_type_ratios = {
            aug_type: len(indices) / self.total_samples
            for aug_type, indices in self.aug_type_to_indices.items()
        }
        
        self.samples_per_batch = {
            aug_type: max(1, int(self.batch_size * ratio))
            for aug_type, ratio in self.aug_type_ratios.items()
        }
        
        # Adjust to match exact batch size
        total_per_batch = sum(self.samples_per_batch.values())
        if total_per_batch > self.batch_size:
            # Reduce from the augmentation type with the highest ratio
            sorted_types = sorted(self.aug_types, key=lambda x: self.aug_type_ratios[x], reverse=True)
            for aug_type in sorted_types:
                if total_per_batch <= self.batch_size:
                    break
                if self.samples_per_batch[aug_type] > 1:
                    self.samples_per_batch[aug_type] -= 1
                    total_per_batch -= 1
        
        # If the total is less than batch_size, complement
        while sum(self.samples_per_batch.values()) < self.batch_size:
            # Add to the one with the highest ratio
            max_type = max(self.aug_types, key=lambda x: self.aug_type_ratios[x])
            self.samples_per_batch[max_type] += 1
        
        print(f"BalancedAugmentationBatchSampler initialized:")
        print(f"   Total samples: {self.total_samples}")
        print(f"   Augmentation type distribution:")
        for aug_type in sorted(self.aug_types):
            count = len(self.aug_type_to_indices[aug_type])
            ratio = self.aug_type_ratios[aug_type]
            per_batch = self.samples_per_batch[aug_type]
            print(f"     {aug_type:20s}: {count:6d} ({ratio*100:5.1f}%) → {per_batch} per batch")
    
    def _extract_audio_ids(self, meta, fallback_idx):
        """Extract youtube_id from metadata"""
        audio_ids = []
        
        if isinstance(meta, dict) and meta.get('mixed', False):
            source_metas = meta.get('source_metas', [])
            for src_meta in source_metas:
                audio_id = src_meta.get('youtube_id') or src_meta.get('audiocap_id') or src_meta.get('audio_idx')
                if audio_id is not None:
                    audio_ids.append(audio_id)
        elif isinstance(meta, dict):
            audio_id = meta.get('youtube_id') or meta.get('audiocap_id') or meta.get('audio_idx')
            if audio_id is not None:
                audio_ids.append(audio_id)
        
        if not audio_ids:
            audio_ids = [fallback_idx]
        
        return audio_ids
    
    def __iter__(self):
        # Shuffle within each augmentation_type and create deque
        aug_type_pools = {}
        for aug_type, indices in self.aug_type_to_indices.items():
            pool = indices.copy()
            if self.shuffle:
                np.random.shuffle(pool)
            aug_type_pools[aug_type] = deque(pool)
        
        batches = []
        max_retry_per_batch = self.batch_size * 10
        
        while True:
            current_batch = []
            batch_audio_ids = set()
            retry_count = 0
            
            # Get specified number from each augmentation_type
            for aug_type in self.aug_types:
                pool = aug_type_pools[aug_type]
                target_count = self.samples_per_batch[aug_type]
                
                added_count = 0
                local_retry = 0
                max_local_retry = len(pool) * 2
                
                # Search within the pool (defer if duplicate)
                while added_count < target_count and local_retry < max_local_retry:
                    if len(pool) == 0:
                        break
                    
                    idx = pool.popleft()
                    local_retry += 1

                    sample_audio_ids = set(self.index_to_audio_ids[idx])
                    
                    if sample_audio_ids & batch_audio_ids:
                        pool.append(idx)
                        retry_count += 1
                        continue
                    
                    current_batch.append(idx)
                    batch_audio_ids.update(sample_audio_ids)
                    added_count += 1

            if len(current_batch) == 0:
                break
            
            # Handle case when batch size is not met
            if len(current_batch) < self.batch_size:
                # Supplement remaining samples
                for aug_type in self.aug_types:
                    pool = aug_type_pools[aug_type]
                    local_retry = 0
                    max_local_retry = len(pool) * 2
                    
                    while len(current_batch) < self.batch_size and local_retry < max_local_retry:
                        if len(pool) == 0:
                            break
                        
                        idx = pool.popleft()
                        local_retry += 1
                        
                        # Check for duplicate audio IDs
                        sample_audio_ids = set(self.index_to_audio_ids[idx])
                        if sample_audio_ids & batch_audio_ids:
                            pool.append(idx)
                            retry_count += 1
                            continue
                        
                        current_batch.append(idx)
                        batch_audio_ids.update(sample_audio_ids)
                    
                    if len(current_batch) >= self.batch_size:
                        break
            
            # If too many retries, fill from any available samples
            if retry_count > max_retry_per_batch and len(current_batch) < self.batch_size:
                for aug_type in self.aug_types:
                    pool = aug_type_pools[aug_type]
                    while len(current_batch) < self.batch_size and len(pool) > 0:
                        idx = pool.popleft()
                        current_batch.append(idx)
                        if len(current_batch) >= self.batch_size:
                            break
            
            if len(current_batch) < self.batch_size and self.drop_last:
                break
            
            if len(current_batch) > 0:
                batches.append(current_batch)
        
        if self.shuffle:
            np.random.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return self.total_samples // self.batch_size
        return (self.total_samples + self.batch_size - 1) // self.batch_size


class ChunkedMemmapDataset:
    def __init__(self, data_dir):
        chunk_info_path = os.path.join(data_dir, 'chunk_info.pkl')
        labels_path = os.path.join(data_dir, 'labels.pkl')

        self.data_dir = data_dir
        self.loaded_chunks = {}

        self._pattern_indices = None

        if os.path.exists(chunk_info_path) and os.path.exists(labels_path):
            self.data_format = 'chunked_memmap'
            self._init_chunked_memmap(chunk_info_path, labels_path)
            print(f"Loaded chunked dataset: {self.total_samples} samples")
        else:
            self.data_format = 'raw_precomputed'
            self._init_raw_precomputed()
            print(f"Loaded raw precomputed dataset: {self.total_samples} samples")

    def _init_chunked_memmap(self, chunk_info_path, labels_path):
        with open(chunk_info_path, 'rb') as f:
            chunk_data = pickle.load(f)

        self.chunks = chunk_data['chunks']
        self.chunk_size = chunk_data['chunk_size']
        self.num_chunks = chunk_data['num_chunks']
        self.total_samples = chunk_data['total_samples']
        self.max_length = chunk_data['max_length']

        with open(labels_path, 'rb') as f:
            labels_data = pickle.load(f)

        self.metas = labels_data['metas']
        self.shape_info = labels_data['shape_info']

        self._build_chunk_index()

    def _init_raw_precomputed(self):
        audio_dir = os.path.join(self.data_dir, 'audio')
        label_dir = os.path.join(self.data_dir, 'label')

        if not os.path.isdir(audio_dir) or not os.path.isdir(label_dir):
            raise FileNotFoundError(
                f"No supported dataset format found in {self.data_dir}. "
                f"Expected either chunked files (chunk_info.pkl, labels.pkl) "
                f"or raw directories (audio/, label/)."
            )

        audio_files = sorted(
            f for f in os.listdir(audio_dir)
            if f.startswith('audio_') and f.endswith('.npy')
        )

        if not audio_files:
            raise FileNotFoundError(f"No audio files found in: {audio_dir}")

        self.sample_files = []
        self.metas = []
        self.shape_info = []

        for audio_file in audio_files:
            label_file = audio_file.replace('audio_', 'label_').replace('.npy', '.pkl')
            label_path = os.path.join(label_dir, label_file)
            if not os.path.exists(label_path):
                continue

            audio_path = os.path.join(audio_dir, audio_file)
            self.sample_files.append((audio_path, label_path))

            with open(label_path, 'rb') as f:
                meta = pickle.load(f)
            self.metas.append(meta)
            self.shape_info.append(None)

        if not self.sample_files:
            raise FileNotFoundError(
                f"No valid audio/label pairs found under {self.data_dir}"
            )

        self.total_samples = len(self.sample_files)
        self.chunks = []
        self.chunk_size = 0
        self.num_chunks = 0
        self.max_length = 0

        print(
            f"Raw format detected in {self.data_dir}: "
            f"{self.total_samples} paired samples"
        )
    
    def _build_chunk_index(self):
        """Build a fast index mapping sample index to (chunk_id, local_index)"""
        self.sample_to_chunk = {}
        for chunk in self.chunks:
            chunk_id = chunk['chunk_id']
            start_idx = chunk['start_idx']
            end_idx = chunk['end_idx']
            for sample_idx in range(start_idx, end_idx):
                self.sample_to_chunk[sample_idx] = (chunk_id, sample_idx - start_idx)
    
    def _load_chunk(self, chunk_id):
        if chunk_id in self.loaded_chunks:
            return self.loaded_chunks[chunk_id]
        
        chunk_file = f'audio_chunk_{chunk_id:03d}.npy'
        chunk_path = os.path.join(self.data_dir, chunk_file)
        
        if not os.path.exists(chunk_path):
            raise FileNotFoundError(f"Chunk file not found: {chunk_path}")
        
        chunk_data = np.load(chunk_path, mmap_mode='r')
        
        if len(self.loaded_chunks) >= 3:
            oldest_key = next(iter(self.loaded_chunks))
            del self.loaded_chunks[oldest_key]
        
        self.loaded_chunks[chunk_id] = chunk_data
        return chunk_data
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range")

        if self.data_format == 'raw_precomputed':
            audio_path, _ = self.sample_files[idx]
            wav_np = np.load(audio_path)
            if not wav_np.flags['C_CONTIGUOUS']:
                wav = torch.from_numpy(np.ascontiguousarray(wav_np)).float()
            else:
                wav = torch.from_numpy(wav_np).float()
            meta = self.metas[idx]
            return wav, meta

        if idx in self.sample_to_chunk:
            chunk_id, local_idx = self.sample_to_chunk[idx]
        else:
            chunk_id = idx // self.chunk_size
            local_idx = idx % self.chunk_size

        chunk_data = self._load_chunk(chunk_id)
        actual_length = self.shape_info[idx]

        wav_view = chunk_data[local_idx, :, :actual_length]

        if not wav_view.flags['C_CONTIGUOUS']:
            wav = torch.from_numpy(np.ascontiguousarray(wav_view)).float()
        else:
            wav = torch.from_numpy(wav_view).float()

        meta = self.metas[idx]

        return wav, meta
    
    def get_pattern_indices(self, single_only=True):
        if self._pattern_indices is not None:
            return self._pattern_indices
        
        pattern_indices = {}
        
        for idx in range(len(self)):
            meta = self.metas[idx]
            
            is_mixed = isinstance(meta, dict) and meta.get('mixed', False)
            
            if single_only and is_mixed:
                continue
            
            if not single_only and not is_mixed:
                continue
            
            if not is_mixed:
                start_zone = meta.get('start_zone')
                end_zone = meta.get('end_zone')
                
                if start_zone is not None and end_zone is not None:
                    pattern = (start_zone, end_zone)
                    if pattern not in pattern_indices:
                        pattern_indices[pattern] = []
                    pattern_indices[pattern].append(idx)
        
        self._pattern_indices = pattern_indices
        return pattern_indices


def test_collate_fn(batch):
    wavs = []
    metas = []
    
    for wav, meta in batch:
        wavs.append(wav)
        metas.append(meta)
    
    max_len = 0
    # Choose valid wavs only
    valid_wavs = [w for w in wavs if w is not None and w.shape[-1] > 0]
    if not valid_wavs:
         return None, None
    
    max_len = max(w.shape[-1] for w in valid_wavs)
    
    padded_wavs = []
    valid_metas = []
    
    for w, m in zip(wavs, metas):
        if w is None or w.shape[-1] == 0:
            continue
            
        if w.shape[-1] < max_len:
            pad_len = max_len - w.shape[-1]
            w = torch.nn.functional.pad(w, (0, pad_len))
        padded_wavs.append(w)
        valid_metas.append(m)

    if not padded_wavs:
        return None, None
        
    wavs_tensor = torch.stack(padded_wavs, dim=0)
    return wavs_tensor, valid_metas

class SpatialDataset:
    """
    Filterable Spatial Dataset for CLAP training and evaluation
    """
    
    def __init__(self, precomputed_root, split='train', aug_types='all', exclude_types=None):
        """
        Args:
            aug_types:
                - 'all':
                - ['stationary']:
                - ['moving']:
                - ['stationary', 'moving']: Both single-source types
                - ['mixed_stat_stat']:
                - ['mixed_mov_mov']: 
                - ['mixed_stat_mov']:
                - ['mixed_stat_stat', 'mixed_mov_mov', 'mixed_stat_mov']: All mixed types
            exclude_types:
                - None: No exclusion
                - ['mixed_mov_mov']: Exclude moving+moving mixed sources
                - ['mixed_mov_mov', 'mixed_stat_mov']: Exclude multiple types
        """
        self.full_dataset = ChunkedMemmapDataset(
            os.path.join(precomputed_root, split)
        )
        
        self.metas = self.full_dataset.metas
        self.split = split
        self.aug_types = aug_types
        self.exclude_types = set(exclude_types) if exclude_types else set()
        self.indices = self._create_indices()
        
        print(f"SpatialDataset - Split {split}")
    
    def _get_augmentation_type(self, meta):
        """Get augmentation_type from metadata"""
        if isinstance(meta, dict) and meta.get('mixed', False):
            return meta.get('augmentation_type', 'unknown_mixed')
        
        if isinstance(meta, dict):
            return meta.get('augmentation_type', 'unknown')
        return 'unknown'
    
    def _create_indices(self):
        indices = []
        
        if self.aug_types == 'all':
            # Include all, but respect exclusions
            for i, meta in enumerate(self.metas):
                aug_type = self._get_augmentation_type(meta)
                if aug_type not in self.exclude_types:
                    indices.append(i)
            return indices
        
        aug_types_set = set(self.aug_types) if isinstance(self.aug_types, list) else {self.aug_types}
        
        for i, meta in enumerate(self.metas):
            aug_type = self._get_augmentation_type(meta)
            
            # Include if in aug_types AND not in exclude_types
            if aug_type in aug_types_set and aug_type not in self.exclude_types:
                indices.append(i)
        
        return indices
    
    def get_augmentation_stats(self):
        """Get statistics of augmentation_type in the dataset"""
        stats = {}
        for idx in self.indices:
            meta = self.metas[idx]
            aug_type = self._get_augmentation_type(meta)
            stats[aug_type] = stats.get(aug_type, 0) + 1
        return stats
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.full_dataset[real_idx]

def precomputed_collate_fn(batch):
    """
    Collate function for precomputed dataset
    """
    wavs = []
    metas = []
    
    for wav, meta in batch:
        wavs.append(wav)
        metas.append(meta)
    
    max_len = max(w.shape[-1] for w in wavs)
    padded_wavs = []
    for w in wavs:
        if w.shape[-1] < max_len:
            pad_len = max_len - w.shape[-1]
            w = torch.nn.functional.pad(w, (0, pad_len))
        padded_wavs.append(w)
    
    wavs_tensor = torch.stack(padded_wavs, dim=0)  # (batch, 2, time)
    
    return wavs_tensor, metas

class NonSpatialDataset:
    def __init__(self, non_spatial_root, split='train', deduplicate=True):
        data_path = os.path.join(non_spatial_root, split)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Non-spatial data not found: {data_path}")
        
        self.dataset = ChunkedMemmapDataset(data_path)
        self.split = split
        
        if split in ['val', 'test'] and deduplicate:
            self.indices = self._deduplicate_by_audio()
            print(f"NonSpatialDataset - Split {split}: {len(self.indices)} samples "
                  f"(deduplicated from {len(self.dataset)})")
        else:
            self.indices = list(range(len(self.dataset)))
            print(f"NonSpatialDataset - Split {split}: {len(self.indices)} samples")
    
    def _deduplicate_by_audio(self):
        seen_ids = set()
        unique_indices = []
        
        for idx in range(len(self.dataset)):
            meta = self.dataset.metas[idx]
            
            youtube_id = meta.get('youtube_id', idx)
            if youtube_id not in seen_ids:
                seen_ids.add(youtube_id)
                unique_indices.append(idx)
        
        return unique_indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        wav, meta = self.dataset[real_idx]
        return wav, meta


class AllPatternDataset:
    """
    Dataset for all spatial patterns per audio source or source pair
    """
    def __init__(self, pattern_root, pattern_type='single'):
        if pattern_type == 'single':
            data_path = os.path.join(pattern_root, 'all_patterns_single', 'test')
        elif pattern_type == 'mixed':
            data_path = os.path.join(pattern_root, 'all_patterns_mixed', 'test')
        else:
            raise ValueError(f"Invalid pattern_type: {pattern_type}. Use 'single' or 'mixed'.")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Pattern data not found: {data_path}")
        
        self.dataset = ChunkedMemmapDataset(data_path)
        self.pattern_type = pattern_type
        
        self._build_pattern_groups()
        
        print(f"AllPatternDataset - Type: {pattern_type}")
        print(f"  Total samples: {len(self.dataset)}")
        print(f"  Audio/Pair groups: {len(self.pattern_groups)}")

    def _build_pattern_groups(self):
        self.pattern_groups = defaultdict(list)
        
        for idx in range(len(self.dataset)):
            meta = self.dataset.metas[idx]
            
            if self.pattern_type == 'single':
                group_key = meta.get('audio_idx')
            else:  # mixed
                group_key = meta.get('pair_idx')
            
            if group_key is not None:
                self.pattern_groups[group_key].append(idx)
        
        # ソート（pattern_idで並べる）
        for group_key in self.pattern_groups:
            self.pattern_groups[group_key] = sorted(
                self.pattern_groups[group_key],
                key=lambda idx: self.dataset.metas[idx].get('pattern_id', '')
            )
    
    def get_audio_ids(self):
        return sorted(self.pattern_groups.keys())
    
    def get_patterns_for_audio(self, audio_id):
        indices = self.pattern_groups.get(audio_id, [])
        patterns = []
        
        for pattern_idx, idx in enumerate(indices):
            wav, meta = self.dataset[idx]
            if isinstance(meta, dict):
                meta = meta.copy()
                meta['pattern_idx'] = pattern_idx
            patterns.append((wav, meta))
        
        return patterns
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]