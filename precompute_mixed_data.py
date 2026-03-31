import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import argparse
from tqdm import tqdm
from audio_dataset import AudioCapsDataset
from moving_collate import SpatialCollate

def save_precomputed_simulated_spatial(split, out_dir, audio_config, rir_base_dir, device):
    """
    Simulated spatial sound source generation
    """
    os.makedirs(out_dir, exist_ok=True)
    audio_out = os.path.join(out_dir, 'audio')
    label_out = os.path.join(out_dir, 'label')
    os.makedirs(audio_out, exist_ok=True)
    os.makedirs(label_out, exist_ok=True)

    audio_dataset = AudioCapsDataset(audio_config)
    N = len(audio_dataset)
    half_N = N // 2
    # Collate functions for different source types
    collate_stationary = SpatialCollate(
        rir_base_dir, split, source_type='stationary', 
        enable_mixing=False, device=device
    )
    
    collate_moving = SpatialCollate(
        rir_base_dir, split, source_type='moving',
        enable_mixing=False, device=device
    )
    
    collate_mixed_stat_stat = SpatialCollate(
        rir_base_dir, split, source_type='stationary',
        enable_mixing=True, zone_check='same_zone', device=device
    )
    
    collate_mixed_stat_mov = SpatialCollate(
        rir_base_dir, split, source_type='mixed_stat_mov',
        enable_mixing=True, zone_check='same_zone', device=device
    )
    
    global_idx = 0
    stats = {
        'stationary': 0,
        'moving': 0,
        'mixed_stat_stat': 0,
        'mixed_stat_mov': 0
    }
    audio_events = {}
    
    # ===== Phase 1: Single source generation =====    
    # 1-1. Stationary sources (first half)
    print(f"  Generating {half_N} stationary sources (first half)")
    for audio_idx in tqdm(range(0, half_N), desc=f"Stationary"):
        try:
            audio_sample = audio_dataset[audio_idx]
            if audio_sample is None:
                continue
            
            wavs, metas = collate_stationary([audio_sample])
            wav = wavs[0].cpu().numpy()
            meta = metas[0]
            
            meta['augmentation_type'] = 'stationary'
            meta['audio_idx'] = audio_idx
            
            np.save(os.path.join(audio_out, f'audio_{global_idx:06d}.npy'), wav)
            with open(os.path.join(label_out, f'label_{global_idx:06d}.pkl'), 'wb') as f:
                pickle.dump(meta, f)
            
            stats['stationary'] += 1
            global_idx += 1
            
        except Exception as e:
            print(f"Error at stationary {audio_idx}: {e}")
            continue
    
    # 1-2. Moving sources
    print(f"  Generating {N} moving sources (second half)")
    for audio_idx in tqdm(range(0, N), desc=f"Moving"):
        try:
            audio_sample = audio_dataset[audio_idx]
            if audio_sample is None:
                continue
                
            wavs, metas = collate_moving([audio_sample])
            wav = wavs[0].cpu().numpy()
            meta = metas[0]
            
            meta['augmentation_type'] = 'moving'
            meta['audio_idx'] = audio_idx

            audio_events[audio_idx] = {
                'events': set(meta.get('event', []) if isinstance(meta.get('event', []), list) else [meta.get('event', 0)]),
            }
            
            np.save(os.path.join(audio_out, f'audio_{global_idx:06d}.npy'), wav)
            with open(os.path.join(label_out, f'label_{global_idx:06d}.pkl'), 'wb') as f:
                pickle.dump(meta, f)
            
            stats['moving'] += 1
            global_idx += 1
            
        except Exception as e:
            print(f"Error at moving {audio_idx}: {e}")
            continue
    
    # Get all metadata
    for audio_idx in range(N):
        if audio_idx in audio_events:
            continue
        try:
            audio_sample = audio_dataset[audio_idx]
            if audio_sample is None:
                continue
            _, metas = collate_moving([audio_sample])
            meta = metas[0]
            audio_events[audio_idx] = {
                'events': set(meta.get('event', []) if isinstance(meta.get('event', []), list) else [meta.get('event', 0)]),
            }
        except Exception as e:
            print(f"Error at event extraction {audio_idx}: {e}")
            continue

    def has_event_overlap(idx1, idx2, audio_events):
        """Check for event overlap"""
        events1 = audio_events[idx1]['events']
        events2 = audio_events[idx2]['events']
        return len(events1 & events2) > 0
    
    def find_non_overlapping_pair(available_indices, audio_events):
        """Find a non-overlapping pair from available indices"""
        if len(available_indices) < 2:
            return None, None
        
        for i, idx1 in enumerate(available_indices):
            for idx2 in available_indices[i+1:]:            
                if has_event_overlap(idx1, idx2, audio_events):
                    continue
                return idx1, idx2
        return None, None
    
    # 2-1. Stationary + Stationary
    print(f"  Generating {half_N} stationary+stationary mixed sources...")
    available_indices = list(range(0, N))
    np.random.shuffle(available_indices)
    stat_stat_count = 0
    target_stat_stat = half_N
    
    with tqdm(total=target_stat_stat, desc="Stat+Stat") as pbar:
        while len(available_indices) >= 2 and stat_stat_count < target_stat_stat:
            idx1, idx2 = find_non_overlapping_pair(available_indices, audio_events)
            
            if idx1 is None:
                print(f"\nNo more non-overlapping stationary pairs found at {stat_stat_count}/{target_stat_stat}")
                break
            try:
                audio_sample1 = audio_dataset[idx1]
                audio_sample2 = audio_dataset[idx2]
                
                wavs, metas = collate_mixed_stat_stat([audio_sample1, audio_sample2])
                mixed = wavs[0].cpu().numpy()
                mixed_meta = metas[0]
                
                mixed_meta['caption'] = [audio_sample1[1].get('caption', ''), audio_sample2[1].get('caption', '')]
                
                np.save(os.path.join(audio_out, f'audio_{global_idx:06d}.npy'), mixed)
                with open(os.path.join(label_out, f'label_{global_idx:06d}.pkl'), 'wb') as f:
                    pickle.dump(mixed_meta, f)
                
                stats['mixed_stat_stat'] += 1
                global_idx += 1
                stat_stat_count += 1
                pbar.update(1)
                
                available_indices.remove(idx1)
                available_indices.remove(idx2)
                
            except Exception as e:
                print(f"\nError at mixed_stat_stat {idx1},{idx2}: {e}")
                available_indices.remove(idx1)
                continue
    
    # 2-2. Stationary + Moving
    print(f"  Generating {half_N} stationary+moving mixed sources...")
    available_indices = list(range(0, N))
    np.random.seed(43)
    np.random.shuffle(available_indices)
    np.random.seed(42)
    stat_mov_count = 0
    target_stat_mov = half_N
    
    with tqdm(total=target_stat_mov, desc="Stat+Mov") as pbar:
        while len(available_indices) >= 2 and stat_mov_count < target_stat_mov:
            idx_stat, idx_mov = find_non_overlapping_pair(available_indices, audio_events)
            
            if idx_stat is None:
                print(f"\nNo more non-overlapping stat+mov pairs found at {stat_mov_count}/{target_stat_mov}")
                break
            
            try:
                audio_sample_stat = audio_dataset[idx_stat]
                audio_sample_mov = audio_dataset[idx_mov]
                
                wavs, metas = collate_mixed_stat_mov([audio_sample_stat, audio_sample_mov])
                mixed = wavs[0].cpu().numpy()
                mixed_meta = metas[0]
                
                mixed_meta['caption'] = [audio_sample_stat[1].get('caption', ''), audio_sample_mov[1].get('caption', '')]
                
                np.save(os.path.join(audio_out, f'audio_{global_idx:06d}.npy'), mixed)
                with open(os.path.join(label_out, f'label_{global_idx:06d}.pkl'), 'wb') as f:
                    pickle.dump(mixed_meta, f)
                
                stats['mixed_stat_mov'] += 1
                global_idx += 1
                stat_mov_count += 1
                pbar.update(1)
                
                available_indices.remove(idx_stat)
                available_indices.remove(idx_mov)
                
            except Exception as e:
                print(f"\nError at mixed_stat_mov {idx_stat},{idx_mov}: {e}")
                available_indices.remove(idx_stat)
                continue

    print(f"  Generating additional {half_N} stationary+moving mixed sources")
    available_indices = list(range(0, N))
    np.random.seed(44)
    np.random.shuffle(available_indices)
    np.random.seed(42)
    stat_mov_count = 0
    target_stat_mov = half_N
    with tqdm(total=target_stat_mov, desc="Stat+Mov") as pbar:
        while len(available_indices) >= 2 and stat_mov_count < target_stat_mov:
            idx_mov, idx_stat = find_non_overlapping_pair(available_indices, audio_events)
            
            if idx_mov is None:
                print(f"\nNo more non-overlapping stat+mov pairs found at {stat_mov_count}/{target_stat_mov}")
                break
            
            try:
                audio_sample_mov = audio_dataset[idx_mov]
                audio_sample_stat = audio_dataset[idx_stat]
                
                wavs, metas = collate_mixed_stat_mov([audio_sample_stat, audio_sample_mov])
                mixed = wavs[0].cpu().numpy()
                mixed_meta = metas[0]
                
                mixed_meta['caption'] = [audio_sample_stat[1].get('caption', ''), audio_sample_mov[1].get('caption', '')]
                
                np.save(os.path.join(audio_out, f'audio_{global_idx:06d}.npy'), mixed)
                with open(os.path.join(label_out, f'label_{global_idx:06d}.pkl'), 'wb') as f:
                    pickle.dump(mixed_meta, f)
                
                stats['mixed_stat_mov'] += 1
                global_idx += 1
                stat_mov_count += 1
                pbar.update(1)
                
                available_indices.remove(idx_stat)
                available_indices.remove(idx_mov)
                
            except Exception as e:
                print(f"\nError at mixed_mov_stat {idx_mov},{idx_stat}: {e}")
                available_indices.remove(idx_mov)
                continue
    
    metadata = {
        'split': split,
        'num_samples': global_idx,
        'base_samples': N,
        'augmentation_ratio': global_idx / N,
        'statistics': stats,
        'sample_rate': 16000,
        'ch_num': 2,
        'augmentation_types': ['stationary', 'moving', 'mixed_stat_stat', 'mixed_stat_mov'],
    }
    
    metadata_path = os.path.join(out_dir, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved to: {metadata_path}")

def consolidate_to_memmap(data_dir, output_dir=None, chunk_size=10000):
    if output_dir is None:
        output_dir = data_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    audio_dir = os.path.join(data_dir, 'audio')
    label_dir = os.path.join(data_dir, 'label')

    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.npy')])
    total_samples = len(audio_files)
    num_chunks = (total_samples + chunk_size - 1) // chunk_size
    
    print(f"{'='*80}")
    print(f"Consolidating {total_samples} samples to chunked memory-mapped format")
    print(f"Source: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Chunk size: {chunk_size} samples/chunk")
    print(f"Total chunks: {num_chunks}")
    print(f"{'='*80}")

    max_length = 0
    lengths = []
    sample_size = min(1000, total_samples)
    
    for audio_file in tqdm(audio_files[:sample_size], desc="Sampling"):
        audio_path = os.path.join(audio_dir, audio_file)
        wav = np.load(audio_path)  # (2, time)
        lengths.append(wav.shape[1])
        max_length = max(max_length, wav.shape[1])
    
    max_length = int(max_length * 1.05)

    all_metas = []
    all_shape_info = []
    chunk_info = []
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_samples)
        chunk_files = audio_files[start_idx:end_idx]
        actual_chunk_size = len(chunk_files)
        
        print(f"\nProcessing chunk {chunk_idx+1}/{num_chunks} ({actual_chunk_size} samples)...")
        
        chunk_audio_path = os.path.join(output_dir, f'audio_chunk_{chunk_idx:03d}.npy')
        
        audio_memmap = np.lib.format.open_memmap(
            chunk_audio_path,
            mode='w+',
            dtype=np.float32,
            shape=(actual_chunk_size, 2, max_length)
        )
        
        chunk_metas = []
        chunk_shapes = []
        
        for i, audio_file in enumerate(tqdm(chunk_files, desc=f"Chunk {chunk_idx+1}")):
            audio_path = os.path.join(audio_dir, audio_file)
            try:
                wav = np.load(audio_path)  # (2, time)
            except Exception as e:
                print(f"Error loading {audio_file}: {e}")
            
            label_file = audio_file.replace('audio_', 'label_').replace('.npy', '.pkl')
            label_path = os.path.join(label_dir, label_file)
            with open(label_path, 'rb') as f:
                meta = pickle.load(f)
            
            actual_length = wav.shape[1]
            if actual_length > max_length:
                wav = wav[:, :max_length]
                actual_length = max_length
            
            audio_memmap[i, :, :actual_length] = wav
            
            chunk_metas.append(meta)
            chunk_shapes.append(actual_length)
        
        audio_memmap.flush()
        del audio_memmap
        all_metas.extend(chunk_metas)
        all_shape_info.extend(chunk_shapes)
        chunk_info.append({
            'chunk_id': chunk_idx,
            'file': f'audio_chunk_{chunk_idx:03d}.npy',
            'start_idx': start_idx,
            'end_idx': end_idx,
            'size': actual_chunk_size
        })
        
        print(f"Chunk {chunk_idx+1} saved: {chunk_audio_path}")
    
    print(f"\nSaving metadata...")
    output_label_path = os.path.join(output_dir, 'labels.pkl')
    output_chunk_info_path = os.path.join(output_dir, 'chunk_info.pkl')
    
    with open(output_label_path, 'wb') as f:
        pickle.dump({
            'metas': all_metas,
            'shape_info': all_shape_info,
            'max_length': max_length,
            'total_samples': total_samples
        }, f)
    
    with open(output_chunk_info_path, 'wb') as f:
        pickle.dump({
            'chunks': chunk_info,
            'chunk_size': chunk_size,
            'num_chunks': num_chunks,
            'total_samples': total_samples,
            'max_length': max_length
        }, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate precomputed data')
    parser.add_argument('--consolidate', action='store_true',
                       help='Consolidate precomputed data into memory-mapped format')
    parser.add_argument('--chunk_size', type=int, default=10000,
                       help='Chunk size for memory-mapped files (default: 10000)')
    parser.add_argument('--rir_base_dir', type=str, 
                       default='data/moving_rir_dataset',
                       help='RIR base directory (contains moving/ and stationary/)')
    parser.add_argument('--out_root', type=str, default='output_moving/simulated_spatial_sound')
    args = parser.parse_args()
    
    np.random.seed(42)
    if args.consolidate:
        print("Consolidating data to memory-mapped format...")
        for split in ["train","val"]: #"train", "val", "test"
            data_dir = os.path.join(args.out_root, split)
            if os.path.exists(data_dir):
                consolidate_to_memmap(data_dir, chunk_size=args.chunk_size)
            else:
                print(f"Skipping {split}: directory not found")
        print("All splits consolidated!")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        config = {
            "wav_dir": "data/wav/",
            "csv_path": "data/fixed_audiocaps/train.csv",
            "val_csv_path": "data/fixed_audiocaps/val.csv",
            "test_csv_path": "data/fixed_audiocaps/test.csv",
        }
        
        for split in ["train","val"]:
            if split == "val":
                config["csv_path"] = config["val_csv_path"]
            elif split == "test":
                config["csv_path"] = config["test_csv_path"]
            
            out_dir = os.path.join(args.out_root, split)
            
            save_precomputed_simulated_spatial(
                split, 
                out_dir, 
                config, 
                args.rir_base_dir,
                device
            )
