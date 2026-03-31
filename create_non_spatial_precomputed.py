"""
create non spatial precomputed dataset
python create_non_spatial_precomputed.py --all_splits
python create_non_spatial_precomputed.py --split train
"""
import os
import sys
import numpy as np
import pickle
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import torch
import torchaudio


def create_non_spatial_dataset(
    csv_path: str,
    wav_dir: str,
    output_dir: str,
    split: str = "train",
    sample_rate: int = 16000,
    chunk_size: int = 10000,
    max_length: int = 160000  # 10s @ 16kHz
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    grouped = df.groupby(['youtube_id', 'start_time']).agg({
        'caption': list,
        'audiocap_id': list
    }).reset_index()
    
    captions_per_audio = df.groupby(['youtube_id', 'start_time']).size()
    
    print(f"\n{'='*80}")
    print(f"Creating Non-Spatial Dataset: {split}")
    print(f"{'='*80}")
    print(f"CSV: {csv_path}")
    print(f"WAV dir: {wav_dir}")
    print(f"Total rows in CSV: {len(df)}")
    print(f"Unique audio files: {len(grouped)}")
    print(f"Captions per audio: {captions_per_audio.mean():.1f}")
    print(f"Chunk size: {chunk_size}")
    print(f"Max length: {max_length} samples ({max_length / sample_rate:.2f} sec)")
    
    if split == 'train':
        expand_captions = False 
        print(f"Mode: Train (1 audio = 1 caption, no expansion)")
    else:
        expand_captions = True
        print(f"Mode: Val/Test (1 audio = {captions_per_audio.max():.0f} captions, full expansion)")

    all_metas = []
    all_shape_info = []
    chunks_info = []
    
    chunk_audios = []
    chunk_idx = 0
    total_samples = 0
    skipped = 0
    
    for idx, row in tqdm(grouped.iterrows(), total=len(grouped), desc="Processing audio"):
        youtube_id = row['youtube_id']
        start_time = row['start_time']
        captions = row['caption']
        audiocap_ids = row['audiocap_id']
        
        filename = f"{youtube_id}_{start_time}.wav"
        wav_path = os.path.join(wav_dir, filename)
        
        if not os.path.exists(wav_path):
            skipped += 1
            continue
        
        try:
            waveform, sr = torchaudio.load(wav_path)

            if sr != sample_rate:
                waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            audio = waveform[0].numpy()  # (T,)
            actual_length = len(audio)
            
            if len(audio) > max_length:
                audio = audio[:max_length]
                actual_length = max_length
            elif len(audio) < max_length:
                audio = np.pad(audio, (0, max_length - len(audio)))
            
            audio_stereo = np.stack([audio, audio], axis=0)  # (2, max_length)
            
            num_frames = max(1, actual_length // 1024)
            frame_doas = np.full(num_frames, 0.0, dtype=np.float32)
            frame_times = np.linspace(0.064, actual_length / sample_rate, num_frames).astype(np.float32)
            
            if expand_captions:
                caption_indices = range(len(captions))
            else:
                caption_indices = [0]
            
            for caption_idx in caption_indices:
                caption = captions[caption_idx]
                audiocap_id = audiocap_ids[caption_idx]
                
                chunk_audios.append(audio_stereo)
                all_shape_info.append(actual_length)
                
                meta = {
                    'audiocap_id': audiocap_id,
                    'youtube_id': youtube_id,
                    'start_time': start_time,
                    'caption': caption,
                    'augmentation_type': 'non_spatial',
                    'is_stationary': True,
                    'is_non_spatial': True,
                    'movement_type': 'non_spatial',
                    'doa': 0.0,
                    'start_doa': 0.0,
                    'end_doa': 0.0,
                    'doa_change': 0.0,
                    'start_zone': 'front',
                    'end_zone': 'front',
                    'frame_doas': frame_doas,
                    'frame_times': frame_times,
                    'num_frames': num_frames,
                    'all_captions': captions,
                    'caption_index': caption_idx,
                    'num_captions': len(captions),
                }
                
                all_metas.append(meta)
                total_samples += 1

                if len(chunk_audios) >= chunk_size:
                    _save_chunk(output_dir, chunk_idx, chunk_audios, chunks_info)
                    chunk_idx += 1
                    chunk_audios = []
            
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            skipped += 1
            continue

    if len(chunk_audios) > 0:
        _save_chunk(output_dir, chunk_idx, chunk_audios, chunks_info)
        chunk_idx += 1
    
    print(f"\nProcessed: {len(grouped) - skipped} audio files")
    print(f"Total samples: {total_samples}")
    print(f"Skipped: {skipped} files")
    print(f"Total chunks: {chunk_idx}")
    
    # chunk_info.pkl
    chunk_info = {
        'chunks': chunks_info,
        'chunk_size': chunk_size,
        'num_chunks': chunk_idx,
        'total_samples': total_samples,
        'max_length': max_length
    }
    
    chunk_info_path = os.path.join(output_dir, "chunk_info.pkl")
    with open(chunk_info_path, 'wb') as f:
        pickle.dump(chunk_info, f)
    print(f"Saved: {chunk_info_path}")
    
    # labels.pkl
    labels_data = {
        'metas': all_metas,
        'shape_info': all_shape_info,
        'max_length': max_length,
        'total_samples': total_samples
    }
    
    labels_path = os.path.join(output_dir, "labels.pkl")
    with open(labels_path, 'wb') as f:
        pickle.dump(labels_data, f)
    print(f"Saved: {labels_path}")
    
    print(f"Output directory: {output_dir}")
    print(f"Total samples: {total_samples}")
    print(f"Number of chunks: {chunk_idx}")
    print(f"Chunk size: {chunk_size}")
    print(f"Audio shape per sample: (2, {max_length})")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {max_length / sample_rate:.2f} sec")
    print(f"{'='*80}\n")
    
    return total_samples


def _save_chunk(output_dir, chunk_idx, chunk_audios, chunks_info):
    chunk_array = np.stack(chunk_audios, axis=0).astype(np.float32)  # (N, 2, T)
    chunk_path = os.path.join(output_dir, f"audio_chunk_{chunk_idx:03d}.npy")
    np.save(chunk_path, chunk_array)
    
    start_idx = sum(c['num_samples'] for c in chunks_info)
    chunks_info.append({
        'chunk_id': chunk_idx,
        'start_idx': start_idx,
        'end_idx': start_idx + len(chunk_audios),
        'num_samples': len(chunk_audios)
    })
    
    print(f"  Saved chunk {chunk_idx}: {chunk_array.shape} ({chunk_array.nbytes / 1e9:.2f} GB)")


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--csv_dir', type=str, 
                       default='data/fixed_audiocaps',
                       help='Directory containing AudioCaps CSVs (train.csv, val.csv, test.csv)')
    parser.add_argument('--wav_dir', type=str,
                       default='data/wav',
                       help='Directory containing WAV files')
    parser.add_argument('--output_dir', type=str,
                       default='output_moving/non_spatial_precomputed',
                       help='Output root directory')
    parser.add_argument('--sample_rate', type=int, default=16000,
                       help='Target sample rate (Hz)')
    parser.add_argument('--chunk_size', type=int, default=10000,
                       help='Samples per chunk (smaller = less memory, more files)')
    parser.add_argument('--max_length', type=int, default=160000,
                       help='Max audio length in samples (default: 160000 = 10sec @ 16kHz)')
    
    # Split selection
    parser.add_argument('--split', type=str, default=None,
                       choices=['train', 'val', 'test'],
                       help='Single split to process')
    parser.add_argument('--all_splits', action='store_true',
                       help='Process all splits (train/val/test)')
    
    args = parser.parse_args()
    
    # Determine which splits to process
    if args.all_splits:
        splits = ['train', 'val', 'test']
    elif args.split:
        splits = [args.split]
    else:
        # Default: train and val only (for training)
        splits = ['train', 'val']
        print("  No split specified. Processing train and val by default.")
    
    # Process each split
    success_count = 0
    for split_name in splits:
        csv_path = os.path.join(args.csv_dir, f"{split_name}.csv")
        output_dir = os.path.join(args.output_dir, split_name)
        
        if not os.path.exists(csv_path):
            print(f"CSV not found: {csv_path}, skipping {split_name}")
            continue
        
        count = create_non_spatial_dataset(
            csv_path=csv_path,
            wav_dir=args.wav_dir,
            output_dir=output_dir,
            split=split_name,
            sample_rate=args.sample_rate,
            chunk_size=args.chunk_size,
            max_length=args.max_length
        )
        
        if count > 0:
            success_count += 1
    
    # Final summary
    print("\n" + "="*80)
    print(f"Dataset Creation Complete!")
    print("="*80)
    print(f"Successfully processed: {success_count}/{len(splits)} splits")
    print(f"Output directory: {args.output_dir}")
    print("\nUsage in training:")
    print("  from clap_dataset import NonSpatialDataset")
    print(f"  dataset = NonSpatialDataset('{args.output_dir}', split='train')")
    print("\nMetadata format:")
    print("  - Dict format (new format, compatible with moving_collate_refactored.py)")
    print("  - No list wrapping (meta is dict, not [dict])")
    print("  - augmentation_type: 'non_spatial'")
    print("="*80)

if __name__ == '__main__':
    main()
