#!/usr/bin/env python3
"""
SELDモデル評価スクリプト
使用方法:
    # 単一音源のみ評価
    python evaluate_all_patterns.py --checkpoint output_beats_seldnet/ckpt/best_model.pt --pattern_type single
    
    # 混合音源のみ評価
    python evaluate_all_patterns.py --checkpoint output_beats_seldnet/ckpt/best_model.pt --pattern_type mixed
    
    # 両方評価
    python evaluate_all_patterns.py --checkpoint output_beats_seldnet/ckpt/best_model.pt --pattern_type both
    
    # プロットのみ（パターン毎10枚）
    python evaluate_all_patterns.py --checkpoint output_beats_seldnet/ckpt/best_model.pt --pattern_type both --plot_only --max_viz_per_pattern 10
"""

import torch
import argparse
import os
import sys
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import json
import seaborn as sns
from util import set_seed
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings('ignore')

# 親ディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clap_dataset import AllPatternDataset


def circular_interpolate(doa_seq, target_len):
    """DOAシーケンスをターゲット長に合わせて補間（単位円上での補間）"""
    if isinstance(doa_seq, torch.Tensor):
        doa_np = doa_seq.detach().cpu().numpy()
    else:
        doa_np = np.array(doa_seq, dtype=np.float32)
    
    if doa_np.ndim > 1:
        doa_np = doa_np.flatten()
    
    if len(doa_np) == target_len:
        return torch.from_numpy(doa_np.astype(np.float32)).float()
    
    x_orig = np.linspace(0, 1, len(doa_np))
    x_target = np.linspace(0, 1, target_len)
    
    cos_vals = np.cos(doa_np * np.pi / 2)
    sin_vals = np.sin(doa_np * np.pi / 2)
    
    cos_interp = np.interp(x_target, x_orig, cos_vals)
    sin_interp = np.interp(x_target, x_orig, sin_vals)
    
    norm = np.sqrt(cos_interp**2 + sin_interp**2 + 1e-8)
    cos_interp /= norm
    sin_interp /= norm
    
    angles = np.arctan2(sin_interp, cos_interp) / (np.pi / 2)
    
    return torch.from_numpy(angles.astype(np.float32)).float()


# =====================================================
# Dataset Wrapper
# =====================================================

class AllPatternEvalDataset(Dataset):
    """AllPatternDataset用の評価ラッパー"""
    
    def __init__(self, pattern_root, pattern_type='single'):
        """
        Args:
            pattern_root: パターンデータのルート (output_moving/test_datasets)
            pattern_type: 'single' or 'mixed'
        """
        self.dataset = AllPatternDataset(pattern_root, pattern_type=pattern_type)
        self.pattern_type = pattern_type
        
        print(f"✅ AllPatternEvalDataset - Type: {pattern_type}")
        print(f"   Total samples: {len(self.dataset)}")
        print(f"   Audio/Pair groups: {len(self.dataset.pattern_groups)}")
        print(f"   min audio_ids: {min(self.dataset.pattern_groups.keys())}, max audio_ids: {max(self.dataset.pattern_groups.keys())}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        wav, meta = self.dataset[idx]
        
        # eval_patternを追加
        if isinstance(meta, dict):
            if meta.get('mixed', False):
                trajectory_type = meta.get('trajectory_type', 'unknown')
                if trajectory_type == 'linear':
                    meta['eval_pattern'] = 'mixed_stat_mov_linear'
                elif trajectory_type == 'circular':
                    meta['eval_pattern'] = 'mixed_stat_mov_circular'
                else:
                    meta['eval_pattern'] = f'mixed_stat_stat'
            else:
                trajectory_type = meta.get('trajectory_type', 'unknown')
                if meta.get('is_stationary', True):
                    meta['eval_pattern'] = 'single_stationary'
                elif trajectory_type in ['linear', 'Linear']:
                    meta['eval_pattern'] = 'single_moving_linear'
                elif trajectory_type in ['circular', 'Circular']:
                    meta['eval_pattern'] = 'single_moving_circular'
                else:
                    meta['eval_pattern'] = f'single_{trajectory_type}'
        
        return wav, meta


def collate_fn(batch):
    """共通のcollate関数"""
    wavs = []
    metas = []
    
    for wav, meta in batch:
        wavs.append(wav)
        
        if isinstance(meta, dict):
            if 'event' not in meta and not meta.get('mixed', False):
                meta['event'] = [0]
            if 'frame_doas' not in meta and not meta.get('mixed', False):
                doa = meta.get('doa', 0.0)
                num_frames = meta.get('num_frames', 100)
                meta['frame_doas'] = torch.full((num_frames,), doa)
            metas.append(meta)
        elif isinstance(meta, list):
            for m in meta:
                if isinstance(m, dict):
                    if 'event' not in m:
                        m['event'] = [0]
                    if 'frame_doas' not in m:
                        doa = m.get('doa', 0.0)
                        num_frames = m.get('num_frames', 100)
                        m['frame_doas'] = torch.full((num_frames,), doa)
            metas.append(meta)
        else:
            metas.append(meta)
    
    max_len = max(w.shape[-1] for w in wavs)
    padded_wavs = []
    for w in wavs:
        if w.shape[-1] < max_len:
            pad = max_len - w.shape[-1]
            w = torch.nn.functional.pad(w, (0, pad))
        padded_wavs.append(w)
    
    wavs_tensor = torch.stack(padded_wavs, dim=0)
    
    return wavs_tensor, metas


# =====================================================
# Metrics
# =====================================================

class FrameLevelMetrics:
    """Frame-level SED + DOA評価メトリクス"""
    
    def __init__(self, num_classes, sed_threshold=0.5, doa_threshold=10.0, 
                 temporal_chunk_seconds=2.0, hop_length=512, sr=16000):
        self.num_classes = num_classes
        self.sed_threshold = sed_threshold
        self.doa_threshold = doa_threshold
        self.temporal_chunk_seconds = temporal_chunk_seconds
        self.hop_length = hop_length
        self.sr = sr
        self.frames_per_chunk = int((temporal_chunk_seconds * sr) / hop_length)
        self.reset()
    
    def reset(self):
        self.sed_tp = 0
        self.sed_fp = 0
        self.sed_fn = 0
        self.sed_tn = 0
        self.doa_errors = []
        self.doa_correct = 0
        self.doa_total = 0
        self.temporal_doa_accuracies = []
        self.temporal_chunk_details = []  # 各チャンクの詳細情報
        
        # Single/Mixed別統計
        self.single_sed_tp = 0
        self.single_sed_fp = 0
        self.single_sed_fn = 0
        self.single_doa_errors = []
        self.single_doa_correct = 0
        self.single_doa_total = 0
        self.single_count = 0
        
        self.mixed_sed_tp = 0
        self.mixed_sed_fp = 0
        self.mixed_sed_fn = 0
        self.mixed_doa_errors = []
        self.mixed_doa_correct = 0
        self.mixed_doa_total = 0
        self.mixed_count = 0
        
        # パターン別統計
        self.pattern_stats = defaultdict(lambda: {
            'sed_tp': 0, 'sed_fp': 0, 'sed_fn': 0,
            'doa_errors': [], 'doa_correct': 0, 'doa_total': 0, 'count': 0
        })
        
        # Audio/Pair ID別統計（25パターン/180パターン単位）
        self.audio_stats = defaultdict(lambda: {
            'doa_errors': [], 'count': 0
        })
    
    def update(self, sed_probs, doa_pred, sed_target, doa_target, doa_mask,
               is_mixed=False, pattern=None, audio_id=None):
        """
        Args:
            audio_id: 音源ID（単一音源）またはペアID（混合音源）
        """
        sed_probs = sed_probs.cpu().numpy()
        doa_pred = doa_pred.cpu().numpy()
        sed_target = sed_target.cpu().numpy()
        doa_target = doa_target.cpu().numpy()
        doa_mask = doa_mask.cpu().numpy()
        
        if doa_pred.ndim == 3 and doa_pred.shape[-1] == 2:
            pred_rad = np.arctan2(doa_pred[..., 1], doa_pred[..., 0])
            doa_pred_deg = pred_rad * 180 / np.pi
        else:
            doa_pred_deg = doa_pred * 90

        sed_pred = (sed_probs >= self.sed_threshold).astype(int)
        
        tp = np.sum((sed_pred == 1) & (sed_target == 1))
        fp = np.sum((sed_pred == 1) & (sed_target == 0))
        fn = np.sum((sed_pred == 0) & (sed_target == 1))
        tn = np.sum((sed_pred == 0) & (sed_target == 0))
        
        self.sed_tp += tp
        self.sed_fp += fp
        self.sed_fn += fn
        self.sed_tn += tn
        
        # Audio/Pair ID別のカウント（すべてのサンプルをカウント）
        if audio_id is not None:
            self.audio_stats[audio_id]['count'] += 1
        
        # DOA評価
        gt_active_mask = (doa_mask > 0.5)
        if np.sum(gt_active_mask) > 0:
            pred_angles = doa_pred_deg[gt_active_mask]
            target_angles = doa_target[gt_active_mask] * 90
            diff = np.abs(pred_angles - target_angles)
            self.doa_errors.extend(diff.tolist())
            self.doa_correct += np.sum(diff < self.doa_threshold)
            self.doa_total += len(diff)
            
            # Audio/Pair ID別のDOA誤差集計
            if audio_id is not None:
                self.audio_stats[audio_id]['doa_errors'].extend(diff.tolist())
        
        # 時間区間別評価
        if doa_pred_deg.ndim >= 2:
            num_frames = doa_pred_deg.shape[0]
            num_chunks = (num_frames + self.frames_per_chunk - 1) // self.frames_per_chunk
            
            sample_temporal_accuracies = []
            for chunk_idx in range(num_chunks):
                start_frame = chunk_idx * self.frames_per_chunk
                end_frame = min((chunk_idx + 1) * self.frames_per_chunk, num_frames)
                chunk_mask = gt_active_mask[start_frame:end_frame]
                
                # 時間範囲（秒）を計算
                start_time = start_frame * self.hop_length / self.sr
                end_time = end_frame * self.hop_length / self.sr
                
                if np.sum(chunk_mask) > 0:
                    chunk_pred = doa_pred_deg[start_frame:end_frame][chunk_mask]
                    chunk_target = doa_target[start_frame:end_frame][chunk_mask] * 90
                    chunk_diff = np.abs(chunk_pred - chunk_target)
                    chunk_correct = np.sum(chunk_diff < self.doa_threshold)
                    chunk_total = len(chunk_diff)
                    chunk_accuracy = chunk_correct / chunk_total if chunk_total > 0 else 0.0
                    sample_temporal_accuracies.append(chunk_accuracy)
                    
                    # チャンクの詳細情報を保存
                    self.temporal_chunk_details.append({
                        'chunk_idx': chunk_idx,
                        'start_time': start_time,
                        'end_time': end_time,
                        'accuracy': chunk_accuracy,
                        'correct': int(chunk_correct),
                        'total': int(chunk_total),
                        'pattern': pattern,
                        'audio_id': audio_id
                    })
            
            if sample_temporal_accuracies:
                self.temporal_doa_accuracies.extend(sample_temporal_accuracies)
        
        # Single/Mixed別集計
        if is_mixed:
            self.mixed_sed_tp += tp
            self.mixed_sed_fp += fp
            self.mixed_sed_fn += fn
            self.mixed_count += 1
            if np.sum(gt_active_mask) > 0:
                pred_angles = doa_pred_deg[gt_active_mask]
                target_angles = doa_target[gt_active_mask] * 90
                diff = np.abs(pred_angles - target_angles)
                self.mixed_doa_errors.extend(diff.tolist())
                self.mixed_doa_correct += np.sum(diff < self.doa_threshold)
                self.mixed_doa_total += len(diff)
        else:
            self.single_sed_tp += tp
            self.single_sed_fp += fp
            self.single_sed_fn += fn
            self.single_count += 1
            if np.sum(gt_active_mask) > 0:
                pred_angles = doa_pred_deg[gt_active_mask]
                target_angles = doa_target[gt_active_mask] * 90
                diff = np.abs(pred_angles - target_angles)
                self.single_doa_errors.extend(diff.tolist())
                self.single_doa_correct += np.sum(diff < self.doa_threshold)
                self.single_doa_total += len(diff)
        
        # パターン別集計
        if pattern:
            pstats = self.pattern_stats[pattern]
            pstats['sed_tp'] += tp
            pstats['sed_fp'] += fp
            pstats['sed_fn'] += fn
            pstats['count'] += 1
            if np.sum(gt_active_mask) > 0:
                pred_angles = doa_pred_deg[gt_active_mask]
                target_angles = doa_target[gt_active_mask] * 90
                diff = np.abs(pred_angles - target_angles)
                pstats['doa_errors'].extend(diff.tolist())
                pstats['doa_correct'] += np.sum(diff < self.doa_threshold)
                pstats['doa_total'] += len(diff)
    
    def compute(self):
        sed_precision = self.sed_tp / (self.sed_tp + self.sed_fp) if (self.sed_tp + self.sed_fp) > 0 else 0.0
        sed_recall = self.sed_tp / (self.sed_tp + self.sed_fn) if (self.sed_tp + self.sed_fn) > 0 else 0.0
        sed_f1 = 2 * sed_precision * sed_recall / (sed_precision + sed_recall) if (sed_precision + sed_recall) > 0 else 0.0
        sed_accuracy = (self.sed_tp + self.sed_tn) / (self.sed_tp + self.sed_fp + self.sed_fn + self.sed_tn) if (self.sed_tp + self.sed_fp + self.sed_fn + self.sed_tn) > 0 else 0.0
        
        if self.doa_errors:
            doa_mae = np.mean(self.doa_errors)
            doa_std = np.std(self.doa_errors)
            doa_median = np.median(self.doa_errors)
            doa_accuracy = self.doa_correct / self.doa_total if self.doa_total > 0 else 0.0
        else:
            doa_mae = doa_std = doa_median = doa_accuracy = 0.0
        
        if self.temporal_doa_accuracies:
            temporal_doa_mean = np.mean(self.temporal_doa_accuracies)
            temporal_doa_std = np.std(self.temporal_doa_accuracies)
        else:
            temporal_doa_mean = temporal_doa_std = 0.0
        
        # チャンク別統計を計算
        chunk_stats = {}
        if self.temporal_chunk_details:
            # チャンクインデックス別に集計
            chunk_groups = defaultdict(lambda: {'accuracies': [], 'correct': 0, 'total': 0})
            for detail in self.temporal_chunk_details:
                chunk_idx = detail['chunk_idx']
                chunk_groups[chunk_idx]['accuracies'].append(detail['accuracy'])
                chunk_groups[chunk_idx]['correct'] += detail['correct']
                chunk_groups[chunk_idx]['total'] += detail['total']
                if 'start_time' not in chunk_groups[chunk_idx]:
                    chunk_groups[chunk_idx]['start_time'] = detail['start_time']
                    chunk_groups[chunk_idx]['end_time'] = detail['end_time']
            
            for chunk_idx, stats in chunk_groups.items():
                chunk_stats[chunk_idx] = {
                    'start_time': stats['start_time'],
                    'end_time': stats['end_time'],
                    'mean_accuracy': np.mean(stats['accuracies']),
                    'correct': stats['correct'],
                    'total': stats['total'],
                    'samples': len(stats['accuracies'])
                }
        
        results = {
            'sed': {
                'precision': sed_precision,
                'recall': sed_recall,
                'f1': sed_f1,
                'accuracy': sed_accuracy,
                'tp': self.sed_tp,
                'fp': self.sed_fp,
                'fn': self.sed_fn
            },
            'doa': {
                'mae': doa_mae,
                'std': doa_std,
                'median': doa_median,
                'accuracy': doa_accuracy,
                'threshold': self.doa_threshold,
                'correct': self.doa_correct,
                'total': self.doa_total
            },
            'temporal_doa': {
                'mean_accuracy': temporal_doa_mean,
                'std_accuracy': temporal_doa_std,
                'chunk_seconds': self.temporal_chunk_seconds,
                'chunk_stats': chunk_stats,
                'total_chunks': len(self.temporal_doa_accuracies)
            }
        }
        
        # Single
        if self.single_count > 0:
            single_precision = self.single_sed_tp / (self.single_sed_tp + self.single_sed_fp) if (self.single_sed_tp + self.single_sed_fp) > 0 else 0.0
            single_recall = self.single_sed_tp / (self.single_sed_tp + self.single_sed_fn) if (self.single_sed_tp + self.single_sed_fn) > 0 else 0.0
            single_f1 = 2 * single_precision * single_recall / (single_precision + single_recall) if (single_precision + single_recall) > 0 else 0.0
            
            if self.single_doa_errors:
                single_doa_mae = np.mean(self.single_doa_errors)
                single_doa_accuracy = self.single_doa_correct / self.single_doa_total if self.single_doa_total > 0 else 0.0
            else:
                single_doa_mae = single_doa_accuracy = 0.0
            
            results['single'] = {
                'sed': {'precision': single_precision, 'recall': single_recall, 'f1': single_f1},
                'doa': {'mae': single_doa_mae, 'accuracy': single_doa_accuracy},
                'count': self.single_count
            }
        
        # Mixed
        if self.mixed_count > 0:
            mixed_precision = self.mixed_sed_tp / (self.mixed_sed_tp + self.mixed_sed_fp) if (self.mixed_sed_tp + self.mixed_sed_fp) > 0 else 0.0
            mixed_recall = self.mixed_sed_tp / (self.mixed_sed_tp + self.mixed_sed_fn) if (self.mixed_sed_tp + self.mixed_sed_fn) > 0 else 0.0
            mixed_f1 = 2 * mixed_precision * mixed_recall / (mixed_precision + mixed_recall) if (mixed_precision + mixed_recall) > 0 else 0.0
            
            if self.mixed_doa_errors:
                mixed_doa_mae = np.mean(self.mixed_doa_errors)
                mixed_doa_accuracy = self.mixed_doa_correct / self.mixed_doa_total if self.mixed_doa_total > 0 else 0.0
            else:
                mixed_doa_mae = mixed_doa_accuracy = 0.0
            
            results['mixed'] = {
                'sed': {'precision': mixed_precision, 'recall': mixed_recall, 'f1': mixed_f1},
                'doa': {'mae': mixed_doa_mae, 'accuracy': mixed_doa_accuracy},
                'count': self.mixed_count
            }
        
        # パターン別結果
        if self.pattern_stats:
            results['patterns'] = {}
            for pattern, pstats in self.pattern_stats.items():
                if pstats['count'] > 0:
                    p_precision = pstats['sed_tp'] / (pstats['sed_tp'] + pstats['sed_fp']) if (pstats['sed_tp'] + pstats['sed_fp']) > 0 else 0.0
                    p_recall = pstats['sed_tp'] / (pstats['sed_tp'] + pstats['sed_fn']) if (pstats['sed_tp'] + pstats['sed_fn']) > 0 else 0.0
                    p_f1 = 2 * p_precision * p_recall / (p_precision + p_recall) if (p_precision + p_recall) > 0 else 0.0
                    
                    if pstats['doa_errors']:
                        p_doa_mae = np.mean(pstats['doa_errors'])
                        p_doa_accuracy = pstats['doa_correct'] / pstats['doa_total'] if pstats['doa_total'] > 0 else 0.0
                    else:
                        p_doa_mae = p_doa_accuracy = 0.0
                    
                    results['patterns'][pattern] = {
                        'sed': {'precision': p_precision, 'recall': p_recall, 'f1': p_f1},
                        'doa': {'mae': p_doa_mae, 'accuracy': p_doa_accuracy},
                        'count': pstats['count']
                    }
        
        # Audio/Pair ID別結果
        if self.audio_stats:
            results['audio_stats'] = {}
            for audio_id, astats in self.audio_stats.items():
                if astats['doa_errors']:
                    results['audio_stats'][audio_id] = {
                        'doa_mae': np.mean(astats['doa_errors']),
                        'count': astats['count']
                    }
        
        return results


# =====================================================
# Model Loading
# =====================================================

def load_model(num_classes, device, checkpoint_path):
    from beats_seldnet_model import BEATsSELDnetModel
    model = BEATsSELDnetModel(
        num_classes=num_classes,
        use_temporal_decoder=True,
        pretrained_model_name='BEATs',
        freeze_pretrained=True
    ).to(device)
    print("📦 BEATs SELDNet Model")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"✅ Checkpoint loaded: {checkpoint_path}")
    
    return model


# =====================================================
# Evaluation
# =====================================================

def evaluate(model, dataloader, device, num_classes,
             sed_threshold=0.5, doa_threshold=10.0):
    """評価実行"""
    model.eval()
    metrics = FrameLevelMetrics(num_classes, sed_threshold, doa_threshold)
    
    with torch.no_grad():
        for batch_idx, (audios, metas) in enumerate(tqdm(dataloader, desc="Evaluating BEATs SELDNet")):
            audios = audios.to(device)
            batch_size = audios.shape[0]
            
            # モデル出力
            sed_output, doa_output = model(audios)
            sed_probs = torch.sigmoid(sed_output)
            
            # ターゲット生成と評価
            for b in range(batch_size):
                meta = metas[b]
                
                # パターン情報を取得
                eval_pattern = None
                audio_id = None
                if isinstance(meta, dict):
                    eval_pattern = meta.get('eval_pattern', None)
                    # audio_idx (単一音源) または pair_idx (混合音源) を取得
                    # 0も有効な値なので、or演算子ではなく明示的にNoneチェック
                    audio_id = meta.get('audio_idx') if 'audio_idx' in meta else meta.get('pair_idx')
                
                # BEATs SELDNet: フレーム単位評価
                num_frames = sed_probs.shape[1]

                sed_target = torch.zeros((num_frames, num_classes), device=device)
                doa_target = torch.zeros((num_frames, num_classes), device=device)
                doa_mask = torch.zeros((num_frames, num_classes), device=device)

                if isinstance(meta, dict):
                    if meta.get('mixed', False):
                        meta_list = meta.get('source_metas', [meta])
                    else:
                        meta_list = [meta]
                else:
                    meta_list = [meta] if not isinstance(meta, list) else meta

                # 混合音源の場合
                if isinstance(meta, dict) and meta.get('mixed', False):
                    source_metas = meta.get('source_metas', [])
                    frame_doas_mixed = meta.get('frame_doas', None)

                    if frame_doas_mixed is not None and len(source_metas) == 2:
                        if not isinstance(frame_doas_mixed, torch.Tensor):
                            if isinstance(frame_doas_mixed, np.ndarray):
                                frame_doas_t = torch.from_numpy(frame_doas_mixed).float()
                            else:
                                frame_doas_t = torch.tensor(frame_doas_mixed).float()
                        else:
                            frame_doas_t = frame_doas_mixed.clone()

                        if frame_doas_t.ndim == 2 and frame_doas_t.shape[1] == 2:
                            for source_idx, source_meta in enumerate(source_metas):
                                if source_idx >= 2:
                                    break

                                events = source_meta.get('event', [0])
                                if not isinstance(events, list):
                                    events = [events]

                                src_doas = frame_doas_t[:, source_idx]

                                if len(src_doas) != num_frames:
                                    src_doas = circular_interpolate(src_doas, num_frames)

                                for event_idx in events:
                                    if event_idx < num_classes:
                                        sed_target[:, event_idx] = 1.0
                                        doa_target[:, event_idx] = src_doas.to(device)
                                        doa_mask[:, event_idx] = 1.0
                else:
                    # 単一音源
                    for m in meta_list:
                        if not isinstance(m, dict):
                            continue

                        events = m.get('event', [0])
                        frame_doas = m.get('frame_doas', None)

                        if not isinstance(events, list):
                            events = [events]

                        for event_idx in events:
                            if event_idx < num_classes:
                                sed_target[:, event_idx] = 1.0

                                if frame_doas is not None:
                                    if not isinstance(frame_doas, torch.Tensor):
                                        if isinstance(frame_doas, np.ndarray):
                                            frame_doas_t = torch.from_numpy(frame_doas).float()
                                        else:
                                            frame_doas_t = torch.tensor(frame_doas).float()
                                    else:
                                        frame_doas_t = frame_doas

                                    if len(frame_doas_t) != num_frames:
                                        d = circular_interpolate(frame_doas_t, num_frames)
                                    else:
                                        d = frame_doas_t

                                    doa_target[:, event_idx] = d.to(device)
                                    doa_mask[:, event_idx] = 1.0

                is_mixed = (len(meta_list) > 1) or (isinstance(meta, dict) and meta.get('mixed', False))

                metrics.update(
                    sed_probs[b], doa_output[b],
                    sed_target, doa_target, doa_mask,
                    is_mixed=is_mixed, pattern=eval_pattern, audio_id=audio_id
                )
    
    return metrics.compute()


def plot_only_from_dataset(model, dataloader, device, num_classes, output_dir,
                           max_viz_per_pattern=15, hop_length=512, sr=16000):
    """
    データセットから指定枚数のサンプルを読み込んで可視化のみ実行
    評価は行わず、プロット生成のみ
    """
    model.eval()
    viz_dir = os.path.join(output_dir, "viz_trajectories_all_patterns_beats_seldnet")
    os.makedirs(viz_dir, exist_ok=True)
    
    pattern_viz_count = defaultdict(int)
    total_patterns_needed = {}
    
    print(f"\n📊 可視化モード: パターン毎に最大 {max_viz_per_pattern} 枚生成")
    print(f"   出力先: {viz_dir}\n")
    
    with torch.no_grad():
        for batch_idx, (audios, metas) in enumerate(tqdm(dataloader, desc="Visualizing BEATs SELDNet")):
            audios = audios.to(device)
            batch_size = audios.shape[0]
            
            # まず、必要なパターンがすべて完了しているかチェック
            # 初回のバッチで見つかったパターンをトラッキング
            for b in range(batch_size):
                meta = metas[b]
                eval_pattern = None
                
                if isinstance(meta, dict):
                    eval_pattern = meta.get('eval_pattern', None)
                
                if eval_pattern and eval_pattern not in total_patterns_needed:
                    total_patterns_needed[eval_pattern] = max_viz_per_pattern
            
            # すべてのパターンが完了したら終了
            all_complete = all(
                pattern_viz_count[p] >= max_viz_per_pattern 
                for p in total_patterns_needed.keys()
            )
            if all_complete and len(total_patterns_needed) > 0:
                break
            
            # モデル推論（可視化に必要）
            sed_output, doa_output = model(audios)
            
            for b in range(batch_size):
                meta = metas[b]
                eval_pattern = None
                
                if isinstance(meta, dict):
                    eval_pattern = meta.get('eval_pattern', None)
                
                if not eval_pattern:
                    continue
                
                # このパターンがまだ必要な枚数に達していない場合のみ処理
                if pattern_viz_count[eval_pattern] >= max_viz_per_pattern:
                    continue
                
                # DOA予測の取得
                doa_pred_np = doa_output[b].cpu().numpy()
                if doa_pred_np.ndim == 3 and doa_pred_np.shape[-1] == 2:
                    pred_rad = np.arctan2(doa_pred_np[..., 1], doa_pred_np[..., 0])
                    pred_deg = pred_rad * 180 / np.pi
                else:
                    pred_deg = doa_pred_np * 90
                
                # Ground truthの構築
                num_frames = pred_deg.shape[0]
                doa_target = torch.zeros((num_frames, num_classes), device=device)
                doa_mask = torch.zeros((num_frames, num_classes), device=device)
                
                if isinstance(meta, dict):
                    if meta.get('mixed', False):
                        meta_list = meta.get('source_metas', [meta])
                    else:
                        meta_list = [meta]
                else:
                    meta_list = [meta] if not isinstance(meta, list) else meta
                
                # 混合音源の場合
                if isinstance(meta, dict) and meta.get('mixed', False):
                    source_metas = meta.get('source_metas', [])
                    frame_doas_mixed = meta.get('frame_doas', None)
                    
                    if frame_doas_mixed is not None and len(source_metas) == 2:
                        if not isinstance(frame_doas_mixed, torch.Tensor):
                            if isinstance(frame_doas_mixed, np.ndarray):
                                frame_doas_t = torch.from_numpy(frame_doas_mixed).float()
                            else:
                                frame_doas_t = torch.tensor(frame_doas_mixed).float()
                        else:
                            frame_doas_t = frame_doas_mixed.clone()
                        
                        if frame_doas_t.ndim == 2 and frame_doas_t.shape[1] == 2:
                            for source_idx, source_meta in enumerate(source_metas):
                                if source_idx >= 2:
                                    break
                                
                                events = source_meta.get('event', [0])
                                if not isinstance(events, list):
                                    events = [events]
                                
                                src_doas = frame_doas_t[:, source_idx]
                                
                                if len(src_doas) != num_frames:
                                    src_doas = circular_interpolate(src_doas, num_frames)
                                
                                for event_idx in events:
                                    if event_idx < num_classes:
                                        doa_target[:, event_idx] = src_doas.to(device)
                                        doa_mask[:, event_idx] = 1.0
                else:
                    # 単一音源
                    for m in meta_list:
                        if not isinstance(m, dict): continue
                        
                        events = m.get('event', [0])
                        frame_doas = m.get('frame_doas', None)
                        
                        if not isinstance(events, list): events = [events]
                        
                        for event_idx in events:
                            if event_idx < num_classes:
                                if frame_doas is not None:
                                    if not isinstance(frame_doas, torch.Tensor):
                                        if isinstance(frame_doas, np.ndarray):
                                            frame_doas_t = torch.from_numpy(frame_doas).float()
                                        else:
                                            frame_doas_t = torch.tensor(frame_doas).float()
                                    else:
                                        frame_doas_t = frame_doas
                                    
                                    if len(frame_doas_t) != num_frames:
                                        d = circular_interpolate(frame_doas_t, num_frames)
                                    else:
                                        d = frame_doas_t
                                    
                                    doa_target[:, event_idx] = d.to(device)
                                    doa_mask[:, event_idx] = 1.0
                
                # 可視化
                target_deg = doa_target.cpu().numpy() * 90
                mask_np = doa_mask.cpu().numpy()
                
                pattern_short = eval_pattern.replace('single_', '').replace('mixed_', 'm_')
                save_path = os.path.join(viz_dir, f"{pattern_short}_{pattern_viz_count[eval_pattern]:02d}.png")
                viz_meta = meta if (isinstance(meta, dict) and meta.get('mixed', False)) else (meta_list[0] if meta_list else meta)
                visualize_trajectory(pred_deg, target_deg, mask_np, save_path, viz_meta, eval_pattern,
                                   hop_length=hop_length, sr=sr)
                pattern_viz_count[eval_pattern] += 1
    
    # 統計出力
    print(f"\n📊 可視化完了:")
    print(f"   出力ディレクトリ: {viz_dir}")
    total_viz = sum(pattern_viz_count.values())
    print(f"   総可視化数: {total_viz}")
    for pattern in sorted(pattern_viz_count.keys()):
        count = pattern_viz_count[pattern]
        print(f"     - {pattern}: {count} samples")


def visualize_trajectory(pred_deg, target_deg, active_mask, save_path, meta, pattern_name=None, 
                        hop_length=512, sr=16000):
    """
    DOA軌跡の可視化（論文掲載用・修正版）
    - 凡例順序: [GT Source 1] -> [Pred Src1 Class A] -> ...
    - 色: 全ての線で異なる色を使用し、区別を明確化
    - 枠線: 図の縁を黒色ではっきりと表示
    """
    
    # --- スタイル設定 ---
    # グリッド付きのスタイルをベースにする
    # sns.set_style("whitegrid")
    
    # フォントと枠線の基本設定
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 22,
        'axes.titlesize': 24,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 24,
        # Axesの初期設定（後でax.spinesでも上書きしますが念のため）
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.2
    })
    
    # FigureとAxesオブジェクトを作成
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # --- データ準備 ---
    num_frames = pred_deg.shape[0]
    time_per_frame = hop_length / sr
    times = np.arange(num_frames) * time_per_frame
    active_classes = np.where(active_mask.sum(axis=0) > 0)[0]
    
    # --- メタデータ解析 & マッピング ---
    # (このセクションのロジックは前回と同じです)
    source_mapping = {} 
    is_mixed = False
    if isinstance(meta, dict):
        is_mixed = meta.get('mixed', False)
        if is_mixed:
            source_metas = meta.get('source_metas', [])
            for s_idx, s_meta in enumerate(source_metas):
                evs = s_meta.get('event', [0])
                if not isinstance(evs, list): evs = [evs]
                source_mapping[s_idx] = {'events': evs, 'classes': [], 'label': f'Source {s_idx + 1}'}
        else:
            evs = meta.get('event', [0])
            if not isinstance(evs, list): evs = [evs]
            source_mapping[0] = {'events': evs, 'classes': [], 'label': 'Source'}
    for c in active_classes:
        assigned = False
        for s_idx, data in source_mapping.items():
            if c in data['events']:
                data['classes'].append(c)
                assigned = True
                break
        if not assigned and len(source_mapping) > 0:
             source_mapping[0]['classes'].append(c)

    # --- 【変更点】色の準備 ---
    # プロットする線の総数を計算（有効な音源の数 + 有効な予測クラスの数）
    num_gt_lines = len([s for s in source_mapping.values() if s['classes']])
    num_pred_lines = len(active_classes)
    total_lines = num_gt_lines + num_pred_lines
    
    # 線の数に応じて、色相が均等に分散するパレットを生成
    #黄色(６番目）は見ずらいため除外
    palette = sns.color_palette("Set1", max(total_lines+3, 1))
    palette = [color for i, color in enumerate(palette) if i != 5]  # 6番目の色（黄色）を除外
    color_idx = 0 # 色を取り出すためのカウンター

    has_plot = False

    # --- 描画ループ ---
    for s_idx in sorted(source_mapping.keys()):
        data = source_mapping[s_idx]
        s_classes = data['classes']
        if not s_classes: continue

        # 1. 正解ラベルの描画
        rep_c = s_classes[0]
        mask_c = active_mask[:, rep_c] > 0.0
        if mask_c.sum() > 0:
            has_plot = True
            # パレットから順番に色を取得
            current_color = palette[color_idx % len(palette)]
            color_idx += 1
            
            ax.plot(times[mask_c], target_deg[mask_c, rep_c],
                     label=data['label'],
                     color=current_color,
                     linewidth=5.0, linestyle='-', alpha=0.9)

        # 2. 予測ラベルの描画
        for c in s_classes:
            mask_c = active_mask[:, c] > 0.0
            if mask_c.sum() > 0:
                # 次の色を取得
                current_color = palette[color_idx % len(palette)]
                color_idx += 1

                label_pred = f"Pred: Class {c}"
                ax.plot(times[mask_c], pred_deg[mask_c, c],
                         label=label_pred,
                         color=current_color,
                         linewidth=4.0, linestyle='--',
                         alpha=0.8)

    # --- グラフ装飾 ---
    if has_plot:
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('DoA (Degree)', fontweight='bold')
        
        # 【変更点】図の枠線（Spines）を黒色に設定
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1) # 少し太くして強調
        
        ax.set_ylim(-95, 95)
        time_margin = times[-1] * 0.05
        ax.set_xlim(-time_margin, times[-1] + time_margin)
        
        # 凡例 (枠線も黒く)
        ax.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='gray', fontsize=16)
        ax.grid(True, linestyle=':', alpha=0.8)
        
        # レイアウト調整と保存
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # --- メタデータ保存 (前回と同じため省略) ---
        meta_save_path = save_path.replace('.png', '_meta.json')
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)): return int(obj)
            elif isinstance(obj, (np.float64, np.float32)): return float(obj)
            elif isinstance(obj, torch.Tensor): return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict): return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list): return [convert_to_serializable(item) for item in obj]
            return obj
        save_meta = convert_to_serializable(meta) if meta else {}
        save_meta['viz_info'] = {'pattern': pattern_name, 'duration': float(times[-1]), 'active_classes': active_classes.tolist()}
        with open(meta_save_path, 'w', encoding='utf-8') as f:
            json.dump(save_meta, f, indent=2, ensure_ascii=False)

    # Figureを閉じる
    plt.close(fig)


def print_results(results, pattern_type):
    """結果表示"""
    print("\n" + "="*80)
    print(f"🧪 BEATs SELDNet 評価結果 ({pattern_type.upper()})")
    print("="*80)
    
    print(f"\n📊 Sound Event Detection:")
    print(f"  Precision: {results['sed']['precision']:.4f}")
    print(f"  Recall:    {results['sed']['recall']:.4f}")
    print(f"  F1 Score:  {results['sed']['f1']:.4f}")
    print(f"  Accuracy:  {results['sed']['accuracy']:.4f}")
    
    print(f"\n🎯 Direction of Arrival:")
    print(f"  Mean Absolute Error: {results['doa']['mae']:.2f}°")
    print(f"  Accuracy (<{results['doa']['threshold']}°): {results['doa']['accuracy']:.2%}")
    
    if results['temporal_doa']['mean_accuracy'] > 0:
        print(f"\n⏱️  時間区間ごとのDOA精度 ({results['temporal_doa']['chunk_seconds']}秒区間):")
        print(f"  平均正答率: {results['temporal_doa']['mean_accuracy']:.2%}")
        print(f"  総チャンク数: {results['temporal_doa']['total_chunks']}")
        
        # チャンク別の詳細を表示
        chunk_stats = results['temporal_doa'].get('chunk_stats', {})
        if chunk_stats:
            print(f"\n  チャンク別詳細:")
            for chunk_idx in sorted(chunk_stats.keys()):
                stats = chunk_stats[chunk_idx]
                print(f"    Chunk {chunk_idx} ({stats['start_time']:.1f}s - {stats['end_time']:.1f}s): "
                      f"精度 {stats['mean_accuracy']:.2%} ({stats['correct']}/{stats['total']} frames, "
                      f"{stats['samples']} samples)")
    
    if 'single' in results:
        print(f"\n🎵 単一音源 ({results['single']['count']}サンプル):")
        print(f"  SED F1:  {results['single']['sed']['f1']:.4f}")
        print(f"  DOA MAE: {results['single']['doa']['mae']:.2f}°")
        print(f"  DOA Acc: {results['single']['doa']['accuracy']:.2%}")
    
    if 'mixed' in results:
        print(f"\n🎼 混合音源 ({results['mixed']['count']}サンプル):")
        print(f"  SED F1:  {results['mixed']['sed']['f1']:.4f}")
        print(f"  DOA MAE: {results['mixed']['doa']['mae']:.2f}°")
        print(f"  DOA Acc: {results['mixed']['doa']['accuracy']:.2%}")
    
    # パターン別結果
    if 'patterns' in results:
        print(f"\n📋 パターン別結果:")
        for pattern in sorted(results['patterns'].keys()):
            p = results['patterns'][pattern]
            print(f"\n  {pattern} ({p['count']}サンプル):")
            print(f"    SED F1:  {p['sed']['f1']:.4f}")
            print(f"    DOA MAE: {p['doa']['mae']:.2f}°")
            print(f"    DOA Acc: {p['doa']['accuracy']:.2%}")
    
    # Audio/Pair ID別統計
    if 'audio_stats' in results and len(results['audio_stats']) > 0:
        print(f"\n📈 Audio/Pair ID別統計:")
        print(f"  Total Audio/Pairs: {len(results['audio_stats'])}")
        
        # DOA MAEの分布
        mae_values = [stats['doa_mae'] for stats in results['audio_stats'].values()]
        if mae_values:
            print(f"  DOA MAE (per audio/pair):")
            print(f"    Mean:   {np.mean(mae_values):.2f}°")
            print(f"    Median: {np.median(mae_values):.2f}°")
            print(f"    Std:    {np.std(mae_values):.2f}°")
            print(f"    Min:    {np.min(mae_values):.2f}°")
            print(f"    Max:    {np.max(mae_values):.2f}°")
    
    print("="*80 + "\n")


def save_results(results, output_path, pattern_type):
    """結果をJSON形式で保存"""
    import json
    
    # JSON用にresultsを整形（numpy型を変換）
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    # 完全な統計情報を含むJSONオブジェクトを作成
    output_data = {
        'model': 'BEATsSELDnet',
        'pattern_type': pattern_type.upper(),
        'evaluation_results': convert_to_serializable(results),
        'summary': {
            'sed': {
                'precision': float(results['sed']['precision']),
                'recall': float(results['sed']['recall']),
                'f1': float(results['sed']['f1']),
                'accuracy': float(results['sed']['accuracy']),
                'tp': int(results['sed']['tp']),
                'fp': int(results['sed']['fp']),
                'fn': int(results['sed']['fn'])
            },
            'doa': {
                'mae': float(results['doa']['mae']),
                'std': float(results['doa']['std']),
                'median': float(results['doa']['median']),
                'accuracy': float(results['doa']['accuracy']),
                'threshold': float(results['doa']['threshold']),
                'correct': int(results['doa']['correct']),
                'total': int(results['doa']['total'])
            }
        }
    }
    
    # Temporal DOA情報を追加
    if 'temporal_doa' in results:
        output_data['summary']['temporal_doa'] = convert_to_serializable(results['temporal_doa'])
    
    # Single/Mixed別統計を追加
    if 'single' in results:
        output_data['summary']['single_source'] = convert_to_serializable(results['single'])
    
    if 'mixed' in results:
        output_data['summary']['mixed_source'] = convert_to_serializable(results['mixed'])
    
    # パターン別統計を追加
    if 'patterns' in results:
        output_data['pattern_wise_results'] = convert_to_serializable(results['patterns'])
    
    # Audio/Pair ID別統計を追加
    if 'audio_stats' in results:
        output_data['audio_pair_statistics'] = convert_to_serializable(results['audio_stats'])
    
    # JSON出力パスを設定
    json_output_path = output_path.replace('.txt', '.json')
    
    # JSONファイルに保存
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"📝 Results saved to: {json_output_path}")


# =====================================================
# Main
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="AllPatternDataset SELD Evaluation")
    
    # チェックポイント
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file")
    
    # データ設定
    parser.add_argument("--pattern_root", type=str,
                       default="../all_patterns_test",
                       help="Root directory for AllPatternDataset")
    parser.add_argument("--pattern_type", type=str, default="both",
                       choices=['single', 'mixed', 'both'],
                       help="Evaluate single source, mixed source, or both")
    
    # 評価設定
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=344)
    parser.add_argument("--sed_threshold", type=float, default=0.5)
    parser.add_argument("--doa_threshold", type=float, default=5.0)
    
    # 可視化設定
    parser.add_argument("--max_viz_per_pattern", type=int, default=300,
                       help="Maximum number of visualizations per pattern")
    parser.add_argument("--plot_only", action="store_true",
                       help="Only generate visualizations without full evaluation")
    
    # 出力
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for visualizations and results")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)
    
    print("="*80)
    print("🧪 AllPattern SELD Evaluation - BEATs SELDNet")
    print("="*80)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Pattern Root: {args.pattern_root}")
    print(f"Pattern Type: {args.pattern_type}")
    print("="*80)
    
    # モデルロード
    model = load_model(args.num_classes, device, args.checkpoint)
    
    # 出力ディレクトリ
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.checkpoint)
    
    # プロットのみモード
    if args.plot_only:
        print("\n" + "="*80)
        print(f"📊 可視化モード (パターン毎 最大{args.max_viz_per_pattern}枚)")
        print("="*80)
        
        if args.pattern_type in ['single', 'both']:
            print("\n🎵 単一音源パターンの可視化")
            dataset = AllPatternEvalDataset(args.pattern_root, pattern_type='single')
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn
            )
            plot_only_from_dataset(
                model, dataloader, device, args.num_classes, args.output_dir,
                max_viz_per_pattern=args.max_viz_per_pattern
            )
        
        if args.pattern_type in ['mixed', 'both']:
            print("\n🎼 混合音源パターンの可視化")
            dataset = AllPatternEvalDataset(args.pattern_root, pattern_type='mixed')
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn
            )
            plot_only_from_dataset(
                model, dataloader, device, args.num_classes, args.output_dir,
                max_viz_per_pattern=args.max_viz_per_pattern
            )
        
        print("\n✅ 可視化完了")
        return
    
    # 評価実行（可視化込み）
    if args.pattern_type in ['single', 'both']:
        print("\n" + "="*80)
        print("🎵 単一音源パターンの評価")
        print("="*80)
        
        dataset = AllPatternEvalDataset(args.pattern_root, pattern_type='single')
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )
        
        results = evaluate(
            model, dataloader, device, args.num_classes,
            sed_threshold=args.sed_threshold,
            doa_threshold=args.doa_threshold
        )
        
        print_results(results, 'single')
        
        result_path = os.path.join(args.output_dir, "eval_all_pattern_single_beats_seldnet.txt")
        save_results(results, result_path, 'single')
    
    if args.pattern_type in ['mixed', 'both']:
        print("\n" + "="*80)
        print("🎼 混合音源パターンの評価")
        print("="*80)
        
        dataset = AllPatternEvalDataset(args.pattern_root, pattern_type='mixed')
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )
        
        results = evaluate(
            model, dataloader, device, args.num_classes,
            sed_threshold=args.sed_threshold,
            doa_threshold=args.doa_threshold
        )
        
        print_results(results, 'mixed')
        
        result_path = os.path.join(args.output_dir, "eval_all_pattern_mixed_beats_seldnet.txt")
        save_results(results, result_path, 'mixed')
    
    print("\n✅ 評価完了")


if __name__ == "__main__":
    main()
