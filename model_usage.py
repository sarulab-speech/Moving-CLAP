"""
Moving-CLAP Embedding Calculation and Retrieval Evaluation Samples
"""

import torch
import torchaudio
import numpy as np
import torch.nn.functional as F

# ─────────────────────────────────────────────
# 1. load_model: create and load the Moving-CLAP model
# ─────────────────────────────────────────────

def load_model(
    pattern_name: str,
    checkpoint_path: str | None = None,
    device: str = "cuda",
) -> torch.nn.Module:
    """
    Create and load the Moving-CLAP model.
    """
    from moving_clap_model import MovingCLAP
    from ablation_configs import get_config

    config = get_config(pattern_name)
    
    # if checkpoint_path is not provided, try to get it from config
    if checkpoint_path is None:
        checkpoint_path = config.get('checkpoint_path')

    model = MovingCLAP(
        joint_embed_shape=config.get('joint_embed_shape', 512),
        freeze_spatial_encoder=config.get('freeze_spatial_encoder', True),
        hidden_dim=config.get('hidden_dim', 768),
        pretrained_model_name=config.get('pretrained_model_name', 'BEATs'),
        fusion_type=config.get('fusion_type', 'cross_attn'),
        pooling_type=config.get('pooling_type', 'adaptive'),
        use_text_attention=config.get('use_text_attention', True),
        spatial_bias_value=config.get('spatial_bias_value', 5.0),
    )

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        print(f"Checkpoint loaded: {checkpoint_path}")
    else:
        model.load_default_state_dict()
        print("No checkpoint specified, using default/pretrained weights.")

    model = model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────
# 2. compute_audio_embeddings: calculate audio embeddings
# ─────────────────────────────────────────────

def load_audio(path: str, target_sr: int = 24000) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)  # (C, T)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)

    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    return waveform  # (2, T)


def pad_to_same_length(audios: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(a.shape[-1] for a in audios)
    padded = [
        F.pad(a, (0, max_len - a.shape[-1])) if a.shape[-1] < max_len else a
        for a in audios
    ]
    return torch.stack(padded, dim=0)  # (N, 2, T_max)


@torch.no_grad()
def compute_audio_embeddings(
    model: torch.nn.Module,
    audio_paths: list[str],
    device: str = "cuda",
    batch_size: int = 32,
    target_sr: int = 24000,
) -> torch.Tensor:
    all_embeds = []

    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i : i + batch_size]
        audios = [load_audio(p, target_sr) for p in batch_paths]
        audio_tensor = pad_to_same_length(audios).to(device)  # (B, 2, T)

        embeds = model.encode_audio(audio_tensor)             # (B, D)
        all_embeds.append(embeds.cpu())

    return torch.cat(all_embeds, dim=0)  # (N, D)


# ─────────────────────────────────────────────
# 3. compute_text_embeddings: calculate text embeddings
# ─────────────────────────────────────────────

@torch.no_grad()
def compute_text_embeddings(
    model: torch.nn.Module,
    captions: list[str],
    device: str = "cuda",
    batch_size: int = 64,
) -> torch.Tensor:
    all_embeds = []

    for i in range(0, len(captions), batch_size):
        batch = captions[i : i + batch_size]
        embeds = model.encode_text(batch)   # (B, D)
        all_embeds.append(embeds.cpu())

    return torch.cat(all_embeds, dim=0)  # (N, D)


# ─────────────────────────────────────────────
# 4. compute_similarity_matrix & top_k_retrieval: calculate similarity and evaluate retrieval
# ─────────────────────────────────────────────

def compute_similarity_matrix(
    audio_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    if normalize:
        audio_embeds = F.normalize(audio_embeds, p=2, dim=-1)
        text_embeds  = F.normalize(text_embeds,  p=2, dim=-1)

    return audio_embeds @ text_embeds.T  # (N_a, N_t)


def top_k_retrieval(sim_matrix: torch.Tensor, k: int = 5) -> dict:
    assert sim_matrix.shape[0] == sim_matrix.shape[1], "Similarity matrix must be square (N_a == N_t)"

    n = sim_matrix.shape[0]

    def _metrics(matrix: torch.Tensor):
        top1_count = 0
        topk_count = 0
        rank_sum   = 0
        for i in range(n):
            row   = matrix[i]
            order = torch.argsort(row, descending=True).tolist()
            rank  = order.index(i) + 1   # 1-indexed
            rank_sum   += rank
            top1_count += int(rank == 1)
            topk_count += int(rank <= k)
        return {
            f"top1":      top1_count / n * 100,
            f"top{k}":    topk_count / n * 100,
            "mean_rank":  rank_sum   / n,
        }

    a2t = _metrics(sim_matrix)
    t2a = _metrics(sim_matrix.T)

    return {
        "a2t_top1":      a2t["top1"],
        f"a2t_top{k}":   a2t[f"top{k}"],
        "a2t_mean_rank": a2t["mean_rank"],
        "t2a_top1":      t2a["top1"],
        f"t2a_top{k}":   t2a[f"top{k}"],
        "t2a_mean_rank": t2a["mean_rank"],
    }