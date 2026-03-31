import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from moving_caption import meta_to_caption_moving
from clap_candidates import (
            generate_all_spatial_candidates, 
            find_best_matching_candidate,
            generate_mixed_spatial_candidates,
            find_best_matching_mixed_candidate
        )

class CLAPLoss(nn.Module):
    """
    CLAP Contrastive Loss with optional Label Smoothing
    """
    def __init__(self):
        super().__init__()

    def __call__(self, clap_output, metas, logit_scale):
        """
        audio_features: Tensor of shape (N, D)
        text_features: Tensor of shape (N, D)
        """
        audio_features = clap_output["audio"]
        text_features = clap_output["text"]

        # τ is learnable → use exp(logit_scale)
        temperature = logit_scale.exp().clamp(min=0.01, max=100.0)

        logits_per_audio = audio_features @ text_features.T * temperature
        logits_per_text = text_features @ audio_features.T * temperature

        labels = torch.arange(audio_features.size(0), device=audio_features.device)

        loss = (
            F.cross_entropy(logits_per_audio, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2
        return loss
    
class CLAPLossWrapper(nn.Module):
    def __init__(self, clap_loss):
        super().__init__()
        self.clap_loss = clap_loss
    def __call__(self, clap_output, metas, logit_scale, text_encoder_fn=None, return_components=False):
        loss = self.clap_loss(clap_output, metas, logit_scale)
        if return_components:
            components = {
                'clap': loss.item(), 'swap': 0.0, 'spatial': 0.0, 'total': loss.item()
            }
            return loss, components
        return loss

class MovingCLAPLoss(nn.Module):
    """
    CLAP Contrastive Loss + Swap Loss + Spatial Contrastive Loss

    1. Penalty for reversing moving directions (B→C vs C→B)
    2. Spatial Contrastive Loss selecting the correct one from spatial candidates
    """
    def __init__(self, swap_weight=0.3,
                 spatial_contrastive_weight=0.01,
                 swap_loss_text_margin=0.3,
                 swap_loss_audio_margin=0.1,
                 spatial_loss_skip_mixed=False,
                 hidden_dim=1024, embed_dim=512):
        super().__init__()
        self.swap_weight = swap_weight
        self.spatial_contrastive_weight = spatial_contrastive_weight
        self.swap_loss_text_margin = swap_loss_text_margin
        self.swap_loss_audio_margin = swap_loss_audio_margin
        self.spatial_loss_skip_mixed = spatial_loss_skip_mixed
        self.clap_loss = CLAPLoss()
        
        self._spatial_candidate_cache = {}
        self._spatial_embed_cache = {}
        self._spatial_embed_cache_step = -1
        self._cache_update_interval = 100
        self._current_step = 0
    
    def __call__(self, clap_output, metas, logit_scale, text_encoder_fn=None, return_components=False):
        audio_features = clap_output["audio"]  # (N, D)
        text_features = clap_output["text"]    # (N, D)
        
        if hasattr(logit_scale, 'logit_scale_clap'):
            logit_scale_clap = logit_scale.logit_scale_clap
            logit_scale_swap = logit_scale.logit_scale_swap
            logit_scale_spatial = logit_scale.logit_scale_spatial
        else:
            logit_scale_clap = logit_scale
            logit_scale_swap = logit_scale
            logit_scale_spatial = logit_scale
        
        # Update step counter
        self._current_step += 1
        
        # Record loss components
        loss_components = {}
        
        # 1. CLAP Contrastive Loss
        clap_loss = self.clap_loss(clap_output, metas, logit_scale_clap)
        loss_components['clap'] = clap_loss.item()
        
        total_loss = clap_loss
        
        # 2. Swap Loss
        swap_loss_val = 0.0
        if self.swap_weight > 0:
            swap_text_feature = None
            moving_indices = None
            if text_encoder_fn is not None:
                swap_captions, moving_indices = self._generate_swap_captions(metas)
                if swap_captions is not None:
                    swap_text_feature = text_encoder_fn(swap_captions)
            
            # Audio-Text Swap Loss
            swap_loss_audio = self._audio_swap_loss(
                audio_features, text_features, metas,
                swap_text_feature=swap_text_feature,
                moving_indices=moving_indices,
                margin=self.swap_loss_audio_margin
            )
            
            # Text-Text Swap Loss
            swap_loss_text = self._text_swap_loss(
                text_features, swap_text_feature, moving_indices, self.swap_loss_text_margin
            )
            if swap_loss_audio is None:
                swap_loss_audio = 0.0
            if swap_loss_text is None:
                swap_loss_text = 0.0
            
            # sum of both losses
            swap_loss = swap_loss_audio + swap_loss_text
            
            if isinstance(swap_loss, torch.Tensor):
                swap_loss_val = swap_loss.item()
                if swap_loss_val > 0:
                    total_loss = total_loss + self.swap_weight * swap_loss
        loss_components['swap'] = swap_loss_val
        loss_components['swap_weighted'] = swap_loss_val * self.swap_weight
        
        # 3. Spatial Contrastive Loss
        spatial_loss_val = 0.0
        if self.spatial_contrastive_weight > 0 and text_encoder_fn is not None:
            spatial_loss = self._spatial_contrastive_loss(
                audio_features, metas, logit_scale_spatial, text_encoder_fn
            )
            if spatial_loss is None:
                spatial_loss_val = 0.0
            elif isinstance(spatial_loss, torch.Tensor):
                spatial_loss_val = spatial_loss.item()
                if spatial_loss_val > 0:
                    total_loss = total_loss + self.spatial_contrastive_weight * spatial_loss
        loss_components['spatial'] = spatial_loss_val
        loss_components['spatial_weighted'] = spatial_loss_val * self.spatial_contrastive_weight
        
        loss_components['total'] = total_loss.item()

        if return_components:
            return total_loss, loss_components
        return total_loss
    
    def _generate_swap_captions(self, metas):
        swap_captions = []
        moving_sample_indices = []
        
        for sample_idx, meta in enumerate(metas):
            assert isinstance(meta, dict), "Meta must be a dict"

            if meta.get('mixed', False) and meta.get('source_metas'):
                source_metas = meta['source_metas']
            else:
                source_metas = [meta]
            
            # check if there is any moving source
            num_moving = 0
            for meta in source_metas:
                if meta.get('is_stationary', False) == False:
                    num_moving += 1
            
            if num_moving == 0:
                continue
            
            # create swap meta
            swap_meta_list = []
            for meta in source_metas:
                start_zone = meta.get('start_zone')
                end_zone = meta.get('end_zone')
                
                if meta.get('is_stationary', False) == False and start_zone is not None and end_zone is not None:
                    swap_meta = meta.copy()
                    swap_meta['start_zone'] = end_zone
                    swap_meta['end_zone'] = start_zone

                    if 'start_doa' in meta and 'end_doa' in meta:
                        swap_meta['start_doa'] = meta['end_doa']
                        swap_meta['end_doa'] = meta['start_doa']   
                    swap_meta_list.append(swap_meta)
                else:
                    swap_meta_list.append(meta)
            
            # create caption
            if len(swap_meta_list) == 1:
                swap_caption = meta_to_caption_moving(swap_meta_list[0])
            else:
                # if mixed source
                swap_mixed_meta = {
                    'mixed': True,
                    'source_metas': swap_meta_list,
                    'n_sources': len(swap_meta_list)
                }
                swap_caption = meta_to_caption_moving(swap_mixed_meta)
            
            swap_captions.append(swap_caption)
            moving_sample_indices.append(sample_idx)
        
        if len(swap_captions) == 0:
            return None, None
        
        return swap_captions, moving_sample_indices
    
    def _audio_swap_loss(self, audio_features, text_features, metas, swap_text_feature=None, moving_indices=None, margin=0.1):
        device = audio_features.device

        if moving_indices is None or len(moving_indices) == 0:
            return None

        if swap_text_feature is None:
            return None
        
        moving_audio = audio_features[moving_indices]  # (M, D)
        moving_text = text_features[moving_indices]    # (M, D)

        swap_text = swap_text_feature

        moving_audio = F.normalize(moving_audio, dim=-1)
        moving_text = F.normalize(moving_text, dim=-1)
        swap_text = F.normalize(swap_text, dim=-1)
        
        # calculate cosine similarities
        correct_sims = (moving_audio * moving_text).sum(dim=-1) # (M,)
        swap_sims = (moving_audio * swap_text).sum(dim=-1)  # (M,)
        
        # max(0, margin + swap_sim - correct_sim)
        margin_loss = F.relu(margin + swap_sims - correct_sims)
        
        return margin_loss.mean()
    
    def _text_swap_loss(self, text_features, swap_text_feature, moving_indices, margin=0.3):
        device = text_features.device

        if moving_indices is None or len(moving_indices) == 0:
            return None

        if swap_text_feature is None:
            return None

        moving_text = text_features[moving_indices]  # (M, D)
        
        moving_text = F.normalize(moving_text, dim=-1)
        swap_text = F.normalize(swap_text_feature, dim=-1)
        # calculate cosine similarity
        similarity = (moving_text * swap_text).sum(dim=-1)  # (M,)
        
        # max(0, margin - (1 - similarity))
        text_reversal_loss = F.relu(margin - (1 - similarity))
        
        return text_reversal_loss.mean()
    
    def _spatial_contrastive_loss(self, audio_features, metas, logit_scale, text_encoder_fn):
        """
        Single source: Choose the correct one from 25 candidates
        Mixed source: Choose the correct one from 90 candidates
        """
        device = audio_features.device
        N = audio_features.size(0)
        temperature = logit_scale.exp().clamp(min=0.01, max=100.0)
        
        single_indices = []
        single_metas = []
        mixed_indices = []
        mixed_metas = []
        mixed_indices = []
        mixed_metas = []
        
        for i, meta in enumerate(metas):
            assert isinstance(meta, dict), "Meta must be a dict"
            
            # check mixed source
            if meta.get('mixed', False) and meta.get('source_metas'):
                source_metas = meta['source_metas']
                # check if spatial information exists for all sources
                if all(sm.get('start_zone') and sm.get('end_zone') for sm in source_metas):
                    mixed_indices.append(i)
                    mixed_metas.append(meta)
                continue
            # single source processing
            # exclude non-spatial data
            if meta.get('augmentation_type') == 'non_spatial':
                continue
            # exclude data without spatial information
            if meta.get('start_zone') is None or meta.get('end_zone') is None:
                continue
            single_indices.append(i)
            single_metas.append(meta)
        
        losses = []
        # =====================
        # process single sources
        # =====================
        if len(single_indices) > 0:
            cache_key = "single_candidates"
            if cache_key not in self._spatial_candidate_cache:
                candidates = generate_all_spatial_candidates(spatial_only=True)
                candidate_texts = [c['caption'] for c in candidates]
                self._spatial_candidate_cache[cache_key] = (candidates, candidate_texts)
            else:
                candidates, candidate_texts = self._spatial_candidate_cache[cache_key]
            
            # Calculate text embeddings (periodically recomputed)
            should_update_cache = (
                cache_key not in self._spatial_embed_cache or
                (self._current_step - self._spatial_embed_cache_step) >= self._cache_update_interval
            )
            
            if should_update_cache:
                with torch.no_grad():
                    candidate_embeds = text_encoder_fn(candidate_texts)  # (25, D)
                self._spatial_embed_cache[cache_key] = candidate_embeds
                self._spatial_embed_cache_step = self._current_step
            else:
                candidate_embeds = self._spatial_embed_cache[cache_key]

            true_indices = []
            valid_sample_indices = []

            for i, meta in zip(single_indices, single_metas):
                true_idx = find_best_matching_candidate(meta, candidates)
                if true_idx is not None:
                    true_indices.append(true_idx)
                    valid_sample_indices.append(i)
            
            if len(valid_sample_indices) > 0:
                valid_audio = audio_features[valid_sample_indices]  # (M, D)
                similarities = valid_audio @ candidate_embeds.T * temperature
                true_labels = torch.tensor(true_indices, device=device, dtype=torch.long)
                single_loss = F.cross_entropy(similarities, true_labels)
                losses.append(single_loss)
        
        # =====================
        # process mixed sources
        # =====================
        if len(mixed_indices) > 0 and not self.spatial_loss_skip_mixed:
            cache_key = "mixed_candidates"
            if cache_key not in self._spatial_candidate_cache:
                mixed_candidates = generate_mixed_spatial_candidates(spatial_only=True, non_mov_mov=True)
                mixed_texts = [c['caption'] for c in mixed_candidates]
                self._spatial_candidate_cache[cache_key] = (mixed_candidates, mixed_texts)
            else:
                mixed_candidates, mixed_texts = self._spatial_candidate_cache[cache_key]

            should_update = (
                cache_key not in self._spatial_embed_cache or
                (self._current_step - self._spatial_embed_cache_step) >= self._cache_update_interval
            )
            
            if should_update:
                with torch.no_grad():
                    mixed_embeds = text_encoder_fn(mixed_texts)
                self._spatial_embed_cache[cache_key] = mixed_embeds
            else:
                mixed_embeds = self._spatial_embed_cache[cache_key]
            
            true_indices = []
            valid_indices = []
            
            for i, meta in zip(mixed_indices, mixed_metas):
                true_idx, none = find_best_matching_mixed_candidate(meta, mixed_candidates, spatial_only=True)
                if true_idx is not None:
                    true_indices.append(true_idx)
                    valid_indices.append(i)
            
            if len(valid_indices) > 0:
                valid_audio = audio_features[valid_indices]
                similarities = valid_audio @ mixed_embeds.T * temperature
                true_labels = torch.tensor(true_indices, device=device, dtype=torch.long)
                mixed_loss = F.cross_entropy(similarities, true_labels)
                losses.append(mixed_loss)
        
        if len(losses) == 0:
            return None
        
        # mean of all losses
        return torch.stack(losses).mean()