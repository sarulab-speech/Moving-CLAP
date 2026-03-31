import torch
import torch.nn as nn
import numpy as np

class SELDLoss(nn.Module):
    def __init__(self, num_classes, doa_weight=10.0, device="cuda"):
        super(SELDLoss, self).__init__()
        
        self.num_classes = num_classes
        self.device = device

        self.sed_loss_fn = nn.BCEWithLogitsLoss(
            reduction='none',
            pos_weight=torch.tensor([5.0], device=device)
        )
        
        self.doa_weight = doa_weight
        self.log_var_sed = None
        self.log_var_doa = None

    def __call__(self, sed_output, doa_output, metas):
        return self._temporal_loss(sed_output, doa_output, metas)

    def _temporal_loss(self, sed_output, doa_output, metas):
        batch_size, num_frames, num_classes, _ = doa_output.shape
        
        doa_cos_pred = doa_output[:, :, :, 0]
        doa_sin_pred = doa_output[:, :, :, 1]
        
        sed_target = torch.zeros((batch_size, num_frames, num_classes), device=self.device)
        doa_cos_target = torch.zeros((batch_size, num_frames, num_classes), device=self.device)
        doa_sin_target = torch.zeros((batch_size, num_frames, num_classes), device=self.device)
        doa_mask = torch.zeros((batch_size, num_frames, num_classes), device=self.device)
        
        for b in range(batch_size):
            meta = metas[b]
            if isinstance(meta, dict) and 'source_metas' in meta:
                meta_list = meta['source_metas']
            else:
                meta_list = [meta]
            for source_meta in meta_list:
                self._fill_targets(
                    source_meta, sed_target[b], doa_cos_target[b], doa_sin_target[b], 
                    doa_mask[b], num_frames
                )

        sed_loss = self.sed_loss_fn(sed_output, sed_target).mean()
        
        inner_product = doa_cos_pred * doa_cos_target + doa_sin_pred * doa_sin_target
        doa_loss_per_element = 1.0 - inner_product
        
        n_active = doa_mask.sum()
        if n_active > 0:
            doa_loss = (doa_loss_per_element * doa_mask).sum() / n_active
        else:
            doa_loss = torch.tensor(0.0, device=self.device)

        total_loss = sed_loss + self.doa_weight * doa_loss
        
        return sed_loss, doa_loss, total_loss
    
    def _fill_targets(self, meta, sed_target, doa_cos_target, doa_sin_target, doa_mask, num_frames):
        events = meta.get("event", [])
        frame_doas = meta.get('frame_doas', None)
        
        if frame_doas is None:
            doa = meta.get('doa', 0.0)
            frame_doas = torch.full((num_frames,), doa, device=self.device)
        else:
            if isinstance(frame_doas, list):
                frame_doas = torch.tensor(frame_doas, device=self.device)
            elif isinstance(frame_doas, np.ndarray):
                frame_doas = torch.from_numpy(frame_doas).to(self.device)
            else:
                frame_doas = frame_doas.to(self.device)
            
            if frame_doas.dim() > 1:
                frame_doas = frame_doas[:, 0] if frame_doas.shape[1] > 0 else frame_doas.squeeze()
            
            if len(frame_doas) != num_frames:
                frame_doas = self._interpolate_doas(frame_doas, num_frames)
        
        angles = frame_doas * (torch.pi / 2)
        frame_doa_cos = torch.cos(angles)
        frame_doa_sin = torch.sin(angles)
        
        for event_id in events:
            sed_target[:, event_id] = 1.0
            doa_cos_target[:, event_id] = frame_doa_cos
            doa_sin_target[:, event_id] = frame_doa_sin
            doa_mask[:, event_id] = 1.0
    
    def _interpolate_doas(self, frame_doas, target_length):
        original_length = len(frame_doas)
        if original_length == target_length:
            return frame_doas
        frame_doas = frame_doas.unsqueeze(0).unsqueeze(0)
        interpolated = torch.nn.functional.interpolate(
            frame_doas, size=target_length, mode='linear', align_corners=True
        )
        return interpolated.squeeze(0).squeeze(0)
