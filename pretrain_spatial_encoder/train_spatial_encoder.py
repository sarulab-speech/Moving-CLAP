import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import argparse
import numpy as np
from util import set_seed
from beats_seldnet_model import BEATsSELDnetModel
from loss import SELDLoss
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clap_dataset import ChunkedMemmapDataset
# =====================================================
# Training Functions
# =====================================================

def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, log_f=None):
    model.train()
    
    total_loss = 0
    total_sed_loss = 0
    total_doa_loss = 0
    loss_num = 0
    
    desc = f"Epoch {epoch}"
    for batch_idx, (waveforms, metas) in enumerate(tqdm(dataloader, desc=desc)):
        waveforms = waveforms.to(device)
        
        optimizer.zero_grad()
        
        sed_output, doa_output = model(waveforms)
        sed_loss, doa_loss, loss = loss_fn(sed_output, doa_output, metas)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_sed_loss += sed_loss.item()
        total_doa_loss += doa_loss.item()
        loss_num += 1
        
        if batch_idx % 2000 == 0 and log_f is not None:
            log_line = (
                f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                f"Loss: {loss.item():.4f}, SED: {sed_loss.item():.4f}, DOA: {doa_loss.item():.4f}"
            )
            log_f.write(log_line + "\n")
            log_f.flush()
    
    return total_loss / loss_num, total_sed_loss / loss_num, total_doa_loss / loss_num


def validate(model, dataloader, loss_fn, device):
    model.eval()
    
    total_loss = 0
    total_sed_loss = 0
    total_doa_loss = 0
    loss_num = 0
    
    with torch.no_grad():
        for waveforms, metas in tqdm(dataloader, desc="Validating"):
            waveforms = waveforms.to(device)
            
            sed_output, doa_output = model(waveforms)
            sed_loss, doa_loss, loss = loss_fn(sed_output, doa_output, metas)
            
            total_loss += loss.item()
            total_sed_loss += sed_loss.item()
            total_doa_loss += doa_loss.item()
            loss_num += 1
    
    return total_loss / loss_num, total_sed_loss / loss_num, total_doa_loss / loss_num


class PretrainDataset(Dataset):
    def __init__(self, precomputed_root, split='train', single_source_only=False,
                 balance_augmentation_types=True):
        self.full_dataset = ChunkedMemmapDataset(
            os.path.join(precomputed_root, split)
        )
        
        self.metas = self.full_dataset.metas
        self.single_source_only = single_source_only
        self.balance_augmentation_types = balance_augmentation_types
        
        self.indices, self.type_indices = self._create_indices()
    
    def _create_indices(self):
        indices = []
        type_indices = {
            'stationary': [],
            'moving': [],
            'mixed_stat_stat': [],
            'mixed_stat_mov': [],
        }
        
        for i, meta_list in enumerate(self.metas):
            meta = meta_list[0] if isinstance(meta_list, list) else meta_list
            aug_type = meta.get('augmentation_type', 'unknown')
            
            if self.single_source_only:
                if aug_type in ['stationary', 'moving']:
                    indices.append(i)
                    type_indices[aug_type].append(i)
            else:
                indices.append(i)
                if aug_type in type_indices:
                    type_indices[aug_type].append(i)
                else:
                    continue
        
        if self.balance_augmentation_types and not self.single_source_only:
            # sample indices to balance types
            balanced_indices = self._balance_indices(type_indices)
            return balanced_indices, type_indices
        
        return indices, type_indices
    
    def _balance_indices(self, type_indices):
        for aug_type in type_indices:
            np.random.shuffle(type_indices[aug_type])
        
        max_samples = max(len(idxs) for idxs in type_indices.values() if len(idxs) > 0)
        
        balanced = []
        for i in range(max_samples):
            for aug_type in ['stationary', 'moving', 'mixed_stat_stat', 
                            'mixed_stat_mov']:
                idxs = type_indices[aug_type]
                if i < len(idxs):
                    balanced.append(idxs[i])
        
        return balanced
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.full_dataset[real_idx]


def collate_fn(batch):
    wavs = []
    metas = []
    
    for wav, meta in batch:
        wavs.append(wav)
        
        if isinstance(meta, dict):
            if 'event' not in meta and not meta.get('mixed', False):
                print("Warning: 'event' key missing in meta, adding default [0]")
                break
            if 'frame_doas' not in meta and not meta.get('mixed', False):
                print("Warning: 'frame_doas' key missing in meta, adding default zeros")
                break
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precomputed_root", type=str,
                        default="../output_moving/simulated_spatial_sound/")
    parser.add_argument("--no_balance", action="store_true")
    
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=344)
    

    parser.add_argument("--output_dir", type=str, default="output_beats_seldnet")
    parser.add_argument("--resume", action="store_true")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "log"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ckpt"), exist_ok=True)
    
    print("="*80)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("="*80)
    # Model & Loss
    model = BEATsSELDnetModel(
            num_classes=args.num_classes,
            pretrained_model_name='BEATs',
            freeze_pretrained=True
        ).to(device)
    loss_fn = SELDLoss(
        num_classes=args.num_classes, doa_weight=10.0, device=device
    ).to(device)

    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint_path = os.path.join(output_dir, "ckpt", "last_model.pt")
    
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('loss', float('inf'))
        print(f"   Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    elif args.resume:
        print(f"  Resume flag set but checkpoint not found at {checkpoint_path}")
        print("   Starting from scratch...")
    
    balance_types = not args.no_balance
    
    train_dataset = PretrainDataset(
        args.precomputed_root, split='train', 
        single_source_only=False,
        balance_augmentation_types=balance_types
    )
    print(f"Training dataset size: {len(train_dataset)} samples")
    val_dataset = PretrainDataset(
        args.precomputed_root, split='val', 
        single_source_only=False,
        balance_augmentation_types=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Optimizer & Scheduler
    if hasattr(loss_fn, 'parameters'):
        optimizer = optim.Adam(
            list(model.parameters()) + list(loss_fn.parameters()),
            lr=args.lr
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    if args.resume and os.path.exists(checkpoint_path):
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("   Optimizer state restored")
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )
    
    # log
    log_mode = "a" if (args.resume and os.path.exists(checkpoint_path)) else "w"
    train_log_f = open(os.path.join(output_dir, "log", "train_log.txt"), log_mode)
    val_log_f = open(os.path.join(output_dir, "log", "val_log.txt"), log_mode)
    
    if args.resume and log_mode == "a":
        train_log_f.write(f"\n# Resumed training from epoch {start_epoch}\n")
        val_log_f.write(f"\n# Resumed training from epoch {start_epoch}\n")

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_sed, train_doa = train_epoch(
            model, train_loader, optimizer, loss_fn, device, 
            epoch=epoch, log_f=train_log_f
        )
        
        val_loss, val_sed, val_doa = validate(
            model, val_loader, loss_fn, device
        )
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        print(f"    SED: Train={train_sed:.4f}, Val={val_sed:.4f}")
        print(f"    DOA: Train={train_doa:.4f}, Val={val_doa:.4f}")

        if hasattr(loss_fn, 'get_task_weights'):
            weights = loss_fn.get_task_weights()   
        train_log_f.write(f"{epoch}, {train_loss}, {train_sed}, {train_doa}\n")
        train_log_f.flush()
        val_log_f.write(f"{epoch}, {val_loss}, {val_sed}, {val_doa}\n")
        val_log_f.flush()
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(output_dir, "ckpt", "best_model.pt"))
            print("Best model saved")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, os.path.join(output_dir, "ckpt", "last_model.pt"))
    
    train_log_f.close()
    val_log_f.close()
    
    print(f"\nTraining completed! Best val loss: {best_val_loss:.4f}")
    print(f"   Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
