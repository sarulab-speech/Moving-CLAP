import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import argparse
import copy
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from moving_clap_model import MovingCLAP
from loss import CLAPLoss, CLAPLossWrapper, MovingCLAPLoss
from moving_caption import meta_to_caption_moving
from util import set_seed

from clap_dataset import (
    precomputed_collate_fn,
    BalancedAugmentationBatchSampler,
    NonSpatialDataset,
    SpatialDataset,
)

def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, use_amp=False, scaler=None):
    model.train()
    
    total_loss = 0
    loss_num = 0

    loss_components_sum = {
        'clap': 0.0, 'swap': 0.0, 'spatial': 0.0, 'total': 0.0
    }
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (waveforms, metas) in enumerate(pbar):
        # waveforms: (batch, 2, time)
        # metas: [batch]

        waveforms = waveforms.to(device, non_blocking=True)
        
        captions = [meta_to_caption_moving(meta) for meta in metas]
        
        optimizer.zero_grad()
              
        # Mixed Precision Training
        if use_amp and scaler is not None:
            with autocast():
                clap_output = model(audio=waveforms, text=captions, return_temporal=True)
                loss, components = loss_fn(clap_output, metas, model.logit_scale, 
                                           text_encoder_fn=model.encode_text, return_components=True)       
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            clap_output = model(audio=waveforms, text=captions, return_temporal=True)
            loss, components = loss_fn(clap_output, metas, model.logit_scale, 
                                       text_encoder_fn=model.encode_text, return_components=True)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        loss_num += 1
        
        for key in loss_components_sum:
            if key in components:
                loss_components_sum[key] += components[key]

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'clap': f'{components["clap"]:.3f}',
            'spat': f'{components.get("spatial", 0.0):.3f}',
            'swap': f'{components["swap"]:.3f}'
        })
        
        # Spacific logging
        if batch_idx % 2000 == 0 and batch_idx > 0:
            avg_comps = {k: v/loss_num for k, v in loss_components_sum.items()}
            print(f"\n  Batch {batch_idx}/{len(dataloader)}:")
            print(f"    Total: {avg_comps['total']:.4f} | CLAP: {avg_comps['clap']:.4f} | "
                  f"Spatial: {avg_comps['spatial']:.4f} | Swap: {avg_comps['swap']:.4f}")
            print(f"    Temperature: {model.logit_scale.exp().item():.4f}")
    
    avg_loss = total_loss / loss_num
    avg_components = {k: v/loss_num for k, v in loss_components_sum.items()}
    
    print(f"\n Epoch {epoch} Training Summary:")
    print(f"   Total Loss: {avg_loss:.4f}")
    print(f"   CLAP Loss: {avg_components['clap']:.4f}")
    print(f"   Spatial Contrastive Loss: {avg_components['spatial']:.4f}")
    print(f"   Swap Loss: {avg_components['swap']:.4f}")
    print(f"   Temperature: {model.logit_scale.exp().item():.4f}")

    return avg_loss, avg_components


def validate_epoch(model, dataloader, loss_fn, device, use_amp=False):
    model.eval()
    
    total_loss = 0
    loss_num = 0
    
    loss_components_sum = {
        'clap': 0.0, 'swap': 0.0, 'spatial': 0.0, 'total': 0.0
    }

    with torch.no_grad():
        for batch_idx, (waveforms, metas) in enumerate(tqdm(dataloader, desc="Validation")):
            waveforms = waveforms.to(device, non_blocking=True)
            captions = [meta_to_caption_moving(meta) for meta in metas]
            
            # Mixed Precision Training
            if use_amp:
                with autocast():
                    clap_output = model(audio=waveforms, text=captions, return_temporal=True)
                    loss, components = loss_fn(clap_output, metas, model.logit_scale, 
                                              text_encoder_fn=model.encode_text, return_components=True)
            else:
                clap_output = model(audio=waveforms, text=captions, return_temporal=True)
                loss, components = loss_fn(clap_output, metas, model.logit_scale, 
                                          text_encoder_fn=model.encode_text, return_components=True)
            
            total_loss += loss.item()
            loss_num += 1

            for key in loss_components_sum:
                if key in components:
                    loss_components_sum[key] += components[key]
    
    avg_loss = total_loss / loss_num
    avg_components = {k: v/loss_num for k, v in loss_components_sum.items()}
    
    print(f"\n Validation Summary:")
    print(f"   Total Loss: {avg_loss:.4f}")
    print(f"   CLAP Loss: {avg_components['clap']:.4f}")
    print(f"   Spatial Contrastive Loss: {avg_components['spatial']:.4f}")
    print(f"   Swap Loss: {avg_components['swap']:.4f}")
    
    return avg_loss, avg_components


def main():
    parser = argparse.ArgumentParser(description='Train MovingSpatialCLAP')
    # Phase selection
    parser.add_argument('--phase', type=int, required=True, choices=[0, 1],
                       help='Training phase: 0=Non-spatial, 1 = All Spatial Audio')
    # Data
    parser.add_argument('--precomputed_root', type=str,
                       default='output_moving/simulated_spatial_sound',
                       help='Root directory of precomputed chunked data')
    parser.add_argument('--pretrain_checkpoint', type=str, default=None,
                       help='Path to pretrained checkpoint to resume from')
    
    # Non-spatial data options (Phase 0)
    parser.add_argument('--non_spatial_root', type=str,
                       default='output_moving/non_spatial_precomputed',
                       help='Root directory of non-spatial precomputed data')
    
    # Model architecture
    parser.add_argument('--seldnet_path', type=str,
                       default='pretrain_spatial_encoder/output_beats_seldnet/ckpt/last_model.pt',
                       help='Path to pretrained SELD encoder')
    parser.add_argument('--hidden_dim', type=int, default=768,
                       help='Hidden dimension for fusion layers')
    parser.add_argument('--joint_embed_dim', type=int, default=512,
                       help='Joint embedding dimension')
    parser.add_argument('--fusion_type', type=str, default='cross_attn',
                          choices=['cross_attn', 'none'],
                          help='Type of fusion mechanism between BEATs and spatial features')
    parser.add_argument('--pooling_type', type=str, default='adaptive',
                       choices=['adaptive', 'none'],
                       help='Type of temporal pooling (adaptive or none(mean))')
    parser.add_argument('--use_text_attention', action='store_true', default=False,
                       help='Use text attention mechanism for DOA phrases')
    parser.add_argument('--spatial_bias_value', type=float, default=5.0,
                       help='Bias value added to spatial features before fusion at SBA')
    parser.add_argument('--unfreeze_spatial_encoder', action='store_true', default=False,
                       help='Unfreeze SELDNet spatial encoder weights during training')
    parser.add_argument('--unfreeze_pretrained_encoders', action='store_true', default=False,
                       help='Unfreeze pretrained encoders (e.g., BEATs) during training')
    
    # PretrainedSED options
    parser.add_argument('--pretrained_model_name', type=str, default='BEATs',
                       help='Pretrained model name (BEATs, etc.)')
    parser.add_argument('--unfreeze_beats_layers', type=int, default=2,
                       help='Number of BEATs top layers to unfreeze (0=all frozen)')
    # Spatial-CLAP pretrained weights
    parser.add_argument('--load_spatial_clap', action='store_true',
                       help='Load pretrained Spatial-CLAP weights (text_encoder)')
    # Training
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--num_train_samples', type=int, default=None,
                       help='Number of training samples to use (default: all)')
    parser.add_argument('--num_val_samples', type=int, default=None,
                       help='Number of validation samples to use (default: all)')
    parser.add_argument('--patience', type=int, default=None,
                       help='Patience for early stopping')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--start_epoch', type=int, default=0)
    
    # Performance optimization
    parser.add_argument('--use_amp', action='store_true',
                       help='Use Automatic Mixed Precision training (faster, less memory)')
    
    # Batch sampling strategy
    parser.add_argument('--batch_sampler', type=str, default='balanced',
                       choices=['balanced', 'random'],
                       help='Batch sampling strategy: '
                            'balanced=mix aug types per batch, '
                            'random=standard random sampling')
    # Loss weights
    parser.add_argument('--swap_weight', type=float, default=0.3,
                       help='Swap Loss weight')
    parser.add_argument('--spatial_contrastive_weight', type=float, default=0.01,
                       help='Spatial contrastive loss weight')
    parser.add_argument('--spatial_loss_start_epoch', type=int, default=0,
                       help='Epoch to start spatial contrastive loss (0=from beginning)')
    parser.add_argument('--swap_loss_text_margin', type=float, default=0.3,
                       help='Margin for text swap penalty (default: 0.3)')
    parser.add_argument('--spatial_loss_skip_mixed', action='store_true', default=False,
                       help='Skip mixed sources in spatial contrastive loss (single sources only)')
    parser.add_argument('--swap_loss_audio_margin', type=float, default=0.2,
                       help='Margin for audio swap penalty')
    
    # Data filtering
    parser.add_argument('--exclude_aug_types', type=str, nargs='*', default=['mixed_mov_mov'],
                       help='Augmentation types to exclude from training')
    # Output
    parser.add_argument('--output_dir', type=str, default='output_clap')

    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed()
    
    # Output directory
    phase_dir = os.path.join(args.output_dir, f'phase{args.phase}')
    os.makedirs(phase_dir, exist_ok=True)
    os.makedirs(os.path.join(phase_dir, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(phase_dir, 'log'), exist_ok=True)
    
    print("="*80)
    print(f"CLAP Curriculum Learning - Phase {args.phase}")
    print("="*80)
    
    phase_desc = {
        0: "Phase 0: Non-spatial data only",
        1: "Phase 1: All Spatial data (single + mixed)"
    }
    print(phase_desc[args.phase])
    print("="*80)
    
    # Dataset setup based on phase
    if args.phase == 0:
        train_dataset = NonSpatialDataset(args.non_spatial_root, 'train')
        val_dataset = NonSpatialDataset(args.non_spatial_root, 'val')
    elif args.phase == 1:
        train_dataset = SpatialDataset(args.precomputed_root, 'train', aug_types='all', 
                                    exclude_types=args.exclude_aug_types)
        val_dataset = SpatialDataset(args.precomputed_root, 'val', aug_types='all',
                                    exclude_types=args.exclude_aug_types)

        print("\n Training Dataset Statistics:")
        train_stats = train_dataset.get_augmentation_stats()
        for aug_type, count in sorted(train_stats.items()):
            print(f"   {aug_type}: {count}")
        
        print("\n Validation Dataset Statistics:")
        val_stats = val_dataset.get_augmentation_stats()
        for aug_type, count in sorted(val_stats.items()):
            print(f"   {aug_type}: {count}")
    
    # limiting samples if specified
    if args.num_train_samples is not None and args.num_train_samples < len(train_dataset):
        indices = list(range(min(args.num_train_samples, len(train_dataset))))
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(f"Using {len(train_dataset)} training samples (limited from original)")
    
    if args.num_val_samples is not None and args.num_val_samples < len(val_dataset):
        indices = list(range(min(args.num_val_samples, len(val_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, indices)
        print(f"Using {len(val_dataset)} validation samples (limited from original)")
    
    # DataLoader with batch sampler selection
    if args.phase == 1 and not isinstance(train_dataset, torch.utils.data.Subset):
        if args.batch_sampler == 'balanced':
            print("\nUsing BalancedAugmentationBatchSampler")
            train_sampler = BalancedAugmentationBatchSampler(
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                drop_last=True
            )
        else:
            train_sampler = None
        
        if train_sampler is not None:
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                num_workers=args.num_workers,
                collate_fn=precomputed_collate_fn,
                pin_memory=True
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=precomputed_collate_fn,
                pin_memory=True,
                drop_last=True
            )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=precomputed_collate_fn,
            pin_memory=True,
            drop_last=True
        )
    
    # Validation loader
    if args.phase == 1 and not isinstance(val_dataset, torch.utils.data.Subset):
        if args.batch_sampler == 'balanced':
            print("\nUsing BalancedAugmentationBatchSampler for validation")
            val_sampler = BalancedAugmentationBatchSampler(
                val_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                drop_last=False
            )
        else:
            val_sampler = None
        
        if val_sampler is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_sampler=val_sampler,
                num_workers=args.num_workers,
                collate_fn=precomputed_collate_fn,
                pin_memory=True
            )
        else:
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=precomputed_collate_fn,
                pin_memory=True,
                drop_last=False
            )
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=precomputed_collate_fn,
            pin_memory=True,
            drop_last=False
        )
    
    print(f"\nDataLoaders ready:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Model
    freeze_spatial_encoder = not args.unfreeze_spatial_encoder
    freeze_pretrained_encoders = not args.unfreeze_pretrained_encoders
    print("\nInitializing model...")
    model = MovingCLAP(
        joint_embed_shape=args.joint_embed_dim,
        seldnet_path=args.seldnet_path,
        freeze_spatial_encoder=freeze_spatial_encoder,
        freeze_pretrained_encoders=freeze_pretrained_encoders,
        hidden_dim=args.hidden_dim,
        pretrained_model_name=args.pretrained_model_name,
        fusion_type=args.fusion_type,
        pooling_type=args.pooling_type,
        use_text_attention=args.use_text_attention,
        spatial_bias_value=args.spatial_bias_value,
    ).to(device)
    
    # Unfreeze BEATs top layers if specified
    if args.unfreeze_beats_layers > 0:
        model.unfreeze_beats_top_layers(num_layers=args.unfreeze_beats_layers)
        model.get_trainable_params_summary()
    
    # Weight loading strategy
    if args.pretrain_checkpoint and not args.resume:
        if not os.path.exists(args.pretrain_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.pretrain_checkpoint}")
        
        checkpoint = torch.load(args.pretrain_checkpoint, map_location=device)
        
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        else:
            state_dict = checkpoint
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"Successfully loaded ALL {len(state_dict)} parameters from {args.pretrain_checkpoint}")
        except RuntimeError as e:
            print(f"Strict loading failed: {e}")    
    elif args.load_spatial_clap and not args.resume:
        print("Loading pretrained Spatial-CLAP weights")
        model.load_spatial_clap_pretrained()

    # Loss function setup based on phase
    if args.phase == 0:
        # Phase 0: CLAPLoss
        loss_fn = CLAPLossWrapper(CLAPLoss()).to(device)
    else:
        # Phase 1: MovingCLAPLoss
        initial_spatial_weight = args.spatial_contrastive_weight
        if args.phase == 1 and args.spatial_loss_start_epoch > 0:
            initial_spatial_weight = 0.0
        loss_fn = MovingCLAPLoss(
            swap_weight=args.swap_weight,
            spatial_contrastive_weight=initial_spatial_weight,
            swap_loss_text_margin=args.swap_loss_text_margin,
            spatial_loss_skip_mixed=args.spatial_loss_skip_mixed,
            swap_loss_audio_margin=args.swap_loss_audio_margin,
            hidden_dim=args.hidden_dim,
            embed_dim=args.joint_embed_dim
        ).to(device)
        print(f"\n Using MovingCLAPLoss")
        print(f"   Swap weight: {args.swap_weight}")
        print(f"   Swap loss audio margin: {args.swap_loss_audio_margin}")
        print(f"   Swap loss text margin: {args.swap_loss_text_margin}")
        print(f"   Spatial contrastive weight: {initial_spatial_weight}" + 
              (f" (will be {args.spatial_contrastive_weight} after epoch {args.spatial_loss_start_epoch})" 
               if args.spatial_loss_start_epoch > 0 else ""))
        print(f"   Spatial loss skip mixed: {args.spatial_loss_skip_mixed}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )
    
    # Resume from checkpoint
    start_epoch = args.start_epoch
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Initialize GradScaler for AMP
    scaler = None
    if args.use_amp:
        scaler = GradScaler()
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = start_epoch
    patience_counter = 0
        
    log_mode = 'a' if args.resume else 'w'
    train_log_f = open(os.path.join(phase_dir, 'log', 'train_log.txt'), log_mode)
    val_log_f = open(os.path.join(phase_dir, 'log', 'val_log.txt'), log_mode)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        spatial_loss_enabled = True
        if args.phase == 1 and args.spatial_loss_start_epoch > 0:
            if epoch + 1 == args.spatial_loss_start_epoch:
                # From next epoch, enable Spatial Loss
                loss_fn.spatial_contrastive_weight = args.spatial_contrastive_weight
                print(f"Spatial Contrastive Loss ENABLED (weight={args.spatial_contrastive_weight})")
                train_log_f.write(f"\n=== Spatial Loss enabled at epoch {epoch+1} ===\n")
                # Reset best_val_loss baseline for Spatial Loss phase
                best_val_loss = float('inf')
                patience_counter = 0
            spatial_loss_enabled = (epoch + 1 >= args.spatial_loss_start_epoch)
        
        if args.phase == 0:
            spatial_loss_enabled = False
        
        # Training
        train_loss, train_components = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch+1,
            use_amp=args.use_amp,
            scaler=scaler
        )
        print(f"Train Loss: {train_loss:.4f}")
        train_log_f.write(f"Epoch {epoch+1}: total={train_loss:.4f} clap={train_components['clap']:.4f} "
                          f"spatial={train_components['spatial']:.4f} "
                          f"swap={train_components['swap']:.4f}\n")
        train_log_f.flush()
        
        # Validation
        val_loss, val_components = validate_epoch(
            model, val_loader, loss_fn, device,
            use_amp=args.use_amp
        )
        print(f"Val Loss: {val_loss:.4f}")
        val_log_f.write(f"Epoch {epoch+1}: total={val_loss:.4f} clap={val_components['clap']:.4f} "
                        f"spatial={val_components['spatial']:.4f} "
                        f"swap={val_components['swap']:.4f}\n")
        val_log_f.flush()
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.2e}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'phase': args.phase,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss
        }
        
        # Regular checkpoint
        torch.save(checkpoint, os.path.join(phase_dir, 'ckpt', 'last_model.pt'))
        
        # Best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            torch.save(checkpoint, os.path.join(phase_dir, 'ckpt', 'best_model.pt'))
            print(f"New best model! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1      
        # Early stopping
        if not args.patience is None and patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            print(f"Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
            break
        print("="*80)
    
    train_log_f.close()
    val_log_f.close()
    
    print("\n" + "="*80)
    print(f"Phase {args.phase} Training Complete")
    print("="*80)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {phase_dir}/ckpt/best_model.pt")

if __name__ == '__main__':
    main()
