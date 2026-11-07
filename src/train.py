import os

import fire
import numpy as np
import torch as t
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
import time
import torch.nn.functional as F
from src import Jet
from src.util import bits_per_dim
from src.util import cosine_schedule
from src.util import FeatureWindowDataset, RawFeatureDataset

def train(
    jet_config={}, # configuration bassed to jet
    image_batch_size=4,
    batch_size=64, 
    accumulate_steps=16, # number of mini batches accumulated before optimizer update, this means effective batch size is 64X16
    epochs=50,
    warmup_percentage=0.1, # percentage of training steps for learning rate warmup
    max_grad_norm=1.0, # max grad allowed before clipping
    learning_rate=3e-4,
    weight_decay=1e-5, # AdamW weight decay
    adam_betas=(0.9, 0.95), # coffecients of AdamW
    features_path_train="/mnt/wsl/l/train_features", # feature maps train path
    features_path_valid="/mnt/wsl/l/val_features", # feature maps validation path
    num_workers=2,
    device="cuda",
    checkpoint_path="jet_dino.pt",
):
    t.set_float32_matmul_precision("medium")
    print(f"[INFO] Creating train and val dataset")
    
    dataset = RawFeatureDataset(features_path_train)
    val_dataset = RawFeatureDataset(features_path_valid)
    
    # Get configuration from a sample
    sample_shape = dataset[0].shape # e.g., (64, 128, 384) -> (H, W, C)
    channels = sample_shape[-1]
    
    # FIX 1: dim_count must represent a SINGLE patch passed to the model
    patch_dim_count = 3 * 3 * channels 

    print(f"[INFO] Creating train and val dataloader")
    dataloader = DataLoader(
        dataset, batch_size=image_batch_size, num_workers=num_workers, drop_last=True,
        pin_memory=t.cuda.is_available(), shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=image_batch_size, num_workers=num_workers,
        pin_memory=t.cuda.is_available(), shuffle=False,
    )

    print(f"[INFO] Creating JeT model based on given configuration")
    orig_model = Jet(**jet_config).to(device)
    print(f"# Params: {sum(p.numel() for p in orig_model.parameters() if p.requires_grad):,}")
    
    model = orig_model if device == "cpu" else t.compile(orig_model)

    print(f"[INFO] Configuring optimizer and lr scheduler")
    opt = t.optim.AdamW(model.parameters(), betas=adam_betas, weight_decay=weight_decay, lr=learning_rate)
    
    # Calculate exact total patch mini-batches to fix scheduler and progress bar totals
    # Unfold on (64, 128) with kernel 3, stride 1 yields: (64-3+1) * (128-3+1) = 62 * 126 = 7812 patches per image
    h, w, c = sample_shape
    patches_per_image = (h - 3 + 1) * (w - 3 + 1)
    batches_per_image = int(np.ceil(patches_per_image / batch_size))
    
    total_steps = ((len(dataset) * batches_per_image) // accumulate_steps) * epochs
    warmup_steps = int(total_steps * warmup_percentage)
    
    scheduler = cosine_schedule(
        opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps, min_lr_scale=0.01,
    )

    print(f"[INFO] Starting training loop")
    gn = 0.0 # Initialize to avoid reference errors before first update
    
    for epoch in range(epochs):
        # FIX 2: Set accurate total for the gradient update progress bar
        pbar = tqdm(total=total_steps // epochs, desc=f"Epoch {epoch+1}/{epochs}")
        
        steps = 0
        log_bpd, log_nll, log_logdet = 0.0, 0.0, 0.0
        
        for image_feature_map in dataloader:
            image_feature_map = image_feature_map.to(device)
            b, h, w, c = image_feature_map.shape
            
            x_perm = image_feature_map.permute(0, 3, 1, 2)
            patches = F.unfold(x_perm, kernel_size=3, stride=1)
            patches = patches.transpose(1, 2).reshape(-1, c, 3, 3).permute(0, 2, 3, 1).contiguous()
            
            total_image_patches = patches.shape[0]
            for i in range(0, total_image_patches, batch_size):
                x = patches[i : i + batch_size]
                
                # with t.autocast("cuda", dtype=t.float16):
                logits, logdet = model(x)
                # Use the corrected patch dimensions here
                bpd, nll, logdet = bits_per_dim(logits, logdet, patch_dim_count)
                
                bpd_scaled = bpd / accumulate_steps
                bpd_scaled.backward()
                
                log_bpd += bpd.detach().cpu().item() / accumulate_steps
                log_nll += nll.detach().cpu().item() / accumulate_steps
                log_logdet += logdet.detach().cpu().item() / accumulate_steps
                
                steps += 1
                if steps % accumulate_steps == 0:
                    gn = clip_grad_norm_(model.parameters(), max_grad_norm).cpu().item()
                    opt.step()
                    scheduler.step()
                    opt.zero_grad()
                    
                    pbar.update(1)
                    lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix(dict(bpd=log_bpd, nll=log_nll, logdet=log_logdet, gn=gn, lr=lr))
                    
                    log_bpd, log_nll, log_logdet = 0.0, 0.0, 0.0

        pbar.close()
        
        # Save Checkpoint
        model.eval()
        t.save(dict(model=orig_model.state_dict(), opt=opt.state_dict(), epoch=epoch), checkpoint_path)
        
        # Validation Loop
        val_bpd, val_nll, val_logdet = 0.0, 0.0, 0.0
        val_patch_batches = 0
        
        # FIX 3: Fix validation progress bar total
        val_pbar = tqdm(total=len(val_dataset) * batches_per_image, desc="Validating", leave=False)
        
        with t.no_grad():
            for image_feature_map in val_dataloader:
                image_feature_map = image_feature_map.to(device)
                b, h, w, c = image_feature_map.shape
                
                x_perm = image_feature_map.permute(0, 3, 1, 2)
                patches = F.unfold(x_perm, kernel_size=3, stride=1)
                patches = patches.transpose(1, 2).reshape(-1, c, 3, 3).permute(0, 2, 3, 1).contiguous()
                
                total_image_patches = patches.shape[0]
                for i in range(0, total_image_patches, batch_size):
                    x = patches[i : i + batch_size]
                    
                    # with t.autocast("cuda", dtype=t.float16):
                    logits, logdet = model(x)
                    bpd, nll, logdet = bits_per_dim(logits, logdet, patch_dim_count)
                    
                    val_bpd += bpd.detach().cpu().item()
                    val_nll += nll.detach().cpu().item()
                    val_logdet += logdet.detach().cpu().item()
                    val_patch_batches += 1
                    val_pbar.update(1)
                    
        val_pbar.close()
        
        # FIX 4: Divide by total structural micro-batches evaluated, not total images
        val_bpd /= val_patch_batches
        val_nll /= val_patch_batches
        val_logdet /= val_patch_batches
        print(f"Epoch {epoch+1} Validation Metrics - bpd: {val_bpd:.4f}, nll: {val_nll:.4f}, logdet: {val_logdet:.4f}")
        
        model.train()

# if __name__ == "__main__":
#     fire.Fire(train)
