import os

import fire # allows to execute the python function from command line
import numpy as np
import torch as t
from torch.nn.utils import clip_grad_norm_ # prevents execessively large gradients during training
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from src import Jet
from src.util import bits_per_dim # calculates the normalised nll in bits per dimension
from src.util import cosine_schedule # learning rate scheduler with warmup and cosine decay
from src.util import ImageDataset # Dataset class to load images
from src.util import preprocess # preprocesses the images before training


def train(
    jet_config={}, # configuration bassed to jet
    batch_size=64, 
    accumulate_steps=16, # number of mini batches accumulated before optimizer update, this means effective batch size is 64X16
    epochs=50,
    warmup_percentage=0.1, # percentage of training steps for learning rate warmup
    max_grad_norm=1.0, # max grad allowed before clipping
    learning_rate=3e-4,
    weight_decay=1e-5, # AdamW weight decay
    adam_betas=(0.9, 0.95), # coffecients of AdamW
    images_path_train="/mnt/wsl/l/imagenet_downsampled", # images train path
    images_path_valid="/mnt/wsl/l/imagenet_downsampled", # images validation path
    num_workers=16,
    device="cuda:0",
    checkpoint_path="jet_imagenet.pt", # path where checkpoint is saved
):
    t.set_float32_matmul_precision("medium") # precision for pytorch matrix multiplication set to medium

    # converts images to dataset (from utils)
    dataset = ImageDataset(
        images_path_train,
        transform=v2.ToDtype(t.float32, scale=True), # transform to float 32 and scale it to range 0 to 1
    )
    val_dataset = ImageDataset(
        images_path_valid,
        transform=v2.ToDtype(t.float32, scale=True),
    )

    dim_count = np.prod(dataset[0].shape) # total number of scalar dimensions in one sample, for 64 X 64 X 3 image is 12288
    
    # train and val dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True, # drops the last incomplete batch
        pin_memory=True,
        pin_memory_device=device,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    orig_model = Jet(**jet_config) # initialize Jet model
    orig_model = orig_model.to(device)
    print(orig_model)
    print(f"# Params: {sum(p.numel() for p in orig_model.parameters() if p.requires_grad):,}") # number of trainable params
    model = t.compile(orig_model) # uses PyTorch 2's compilation addon, makes training faster basically

    opt = t.optim.AdamW(
        model.parameters(),
        betas=adam_betas,
        weight_decay=weight_decay, # penalises large model weights
        lr=learning_rate,
    )

    total_steps = (len(dataset) // (batch_size * accumulate_steps)) * epochs
    warmup_steps = int(total_steps * warmup_percentage)

    scheduler = cosine_schedule( # learning rate scheduler, lr decays along cosine shaped curve after warmup
        opt,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_scale=0.01,
    )

    steps = 0 # number of mini batches accumulated since last grad update
    log_bpd = 0.0 # accumulated bits per dim
    log_nll = 0.0 # accumulated nll
    log_logdet = 0.0 # accumulated log determinant

    for epoch in range(epochs):
        pbar = tqdm(dataloader, total=len(dataloader) // accumulate_steps) # progress bar

        for x in dataloader:
            x = preprocess(x).to(device) # preprocesses the batch

            with t.autocast("cuda", dtype=t.bfloat16): # automatic mixed precision, torch selects operations which can go with lower precision bf16
                logits, logdet = model(x) # forward pass
                bpd, nll, logdet = bits_per_dim(logits, logdet, dim_count) # bpd is the training objective

            bpd /= accumulate_steps # divide the training objective by accumulated steps, 
            bpd.backward() # grads will be accumulated 16 times

            nll /= accumulate_steps
            logdet /= accumulate_steps
            log_bpd += bpd.detach().cpu().item()
            log_nll += nll.detach().cpu().item()
            log_logdet += logdet.detach().cpu().item()
            steps += 1

            if steps % accumulate_steps == 0:
                gn = clip_grad_norm_(model.parameters(), max_grad_norm).cpu().item() # clips the gradient
                opt.step() # model param update
                scheduler.step()
                opt.zero_grad()
                pbar.update(1)
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(dict( # displays the metrics alongside the progress bar
                    bpd=log_bpd,
                    nll=log_nll,
                    logdet=log_logdet,
                    gn=gn,
                    lr=lr,
                ))
                steps = 0
                log_bpd = 0.0
                log_nll = 0.0
                log_logdet = 0.0

        model.eval()

        t.save(
            dict(
                model=orig_model.state_dict(),
                opt=opt.state_dict(),
                epoch=epoch,
            ),
            checkpoint_path,
        )

        val_bpd = 0.0
        val_nll = 0.0
        val_logdet = 0.0

        pbar = tqdm(val_dataloader, total=len(val_dataloader))
        with t.no_grad():
            for x in val_dataloader:
                x = preprocess(x).to(device)

                with t.autocast("cuda", dtype=t.bfloat16):
                    logits, logdet = model(x)
                    bpd, nll, logdet = bits_per_dim(logits, logdet, dim_count)

                val_bpd += bpd.detach().cpu().item() # no grad accumulation, no weight update
                val_nll += nll.detach().cpu().item()
                val_logdet += logdet.detach().cpu().item()
                pbar.update(1)

        val_bpd /= len(val_dataloader)
        val_nll /= len(val_dataloader)
        val_logdet /= len(val_dataloader)
        print(f"validation metrics - bpd: {val_bpd:.2f}, nll: {val_nll:.2f}, logdet: {val_logdet:.2f}")
        model.train()


if __name__ == "__main__":
    fire.Fire(train)
