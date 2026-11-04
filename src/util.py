import math
import os
from functools import partial
from pathlib import Path

import torch as t
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.distributions import Normal
from torch.optim.lr_scheduler import LambdaLR # pytorch's learning rate scheduler to schedule the lr using a user defined function
from torch.utils.data import Dataset
from torchvision.io import read_image # reads and converts image to tensor


# def get_pretrained(): # returns pretrained model weights
#     pretrained_path = "models/jet_imagenet64x64_200m/jet_imagenet64x64_200m.safetensors"
#     os.makedirs("models", exist_ok=True)

#     if not os.path.exists(pretrained_path):
#         _ = hf_hub_download(
#             repo_id="btrude/jet_imagenet64x64_200m",
#             filename="jet_imagenet64x64_200m.safetensors",
#             local_dir="models/jet_imagenet64x64_200m",
#         )
#     return load_file(pretrained_path)


def _cosine_schedule( # does both warmup and lr decay
    current_step,
    *,
    num_warmup_steps,
    num_training_steps,
    min_lr_scale,
    num_cycles,
):
    if current_step < num_warmup_steps: # return warmup multiplier, if less than number of warmup steps
        return float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(
        min_lr_scale,
        min_lr_scale + 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * (2.0) * progress)) * (1.0 - min_lr_scale),
    )


def cosine_schedule(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    min_lr_scale=0.1,
    num_cycles=0.5,
    last_epoch=-1,
):
    lr_lambda = partial( # creates a new function where the schedule config is fixed
        _cosine_schedule,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_scale=min_lr_scale,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# class ImageDataset(Dataset): # a very simple image dataset maker
#     def __init__(self, root, transform=None):
#         root = Path(root)
#         self.files = [p for p in Path(root).rglob(f"*.png")]
#         self.transform = transform

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, index):
#         image = read_image(self.files[index])
#         if self.transform is not None:
#             image = self.transform(image)
#         return image


# def preprocess(x):
#     x = x * 2 - 1  # 0,1 -> -1,1 (changes the range)
#     x = x + t.empty(x.shape).uniform_(0.0, 1 / 127.5) # dequantisation, since normalising flow needs to be continuous dentisity estimation, so adding uniform noise can dequantise 
#     return x.permute(0, 2, 3, 1) # (B, C; H, W) to (B, H, W, C)


# def postprocess(x): # inverting preprocess
#     return (x + 1) / 2  # -1,1 -> 0,1


def bits_per_dim(logits, logdet, dim_count, pdf=None): # logits is the transformed latent representation z
    if pdf is None:
        pdf = Normal(0.0, 1.0)

    nll = -pdf.log_prob(logits) # (- log(pz(z))) # where each latent value will get its own nll (z of dim B X H X W X C)
    nll = t.sum(
        nll, # + t.log(t.tensor(127.5, device=nll.device)), # we add the density correction for the uniform noise that we added
        dim=tuple(range(1, nll.dim())) # so t.sum sums the values for all dimension except the batch dimension
    )
    nll = nll.unsqueeze(-1) # adds a dimension, so if its (64,) -> (64,1)

    bits = nll - logdet # so bits is basicall nll optimisation equation on the left side (nll of image/ high dim feature grid)
    normalizer = t.log(t.tensor(2, device=nll.device)) * dim_count # Dlog(2) is normaliser

    return (
        bits.mean() / normalizer,
        nll.mean() / normalizer,
        logdet.mean() / normalizer
    )
    
# ================ Dataset for 3X3 feature map patches ====================


class FeatureWindowDataset(Dataset):
    def __init__(self, feature_dir):
        # Store only file paths (costs almost zero RAM)
        self.feature_paths = sorted(Path(feature_dir).rglob("*.pt"))

        if not self.feature_paths:
            raise RuntimeError(f"No feature maps found in {feature_dir}")

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, index):
        # Load single file dynamically into RAM
        feature_path = self.feature_paths[index]
        features = t.load(feature_path, map_location="cpu")

        if isinstance(features, dict):
            features = features["features"]

        features = features.float()

        # Convert to (1, 384, H, W)
        features = features.permute(2, 0, 1).unsqueeze(0)

        # Extract 3x3 windows
        patches = F.unfold(features, kernel_size=3, stride=1)
        patches = patches.squeeze(0).transpose(0, 1)
        patches = patches.reshape(-1, 384, 3, 3).permute(0, 2, 3, 1).contiguous()

        # Returns all patches extracted from this single image
        return patches

# ===================== Dataset (with purity threshold) for class conditional ==============

class FeatureWindowDatasetPurity(Dataset):
    def __init__(self, feature_dir, purity_dir=None, purity_threshold=0.90):
        self.windows = []

        feature_paths = sorted(Path(feature_dir).rglob("*.pt"))

        for feature_path in feature_paths:
            features = t.load(feature_path, map_location="cpu")

            # Expected shape: (H, W, 768)
            if isinstance(features, dict):
                features = features["features"]

            features = features.float()

            # Convert (H, W, D) -> (1, D, H, W)
            features = features.permute(2, 0, 1).unsqueeze(0)

            # Extract every 3x3 neighbourhood.
            # Output shape: (1, D*9, number_of_windows)
            patches = F.unfold(features, kernel_size=3, stride=1)

            # Convert to (number_of_windows, 3, 3, D)
            patches = patches.squeeze(0).transpose(0, 1)
            patches = patches.reshape(-1, 768, 3, 3)
            patches = patches.permute(0, 2, 3, 1).contiguous()

            # Optionally filter windows using the centre-patch purity.
            if purity_dir is not None:
                purity_path = Path(purity_dir) / feature_path.name
                purity = t.load(purity_path, map_location="cpu")

                # Exclude the boundary, matching the window centres.
                centre_purity = purity[1:-1, 1:-1].reshape(-1)
                valid = centre_purity >= purity_threshold

                patches = patches[valid]

            self.windows.append(patches)

        if not self.windows:
            raise RuntimeError(f"No feature windows found in {feature_dir}")

        self.windows = t.cat(self.windows, dim=0)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]
