import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoModel
from src import Jet
from src.util import bits_per_dim_per_patch

# Assuming your JeT model architecture is imported like this:
# from src import Jet 

def run_patch_inference(
    image_path: str,
    jet_model_path: str,
    jet_config: dict = {},
    dinov3_model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
    patch_batch_size: int = 512,
    hf_token: str = None
):
    """
    Runs dynamic pipeline inference on a single image without resizing:
    1. Extracts DINOv3 ViT-S/16 spatial features at native resolution in half-precision (fp16).
    2. Dynamically calculates spatial grids to prevent distortion.
    3. Slices features into overlapping 3x3 patch grids.
    4. Runs inference through JeT in optimized batches.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ==========================================
    # 1. Load Pretrained Models
    # ==========================================
    print(f"[INFO] Loading DINOv3 backbone: {dinov3_model_name}...")
    dino_model = AutoModel.from_pretrained(dinov3_model_name, token=hf_token).to(device)
    dino_model.eval()

    print("[INFO] Loading trained JeT model...")
    # Initialize your JeT architecture (replace with your exact initialized class setup)
    orig_jet_model = Jet(**jet_config)
    checkpoint = torch.load(jet_model_path, map_location=device)
    orig_jet_model = orig_jet_model.to(device)
    orig_jet_model.load_state_dict(checkpoint['model'])
    orig_jet_model.eval()

    # Freeze weights
    for p in dino_model.parameters(): p.requires_grad = False
    for p in orig_jet_model.parameters(): p.requires_grad = False

    # ==========================================
    # 2. Preprocess Single Image (No Resize)
    # ==========================================
    # Removed transforms.Resize to preserve the original structural aspect ratio
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    print(f"[INFO] Loading image from {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    # Get original image dimensions dynamically
    orig_w, orig_h = image.size 
    img_tensor = image_transform(image).unsqueeze(0).to(device) # Shape: (1, 3, H, W)

    # ==========================================
    # 3. Dynamic DINOv3 Feature Extraction (Half-Precision)
    # ==========================================
    patch_size = 16
    # Dynamically find the feature grid shape for whatever size image is fed in
    h_patches = orig_h // patch_size
    w_patches = orig_w // patch_size

    print(f"[INFO] Image shape: ({orig_h}x{orig_w}). Expected feature grid: ({h_patches}x{w_patches})")
    print("[INFO] Extracting DINOv3 ViT-S/16 feature map...")
    with torch.no_grad():
        outputs = dino_model(img_tensor)
        
        num_prefix_tokens = 1 + dino_model.config.num_register_tokens 
        patch_tokens = outputs.last_hidden_state[:, num_prefix_tokens:]
        
        B, N, D = patch_tokens.shape # D will be 384 for ViT-S/16
        
        assert N == h_patches * w_patches, f"Expected {h_patches*w_patches} patch tokens, got {N}."
        spatial_features = patch_tokens.reshape(B, h_patches, w_patches, D)


    # ==========================================
    # 4. Extract Overlapping 3x3 Patch Grids
    # ==========================================
    print("[INFO] Extracting overlapping 3x3 grids from feature map...")
    x_perm = spatial_features.permute(0, 3, 1, 2) # (B, D, H_patches, W_patches)
    
    # Extract overlapping 3x3 patch neighborhoods (Stride 1)
    patches = F.unfold(x_perm, kernel_size=3, stride=1)
    
    # Reshape back to matching target layout used in your JeT training configuration:
    # Shape: (Total_Patches, 3, 3, D)
    patches = patches.transpose(1, 2).reshape(-1, D, 3, 3).permute(0, 2, 3, 1).contiguous()
    print(f"[INFO] Total generated 3x3 patch grids to process: {patches.shape}")

    # ==========================================
    # 5. Inference Batches Through JeT Model
    # ==========================================
    print(f"[INFO] Streaming patches through JeT model (Batch Size: {patch_batch_size})...")
    
    C_dim = patches.shape[-1] 
    patch_dim_count = 3 * 3 * C_dim

    print(f"[INFO] Patch Dimension Count for BPD calculation: {patch_dim_count}")
    
    all_logits = []
    all_logdets = []
    all_bpds = []  # Added to track individual patch likelihoods
    all_nlls = []

    with torch.no_grad():
        # Step through by patch_batch_size (e.g., 512) to process non-overlapping chunks
        for i in range(0, patches.shape[0], patch_batch_size):
            x_batch = patches[i : i + patch_batch_size].to(device)

            # 1. Forward pass through JeT
            # logits shape: (B, 3, 3, 384), logdet shape: (B,)
            logits, logdet = orig_jet_model(x_batch)

            # 2. Compute per-patch evaluation metrics (Option A: returns (B,) 1D tensors)
            bpd, nll, logdet_metric = bits_per_dim_per_patch(logits, logdet, patch_dim_count)
            # print(f"[INFO] The bpd shape: {bpd.shape}, the nll shape: {nll.shape}, the logdet shape: {logdet_metric.shape}")

            # 3. Store per-patch values (handles standard & trailing batches cleanly)
            all_logits.append(logits.cpu())
            all_logdets.append(logdet_metric.cpu().reshape(-1))
            all_bpds.append(bpd.cpu().reshape(-1))
            all_nlls.append(nll.cpu().reshape(-1))

    # Concatenate or convert lists into continuous tensors
    final_logits = torch.cat(all_logits, dim=0)
    final_logdets = torch.cat(all_logdets, dim=0)
    final_bpds = torch.cat(all_bpds, dim=0)
    final_nlls = torch.cat(all_nlls, dim=0)
    out_h = h_patches - 3 + 1
    out_w = w_patches - 3 + 1

    # Calculate global structural statistics for this image
    mean_image_bpd = final_bpds.mean().item()

    print(f"[SUCCESS] Inference complete!")
    print(f"-> Average Image Score (BPD): {mean_image_bpd:.4f}")
    print(f"-> Logits Shape: {final_logits.shape}")
    print(f"-> BPD Array Shape: {final_bpds.shape}")
    
    return final_bpds, final_nlls, final_logits, final_logdets, (out_h, out_w)

